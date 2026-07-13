from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from app.rag_runtime import iter_configured_corpus_sources, resolve_rag_runtime


APP_DIR = Path(__file__).resolve().parent
API_DIR = APP_DIR.parent
DEFAULT_PROCESSED_PATH = API_DIR / "data" / "processed" / "eureka_interpretations.jsonl"
DEFAULT_LAW_SOURCE_PATHS = (
    API_DIR / "data" / "laws" / "processed" / "excise_act_DU_2026_412.jsonl",
    API_DIR / "data" / "laws" / "processed" / "vat_act_DU_2025_775_codified_2026-05-05.jsonl",
    API_DIR / "data" / "laws" / "processed" / "vat_act_DU_2025_775.jsonl",
    API_DIR / "data" / "laws" / "processed" / "cit_act_DU_2026_554.jsonl",
    API_DIR / "data" / "laws" / "processed" / "pit_act_DU_2025_163.jsonl",
    API_DIR / "data" / "laws" / "processed" / "pit_act_DU_2026_592.jsonl",
    API_DIR / "data" / "laws" / "processed" / "pcc_act_DU_2026_191.jsonl",
    API_DIR / "data" / "laws" / "processed" / "inheritance_gift_tax_act_DU_2026_478.jsonl",
    API_DIR / "data" / "laws" / "processed" / "tax_ordinance_DU_2026_622.jsonl",
    API_DIR / "data" / "laws" / "processed" / "local_taxes_act_DU_2025_707.jsonl",
    API_DIR / "data" / "laws" / "processed" / "tax_treaties_core.jsonl",
    API_DIR / "data" / "laws" / "processed" / "ksef_2_0_current_bundle.jsonl",
    API_DIR / "data" / "laws" / "processed" / "family_foundation_primary_bundle.jsonl",
    API_DIR / "data" / "processed" / "cbosa_nsa_fsk_judgments.jsonl",
)
DEFAULT_RAG_DB_PATH = API_DIR / "data" / "processed" / "eureka_rag.sqlite3"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

WHITESPACE_RE = re.compile(r"\s+")
QUERY_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}")
EMBEDDING_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{2,}")
EXACT_PROVISION_REFERENCE_RE = re.compile(
    r"\bart\.\s*\d+[a-z]?"
    r"(?:\s*(?:ust\.\s*\d+[a-z]?|§\s*\d+[a-z]?))?"
    r"(?:\s*pkt\s*\d+[a-z]?)?"
    r"(?:\s*lit\.\s*[a-z])?",
    re.IGNORECASE,
)
SECTION_BREAK_RE = re.compile(r"\n{2,}")
BOILERPLATE_SECTION_RE = re.compile(
    r"\n(?=(?:Pouczenie o funkcji ochronnej interpretacji|Funkcja ochronna interpretacji|"
    r"Prawo do wniesienia skargi|Pouczenie o prawie do wniesienia skargi na interpretację|"
    r"Mają Państwo prawo do zaskarżenia|Skargę do Sądu wnosi się|Podstawa prawna dla wydania interpretacji))",
    re.IGNORECASE,
)
INTERPRETATION_PROCEDURAL_QUERY_RE = re.compile(
    r"\b(funkcj\w* ochronn\w*|ochron\w* interpretacj\w*|pouczeni\w*|"
    r"prawo do wniesienia skargi|zaskarżeni\w* interpretacj\w*|art\.\s*14k|art\.\s*14na|art\.\s*57a)\b",
    re.IGNORECASE,
)
PROCEDURAL_INTERPRETATION_CHUNK_RE = re.compile(
    r"\b(pouczenie o funkcji ochronnej interpretacji|funkcję ochronną interpretacji|"
    r"pouczenie o prawie do wniesienia skargi|mają państwo prawo do zaskarżenia|"
    r"skargę do sądu wnosi się|skarga na interpretację indywidualną|"
    r"podstawa prawna dla wydania interpretacji|interpretacje indywidualne są wydawane w indywidualnych sprawach|"
    r"nie stanowi źródła powszechnie obowiązującego prawa)\b",
    re.IGNORECASE,
)
INTERPRETATION_MERITS_SECTION_RE = re.compile(
    r"\b(ocena stanowiska|uzasadnienie interpretacji indywidualnej|"
    r"odnosząc się zatem do powyższych różnic|w omawianej sprawie konieczne jest zatem ustalenie|"
    r"zatem w ww\. sytuacji)\b",
    re.IGNORECASE,
)
INTERPRETATION_TAXPAYER_POSITION_RE = re.compile(
    r"\b(państwa stanowisko w sprawie|zdaniem wnioskodawcy|uzasadnienie stanowiska)\b",
    re.IGNORECASE,
)
QUESTION_HEADING_RE = re.compile(r"^Pytani(?:e|a)$", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(
    r"^(?:Opis|Stan faktyczny|Zdarzenie przyszłe|Państwa stanowisko|Ocena stanowiska|"
    r"Uzasadnienie|Rozstrzygnięcie|Dodatkowe informacje|Informacja o zakresie)",
    re.IGNORECASE,
)
SIGNATURE_FAMILY_RE = re.compile(r"\b(KD[A-Z]{2}\d?)\b")
ARTICLE_ID_RE = re.compile(r"\bart\.?\s*(\d+)([a-z]*)\b", re.IGNORECASE)
ARTICLE_SPLIT_SUFFIX_RE = re.compile(r"\bart\.?\s*(\d+)\s+([a-z]{1,4})\b", re.IGNORECASE)
BARE_ARTICLE_SPLIT_SUFFIX_RE = re.compile(r"\b(\d{2,4})\s+([a-z]{1,4})\b", re.IGNORECASE)
OFFLINE_SPLIT_RE = re.compile(r"\boffline\s+(\d{1,3})\b", re.IGNORECASE)
GENERAL_STATUTE_QUERY_RE = re.compile(
    r"\b(co jest|co oznacza|co rozumieć|jak ustawa definiuje|jakie zasady|gdzie uregulowano|"
    r"kiedy .* nie jest|czy .* ma obowiązek)\b",
    re.IGNORECASE,
)
JUDGMENT_SIGNATURE_RE = re.compile(r"\b(?:(I{1,3})\s+)?FSK\s+(\d+)\s*/\s*(\d{2,4})\b", re.IGNORECASE)
JUDGMENT_CHAMBER_RE = re.compile(r"\b(I{1,3})\s+FSK\b", re.IGNORECASE)
JUDGMENT_INTENT_RE = re.compile(r"\b(wyrok\w*|orzecze(?:nie|nia|ń)|orzecznictw\w*|nsa|fsk|sąd\w*|skarg\w* kasacyjn\w*)\b", re.IGNORECASE)
JUDGMENT_ONLY_CONTEXT_RE = re.compile(r"\b(znajdź|pokaż|podaj|dobierz|wyszukaj)\b.{0,80}\b(wyrok\w*|orzecze(?:nie|nia|ń)|orzecznictw\w*)\b", re.IGNORECASE)
KSEF_QUERY_RE = re.compile(r"\bksef\b|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*", re.IGNORECASE)
KSEF_FOREIGN_SALE_QUERY_RE = re.compile(
    r"(wielk\w* brytani\w*|\buk\b|zea|niderland\w*|holandi\w*|państw\w* trzec\w*|"
    r"poza terytorium kraju|poza polsk\w*|terytorium państwa członkowskiego inne niż terytorium kraju|"
    r"sprzedaż lokaln\w*|lokaln\w* sprzedaż|towar\w*.*(?:magazyn|terytorium|znajd))",
    re.IGNORECASE,
)
KSEF_FOREIGN_SALE_MERITS_RE = re.compile(
    r"(polskie regulacje w zakresie fakturowania znajdą również zastosowanie|"
    r"art\.\s*106a\s*pkt\s*2|art\.\s*106ga\s*ust\.\s*1|art\.\s*106ga\s*ust\.\s*2|"
    r"zobowiązani również wystawić faktury ustrukturyzowane|"
    r"nie znajdą zastosowania wyłączenia ustawowe|art\.\s*106gb\s*ust\.\s*4|"
    r"faktura ustrukturyzowana.*udostępnian\w* nabywcy w sposób z nim uzgodniony)",
    re.IGNORECASE,
)
KSEF_FOREIGN_SALE_STATUTE_TARGETS: tuple[tuple[str, str], ...] = (
    ("VAT", "106a"),
    ("VAT", "106b"),
    ("VAT", "106ga"),
    ("VAT", "106gb"),
)
KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS: tuple[str, ...] = ("679542",)
KSEF_OUTSIDE_DEDUCTION_INTERPRETATION_DOCUMENT_IDS: tuple[str, ...] = (
    "695345",
    "695471",
    "695355",
    "695403",
    "694097",
    "693430",
    "693595",
    "693598",
    "693253",
    "693103",
    "696243",
    "696177",
    "693053",
    "694474",
    "692135",
    "695412",
)
KSEF_CURRENT_BUNDLE_DOCUMENT_IDS: tuple[str, ...] = (
    "ksef-2-0-current-law-dzu-2025-1203-transition",
    "ksef-2-0-offline24-operational-modes",
    "ksef-2-0-scope-fixed-establishment-and-foreign-buyers",
    "ksef-2-0-corrections-and-vat-deduction",
)
FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS: tuple[str, ...] = (
    "family-foundation-primary-ufr-art-5-27-29",
    "family-foundation-primary-cit-24q-24r",
    "family-foundation-primary-pit-beneficiary-rates",
    "family-foundation-primary-vat-related-party-transactions",
)
DEBT_ASSUMPTION_INTERPRETATION_DOCUMENT_IDS: tuple[str, ...] = ("695395", "678370")
HOUSING_RELIEF_TEMPORARY_RENTAL_INTERPRETATION_DOCUMENT_IDS: tuple[str, ...] = ("691376",)
MORTGAGE_SETTLEMENT_INTERPRETATION_DOCUMENT_IDS: tuple[str, ...] = ("688486", "693529")

# The search corpus uses both abbreviations and their expanded legal names.  Keeping
# these aliases here makes a user's natural query match either form without an LLM.
QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bksef\b|\bkrajow(?:y|ego) system(?:u)? e[ -]?faktur", re.IGNORECASE), ("Krajowy System e-Faktur", "faktura ustrukturyzowana")),
    (
        re.compile(
            r"(\bksef\b|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*).{0,120}"
            r"(wielk\w* brytani\w*|\buk\b|zea|niderland\w*|holandi\w*|poza terytorium kraju|państw\w* trzec\w*|sprzedaż lokaln\w*)|"
            r"(wielk\w* brytani\w*|\buk\b|zea|niderland\w*|holandi\w*|poza terytorium kraju|państw\w* trzec\w*|sprzedaż lokaln\w*).{0,120}"
            r"(\bksef\b|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*)",
            re.IGNORECASE,
        ),
        (
            "art. 106a pkt 2",
            "art. 106b ust. 1 pkt 1",
            "art. 106ga ust. 1",
            "art. 106ga ust. 2",
            "art. 106gb ust. 4",
            "miejscem świadczenia jest terytorium państwa trzeciego",
            "faktura ustrukturyzowana jest udostępniana nabywcy w sposób z nim uzgodniony",
            "polskie regulacje w zakresie fakturowania znajdą zastosowanie",
        ),
    ),
    (re.compile(r"\bwht\b|podatek u źr[óo]dła|withholding", re.IGNORECASE), ("WHT", "podatek u źródła")),
    (
        re.compile(r"(\bwht\b|podatek u źr[óo]dła|withholding).{0,180}\b(dywidend\w*|odsetk\w*|zarządz\w*|zarzadz\w*|beneficial owner|pay and refund)\b|\b(dywidend\w*|odsetk\w*|zarządz\w*|zarzadz\w*|beneficial owner|pay and refund)\b.{0,180}(\bwht\b|podatek u źr[óo]dła|withholding)", re.IGNORECASE),
        (
            "art. 21",
            "art. 22",
            "art. 22c",
            "art. 26",
            "rzeczywisty właściciel",
            "certyfikat rezydencji",
            "należyta staranność",
            "pay and refund",
        ),
    ),
    (
        re.compile(r"\b(pay and refund|2 mln|2 000 000|art\.\s*26\s*ust\.\s*2e|próg\w* 2 mln|limit\w* 2 mln)\b", re.IGNORECASE),
        (
            "art. 26 ust. 2e",
            "nadwyżka ponad kwotę 2 000 000 zł",
            "na rzecz tego samego podatnika",
            "art. 21 ust. 1 pkt 1",
            "art. 22 ust. 1",
        ),
    ),
    (
        re.compile(r"\b(dywidend\w*|parent-subsidiary|holdingow\w*)\b", re.IGNORECASE),
        (
            "art. 22 ust. 4",
            "art. 22 ust. 4a",
            "art. 22c",
            "dywidendy",
            "udziałów bezpośrednio nieprzerwanie przez okres dwóch lat",
        ),
    ),
    (
        re.compile(r"\b(odsetk\w*|interest and royalties|beneficial owner)\b", re.IGNORECASE),
        (
            "art. 21 ust. 3",
            "art. 21 ust. 3c",
            "rzeczywistym właścicielem",
            "odsetek",
        ),
    ),
    (
        re.compile(r"\b(zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|management fee)\b", re.IGNORECASE),
        (
            "art. 21 ust. 1 pkt 2a",
            "świadczeń doradczych księgowych badania rynku usług prawnych reklamowych zarządzania i kontroli",
            "zyski przedsiębiorstw",
            "zakład",
        ),
    ),
    (
        re.compile(r"\b(transgraniczn\w*|nierezydent\w*|zakład\w*|zaklad\w*|umow\w* o unikaniu podwójnego opodatkowania|upo|podmiot\w* z państw\w* trzec\w*|podmiot\w* zagraniczn\w*)\b", re.IGNORECASE),
        (
            "umowa o unikaniu podwójnego opodatkowania",
            "zyski przedsiębiorstw",
            "zakład",
            "miejsce zamieszkania lub siedziba",
            "nierezydent",
        ),
    ),
    (re.compile(r"\bpcc\b|podatek od czynności cywilnoprawnych", re.IGNORECASE), ("PCC", "podatek od czynności cywilnoprawnych")),
    (
        re.compile(
            r"(\bprzejęci\w*\s+dług\w*|\bprzejęci\w*\s+zobowiązan\w*|\bzwolnieni\w*\s+z dług\w*|"
            r"\bzmian\w*\s+dłużnik\w*|\bzmian\w*\s+dluznik\w*|\bzgod\w*\s+wierzyciel\w*|"
            r"art\.\s*519|art\.\s*520|art\.\s*521|art\.\s*508)\b",
            re.IGNORECASE,
        ),
        (
            "przejęcie długu",
            "przejęcie zobowiązania",
            "zwolnienie z długu",
            "zgoda wierzyciela",
            "art. 519 Kodeksu cywilnego",
            "art. 508 Kodeksu cywilnego",
        ),
    ),
    (
        re.compile(
            r"(\bulg\w*\s+mieszkaniow\w*|\bwłasn\w*\s+cele\s+mieszkaniow\w*|\bwlasn\w*\s+cele\s+mieszkaniow\w*|"
            r"\bczasow\w*\s+wynaj\w*|\bwynajmow\w*\s+lokal\w*|\bwynajem\w*\s+zakupion\w*\s+lokal\w*|"
            r"\bart\.\s*52i\b|\bzaniechan\w*\s+pobor\w*|\bkredyt\w*\s+mieszkaniow\w*|\bumorzen\w*\s+zadłużen\w*|\bumorzen\w*\s+zadluzen\w*)\b",
            re.IGNORECASE,
        ),
        (
            "ulga mieszkaniowa",
            "własne cele mieszkaniowe",
            "czasowy wynajem",
            "zaniechanie poboru",
            "art. 52i",
            "kredyt mieszkaniowy",
            "umorzenie zadłużenia",
        ),
    ),
    (
        re.compile(
            r"(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b).{0,220}"
            r"(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bmałżonk\w*|\bmalzonk\w*)|"
            r"(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bmałżonk\w*|\bmalzonk\w*).{0,220}"
            r"(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b)",
            re.IGNORECASE,
        ),
        (
            "art. 7 ust. 2",
            "przysługiwało w całości lub w części prawo do obniżenia kwoty podatku należnego",
            "art. 15 ust. 1",
            "podatnikami są osoby wykonujące samodzielnie działalność gospodarczą",
            "art. 106b",
            "podatnik jest obowiązany wystawić fakturę",
            "art. 10 ust. 1 pkt 8",
            "przed upływem pół roku licząc od końca miesiąca",
            "art. 10 ust. 2 pkt 4",
            "art. 14 ust. 2 pkt 19",
            "rzeczy ruchome wykorzystywane na podstawie umowy leasingu",
            "art. 23 ust. 1 pkt 46",
            "wydatki i składki w wysokości 20 %",
            "art. 23 ust. 1 pkt 46a",
            "25 % poniesionych wydatków",
            "art. 22 ust. 1d",
            "zgodnie z art. 11 ust. 2-2b został określony przychód",
            "art. 2 ust. 1 pkt 3",
            "przychodów podlegających przepisom o podatku od spadków i darowizn",
            "art. 4a",
            "zgłoszą nabycie własności rzeczy lub praw majątkowych",
            "art. 6 ust. 1 pkt 4",
            "art. 9 ust. 1 pkt 1",
            "art. 14 ust. 3 pkt 1",
        ),
    ),
    (
        re.compile(r"\bfundacj\w*\s+rodzinn\w*\b", re.IGNORECASE),
        (
            "fundacja rodzinna",
            "świadczenie dla beneficjenta",
            "ukryte zyski",
            "art. 24q",
        ),
    ),
    (
        re.compile(r"(\bfundacj\w*\s+rodzinn\w*\b.*\b(pożyczk\w*|spółk\w*|spol[kc]\w*)\b|\b(pożyczk\w*|spółk\w*|spol[kc]\w*)\b.*\bfundacj\w*\s+rodzinn\w*\b)", re.IGNORECASE),
        (
            "fundacja rodzinna",
            "pożyczki mogą być udzielane",
            "spółkom kapitałowym, w których fundacja rodzinna posiada udziały albo akcje",
            "beneficjentom",
        ),
    ),
    (
        re.compile(r"(\bfundacj\w*\s+rodzinn\w*\b.*\b(wypłat\w*|świadczeni\w*|beneficjent\w*)\b|\b(beneficjent\w*|syn\w* fundator\w*)\b.*\bfundacj\w*\s+rodzinn\w*\b)", re.IGNORECASE),
        (
            "fundacja rodzinna",
            "świadczenie dla beneficjenta",
            "art. 21 ust. 1 pkt 157",
            "art. 21 ust. 49",
            "grupy zerowej",
        ),
    ),
    (
        re.compile(r"(pożyczk\w*.*(?:vat|towar[óo]w i usług|art\.\s*2\s*pkt\s*4)|(?:vat|towar[óo]w i usług|art\.\s*2\s*pkt\s*4).*pożyczk\w*)", re.IGNORECASE),
        (
            "umowa pożyczki",
            "pożyczka zwolniona z VAT",
            "wyłączenie z opodatkowania PCC",
            "art. 2 pkt 4 lit. b",
            "usługi finansowe",
            "pożyczkodawca podatnik VAT",
            "pożyczkobiorca",
        ),
    ),
    (
        re.compile(r"((sprzedaż|sprzedaży|nabycie|zakup).*nieruchomości.*(?:vat|pcc|czynności cywilnoprawnych|art\.\s*2\s*pkt\s*4)|(?:vat|pcc|czynności cywilnoprawnych|art\.\s*2\s*pkt\s*4).*nieruchomości)", re.IGNORECASE),
        (
            "sprzedaż nieruchomości",
            "wyłączenie z opodatkowania PCC",
            "art. 2 pkt 4 lit. a",
            "opodatkowanie podatkiem od towarów i usług",
            "umowa sprzedaży nieruchomości",
        ),
    ),
    (
        re.compile(
            r"(\bgrunt\w*\b|\bdziałk\w*\b|\bdzialk\w*\b).{0,220}"
            r"(\bpełnomocnictw\w*\b|\bpelnomocnictw\w*\b|\bdzierżaw\w*\b|\bdzierzaw\w*\b|"
            r"\bdeweloper\w*\b|\bwarunk\w* zabudow\w*\b|\bpozwoleni\w* na budow\w*\b)|"
            r"(\bpełnomocnictw\w*\b|\bpelnomocnictw\w*\b|\bdzierżaw\w*\b|\bdzierzaw\w*\b|"
            r"\bdeweloper\w*\b|\bwarunk\w* zabudow\w*\b|\bpozwoleni\w* na budow\w*\b).{0,220}"
            r"(\bgrunt\w*\b|\bdziałk\w*\b|\bdzialk\w*\b)",
            re.IGNORECASE,
        ),
        (
            "art. 15 ust. 1",
            "art. 15 ust. 2",
            "art. 43 ust. 1 pkt 9",
            "art. 2 pkt 33",
            "tereny budowlane",
            "decyzja o warunkach zabudowy",
            "dzierżawa",
            "pozarolnicza działalność gospodarcza",
            "art. 5a pkt 6",
            "art. 10 ust. 1 pkt 3",
            "art. 10 ust. 1 pkt 8",
            "art. 2 pkt 4",
            "art. 106a",
            "art. 106b",
            "art. 106ga",
            "art. 106gb",
        ),
    ),
    (
        re.compile(r"(zwolnien\w*.*(?:zakup|nabycie).*nieruchomości|(?:zakup|nabycie).*nieruchomości.*zwolnien\w*)", re.IGNORECASE),
        (
            "art. 9 pkt 17",
            "zwolnienie z podatku od czynności cywilnoprawnych",
            "zakup pierwszej nieruchomości mieszkalnej",
            "budynek mieszkalny",
            "własne potrzeby mieszkaniowe",
        ),
    ),
    (
        re.compile(r"(po połowie .*mieszkani|przekazuje .*udzia[łl]|oddając[ay].*udzia[łl]|otrzymuje .*sp[łl]at|przejmuje .*kredyt.*hipoteczn|byli partnerzy)", re.IGNORECASE),
        (
            "odpłatne zniesienie współwłasności",
            "spłata na rzecz byłego współwłaściciela",
            "nabycie ponad dotychczasowy udział",
            "obowiązek podatkowy ciąży na nabywającym",
            "art. 1 ust. 1 pkt 1 lit. f",
            "art. 4 pkt 5",
        ),
    ),
    (re.compile(r"sp[óo]łk[ai] holdingow", re.IGNORECASE), ("Polska Spółka Holdingowa", "PSH")),
    (re.compile(r"ograniczon(?:y|ego) obowi[ąa]zek podatkow", re.IGNORECASE), ("ograniczony obowiązek podatkowy", "nierezydent", "rezydencja podatkowa")),
    (
        re.compile(r"(183\s*dni|centrum interes[óo]w|dochod\w* zagraniczn\w*|pracowa[łl].*za granic[ąa]|mieszka[łl].*za granic[ąa])", re.IGNORECASE),
        (
            "ograniczony obowiązek podatkowy",
            "miejsce zamieszkania dla celów podatkowych",
            "dochody osiągnięte na terytorium Polski",
            "umowa o unikaniu podwójnego opodatkowania",
        ),
    ),
    (re.compile(r"skala podatkow|wyb[óo]r formy opodatkowania", re.IGNORECASE), ("skala podatkowa", "forma opodatkowania", "oświadczenie")),
    (re.compile(r"esto[ńn]sk(?:i|iego)?\s+cit|rycza[łl]t(?:em)? od dochod[óo]w sp[óo][łl]ek", re.IGNORECASE), ("estoński CIT", "ryczałt od dochodów spółek")),
    (re.compile(r"\bip\s*box\b", re.IGNORECASE), ("IP Box", "kwalifikowane prawo własności intelektualnej")),
    (
        re.compile(r"(ip\s*box|kwalifikowan\w* praw\w* własności intelektualnej|autorsk\w* praw\w* majątkow\w* do program\w* komputerow\w*|działalno\w* badawczo-rozwojow\w*|tworzeni\w* i rozwijani\w* oprogramowani\w*|wskaźnik nexus)", re.IGNORECASE),
        (
            "IP Box",
            "kwalifikowane prawo własności intelektualnej",
            "autorskie prawo do programu komputerowego",
            "działalność badawczo-rozwojowa",
            "wskaźnik nexus",
            "dochód z kwalifikowanego prawa własności intelektualnej",
        ),
    ),
    (re.compile(r"\bexit\s+tax\b", re.IGNORECASE), ("exit tax", "dochody z niezrealizowanych zysków", "podatek od dochodów z niezrealizowanych zysków")),
    (
        re.compile(r"(uchylon\w*.*wyrok.*decyzj\w*.*(?:drugiej|ii) instancji|decyzj\w*.*(?:drugiej|ii) instancji.*uchylon\w*)", re.IGNORECASE),
        (
            "Uchylono zaskarżony wyrok i decyzję II instancji",
            "Treść wyniku Uchylono zaskarżony wyrok i decyzję II instancji",
            "decyzję II instancji",
        ),
    ),
    (
        re.compile(r"(sponsor\w*.*faktur\w*|faktur\w*.*sponsor\w*|koszt\w* uzyskania przychod\w*.*faktur\w*)", re.IGNORECASE),
        (
            "sponsorowanie",
            "zakwestionowane faktury",
            "koszty uzyskania przychodów",
        ),
    ),
    (
        re.compile(r"(art\.\s*70.*(?:par\.|§)\s*6.*pkt\s*1.*amortyz|amortyz.*art\.\s*70.*(?:par\.|§)\s*6.*pkt\s*1)", re.IGNORECASE),
        (
            "art. 70 § 6 pkt 1 Ordynacji podatkowej",
            "art. 70 par. 6 pkt 1",
            "amortyzacja",
            "podatek dochodowy od osób prawnych",
        ),
    ),
    (
        re.compile(r"(własn\w* cele mieszkaniow\w*|rat\w* kredyt\w*|sprzeda[żz].*mieszkani\w*|zaci[ąa]gni[ęe]t\w*.*przed sprzeda[żz])", re.IGNORECASE),
        (
            "własne cele mieszkaniowe",
            "spłata kredytu wraz z odsetkami",
            "przed dniem uzyskania przychodu",
            "art. 21 ust. 25 pkt 2",
        ),
    ),
    (
        re.compile(r"(dwoma kredytami|raty wraz z odsetkami|drugie mieszkanie|kolejne raty kredyt[óo]w)", re.IGNORECASE),
        (
            "wspólnie z małżonkiem",
            "małżeństwo",
            "pożyczka",
            "art. 21 ust. 25 pkt 2",
            "pit-39",
            "ustawowym terminie 3 lat",
            "każdą ratę kredytu",
        ),
    ),
    (
        re.compile(r"(art\.\s*10\s*ust\.\s*1\s*pkt\s*8|art\.\s*10\s*ust\.\s*5|art\.\s*30e|sprzeda[żz].*mieszkani\w*.*202[67]|odpłatn\w* zbyci\w* nieruchomości)", re.IGNORECASE),
        (
            "odpłatne zbycie nieruchomości",
            "pięcioletni termin podatkowy",
            "data nabycia nieruchomości",
            "art. 10 ust. 1 pkt 8",
            "art. 10 ust. 5",
            "art. 30e",
        ),
    ),
    (
        re.compile(r"(pierwsz\w* wyp[łl]at\w* (?:świadczeni|emerytur)|przyzna[łl].*emerytur|prawo do emerytur|odpraw\w* emerytaln\w*)", re.IGNORECASE),
        (
            "wypłata wynagrodzenia przed pierwszą emeryturą",
            "korekta zeznania",
            "art. 21 ust. 1 pkt 154",
            "ulga dla pracującego seniora",
        ),
    ),
    (
        re.compile(r"(sprzeda[żz] udział[óo]w|sp[óo][łl]k[ai] komandytow\w*|przekszta[łl]con\w* w sp[óo][łl]k[ęe] z o\.o\.|nie wnosi[łl].*nowych wk[łl]ad[óo]w|udzia[łl]owi w maj[ąa]tku)", re.IGNORECASE),
        (
            "skutki podatkowe sprzedaży udziałów w spółce przekształconej",
            "art. 23 ust. 1 pkt 38",
            "spółka z ograniczoną odpowiedzialnością",
            "koszt uzyskania przychodów",
            "kapitał własny spółki komandytowej",
            "wartość majątku spółki komandytowej",
            "objętych w wyniku przekształcenia",
        ),
    ),
    (
        re.compile(r"(ulga\s*4\+|ulga\s*4\s*plus|co najmniej czworg\w* dzieci|czworo dzieci|wychowywani\w* co najmniej czworg\w* dzieci|władz\w* rodzicielsk\w*)", re.IGNORECASE),
        (
            "ulga 4+",
            "wychowywanie co najmniej czworga dzieci",
            "zwolnienie dla rodzin 4+",
            "wykonywanie władzy rodzicielskiej",
        ),
    ),
    (
        re.compile(r"(ugod\w* z bankiem|kredyt\w* frankow\w*|świadczeni\w* nienależn\w*|zwrot .*rat|umorzeni\w* części kredytu hipotecznego|koszt\w* zastępstwa procesowego)", re.IGNORECASE),
        (
            "ugoda z bankiem dotycząca kredytu hipotecznego",
            "zwrot świadczenia nienależnego",
            "umorzenie części kredytu hipotecznego",
            "zwrot kosztów zastępstwa procesowego",
            "zwrot własnych rat kredytu",
        ),
    ),
)
STATUTE_QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(definic|definiuj|oznacza|pojęci|pojęcia)\b", re.IGNORECASE), ("użyte w ustawie określenia oznaczają",)),
    (re.compile(r"\b(deklarac|wpłac|zapłat|termin)\b", re.IGNORECASE), ("deklaracje obliczać wpłacać miesięczne okresy rozliczeniowe",)),
    (re.compile(r"\b(odzyska|zwrot|eksport|wywóz)\b", re.IGNORECASE), ("zwrot akcyzy dostawa wewnątrzwspólnotowa eksport",)),
    (re.compile(r"\b(przedmiot\w* opodatkowani\w*|doch[óo]d stanowi|zysk[óo]w kapita[łl]ow\w*)\b", re.IGNORECASE), ("przedmiotem opodatkowania podatkiem dochodowym jest dochód",)),
    (re.compile(r"\b(przychod\w* należn\w*|przychod\w*|nie zalicza si[eę] do przychod[óo]w)\b", re.IGNORECASE), ("przychodami są", "za przychody związane z działalnością gospodarczą", "do przychodów nie zalicza się")),
    (re.compile(r"\b(koszt\w* bezpośredni\w*|koszt\w* bezposredni\w*|potr[ąa]ci[ćc].*po zako[nń]czeniu roku)\b", re.IGNORECASE), ("koszty uzyskania przychodów bezpośrednio związane z przychodami",)),
    (re.compile(r"\b(darowizn\w*|odliczeni\w* od podstawy opodatkowania)\b", re.IGNORECASE), ("po odliczeniu", "darowizn przekazanych na cele określone",)),
    (re.compile(r"\b(z[łl]e d[łl]ugi|90 dni od dnia up[łl]ywu terminu zap[łl]aty)\b", re.IGNORECASE), ("może być zmniejszona o zaliczaną do przychodów należnych wartość wierzytelności", "upłynęło 90 dni od dnia upływu terminu zapłaty")),
    (re.compile(r"\b(podatek u źr[óo]dła|odsetek|należno[śs]ci licencyjn\w*|know-how)\b", re.IGNORECASE), ("podatek dochodowy z tytułu uzyskanych", "z odsetek", "know-how")),
    (re.compile(r"\b(esto[ńn]sk(?:i|iego)?\s+cit|rycza[łl]t(?:em)? od dochod[óo]w sp[óo][łl]ek)\b", re.IGNORECASE), ("opodatkowaniu ryczałtem może podlegać podatnik", "przepisów niniejszego rozdziału nie stosuje się do")),
    (re.compile(r"\b(zwolnion\w* od podatku|zwolnienia podmiotow\w*|katalog podmiot\w* zwolnion\w*)\b", re.IGNORECASE), ("zwalnia się od podatku",)),
    (re.compile(r"\b(stawk\w*\s*9\s*%|9\s*proc\.?|9-proc)\b", re.IGNORECASE), ("9 % podstawy opodatkowania",)),
    (re.compile(r"\b(ip\s*box|kwalifikowan\w* praw\w* własności intelektualnej|obowiązk\w* ewidencyjn\w*)\b", re.IGNORECASE), ("kwalifikowane prawo własności intelektualnej", "ewidencji rachunkowej",)),
    (re.compile(r"\b(exit\s+tax|niezrealizowanych zysk\w*)\b", re.IGNORECASE), ("podatek od dochodów z niezrealizowanych zysków", "dochód z niezrealizowanych zysków")),
    (
        re.compile(
            r"(\bprzejęci\w*\s+dług\w*|\bprzejęci\w*\s+zobowiązan\w*|\bzwolnieni\w*\s+z dług\w*|"
            r"\bzmian\w*\s+dłużnik\w*|\bzmian\w*\s+dluznik\w*|\bzgod\w*\s+wierzyciel\w*|"
            r"art\.\s*519|art\.\s*520|art\.\s*521|art\.\s*508)\b",
            re.IGNORECASE,
        ),
        (
            "przejęcie długu",
            "przejęcie zobowiązania",
            "zwolnienie z długu",
            "zgoda wierzyciela",
            "art. 519 Kodeksu cywilnego",
            "art. 508 Kodeksu cywilnego",
        ),
    ),
    (
        re.compile(
            r"(\bulg\w*\s+mieszkaniow\w*|\bwłasn\w*\s+cele\s+mieszkaniow\w*|\bwlasn\w*\s+cele\s+mieszkaniow\w*|"
            r"\bczasow\w*\s+wynaj\w*|\bwynajmow\w*\s+lokal\w*|\bwynajem\w*\s+zakupion\w*\s+lokal\w*|"
            r"\bart\.\s*52i\b|\bzaniechan\w*\s+pobor\w*|\bkredyt\w*\s+mieszkaniow\w*|\bumorzen\w*\s+zadłużen\w*|\bumorzen\w*\s+zadluzen\w*)\b",
            re.IGNORECASE,
        ),
        (
            "ulga mieszkaniowa",
            "własne cele mieszkaniowe",
            "czasowy wynajem",
            "zaniechanie poboru",
            "art. 52i",
            "kredyt mieszkaniowy",
            "umorzenie zadłużenia",
        ),
    ),
)
STATUTORY_CONCEPTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(terytor\w* kraju|defini\w*|oznacza\w*|ilekroć)\b", re.IGNORECASE), ("ilekroć w dalszych przepisach jest mowa",)),
    (re.compile(r"\b(defini\w* działalno\w* gospodarc\w*|działalno\w* gospodarc\w*)\b", re.IGNORECASE), ("wszelką działalność producentów handlowców lub usługodawców",)),
    (re.compile(r"\b(podatnik\w*|działalno\w* gospodarc\w*|organ\w* władzy)\b", re.IGNORECASE), ("podatnikami są", "wykonujące samodzielnie działalność gospodarczą")),
    (re.compile(r"\b(czynno\w*.*podleg\w*|podleg\w*.*vat|przedmiot\w* opodatkowan\w*)\b", re.IGNORECASE), ("opodatkowaniu podatkiem podlegają",)),
    (re.compile(r"\b(dostaw\w* towar\w*)\b", re.IGNORECASE), ("przeniesienie prawa do rozporządzania towarami jak właściciel",)),
    (re.compile(r"\b(świadczen\w* usług\w*)\b", re.IGNORECASE), ("każde świadczenie które nie stanowi dostawy towarów",)),
    (re.compile(r"\b(obowiązk\w* podatkow\w*|zaliczk\w*|usług\w* ciągł\w*)\b", re.IGNORECASE), ("obowiązek podatkowy powstaje", "otrzymano całość lub część zapłaty")),
    (re.compile(r"\b(podstaw\w* opodatkowan\w*|rabat\w*|obniżk\w* cen\w*)\b", re.IGNORECASE), ("podstawą opodatkowania jest wszystko co stanowi zapłatę", "podstawę opodatkowania obniża się")),
    (re.compile(r"\b(stawk\w*.*vat|vat.*stawk\w*|stawka podstawow\w*|stawka obniżon\w*)\b", re.IGNORECASE), ("stawka podatku wynosi", "dla towarów i usług wymienionych")),
    (re.compile(r"\b(zwolnieni\w*.*vat|vat.*zwolnieni\w*|zwolnienie podmiotow\w*)\b", re.IGNORECASE), ("zwalnia się od podatku", "zwalnia się od podatku sprzedaż")),
    (re.compile(r"\b(odlicz\w*|podatek naliczon\w*)\b", re.IGNORECASE), ("przysługuje prawo do obniżenia kwoty podatku należnego o kwotę podatku naliczonego",)),
    (re.compile(r"\b(wyłączeni\w*.*odlicz\w*|nie można odliczyć|faktur\w*.*nieistniej\w*)\b", re.IGNORECASE), ("nie stanowią podstawy do obniżenia podatku należnego", "obniżenia kwoty lub zwrotu różnicy podatku należnego nie stosuje się")),
    (re.compile(r"\b(zwrot\w*.*vat|nadwyżk\w*.*naliczon\w*)\b", re.IGNORECASE), ("kwota podatku naliczonego jest wyższa od kwoty podatku należnego", "zwrotu różnicy na rachunek bankowy")),
    (re.compile(r"\b(faktur\w*|fakturow\w*)\b", re.IGNORECASE), ("podatnik jest obowiązany wystawić fakturę", "faktura powinna zawierać")),
    (re.compile(r"\b(rejestrac\w*.*vat|zarejestrow\w*)\b", re.IGNORECASE), ("przed dniem wykonania pierwszej czynności", "zgłoszenie rejestracyjne")),
    (re.compile(r"\b(deklarac\w*.*vat|wpłaci\w*.*vat|zapłat\w*.*vat)\b", re.IGNORECASE), ("są obowiązani składać", "obowiązani bez wezwania naczelnika urzędu skarbowego")),
    (re.compile(r"\b(kas\w* rejestrując\w*|ewidencj\w*.*sprzedaż\w*)\b", re.IGNORECASE), ("są obowiązani prowadzić ewidencję sprzedaży przy zastosowaniu kas rejestrujących",)),
    (re.compile(r"\b(wewnątrzwspólnotow.*nabyci|\bwnt\b)\b", re.IGNORECASE), ("wewnątrzwspólnotowe nabycie towarów",)),
    (re.compile(r"\b(wewnątrzwspólnotow.*dostaw|\bwdt\b)\b", re.IGNORECASE), ("wewnątrzwspólnotowa dostawa towarów",)),
    (re.compile(r"\b(dostaw\w*.*wewnątrzwspólnotow|wewnątrzwspólnotow.*dostaw)\b", re.IGNORECASE), ("wywóz towarów z terytorium kraju",)),
    (re.compile(r"\b(import\w* towar\w*)\b", re.IGNORECASE), ("przywóz towarów z terytorium państwa trzeciego",)),
    (re.compile(r"\b(miejsce.*dostaw\w*.*(wysył|transport)|dostaw\w*.*(wysył|transport))\b", re.IGNORECASE), ("miejscem dostawy towarów wysyłanych lub transportowanych",)),
    (re.compile(r"\b(usług[ai].*podatnik.*innego państwa|\bb2b\b)\b", re.IGNORECASE), ("miejscem świadczenia usług w przypadku świadczenia usług na rzecz podatnika",)),
    (re.compile(r"\b(konsument\w*|niebędąc\w* podatnik\w*|\bb2c\b)\b", re.IGNORECASE), ("miejscem świadczenia usług na rzecz podmiotów niebędących podatnikami",)),
    (re.compile(r"\b(nieruchomoś|usług[ai].*nieruchomo)\b", re.IGNORECASE), ("miejscem świadczenia usług związanych z nieruchomościami",)),
    (re.compile(r"\b(podzielon[ae] płatno|split payment)\b", re.IGNORECASE), ("mechanizm podzielonej płatności",)),
    (re.compile(r"\b(marż[ay].*towar|towarów używanych)\b", re.IGNORECASE), ("podstawą opodatkowania podatkiem jest marża",)),
    (re.compile(r"\b(towar\w* używan\w*.*zwolnion|zwolnion\w*.*towar\w* używan\w*)\b", re.IGNORECASE), ("dostawę towarów używanych wyłącznie na cele działalności zwolnionej",)),
    (re.compile(r"\b(kwot\w* podatk\w* naliczon\w*)\b", re.IGNORECASE), ("kwotę podatku naliczonego stanowi suma",)),
    (re.compile(r"\b(termin\w*.*wystaw\w* faktur|do kiedy.*faktur\w*)\b", re.IGNORECASE), ("fakturę wystawia się nie później niż",)),
    (re.compile(r"\b(faktur\w* koryguj\w*)\b", re.IGNORECASE), ("w przypadku gdy po wystawieniu faktury",)),
    (re.compile(r"\b(duplikat\w* faktur|faktur\w*.*(zagin|zniszcz))\b", re.IGNORECASE), ("w przypadku gdy faktura ulegnie zniszczeniu albo zaginie",)),
    (re.compile(r"\b(ewidencj\w*.*vat|ewidencj\w*.*jpk|\bjpk\b)\b", re.IGNORECASE), ("podatnicy są obowiązani prowadzić ewidencję",)),
    (re.compile(r"\b(pust\w* faktur|faktur\w*.*wykazanym.*vat|wystawc\w* faktur\w*.*zapłaci)\b", re.IGNORECASE), ("w przypadku gdy osoba prawna wystawi fakturę w której wykaże kwotę podatku", "jest obowiązana do jego zapłaty")),
    (re.compile(r"\b(stawk\w*.*zero.*eksport|eksport\w*.*stawk\w*.*0)\b", re.IGNORECASE), ("w eksporcie towarów stawka podatku wynosi 0",)),
)
RANKING_STOPWORDS = {
    "aby", "albo", "bez", "będzie", "była", "było", "był", "czy", "dla", "jego", "jej", "jest",
    "jeżeli", "która", "które", "który", "lub", "może", "muszę", "musi", "nad", "nie", "oraz",
    "pod", "po", "przez", "przy", "się", "tak", "tego", "tym", "ustawy", "ustawie", "wartości",
    "wraz", "wtedy", "został", "została", "zostały", "związku",
}
DOMAIN_MARKERS: dict[str, tuple[str, ...]] = {
    "vat": (
        "vat",
        "ksef",
        "faktur",
        "odliczen",
        "sprzedaż",
        "sprzedaz",
        "podatku od towarów i usług",
        "podatku od towarow i uslug",
        "towarów i usług",
        "towarow i uslug",
        "ustawa o vat",
        "ustawy o vat",
    ),
    "cit": (
        "cit",
        "estońsk",
        "estonsk",
        "spółk",
        "spolk",
        "holding",
        "podatku dochodowym od osób prawnych",
        "podatku dochodowym od osob prawnych",
        "dochodowym od osób prawnych",
        "dochodowym od osob prawnych",
        "ustawa o cit",
        "ustawy o cit",
    ),
    "pit": (
        "pit",
        "ryczałt",
        "ryczalt",
        "ulga",
        "rezydenc",
        "podatku dochodowym od osób fizycznych",
        "podatku dochodowym od osob fizycznych",
        "dochodowym od osób fizycznych",
        "dochodowym od osob fizycznych",
        "ustawa o pit",
        "ustawy o pit",
    ),
    "pcc": (
        "pcc",
        "czynności cywilnoprawnych",
        "czynnosci cywilnoprawnych",
        "podatek od czynności cywilnoprawnych",
        "podatek od czynnosci cywilnoprawnych",
        "ustawa o pcc",
        "ustawy o pcc",
        "aport",
        "współwłas",
        "wspolwlas",
    ),
    "sd": ("podatek od spadków i darowizn", "podatek od spadkow i darowizn", "sd-z2", "spadków", "spadkow", "darowizn"),
    "nieruchomości": (
        "nieruchomoś",
        "nieruchomos",
        "u.p.o.l",
        "podatki lokalne",
        "budynk",
        "budowl",
        "grunt",
        "powierzchni użytkow",
        "powierzchni uzytkow",
    ),
    "wht": ("wht", "źródła", "zrodla", "withholding", "beneficial", "certyfikat rezydencji", "nierezydent", "zakład", "zaklad", "upo"),
    "akcyza": ("akcyza", "akcyzow", "skład podatkowy", "sklad podatkowy"),
    "ordynacja": (
        "ordynac",
        "przedawnien",
        "nadpłat",
        "nadplat",
        "oprocentowan",
        "zaległoś",
        "zaleglos",
        "odsetk",
        "korekt",
        "deklarac",
        "interpretacj",
        "odwołan",
        "odwolan",
        "wznowien",
        "rygor",
        "postępowan",
        "postepowan",
        "ordynacja podatkowa",
        "ordynacji podatkowej",
    ),
}
MECHANISM_RULES: dict[str, tuple[str, ...]] = {
    "invoice_outside_ksef": ("poza ksef", "faktura papierowa", "faktura pdf"),
    "input_vat_deduction": ("odliczyć vat", "prawo do odliczenia"),
    "ksef_foreign_local_sale": (
        "sprzedaż lokalna",
        "lokalna sprzedaż",
        "poza terytorium kraju",
        "państwo trzecie",
        "wielka brytania",
        "uk",
        "towar znajduje się",
        "towary znajdują się",
        "miejsce dostawy poza terytorium kraju",
        "art. 106a pkt 2",
        "art. 106ga",
        "art. 106gb ust. 4",
        "faktura ustrukturyzowana jest udostępniana nabywcy",
    ),
    "ksef_offline_input_vat_deduction": (
        "offline24",
        "tryb offline24",
        "art. 106nda",
        "art 106nda",
        "106nda",
        "art. 106nh",
        "art 106nh",
        "106nh",
        "przydzielenia numeru identyfikującego",
        "data przydzielenia numeru identyfikującego",
        "numer ksef",
    ),
    "limited_tax_liability": ("ograniczony obowiązek", "183 dni", "centrum interesów"),
    "foreign_employment_income": (
        "dochody zagraniczne",
        "dochody osiągnięte za granicą",
        "dochody osiągnięte na terytorium polski",
        "czy wykazać dochody zagraniczne w polsce",
        "pracował za granicą",
        "mieszkał za granicą",
        "większość roku za granicą",
        "krocej niż 183 dni",
        "krócej niż 183 dni",
        "centrum interesów życiowych",
        "miejsce zamieszkania dla celów podatkowych",
        "umowa o unikaniu podwójnego opodatkowania",
    ),
    "return_relief": ("ulga na powrót",),
    "return_relief_residency_change": (
        "bez zmiany rezydencji",
        "bez zmiany rezydencji podatkowej",
        "zmiana miejsca zamieszkania do polski",
        "przeniósł miejsce zamieszkania do polski",
        "przeniosl miejsce zamieszkania do polski",
    ),
    "termination_of_co_ownership": ("zniesienie współwłasności", "zniesienie wspólwłasności"),
    "equalization_payment": ("spłata", "splata"),
    "thermomodernization_relief": ("termomoderniz",),
    "existing_residential_building_requirement": (
        "dom w budowie",
        "budynek w budowie",
        "istniejący budynek mieszkalny",
        "istniejacy budynek mieszkalny",
        "przed zakończeniem budowy",
        "przed zakonczeniem budowy",
    ),
    "housing_relief": ("ulga mieszkaniowa",),
    "housing_relief_loan_timing": (
        "kredyt zaciągnięty przed sprzedażą",
        "kredyt zaciagniety przed sprzedaza",
        "spłata rat kredytu",
        "splata rat kredytu",
        "raty wraz z odsetkami",
        "spłata kredytu wraz z odsetkami",
        "splata kredytu wraz z odsetkami",
        "własne cele mieszkaniowe",
        "wlasne cele mieszkaniowe",
        "po sprzedaży mieszkania",
        "po sprzedazy mieszkania",
        "przychód ze sprzedaży na spłatę kredytu",
        "przychod ze sprzedazy na splate kredytu",
        "przed dniem uzyskania przychodu",
    ),
    "temporary_rental": ("czasowy wynajem", "wynajmować lokal"),
    "dropshipping": ("dropshipping", "klient jako importer"),
    "land_sale_vat": ("sprzedaż działki", "sprzedaz dzialki"),
    "buyer_power_of_attorney": ("pełnomocnictw", "pelnomocnictw"),
    "developer_land_sale_preparation": (
        "deweloper",
        "warunki zabudowy",
        "warunków zabudowy",
        "pozwolenie na budowę",
        "pozwolenia na budowę",
        "podział działki",
        "podzial dzialki",
        "warunki przyłączenia mediów",
        "warunki przylaczenia mediow",
        "dzierżawa",
        "dzierzawa",
        "grunt",
        "działki",
        "dzialki",
    ),
    "private_leased_vehicle_sale": ("samochód leasing", "samochod leasing", "wykup", "majątku prywat", "po leasingu"),
    "post_leasing_vehicle_gift_sale": (
        "samochód po leasingu",
        "samochod po leasingu",
        "wykup samochodu",
        "wykup do majątku prywatnego",
        "wykup do majatku prywatnego",
        "darowizna małżonce",
        "darowizna malzonce",
        "sprzedaż przez małżonkę",
        "sprzedaz przez malzonke",
    ),
    "senior_relief": ("ulga dla pracujących seniorów", "ulga dla senior"),
    "senior_relief_payment_timing": (
        "pierwsza wypłata emerytury",
        "pierwsza wyplata emerytury",
        "przed pierwszą wypłatą emerytury",
        "przed pierwsza wyplata emerytury",
        "nabycie prawa do emerytury",
        "wypłata wynagrodzenia przed pierwszą emeryturą",
        "wyplata wynagrodzenia przed pierwsza emerytura",
        "korekta zeznania",
        "po ukończeniu 65 lat",
        "po ukonczeniu 65 lat",
    ),
    "ip_box_software_development": (
        "ip box",
        "kwalifikowane prawo własności intelektualnej",
        "autorskie prawo do programu komputerowego",
        "autorskie prawa majątkowe do programu komputerowego",
        "działalność badawczo-rozwojowa",
        "tworzenie i rozwijanie oprogramowania",
        "tworzenie oprogramowania",
        "rozwijanie oprogramowania",
        "wskaźnik nexus",
        "dochód z kwalifikowanego prawa własności intelektualnej",
    ),
    "property_sale_tax_timing": (
        "odpłatne zbycie nieruchomości",
        "sprzedaż mieszkania w 2026",
        "sprzedaż mieszkania w 2027",
        "sprzedaz mieszkania w 2026",
        "sprzedaz mieszkania w 2027",
        "pięcioletni termin podatkowy",
        "piecioletni termin podatkowy",
        "data nabycia nieruchomości",
        "art. 10 ust. 1 pkt 8",
        "art. 10 ust. 5",
        "art. 30e",
    ),
    "large_family_relief": (
        "ulga 4+",
        "ulga 4 plus",
        "wychowywanie co najmniej czworga dzieci",
        "co najmniej czworo dzieci",
        "czworo dzieci",
        "zwolnienie dla rodzin 4+",
        "wykonywanie władzy rodzicielskiej",
    ),
    "mortgage_settlement_refund": (
        "ugoda z bankiem dotycząca kredytu hipotecznego",
        "kredyt frankowy",
        "zwrot świadczenia nienależnego",
        "zwrot własnych rat kredytu",
        "umorzenie części kredytu hipotecznego",
        "zwrot kosztów zastępstwa procesowego",
        "zwrot kosztów zastepstwa procesowego",
    ),
    "debt_assumption": (
        "przejęcie długu",
        "przejęcie zobowiązania",
        "zwolnienie z długu",
        "zmiana dłużnika",
        "zgoda wierzyciela",
        "art. 519 kodeksu cywilnego",
    ),
}
STATUTE_PROCEDURAL_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    (
        re.compile(r"\b(co reguluje ordynacja podatkowa|zakres ordynacji podatkowej|ustawa normuje)\b", re.IGNORECASE),
        (),
        ("1",),
        ("Ustawa normuje", "zobowiązania podatkowe", "postępowanie podatkowe"),
    ),
    (
        re.compile(r"\b(definicj\w*.*ordynacj\w*|podstawow\w* definicj\w* ustawow\w*|ilekro[ćc] w ustawie jest mowa)\b", re.IGNORECASE),
        (),
        ("3",),
        ("Ilekroć w ustawie jest mowa o",),
    ),
    (
        re.compile(r"\b(jak ordynacja podatkowa definiuje podatek|definicj\w* podatku|co to jest podatek)\b", re.IGNORECASE),
        (),
        ("6",),
        ("Podatkiem jest publicznoprawne",),
    ),
    (
        re.compile(r"\b(kto wydaje interpretacj\w* indywidualn\w*|organ.*interpretacj\w* indywidualn\w*)\b", re.IGNORECASE),
        (),
        ("14b",),
        ("Dyrektor Krajowej Informacji Skarbowej", "interpretację indywidualną"),
    ),
    (
        re.compile(r"\b(co powinna zawiera[ćc] interpretacj\w* indywidualn\w*|interpretacj\w* indywidualn\w* zawiera)\b", re.IGNORECASE),
        (),
        ("14c",),
        ("Interpretacja indywidualna zawiera", "ocenę stanowiska wnioskodawcy"),
    ),
    (
        re.compile(r"\b(w jakim terminie wydaje si[ęe] interpretacj\w* indywidualn\w*|termin wydania interpretacj\w*)\b", re.IGNORECASE),
        (),
        ("14d",),
        ("w terminie 3 miesięcy od dnia otrzymania wniosku",),
    ),
    (
        re.compile(r"\b(zastosowanie si[ęe] do interpretacj\w* indywidualn\w*|ochron\w* interpretacj\w* indywidualn\w*|nie mo[żz]e szkodzi[ćc] wnioskodawcy)\b", re.IGNORECASE),
        (),
        ("14k",),
        ("nie może szkodzić wnioskodawcy",),
    ),
    (
        re.compile(r"\b(kiedy powstaje zobowi[ąa]zanie podatkow\w*|powstaje z dniem)\b", re.IGNORECASE),
        (),
        ("21",),
        ("Zobowiązanie podatkowe powstaje z dniem",),
    ),
    (
        re.compile(
            r"(\bksef\b|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*).{0,160}"
            r"(wielk\w* brytani\w*|\buk\b|zea|niderland\w*|holandi\w*|poza terytorium kraju|państw\w* trzec\w*|sprzedaż lokaln\w*)|"
            r"(wielk\w* brytani\w*|\buk\b|zea|niderland\w*|holandi\w*|poza terytorium kraju|państw\w* trzec\w*|sprzedaż lokaln\w*).{0,160}"
            r"(\bksef\b|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*)",
            re.IGNORECASE,
        ),
        (),
        ("106a", "106b", "106ga", "106gb"),
        (
            "Podatnicy są obowiązani wystawiać faktury ustrukturyzowane",
            "Obowiązek, o którym mowa w ust. 1, nie dotyczy",
            "miejscem świadczenia jest terytorium państwa trzeciego",
            "faktura ustrukturyzowana jest udostępniana nabywcy w sposób z nim uzgodniony",
        ),
    ),
    (
        re.compile(
            r"(\bmał\w* podatnik\w*|\bmaly podatnik\w*|art\.\s*4a\s*pkt\s*10).{0,180}"
            r"(vat poza polsk\w*|poza polsk\w*|odwrotne obci[aą]żenie|reverse charge|warto[śs]ci dodanej|lokaln\w* stawk\w*|nabywc\w* rozlicza vat)|"
            r"(vat poza polsk\w*|poza polsk\w*|odwrotne obci[aą]żenie|reverse charge|warto[śs]ci dodanej|lokaln\w* stawk\w*|nabywc\w* rozlicza vat).{0,180}"
            r"(\bmał\w* podatnik\w*|\bmaly podatnik\w*|art\.\s*4a\s*pkt\s*10)",
            re.IGNORECASE,
        ),
        ("4a", "19"),
        ("4a",),
        (
            "małego podatnika",
            "wartość przychodu ze sprzedaży",
            "wraz z kwotą należnego podatku od towarów i usług",
            "wartość przychodu ze sprzedaży",
        ),
    ),
    (
        re.compile(r"\b(zaniechan\w* poboru podatk\w*|zwolni[ćc] niektóre grupy p[łl]atnik[óo]w z obowi[ąa]zku pobierania)\b", re.IGNORECASE),
        (),
        ("22",),
        ("zaniechać w całości lub w części poboru podatków",),
    ),
    (
        re.compile(r"\b(odpowiada ca[łl]ym swoim maj[ąa]tkiem|odpowiedzialno[śs][ćc] podatnika.*ca[łl]ym maj[ąa]tkiem)\b", re.IGNORECASE),
        (),
        ("26",),
        ("Podatnik odpowiada całym swoim majątkiem",),
    ),
    (
        re.compile(r"\b(termin p[łl]atno[śs]ci podatku.*14 dni|dor[ęe]czeni[au] decyzji ustalaj[ąa]cej wysoko[śs][ćc] zobowi[ąa]zania)\b", re.IGNORECASE),
        (),
        ("47",),
        ("Termin płatności podatku wynosi 14 dni",),
    ),
    (
        re.compile(r"\b(co jest zaleg[łl]o[śs]ci[ąa] podatkow[ąa]|zaleg[łl]o[śs][ćc] podatkowa)\b", re.IGNORECASE),
        (),
        ("51",),
        ("Zaległością podatkową jest podatek niezapłacony w terminie płatności",),
    ),
    (
        re.compile(r"\b(kiedy nie nalicza si[ęe] odsetek za zw[łl]ok[ęe]|odsetek za zw[łl]ok[ęe] nie nalicza si[ęe]|brak odsetek za zw[łl]ok[ęe])\b", re.IGNORECASE),
        (),
        ("54",),
        ("Odsetek za zwłokę nie nalicza się",),
    ),
    (
        re.compile(r"\b(odsetki za zw[łl]ok[ęe]|naliczane s[ąa] odsetki)\b", re.IGNORECASE),
        (),
        ("53", "54"),
        ("Od zaległości podatkowych naliczane są odsetki za zwłokę", "Odsetek za zwłokę nie nalicza się"),
    ),
    (
        re.compile(r"\b(w jaki spos[óo]b mo[żz]e wygasn[ąa][ćc] zobowi[ąa]zanie podatkow\w*|zobowi[ąa]zanie podatkowe wygasa)\b", re.IGNORECASE),
        (),
        ("59",),
        ("Zobowiązanie podatkowe wygasa",),
    ),
    (
        re.compile(r"\b(odroczy[ćc] termin p[łl]atno[śs]ci|roz[łl]o[żz]y[ćc] zap[łl]at[ęe] podatku na raty|umorzy[ćc] zaleg[łl]o[śs][ćc])\b", re.IGNORECASE),
        (),
        ("67a",),
        ("może odroczyć termin płatności podatku", "rozłożyć zapłatę podatku na raty", "umorzyć w całości lub w części zaległości podatkowe"),
    ),
    (
        re.compile(r"\b(co uwa[żz]a si[ęe] za nadp[łl]at[ęe]|definicj\w* nadp[łl]aty)\b", re.IGNORECASE),
        (),
        ("72",),
        ("Za nadpłatę uważa się kwotę",),
    ),
    (
        re.compile(r"\b(kiedy powstaje nadp[łl]ata|nadp[łl]ata powstaje z dniem)\b", re.IGNORECASE),
        (),
        ("73",),
        ("Nadpłata powstaje z dniem",),
    ),
    (
        re.compile(r"\b(wniosek o stwierdzenie nadp[łl]aty|kto mo[żz]e z[łl]o[żz]y[ćc] wniosek o stwierdzenie nadp[łl]aty)\b", re.IGNORECASE),
        (),
        ("75",),
        ("wniosek o stwierdzenie nadpłaty podatku",),
    ),
    (
        re.compile(r"\b(zaliczeni[ea] nadp[łl]aty|nadp[łl]aty wraz z ich oprocentowaniem podlegaj[ąa] zaliczeniu)\b", re.IGNORECASE),
        (),
        ("76",),
        ("Nadpłaty wraz z ich oprocentowaniem podlegają zaliczeniu z urzędu",),
    ),
    (
        re.compile(r"\b(w jakim terminie powinna zosta[ćc] zwr[óo]cona nadp[łl]ata|zwrot nadp[łl]aty w terminie)\b", re.IGNORECASE),
        (),
        ("77",),
        ("Nadpłata podlega zwrotowi w terminie",),
    ),
    (
        re.compile(r"\b(oprocentowani[ea] nadp[łl]at|czy nadp[łl]ata podlega oprocentowaniu)\b", re.IGNORECASE),
        (),
        ("78",),
        ("Nadpłaty podlegają oprocentowaniu",),
    ),
    (
        re.compile(r"\b(post[ęe]powani[ea] w sprawie stwierdzenia nadp[łl]aty nie mo[żz]na wszcz[ąa][ćc]|podczas trwania post[ęe]powania podatkowego.*nadp[łl]at)\b", re.IGNORECASE),
        (),
        ("79",),
        ("Postępowanie w sprawie stwierdzenia nadpłaty nie może zostać wszczęte",),
    ),
    (
        re.compile(r"\b(raportowani\w* schemat[óo]w podatkow\w*|schemat\w* podatkow\w*|mdr\b|cech\w* rozpoznawcz\w*)\b", re.IGNORECASE),
        (),
        ("86a",),
        ("Ilekroć w niniejszym rozdziale jest mowa o", "cesze rozpoznawczej"),
    ),
    (
        re.compile(r"\b(sukcesj\w* podatkow\w*.*przekszta[łl]ceni\w*|przekszta[łl]ceni\w*.*sukcesj\w* podatkow\w*|przekszta[łl]cenia innej osoby prawnej)\b", re.IGNORECASE),
        (),
        ("93a",),
        ("przekształcenia innej osoby prawnej", "wstępuje we wszelkie przewidziane w przepisach prawa podatkowego prawa i obowiązki"),
    ),
    (
        re.compile(r"\b(w jakim terminie wnosi si[ęe] odwo[łl]anie|za czyim po[śs]rednictwem wnosi si[ęe] odwo[łl]anie|odwo[łl]anie wnosi si[ęe] do w[łl]a[śs]ciwego organu odwo[łl]awczego)\b", re.IGNORECASE),
        (),
        ("223",),
        ("Odwołanie wnosi się do właściwego organu odwoławczego za pośrednictwem organu podatkowego", "Odwołanie wnosi się w terminie 14 dni"),
    ),
    (
        re.compile(r"\b(decyzja nieostateczna.*nie podlega wykonaniu|je[śs]li nie nadano.*rygor\w* natychmiastowej wykonalno[śs]ci|chyba [żz]e decyzji nadano rygor natychmiastowej wykonalno[śs]ci)\b", re.IGNORECASE),
        (),
        ("239a",),
        ("Decyzja nieostateczna", "nie podlega wykonaniu", "rygor natychmiastowej wykonalności"),
    ),
    (
        re.compile(r"\b(co reguluje ustawa o podatkach i opłatach lokalnych|zakres ustawy o podatkach i opłatach lokalnych|czy ustawa o podatkach i opłatach lokalnych normuje podatek od nieruchomo[śs]ci|ustawa normuje opłat[ęe] reklamow[ąa]|zalicza opłat[ęe] uzdrowiskow[ąa] do spraw przez ni[ąa] normowanych|opłat[ęe] uzdrowiskow[ąa].*normowanych)\b", re.IGNORECASE),
        (),
        ("1",),
        ("Ustawa normuje następujące podatki i opłaty lokalne", "opłatę reklamową", "opłatę uzdrowiskową"),
    ),
    (
        re.compile(r"\b(definicj\w* budynku|jak ustawa definiuje budynek|co oznacza budynek)\b", re.IGNORECASE),
        (),
        ("1a",),
        ("Użyte w ustawie określenia oznaczają", "budynek"),
    ),
    (
        re.compile(r"\b(jak ustawa(?: o podatkach i opłatach lokalnych)? definiuje budynek|definicj\w* budynku|co oznacza budynek)\b", re.IGNORECASE),
        (),
        ("1a",),
        ("Użyte w ustawie określenia oznaczają", "budynek"),
    ),
    (
        re.compile(r"\b(jak ustawa(?: o podatkach i opłatach lokalnych)? definiuje budowl\w*|co oznacza budowla|definicj\w* budowli)\b", re.IGNORECASE),
        (),
        ("1a",),
        ("Użyte w ustawie określenia oznaczają", "budowla"),
    ),
    (
        re.compile(
            r"\b("
            r"gdzie zdefiniowano powierzchni[ęe] użytkow[ąa] budynku|"
            r"jak ustawa rozumie powierzchni[ęe] użytkow[ąa]|"
            r"co oznacza powierzchnia użytkowa budynku|"
            r"budynek musi mie[ćc] fundamenty i dach|"
            r"w kt[óo]rym przepisie .* budynek .* fundamenty i dach|"
            r"klatek schodowych i szyb[óo]w d[źz]wigowych|"
            r"za kondygnacj[ęe] uwa[żz]a si[ęe] r[óo]wnie[żz] gara[żz]e podziemne"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1a",),
        ("powierzchnia użytkowa budynku lub jego części", "klatek schodowych", "szybów dźwigowych", "garaże podziemne"),
    ),
    (
        re.compile(
            r"\b("
            r"jak ustawa .* rozumie działalno[śs][ćc] gospodarcz\w*|"
            r"definicj\w* działalno[śs]ci gospodarczej|"
            r"czego nie uważa si[ęe] za działalno[śs][ćc] gospodarcz\w*|"
            r"co nie jest działalno[śs]ci[ąa] gospodarcz\w* w rozumieniu tej ustawy|"
            r"poj[ęe]cie grunt[óo]w, budynk[óo]w i budowli zwi[ąa]zanych z prowadzeniem działalno[śs]ci gospodarczej|"
            r"gdzie uregulowano poj[ęe]cie grunt[óo]w, budynk[óo]w i budowli zwi[ąa]zanych z prowadzeniem działalno[śs]ci gospodarczej|"
            r"wynajem do 5 pokoi go[śs]cinnych|"
            r"turystom na obszarach wiejskich .* nie jest działalno[śs]cią gospodarcz|"
            r"art\.? 6 ust\.? 1 pkt 4 prawa przedsi[ęe]biorc[óo]w|"
            r"art\.? 6 ust\.? 1 pkt 4 prawa przedsi[ęe]biorc[óo]w .* nie uwa[żz]a si[ęe] za działalno[śs][ćc] gospodarcz|"
            r"działalno[śs]ci wskazanej w art\.? 6 ust\.? 1 pkt 4 prawa przedsi[ęe]biorc[óo]w nie uwa[żz]a si[ęe] za działalno[śs][ćc] gospodarcz|"
            r"prawo przedsi[ęe]biorc[óo]w .* nie uwa[żz]a si[ęe] za działalno[śs][ćc] gospodarcz"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1a",),
        ("działalność gospodarcza", "Za działalność gospodarczą w rozumieniu ustawy nie uważa się", "art. 6 ust. 1 pkt 4 ustawy z dnia 6 marca 2018 r. – Prawo przedsiębiorców", "Prawo przedsiębiorców"),
    ),
    (
        re.compile(
            r"\b("
            r"co oznacza trwa[łl]e zwi[ąa]zanie z gruntem|"
            r"jak ustawa definiuje trwa[łl]e zwi[ąa]zanie z gruntem"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1a",),
        ("trwałe związanie z gruntem",),
    ),
    (
        re.compile(
            r"\b("
            r"fundamenty pod maszyn\w*|"
            r"urz[ąa]dzenia techniczne mog[ąa] by[ćc] budowl[ąa]|"
            r"obiekt budowlany jako budynek lub budowl[ęe]|"
            r"definicj\w* obiektu budowlanego|"
            r"definicj\w* rob[óo]t budowlanych|"
            r"budow[ęe], odbudow[ęe], rozbudow[ęe], nadbudow[ęe], przebudow[ęe] lub monta[żz]|"
            r"obiekty kultu religijnego|kapliczki|krzy[żz]e przydro[żz]ne|"
            r"nieu[żz]ytk[óo]w .* ewidencji grunt[óo]w i budynk[óo]w|"
            r"trwale wy[łl][ąa]czone z u[żz]ytkowania .* nie s[ąa] uznawane za zwi[ąa]zane z prowadzeniem działalno[śs]ci gospodarczej|"
            r"za cz[ęe][śs][ćc] mieszkaln[ąa] budynku mieszkalnego uznaje si[ęe] tak[żz]e pomieszczenie przeznaczone do przechowywania pojazd[óo]w"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1a",),
        ("fundamenty pod maszyny", "obiekt budowlany", "roboty budowlane", "obiekty kultu religijnego", "nieużytki", "trwale wyłączono budynek, budowlę lub ich części z użytkowania", "Za część mieszkalną budynku mieszkalnego uznaje się także pomieszczenie przeznaczone do przechowywania pojazdów"),
    ),
    (
        re.compile(
            r"\b("
            r"ulgi .* ko[śs]cio[łl][óo]w|"
            r"zwolnienia .* ko[śs]cio[łl][óo]w|"
            r"zwi[ąa]zk[óo]w wyznaniowych.*odr[ęe]bne ustawy|"
            r"ustawa odsy[łl]a do odr[ęe]bnych ustaw .* ko[śs]cio[łl][óo]w"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1b",),
        ("Ulgi i zwolnienia podatkowe", "kościołom i związkom wyznaniowym regulują odrębne ustawy"),
    ),
    (
        re.compile(
            r"\b("
            r"specjalnych stref ekonomicznych|"
            r"sse.*zwolnieni\w*.*podatku od nieruchomo[śs]ci|"
            r"gdzie uregulowano zwolnienia z podatku od nieruchomo[śs]ci .* stref ekonomicznych|"
            r"odsy[łl]a do odr[ęe]bnych przepis[óo]w .* specjalnych stref ekonomicznych"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1b",),
        ("Zwolnienia z podatku od nieruchomości", "specjalnych stref ekonomicznych"),
    ),
    (
        re.compile(
            r"\b("
            r"dr[óo]g publicznych|"
            r"budow[ęe] dr[óo]g publicznych|"
            r"grunty i budynki przeznaczone na budow[ęe] dr[óo]g publicznych"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1b",),
        ("budowę dróg publicznych", "szczególnych zasadach przygotowania i realizacji inwestycji w zakresie dróg publicznych"),
    ),
    (
        re.compile(
            r"\b("
            r"jaki organ podatkowy jest właściwy|"
            r"kt[óo]ry organ gminy jest organem podatkowym|"
            r"w[óo]jt|burmistrz|prezydent miasta.*organem podatkowym"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("1c",),
        ("Organem podatkowym właściwym", "wójt", "burmistrz", "prezydent miasta"),
    ),
    (
        re.compile(r"\b(co podlega opodatkowaniu podatkiem od nieruchomo[śs]ci|przedmiot opodatkowania podatkiem od nieruchomo[śs]ci|jakie nieruchomo[śs]ci podlegaj[ąa]|jakie przedmioty podlegaj[ąa] opodatkowaniu podatkiem od nieruchomo[śs]ci|grunty, budynki lub ich cz[ęe][śs]ci oraz budowle zwi[ąa]zane z działalno[śs]ci[ąa] gospodarcz[ąa]|budowle lub ich cz[ęe][śs]ci zwi[ąa]zane z prowadzeniem działalno[śs]ci gospodarczej)\b", re.IGNORECASE),
        (),
        ("2",),
        ("Opodatkowaniu podatkiem od nieruchomości podlegają",),
    ),
    (
        re.compile(r"\b(czy użytki rolne i lasy podlegaj[ąa] podatkowi od nieruchomo[śs]ci|użytki rolne lub lasy.*nie podlegaj[ąa])\b", re.IGNORECASE),
        (),
        ("2",),
        ("Opodatkowaniu podatkiem od nieruchomości nie podlegają użytki rolne lub lasy",),
    ),
    (
        re.compile(
            r"\b("
            r"kiedy użytki rolne .* podlegaj[ąa] podatkowi od nieruchomo[śs]ci|"
            r"las zaj[ęe]ty na prowadzenie działalno[śs]ci gospodarczej|"
            r"użytki rolne .* zaj[ęe]te na prowadzenie działalno[śs]ci gospodarczej|"
            r"nieruchomo[śs]ci pa[ńn]stw obcych .* przedstawicielstw dyplomatycznych|"
            r"warunek wzajemno[śs]ci .* pa[ńn]stw obcych|"
            r"grunty pod wodami powierzchniowymi p[łl]yn[ąa]cymi, z wyj[ąa]tkiem jezior i zbiornik[óo]w sztucznych|"
            r"nieruchomo[śs]ci lub ich cz[ęe][śs]ci zaj[ęe]te na potrzeby organ[óo]w jednostek samorz[ąa]du terytorialnego|"
            r"urz[ęe]d[óo]w gmin i starostw|"
            r"grunty zaj[ęe]te pod pasy drogowe dr[óo]g publicznych .* budowle|"
            r"krajowego zasobu nieruchomo[śs]ci"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("2",),
        ("z wyjątkiem zajętych na prowadzenie działalności gospodarczej", "pod warunkiem wzajemności", "przeznaczone na siedziby przedstawicielstw dyplomatycznych", "grunty pod wodami powierzchniowymi płynącymi", "zajęte na potrzeby organów jednostek samorządu terytorialnego", "grunty zajęte pod pasy drogowe dróg publicznych", "wchodzą w skład Zasobu Nieruchomości"),
    ),
    (
        re.compile(r"\b(kto jest podatnikiem podatku od nieruchomo[śs]ci|podatnikami podatku od nieruchomo[śs]ci s[ąa])\b", re.IGNORECASE),
        (),
        ("3",),
        ("Podatnikami podatku od nieruchomości są",),
    ),
    (
        re.compile(
            r"\b("
            r"użytkownik wieczysty .* podatnikiem|"
            r"posiadacz samoistny .* podatnikiem|"
            r"je[żz]eli przedmiot opodatkowania znajduje si[ęe] w posiadaniu samoistnym|"
            r"wsp[óo][łl]w[łl]asno[śs][ćc].*solidarn|"
            r"obowi[ąa]zek podatkowy ci[ąa][żz]y solidarnie na wsp[óo][łl]w[łl]a[śs]cicielach|"
            r"zasad[ęe] solidarnej odpowiedzialno[śs]ci wsp[óo][łl]w[łl]a[śs]cicieli|"
            r"posiadacz nieruchomo[śs]ci skarbu pa[ńn]stwa bez tytu[łl]u prawnego|"
            r"zasobu w[łl]asno[śs]ci rolnej skarbu pa[ńn]stwa|"
            r"las[óo]w pa[ńn]stwowych|"
            r"wyodr[ęe]bnionej własno[śs]ci lokali|"
            r"lokalu mieszkalnego niestanowi[ąa]cego odr[ęe]bnej nieruchomo[śs]ci|"
            r"lokal mieszkalny niestanowi[ąa]cy odr[ęe]bnej nieruchomo[śs]ci na podstawie umowy z w[łl]a[śs]cicielem|"
            r"jeden ze wsp[óo][łl]w[łl]a[śs]cicieli jest zwolniony|"
            r"gara[żz]u wielostanowiskowego w budynku mieszkalnym|"
            r"gara[żz]u wielostanowiskowego.*odpowiedzialno[śs][ćc] solidarn|"
            r"gara[żz]u wielostanowiskowego.*stosuje si[ęe] odpowiedzialno[śs][ćc] solidarn"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("3",),
        (
            "użytkownikami wieczystymi gruntów",
            "posiadaczami samoistnymi nieruchomości",
            "obowiązek podatkowy w zakresie podatku od nieruchomości ciąży na posiadaczu samoistnym",
            "ciąży solidarnie na wszystkich współwłaścicielach",
            "jest bez tytułu prawnego",
            "Zasobu Własności Rolnej Skarbu Państwa",
            "nieruchomości wspólnej",
            "lokali mieszkalnych niestanowiących odrębnych nieruchomości",
            "lokalu mieszkalnego niestanowiącego odrębnej nieruchomości",
            "jeden lub kilku współwłaścicieli",
            "garażu wielostanowiskowego",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"podstawa opodatkowania.*podatku od nieruchomo[śs]ci|"
            r"dla grunt[óo]w.*powierzchnia|"
            r"dla budynk[óo]w.*powierzchnia użytkowa|"
            r"podstawa opodatkowania dla budowli.*zwi[ąa]zanych z prowadzeniem działalno[śs]ci gospodarczej|"
            r"dla budowli.*warto[śs][ćc]|"
            r"budowl[ai] .* przedmiotem umowy leasingu|"
            r"budowli b[ęe]d[ąa]cej przedmiotem leasingu|"
            r"warto[śs][ćc] cz[ęe][śs]ci budowli po[łl]o[żz]onych .* dw[óo]ch lub wi[ęe]cej gmin|"
            r"obowi[ąa]zek podatkowy .* budowli .* powsta[łl] w ci[ąa]gu roku podatkowego|"
            r"jaka jest podstawa opodatkowania budowli, gdy obowi[ąa]zek podatkowy powsta[łl] w ci[ąa]gu roku podatkowego|"
            r"po ulepszeniu budowli|"
            r"aktualizacji wyceny [śs]rodk[óo]w trwa[łl]ych|"
            r"poda[łl] warto[śs][ćc] nieodpowiadaj[ąa]c[ąa] warto[śs]ci rynkowej|"
            r"organ podatkowy powo[łl]a bieg[łl]ego|"
            r"rzeczoznawc[óo]w maj[ąa]tkowych .* ustalenia warto[śs]ci budowli|"
            r"jak liczy si[ęe] powierzchni[ęe] użytkow[ąa].*1,40 m do 2,20 m|"
            r"powierzchni[ęe] użytkow[ąa].*1,40 m do 2,20 m|"
            r"powierzchni[ęe] pomieszcze[ńn] o wysoko[śs]ci mniejszej ni[żz] 1,40 m|"
            r"mniejsza ni[żz] 1,40 m|"
            r"nie dokonuje si[ęe] odpis[óo]w amortyzacyjnych"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("4",),
        (
            "Podstawę opodatkowania stanowi",
            "dla gruntów – powierzchnia",
            "dla budynków lub ich części – powierzchnia użytkowa",
            "Powierzchnię pomieszczeń lub ich części oraz część kondygnacji o wysokości w świetle od 1,40 m do 2,20 m",
            "a jeżeli wysokość jest mniejsza niż 1,40 m, powierzchnię tę pomija się",
            "Jeżeli od budowli lub ich części", "wartość rynkowa",
            "jest przedmiotem umowy leasingu",
            "budowli będącej przedmiotem leasingu",
            "Wartość części budowli położonych w danej gminie",
            "obowiązek podatkowy powstał w ciągu roku podatkowego",
            "organ podatkowy powoła biegłego",
            "spośród rzeczoznawców majątkowych",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"kto okre[śs]la wysoko[śs][ćc] stawek podatku od nieruchomo[śs]ci|"
            r"kto okre[śs]la stawki podatku od nieruchomo[śs]ci|"
            r"rada gminy.*stawki podatku od nieruchomo[śs]ci|"
            r"stawki nie mog[ąa] przekroczy[ćc]|"
            r"r[óo]żnicowa[ćc] wysoko[śs][ćc] stawek|"
            r"jakie kryteria rada gminy mo[żz]e uwzgl[ęe]dnia[ćc] przy r[óo]żnicowaniu stawek|"
            r"r[óo]żnicowanie stawek dla budynk[óo]w wed[łl]ug lokalizacji|"
            r"grunty obj[ęe]te obszarem rewitalizacji|"
            r"niezabudowanych grunt[óo]w obj[ęe]tych obszarem rewitalizacji|"
            r"g[óo]rn[ąa] stawk[ęe] dla grunt[óo]w zwi[ąa]zanych z prowadzeniem działalno[śs]ci gospodarczej|"
            r"budynk[óo]w zwi[ąa]zanych z udzielaniem [śs]wiadcze[ńn] zdrowotnych|"
            r"grunt[óo]w pod wodami powierzchniowymi stoj[ąa]cymi|"
            r"budynk[óo]w mieszkalnych|"
            r"ile wynosi stawka .* od budowli|"
            r"rada gminy mo[żz]e r[óo]żnicowa[ćc] stawki .* rodzaj prowadzonej działalno[śs]ci"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("5",),
        (
            "Rada gminy określa wysokość stawek podatku od nieruchomości",
            "rada gminy może różnicować ich wysokość",
            "związanych z prowadzeniem działalności gospodarczej",
            "lokalizację, sposób wykorzystywania, rodzaj zabudowy, stan techniczny oraz wiek budynków",
            "rodzaj prowadzonej działalności",
            "objętych obszarem rewitalizacji",
            "niezabudowanych gruntów objętych obszarem rewitalizacji",
            "związanych z udzielaniem świadczeń zdrowotnych",
            "pod wodami powierzchniowymi stojącymi",
            "mieszkalnych",
            "od budowli – 2 % ich wartości",
        ),
    ),
    (
        re.compile(
            r"osoby prawne .* podatek od nieruchomo[śs]ci .* 15\.\s*dnia ka[żz]dego miesi[ąa]ca.*31 stycznia",
            re.IGNORECASE,
        ),
        (),
        ("6",),
        ("do dnia 15. każdego miesiąca, a za styczeń do dnia 31 stycznia",),
    ),
    (
        re.compile(
            r"\b("
            r"kiedy wygasa obowi[ąa]zek podatkowy|"
            r"w jakim terminie osoba fizyczna sk[łl]ada informacj[ęe] o nieruchomo[śs]ciach|"
            r"gdzie uregulowano 14-dniowy termin dla osoby fizycznej|"
            r"do kiedy osoby prawne sk[łl]adaj[ąa] deklaracj[ęe] na podatek od nieruchomo[śs]ci|"
            r"31 stycznia.*deklaracj\w* na podatek od nieruchomo[śs]ci|"
            r"w jakich terminach osoby prawne wp[łl]acaj[ąa] podatek od nieruchomo[śs]ci bez wezwania|"
            r"w kt[óo]rym przepisie wskazano, [żz]e osoby prawne p[łl]ac[ąa] podatek od nieruchomo[śs]ci do 15\\. dnia ka[żz]dego miesi[ąa]ca, a za stycze[ńn] do 31 stycznia|"
            r"osoby prawne p[łl]ac[ąa] podatek od nieruchomo[śs]ci do 15\\. dnia ka[żz]dego miesi[ąa]ca|"
            r"terminy rat podatku od nieruchomo[śs]ci dla os[óo]b fizycznych|"
            r"organ podatkowy zmienia decyzj[ęe] ustalaj[ąa]c[ąa] podatek|"
            r"obowi[ąa]zek składania informacji i deklaracji dotyczy tak[żz]e podatnik[óo]w korzystaj[ąa]cych ze zwolnie[ńn]|"
            r"je[śs]li obowiązek podatkowy powstał lub wygasł w ci[ąa]gu roku|"
            r"skorygowa[ćc] deklaracj[ęe] .* w terminie 14 dni od dnia zaistnienia tego zdarzenia|"
            r"skorygowania deklaracji na podatek od nieruchomo[śs]ci w terminie 14 dni od zdarzenia wpływaj[ąa]cego na wysoko[śs][ćc] podatku|"
            r"nie wszczyna si[ęe] post[ęe]powania, a post[ęe]powanie wszcz[ęe]te umarza|"
            r"nie wszczyna si[ęe] post[ęe]powania albo je umarza, je[śs]li wysoko[śs][ćc] zobowi[ąa]zania podatkowego by[łl]aby ni[żz]sza ni[żz] koszty dor[ęe]czenia przesy[łl]ki poleconej|"
            r"kwota podatku nie przekracza 100 z[łl]|"
            r"do 15\\. ka[żz]dego miesi[ąa]ca|"
            r"za stycze[ńn] do dnia 31 stycznia|"
            r"czy .* deklaracje .* mog[ąa] by[ćc] sk[łl]adane elektronicznie|"
            r"czy rada gminy mo[żz]e zarz[ąa]dzi[ćc] pob[óo]r .* w drodze inkasa|"
            r"okre[śs]lenia sposobu przesy[łl]ania informacji i deklaracji .* drog[ąa] elektroniczn[ąa] .* rodzaj[óo]w podpisu elektronicznego"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("6",),
        (
            "Obowiązek podatkowy wygasa z upływem miesiąca",
            "w terminie 14 dni",
            "w terminie do dnia 31 stycznia",
            "do dnia 15. każdego miesiąca, a za styczeń do dnia 31 stycznia",
            "w terminach: do dnia 15 marca, 15 maja, 15 września i 15 listopada",
            "organ podatkowy dokonuje zmiany decyzji",
            "dotyczy również podatników korzystających ze zwolnień",
            "podatek za ten rok ustala się proporcjonalnie do liczby miesięcy",
            "w terminie 14 dni od dnia zaistnienia tego zdarzenia",
            "odpowiednio skorygować deklaracje w razie zaistnienia zdarzenia",
            "Nie wszczyna się postępowania",
            "wysokość zobowiązania podatkowego nie przekraczałaby kosztów doręczenia przesyłki poleconej",
            "W przypadku gdy kwota podatku nie przekracza 100 zł",
            "mogą być składane za pomocą środków komunikacji elektronicznej",
            "sposób przesyłania informacji o nieruchomościach i obiektach budowlanych oraz deklaracji na podatek od nieruchomości",
            "Rada gminy może zarządzać pobór podatku od nieruchomości",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"kiedy powstaje obowi[ąa]zek podatkowy w podatku od nieruchomo[śs]ci|"
            r"od kiedy powstaje obowi[ąa]zek podatkowy dla nowo wybudowanego budynku lub budowli|"
            r"od kiedy zmiana sposobu wykorzystywania nieruchomo[śs]ci wp[łl]ywa na wysoko[śs][ćc] podatku od nieruchomo[śs]ci|"
            r"od kiedy po zmianie sposobu wykorzystywania przedmiotu opodatkowania podatek ulega obni[żz]eniu albo podwy[żz]szeniu|"
            r"obowi[ąa]zek podatkowy powstaje od pierwszego dnia miesi[ąa]ca|"
            r"obowi[ąa]zek podatkowy .* mo[żz]e powsta[ćc] ju[żz] po rozpocz[ęe]ciu użytkowania przed ostatecznym wyko[ńn]czeniem|"
            r"budowa zosta[łl]a zako[ńn]czona|"
            r"zmiana sposobu wykorzystywania przedmiotu opodatkowania"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("6",),
        (
            "Obowiązek podatkowy powstaje od pierwszego dnia miesiąca",
            "budowa została zakończona",
            "zmiana sposobu wykorzystywania przedmiotu opodatkowania",
        ),
    ),
    (
        re.compile(
            r"\b("
            r"jakie s[ąa] zwolnienia z podatku od nieruchomo[śs]ci|"
            r"w kt[óo]rym przepisie znajduje si[ęe] katalog zwolnie[ńn] z podatku od nieruchomo[śs]ci|"
            r"katalog zwolnie[ńn] z podatku od nieruchomo[śs]ci|"
            r"zwalnia si[ęe] od podatku od nieruchomo[śs]ci|"
            r"inne zwolnienia przedmiotowe|"
            r"podstaw[ęe] do wprowadzania .* innych zwolnie[ńn] przedmiotowych|"
            r"infrastruktury kolejowej|"
            r"zwolnienie od podatku od nieruchomo[śs]ci dla uczelni|"
            r"budowle infrastruktury portowej w portach morskich|"
            r"wpisanych indywidualnie do rejestru zabytk[óo]w|"
            r"rodzinnego ogrodu działkowego|altan działkowych|obiekt[óo]w gospodarczych o powierzchni zabudowy do 35 m2"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("7",),
        ("Zwalnia się od podatku od nieruchomości", "Rada gminy, w drodze uchwały, może wprowadzić inne zwolnienia przedmiotowe", "wchodzące w skład infrastruktury kolejowej", "uczelnie, zwolnienie nie dotyczy", "budowle infrastruktury portowej w portach morskich", "wpisane indywidualnie do rejestru zabytków", "położone na terenie rodzinnego ogrodu działkowego"),
    ),
    (
        re.compile(r"\b(ewidencj\w* podatkow\w* nieruchomo[śs]ci|organy podatkowe prowadz[ąa] ewidencj\w* podatkow\w* nieruchomo[śs]ci|z jakich [źz]r[óo]de[łl] pochodz[ąa] dane .* ewidencji podatkowej nieruchomo[śs]ci|ksi[ąa]g wieczystych.*ewidencji grunt[óo]w i budynk[óo]w|ewidencji prowadzonych przez urz[ęe]dy skarbowe)\b", re.IGNORECASE),
        (),
        ("7a",),
        ("organy podatkowe prowadzą ewidencję podatkową nieruchomości", "księgach wieczystych", "ewidencji gruntów i budynków", "ewidencji prowadzonych przez urzędy skarbowe"),
    ),
    (
        re.compile(
            r"\b("
            r"sprawozdanie podatkowe .* podatku od nieruchomo[śs]ci|"
            r"jakie informacje zawiera sprawozdanie podatkowe|"
            r"w jakich terminach przekazuje si[ęe] cz[ęe][śs]ci sprawozdania|"
            r"regionalnych izb obrachunkowych|"
            r"sprawozdanie .* elektronicznie .* rio|"
            r"w kt[óo]rym przepisie zapisano, [żz]e sprawozdanie podatkowe przekazuje si[ęe] ministrowi finans[óo]w elektronicznie przez regionalne izby obrachunkowe|"
            r"wed[łl]ug stanu na dzie[ńn] 30 czerwca roku podatkowego|"
            r"aktualizacji sprawozdania podatkowego|"
            r"do ko[ńn]ca i kwarta[łl]u .* do ko[ńn]ca iii kwarta[łl]u|"
            r"czę[śs][ćc] sprawozdania dotycz[ąa]ca stawek przekazywana jest do ko[ńn]ca i kwarta[łl]u, a czę[śs][ćc] dotycz[ąa]ca podstaw opodatkowania do ko[ńn]ca iii kwarta[łl]u"
            r")\b",
            re.IGNORECASE,
        ),
        (),
        ("7b",),
        (
            "sporządzają co roku sprawozdanie podatkowe",
            "Sprawozdanie zawiera informacje o",
            "nie później niż do końca I kwartału",
            "nie później niż do końca III kwartału",
            "regionalnych izb obrachunkowych",
            "według stanu na dzień 30 czerwca",
            "Sprawozdanie jest aktualizowane",
        ),
    ),
    (
        re.compile(r"\b(g[óo]rne granice stawek kwotowych|minister.*og[łl]asza.*g[óo]rne granice stawek|art\. 20|komunikatu prezesa g[łl][óo]wnego urz[ęe]du statystycznego|terminie 20 dni po up[łl]ywie pierwszego p[óo][łl]rocza)\b", re.IGNORECASE),
        (),
        ("20",),
        ("Górne granice stawek kwotowych", "Minister właściwy do spraw finansów publicznych ogłasza", "komunikatu Prezesa Głównego Urzędu Statystycznego", "w terminie 20 dni po upływie pierwszego półrocza"),
    ),
    (
        re.compile(
            r"\b("
            r"na jakiej zasadzie .* g[óo]rne granice stawek|"
            r"co stosuje si[ęe], gdy rada gminy nie uchwali stawek|"
            r"stawki z roku poprzedniego|"
            r"uchwa[łl]a rady gminy przewiduj[ąa]ca pomoc publiczn\w* .* uwzgl[ęe]dnia[ćc] przepisy|"
            r"pomoc publiczna .* uchwa[łl]a .* uwzgl[ęe]dnia[ćc] przepisy|"
            r"uchwa[łl]a rady gminy przewiduje pomoc publiczn\w* w przypadku zwolnie[ńn] z art\.? 7 ust\.? 3|"
            r"pomoc de minimis|"
            r"w kt[óo]rym przepisie .* udzielanie pomocy jako pomocy de minimis|"
            r"nakazano, aby uchwa[łl]a rady gminy przewiduj[ąa]ca pomoc publiczn\w* by[łl]a podejmowana z uwzgl[ęe]dnieniem przepis[óo]w o pomocy publicznej|"
            r"kto okre[śs]la warunki udzielania zwolnie[ńn] stanowi[ąa]cych pomoc publiczn\w*|"
            r"rada ministr[óo]w okre[śs]la warunki udzielania zwolnie[ńn] stanowi[ąa]cych pomoc publiczn\w* dotycz[ąa]cych art\.? 7 ust\.? 3 i art\.? 12 ust\.? 4 ustawy|"
            r"notyfikacji Komisji Europejskiej"
            r")\b",
            re.IGNORECASE,
        ),
        ("20d",),
        ("20a", "20b", "20c", "20d"),
        (
            "W przypadku nieuchwalenia stawek",
            "uchwała ta powinna być podjęta z uwzględnieniem przepisów dotyczących pomocy publicznej",
            "pomoc ta jest udzielana jako pomoc de minimis",
            "Rada Ministrów określi",
            "warunki udzielania zwolnień stanowiących pomoc publiczną dotyczących art. 7 ust. 3 i art. 12 ust. 4",
            "podlega notyfikacji Komisji Europejskiej",
        ),
    ),
    (
        re.compile(r"\b(niedaj[ąa]c\w* si[ęe] usun[ąa][ćc].*w[ąa]tpliwo[śs]ci|korzy[śs][ćc] podatnika|in dubio pro tributario)\b", re.IGNORECASE),
        (),
        ("2a",),
        ("Niedające się usunąć wątpliwości co do treści przepisów prawa podatkowego",),
    ),
    (
        re.compile(r"\b(przedawn\w*.*zobowi[ąa]zan\w* podatkow\w*|zobowi[ąa]zan\w* podatkow\w*.*przedawn\w*)\b", re.IGNORECASE),
        (),
        ("70",),
        ("Zobowiązanie podatkowe przedawnia się",),
    ),
    (
        re.compile(r"\b(zawiadom\w*.*zawieszen\w*.*przedawn\w*|zawieszen\w*.*bieg\w* terminu przedawn\w*)\b", re.IGNORECASE),
        (),
        ("70c",),
        ("zawiadamia podatnika o nierozpoczęciu lub zawieszeniu biegu terminu przedawnienia",),
    ),
    (
        re.compile(r"\b(korekt\w* deklarac\w*|skoryg\w* deklarac\w*)\b", re.IGNORECASE),
        (),
        ("81",),
        ("skorygować uprzednio złożoną deklarację",),
    ),
    (
        re.compile(r"\b(sukcesj\w*.*(?:[łl][ąa]czeni\w*|os[óo]b prawnych)|(?:[łl][ąa]czeni\w*|os[óo]b prawnych).*sukcesj\w*)\b", re.IGNORECASE),
        (),
        ("93",),
        ("wstępuje we wszelkie przewidziane w przepisach prawa podatkowego prawa i obowiązki",),
    ),
    (
        re.compile(r"\b(sukcesj\w*.*podzia[łl]\w*.*wydzielen\w*|podzia[łl]\w*.*wydzielen\w*.*sukcesj\w*)\b", re.IGNORECASE),
        (),
        ("93c",),
        ("osoby prawne przejmujące wstępują z dniem podziału",),
    ),
    (
        re.compile(r"\b(na podstawie przepis[óo]w prawa|zasad\w* legalizmu|dzia[łl]a\w*.*podstawie prawa)\b", re.IGNORECASE),
        (),
        ("120",),
        ("Organy podatkowe działają na podstawie przepisów prawa",),
    ),
    (
        re.compile(r"\b(budz[ąa]c\w* zaufanie|zaufani\w* do organ[óo]w podatkow\w*)\b", re.IGNORECASE),
        (),
        ("121",),
        ("w sposób budzący zaufanie do organów podatkowych",),
    ),
    (
        re.compile(r"\b(dok[łl]adn\w* wyja[śs]nien\w* stanu faktycznego|prawd\w* obiektywn\w*)\b", re.IGNORECASE),
        (),
        ("122",),
        ("wszelkie niezbędne działania w celu dokładnego wyjaśnienia stanu faktycznego",),
    ),
    (
        re.compile(r"\b(wszcz[ęe]ci\w* post[ęe]powani\w* podatkow\w*)\b", re.IGNORECASE),
        (),
        ("165",),
        ("postępowanie podatkowe wszczyna się",),
    ),
    (
        re.compile(r"\b(jako dow[óo]d.*dopu[śs]ci[ćc].*wszystko|dowod\w*.*wyja[śs]nien\w* sprawy)\b", re.IGNORECASE),
        (),
        ("180",),
        ("Jako dowód należy dopuścić wszystko",),
    ),
    (
        re.compile(r"\b(ca[łl]y materia[łl] dowodow\w*|zebra[ćc].*rozpatrzy[ćc].*materia[łl] dowodow\w*)\b", re.IGNORECASE),
        (),
        ("187",),
        ("zebrać i w sposób wyczerpujący rozpatrzyć cały materiał dowodowy",),
    ),
    (
        re.compile(r"\b(swobodn\w* ocen\w* dowod\w*|ocen\w*.*ca[łl]okszta[łl]t\w* materia[łl]u dowodow\w*)\b", re.IGNORECASE),
        (),
        ("191",),
        ("organ podatkowy ocenia na podstawie całego zebranego materiału dowodowego",),
    ),
    (
        re.compile(r"\b(element\w* decyzj\w*|uzasadnieni\w* decyzj\w*|decyzj\w* podatkow\w*)\b", re.IGNORECASE),
        (),
        ("210",),
        ("Decyzja zawiera", "uzasadnienie faktyczne i prawne"),
    ),
    (
        re.compile(r"\b(rygor\w* natychmiastow\w* wykonalno\w*|decyzj\w* nieostateczn\w*.*wykonalno\w*)\b", re.IGNORECASE),
        (),
        ("239b",),
        ("decyzji nieostatecznej może być nadany rygor natychmiastowej wykonalności",),
    ),
    (
        re.compile(r"\b(ksef|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*|awari\w* ksef|niedostępno\w* ksef|niedostepno\w* ksef)\b", re.IGNORECASE),
        ("106g", "106n"),
        (),
        ("Krajowy System e-Faktur", "faktura ustrukturyzowana", "awaria Krajowego Systemu e-Faktur", "niedostępności Krajowego Systemu e-Faktur"),
    ),
    (
        re.compile(r"\b(paragon\w*|kas\w* rejestrując\w*|kas\w* rejestruj\w*|faktur\w* do paragon\w*)\b", re.IGNORECASE),
        (),
        ("106h",),
        ("sprzedaży zaewidencjonowanej przy zastosowaniu kasy rejestrującej",),
    ),
    (
        re.compile(r"\b(not\w* korygując\w*|nota koryguj\w*|pomyłk\w*.*faktur\w*)\b", re.IGNORECASE),
        (),
        ("106k",),
        ("notą korygującą",),
    ),
    (
        re.compile(r"\b(podzielon\w* płatno\w*|split payment|rachunk\w* vat|załącznik(?:a)? nr 15|zalacznik(?:a)? nr 15)\b", re.IGNORECASE),
        ("108",),
        (),
        ("mechanizm podzielonej płatności", "rachunek VAT", "załączniku nr 15 do ustawy"),
    ),
    (
        re.compile(r"\b(platform\w*|ułatwiaj\w* dostaw\w*|ulatwiaj\w* dostaw\w*|rozporządzeni\w* 282/2011|rozporzadzeni\w* 282/2011)\b", re.IGNORECASE),
        (),
        ("103b",),
        ("podatnik ułatwiający dostawy towarów", "art. 5c rozporządzenia 282/2011"),
    ),
    (
        re.compile(r"\b(przechowywani\w* faktur|przechowywani\w* ewidencj\w*|dokumentacj\w*.*vat)\b", re.IGNORECASE),
        (),
        ("112", "112a", "112aa"),
        ("Podatnicy są obowiązani przechowywać ewidencje", "Podatnicy przechowują", "Faktury ustrukturyzowane są przechowywane"),
    ),
    (
        re.compile(r"\b(tax free|podróżn\w*|podrozn\w*|zwrot\w* podatku podróżnemu|zwrot\w* podatku podroznemu)\b", re.IGNORECASE),
        (),
        ("127a", "129"),
        ("dokument elektroniczny TAX FREE", "zwrotu podatku podróżnemu"),
    ),
    (
        re.compile(r"\b(oss\b|procedur\w* unijn\w*|państw\w* członkowsk\w* identyfikacji|panstw\w* czlonkowsk\w* identyfikacji)\b", re.IGNORECASE),
        ("130",),
        (),
        ("procedura unijna", "państwie członkowskim identyfikacji"),
    ),
    (
        re.compile(r"\b(procedur\w* nieunijn\w*|podmiot\w* zagraniczn\w*)\b", re.IGNORECASE),
        (),
        ("131", "132", "133", "134"),
        ("procedura nieunijna", "podmioty zagraniczne"),
    ),
    (
        re.compile(r"\b(przewoz\w* okazjonaln\w*|przewoz\w* osób autobus\w*|przewoz\w* osob autobus\w*|autobus\w*)\b", re.IGNORECASE),
        (),
        ("134a", "134b", "134c"),
        ("przewozu osób autobusami", "przewozy okazjonalne"),
    ),
    (
        re.compile(r"\b(ioss\b|procedur\w* importu|sprzedaż na odległość towarów importowanych|sprzedaz na odleglosc towarow importowanych|pośrednik\w*.*ioss|posrednik\w*.*ioss|pośrednik\w*.*importu|posrednik\w*.*importu)\b", re.IGNORECASE),
        ("138",),
        (),
        ("procedura importu", "sprzedaży na odległość towarów importowanych", "pośrednikiem może być"),
    ),
    (
        re.compile(r"\b(transakcj\w* trójstronn\w*|transakcj\w* trojstronn\w*|procedur\w* uproszczon\w*)\b", re.IGNORECASE),
        (),
        ("136", "138"),
        ("procedura uproszczona",),
    ),
    (
        re.compile(r"\b(przedmiot\w* opodatkowani\w*|co jest przedmiotem opodatkowania)\b", re.IGNORECASE),
        (),
        ("7",),
        ("przedmiotem opodatkowania podatkiem dochodowym jest dochód",),
    ),
    (
        re.compile(r"\b(definiuje przychod\w*|jak ustawa .*przychod\w*|przychod\w* należn\w*|przychod\w* nalezn\w*|nie zalicza si[eę] do przychod[óo]w)\b", re.IGNORECASE),
        (),
        ("12",),
        ("przychodami są", "za przychody związane z działalnością gospodarczą", "do przychodów nie zalicza się"),
    ),
    (
        re.compile(r"\b(koszt\w* bezpośredni\w*|koszt\w* bezposredni\w*|potr[ąa]ci[ćc].*po zako[nń]czeniu roku)\b", re.IGNORECASE),
        (),
        ("15",),
        ("koszty uzyskania przychodów bezpośrednio związane z przychodami",),
    ),
    (
        re.compile(r"\b(darowizn\w*|odliczeni\w* od podstawy opodatkowania)\b", re.IGNORECASE),
        (),
        ("18",),
        ("darowizn przekazanych na cele określone",),
    ),
    (
        re.compile(r"\b(z[łl]e d[łl]ugi|90 dni od dnia up[łl]ywu terminu zap[łl]aty)\b", re.IGNORECASE),
        (),
        ("18f",),
        ("może być zmniejszona o zaliczaną do przychodów należnych wartość wierzytelności", "upłynęło 90 dni od dnia upływu terminu zapłaty"),
    ),
    (
        re.compile(r"\b(podatek u źr[óo]dła|odsetek|należno[śs]ci licencyjn\w*|know-how)\b", re.IGNORECASE),
        (),
        ("21",),
        ("podatek dochodowy z tytułu uzyskanych", "z odsetek", "know-how"),
    ),
    (
        re.compile(r"\b(wht|podatek u źr[óo]dła|certyfikat\w* rezydencji|rzeczywist\w* właściciel\w*|beneficial owner|należyt\w* staranno\w*|look-through|nie pobra\w* podatku u źr[óo]dła|zagraniczn\w* zakład\w*)\b", re.IGNORECASE),
        (),
        ("21", "22", "26", "41"),
        (
            "certyfikat rezydencji",
            "rzeczywistym właścicielem",
            "dochowania należytej staranności",
            "nie pobierać podatku",
            "zagraniczny zakład",
        ),
    ),
    (
        re.compile(r"\b(esto[ńn]sk(?:i|iego)?\s+cit|rycza[łl]t(?:em)? od dochod[óo]w sp[óo][łl]ek)\b", re.IGNORECASE),
        ("28",),
        (),
        ("opodatkowaniu ryczałtem może podlegać podatnik", "przepisów niniejszego rozdziału nie stosuje się do"),
    ),
    (
        re.compile(r"\b(warunk\w* wej[śs]cia|warunki wej[śs]cia|może podlega[ćc] rycza[łl]towi|moze podlegac ryczaltowi)\b", re.IGNORECASE),
        (),
        ("28j",),
        ("opodatkowaniu ryczałtem może podlegać podatnik",),
    ),
    (
        re.compile(r"\b(wy[łl][ąa]czeni\w* z mo[żz]liwo[śs]ci stosowania|kto nie mo[żz]e stosowa[ćc]|przepis[óo]w niniejszego rozdzia[łl]u nie stosuje si[eę])\b", re.IGNORECASE),
        (),
        ("28k",),
        ("przepisów niniejszego rozdziału nie stosuje się do",),
    ),
    (
        re.compile(r"\b(słowniczek poję[ćc]|zawiera słowniczek|definicje esto[ńn]skiego cit|pojęcia używane w rozdziale)\b", re.IGNORECASE),
        (),
        ("28c",),
        ("Ilekroć w niniejszym rozdziale jest mowa o",),
    ),
    (
        re.compile(r"\b(jakich innych podatk\w* i rozdział\w* nie stosuje si[eę]|nie stosuje się do podatnika w esto[ńn]skim cit|wyłączenie innych podatk\w*)\b", re.IGNORECASE),
        (),
        ("28h",),
        ("Podatnik opodatkowany ryczałtem nie podlega opodatkowaniu na zasadach określonych",),
    ),
    (
        re.compile(r"\b(przedmiot opodatkowani\w* esto[ńn]sk(?:iego|im)? cit|jakie kategori\w* dochodu podlegaj\w* opodatkowaniu esto[ńn]skim cit)\b", re.IGNORECASE),
        (),
        ("28m",),
        ("Opodatkowaniu ryczałtem podlega dochód odpowiadający",),
    ),
    (
        re.compile(r"\b(stawki esto[ńn]skiego cit|stawka esto[ńn]skiego cit|ryczałt wynosi)\b", re.IGNORECASE),
        (),
        ("28o",),
        ("Ryczałt wynosi",),
    ),
    (
        re.compile(r"\b(odliczeni\w* podatk\w* zagraniczn\w* esto[ńn]sk(?:iego|im)? cit|podatk\w* zapłacon\w* za granicą.*esto[ńn]sk(?:iego|im)? cit)\b", re.IGNORECASE),
        (),
        ("28p",),
        ("odlicza się kwotę równą podatkowi zapłaconemu w obcym państwie",),
    ),
    (
        re.compile(r"\b(podatkow\w* grup\w* kapitał\w*.*podatk\w* od budynk\w*|podatk\w* od budynk\w*.*podatkow\w* grup\w* kapitał\w*)\b", re.IGNORECASE),
        (),
        ("24c",),
        ("W przypadku podatkowej grupy kapitałowej suma przychodów",),
    ),
    (
        re.compile(r"\b(ip\s*box.*ewidenc|ewidencyjn\w*.*ip\s*box|kwalifikowan\w* praw\w* własności intelektualnej.*ewidenc)\b", re.IGNORECASE),
        (),
        ("24e",),
        ("Podatnicy podlegający opodatkowaniu na podstawie art. 24d są obowiązani", "ewidencji rachunkowej"),
    ),
    (
        re.compile(r"\b(exit\s+tax.*sp[óo]łk\w* niebędąc\w* osobą prawną|sp[óo]łk\w* niebędąc\w* osobą prawną.*exit\s+tax|sp[óo]łk\w* niebędąc\w* osobą prawną.*niezrealizowanych zysk\w*)\b", re.IGNORECASE),
        (),
        ("24k",),
        ("W przypadku gdy przenoszącym składnik majątku jest spółka niebędąca osobą prawną",),
    ),
    (
        re.compile(r"\b(traci prawo albo obowiązek kontynuowania|kontynuowania części ulg|po wejściu w esto[ńn]ski cit.*ulg)\b", re.IGNORECASE),
        (),
        ("18aa",),
        ("traci odpowiednio prawo albo obowiązek do ich kontynuowania",),
    ),
)

_cross_encoder: Any = None
_cross_encoder_load_failed = False
_cross_encoder_lock = threading.Lock()
_index_refresh_lock = threading.Lock()

INDEX_BUILD_VERSION = "provision_units_v3"


@dataclass(frozen=True)
class RagConfig:
    processed_path: Path
    additional_source_paths: tuple[Path, ...]
    db_path: Path
    chunk_target_chars: int
    chunk_overlap_chars: int
    retrieval_limit: int
    max_context_chars: int
    document_context_enabled: bool
    document_context_document_limit: int
    document_context_max_chars: int
    supabase_batch_size: int
    supabase_chunk_batch_size: int
    supabase_request_timeout: float
    supabase_max_retries: int
    supabase_state_path: Path
    retrieval_max_chunks_per_document: int
    embedding_dimensions: int
    hybrid_lexical_weight: float
    hybrid_semantic_weight: float
    candidate_pool_limit: int
    legal_match_weight: float
    cross_encoder_model: str
    cross_encoder_enabled: bool
    cross_encoder_cache_path: Path
    cross_encoder_candidate_limit: int
    cross_encoder_weight: float
    cross_encoder_device: str
    facts_channel_enabled: bool
    domain_filter_enabled: bool
    mechanism_match_weight: float
    mechanism_lexicon_path: Path
    facts_rerank_weight: float
    judgment_match_weight: float


def index_content_fingerprint(record: dict[str, Any]) -> str:
    """Invalidate persisted chunks when parser/chunker semantics change."""
    source_hash = str(record.get("content_sha256") or "")
    return hashlib.sha256(
        f"{source_hash}\0{INDEX_BUILD_VERSION}".encode("utf-8")
    ).hexdigest()


def extract_normalized_provision_references(
    text: str,
    declared: Iterable[str] = (),
) -> list[str]:
    values = [match.group(0) for match in EXACT_PROVISION_REFERENCE_RE.finditer(text)]
    for declared_value in declared:
        raw_value = str(declared_value)
        # Eureka metadata encodes hierarchy with hyphens, for example
        # ``...-art. 21-ust. 1-pkt 131``.  Convert only structural separators,
        # then persist the same canonical citation used for natural text.
        expanded_value = re.sub(
            r"-(?=(?:art|ust|pkt|lit)(?:\.|\s))",
            " ",
            raw_value,
            flags=re.IGNORECASE,
        )
        declared_matches = [
            match.group(0)
            for match in EXACT_PROVISION_REFERENCE_RE.finditer(expanded_value)
        ]
        values.extend(declared_matches or [raw_value])
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_provision_reference(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float
    chunk_text: str
    subject: str
    signature: Optional[str]
    published_date: Optional[str]
    source_url: Optional[str]
    category: Optional[str]
    source: str = ""
    source_type: str = "interpretation"
    source_subtype: Optional[str] = None
    authority: Optional[str] = None
    publication: Optional[str] = None
    legal_state_date: Optional[str] = None
    source_pages: list[int] = field(default_factory=list)
    legal_provisions: list[str] = field(default_factory=list)
    evidence_role: str = ""


@dataclass(frozen=True)
class RagDocumentContext:
    document_id: str
    subject: str
    signature: Optional[str]
    published_date: Optional[str]
    source_url: Optional[str]
    category: Optional[str]
    source: str
    source_type: str
    source_subtype: Optional[str]
    authority: Optional[str]
    publication: Optional[str]
    legal_state_date: Optional[str]
    source_pages: list[int]
    legal_provisions: list[str]
    text: str
    seed_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalInspection:
    query: str
    match_query: Optional[str]
    requested_limit: int
    retrieved_count: int
    selected_count: int
    selected_context_chars: int
    hits: list[dict[str, Any]]
    chunks: list[RagChunk]
    raw_candidate_pool: list[dict[str, Any]]


@dataclass(frozen=True)
class LegalRetrievalAxis:
    axis_id: str
    label: str
    query: str
    source_types: Optional[set[str]] = None
    tax_domains: Optional[set[str]] = None
    preferred_targets: tuple[tuple[str, str], ...] = ()
    direct_subject_prefix: Optional[str] = None
    limit_fraction: float = 1.0


@dataclass(frozen=True)
class LegalSourcePlan:
    query: str
    axes: tuple[LegalRetrievalAxis, ...]
    primary_source_types: tuple[str, ...]
    secondary_source_types: tuple[str, ...]
    statute_targets: tuple[tuple[str, str], ...]
    explicit_statute_targets: tuple[tuple[str, str], ...]
    tax_domains: tuple[str, ...]
    primary_required: bool
    stage_order: tuple[str, ...]


@dataclass(frozen=True)
class SourceRequirement:
    axis_id: str
    mandatory_primary_sources: list[str] = field(default_factory=list)
    optional_secondary_sources: list[str] = field(default_factory=list)
    controlling_rule_required: bool = True
    current_law_required: bool = True
    treaty_required: bool = False
    eu_source_required: bool = False
    official_guidance_required: bool = False


@dataclass(frozen=True)
class AxisCoverage:
    axis_id: str
    label: str
    controlling_rule_present: bool
    current_law_source_present: bool
    relevant_resolution_present: bool
    primary_source_present: bool
    required_treaty_present: Optional[bool]
    missing_source_types: list[str]
    misleading_neighbor_present: bool
    coverage_score: float
    status: str
    supporting_source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalRule:
    source_id: str
    act_title: str
    publication: Optional[str]
    legal_state_date: Optional[str]
    provision_id: str
    citation: str
    article_key: str
    paragraph: Optional[str]
    point: Optional[str]
    letter: Optional[str]
    rule_type: str
    condition: str
    directive: str
    exact_source_span: str
    required_facts: list[str] = field(default_factory=list)
    definition_dependencies: list[str] = field(default_factory=list)
    scope_subject_terms: list[str] = field(default_factory=list)
    scope_object_terms: list[str] = field(default_factory=list)
    specificity_rank: int = 0
    retrieval_stage: str = "primary_source_exact_lookup"
    supporting_chunk_ids: list[str] = field(default_factory=list)
    source_url: Optional[str] = None


HISTORICAL_PROVISION_END_DATES: dict[
    tuple[str, str, Optional[str], Optional[str], Optional[str]], str
] = {
    # Historical VAT bad-debt-relief conditions. They must remain available
    # for historical analysis, but never control a claim after their repeal.
    ("VAT", "89a", "2", "1", None): "2021-09-30",
    ("VAT", "89a", "2", "2", None): "2021-09-30",
    ("VAT", "89a", "2", "3", "b"): "2021-09-30",
}


def legal_rule_effective_to(rule: LegalRule) -> Optional[str]:
    domain = "VAT" if "towarów i usług" in rule.act_title.lower() or "vat" in rule.source_id.lower() else ""
    exact = HISTORICAL_PROVISION_END_DATES.get(
        (domain, rule.article_key.lower(), rule.paragraph, rule.point, rule.letter)
    )
    return exact or HISTORICAL_PROVISION_END_DATES.get(
        (domain, rule.article_key.lower(), rule.paragraph, rule.point, None)
    )


def legal_rule_is_effective_on(rule: LegalRule, target_date: str) -> bool:
    try:
        parsed_target = date.fromisoformat(target_date[:10])
    except ValueError:
        return False
    effective_to = legal_rule_effective_to(rule)
    if effective_to and parsed_target > date.fromisoformat(effective_to):
        return False
    if rule.rule_type == "repealed":
        return False
    effective_from = legal_rule_effective_from(rule)
    if effective_from and date.fromisoformat(effective_from) > parsed_target:
        return False
    return True


def filter_legal_rules_for_target_date(
    rules: list[LegalRule], target_date: str
) -> list[LegalRule]:
    return [rule for rule in rules if legal_rule_is_effective_on(rule, target_date)]


def legal_rule_effective_from(rule: LegalRule) -> str:
    value = normalize_whitespace(rule.legal_state_date or rule.publication or "")
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    return match.group(0) if match else ""


def get_rag_config() -> RagConfig:
    configured_extra_sources = os.getenv("ALITIGATOR_RAG_ADDITIONAL_SOURCE_PATHS", "").strip()
    additional_source_paths = tuple(
        Path(value.strip())
        for value in configured_extra_sources.split(",")
        if value.strip()
    )
    if not configured_extra_sources:
        additional_source_paths = tuple(path for path in DEFAULT_LAW_SOURCE_PATHS if path.exists())
    return RagConfig(
        processed_path=Path(os.getenv("ALITIGATOR_RAG_SOURCE_PATH", DEFAULT_PROCESSED_PATH)),
        additional_source_paths=additional_source_paths,
        db_path=Path(os.getenv("ALITIGATOR_RAG_DB_PATH", DEFAULT_RAG_DB_PATH)),
        chunk_target_chars=int(os.getenv("ALITIGATOR_RAG_CHUNK_TARGET_CHARS", "1400")),
        chunk_overlap_chars=int(os.getenv("ALITIGATOR_RAG_CHUNK_OVERLAP_CHARS", "220")),
        retrieval_limit=int(os.getenv("ALITIGATOR_RAG_TOP_K", "6")),
        max_context_chars=int(os.getenv("ALITIGATOR_RAG_MAX_CONTEXT_CHARS", "9000")),
        document_context_enabled=os.getenv("ALITIGATOR_RAG_DOCUMENT_CONTEXT_ENABLED", "true").lower()
        in {"1", "true", "yes"},
        document_context_document_limit=int(
            os.getenv("ALITIGATOR_RAG_DOCUMENT_CONTEXT_DOCUMENTS", os.getenv("ALITIGATOR_RAG_TOP_K", "6"))
        ),
        document_context_max_chars=int(os.getenv("ALITIGATOR_RAG_DOCUMENT_CONTEXT_MAX_CHARS", "120000")),
        supabase_batch_size=int(os.getenv("ALITIGATOR_RAG_SUPABASE_BATCH_SIZE", "200")),
        supabase_chunk_batch_size=int(os.getenv("ALITIGATOR_RAG_SUPABASE_CHUNK_BATCH_SIZE", "500")),
        supabase_request_timeout=float(os.getenv("ALITIGATOR_RAG_SUPABASE_TIMEOUT_SECONDS", "60")),
        supabase_max_retries=int(os.getenv("ALITIGATOR_RAG_SUPABASE_MAX_RETRIES", "4")),
        supabase_state_path=Path(
            os.getenv(
                "ALITIGATOR_RAG_SUPABASE_STATE_PATH",
                str(API_DIR / "data" / "processed" / "eureka_supabase_sync_state.json"),
            )
        ),
        retrieval_max_chunks_per_document=int(
            os.getenv("ALITIGATOR_RAG_MAX_CHUNKS_PER_DOCUMENT", "2")
        ),
        embedding_dimensions=int(os.getenv("ALITIGATOR_RAG_EMBEDDING_DIMENSIONS", "256")),
        hybrid_lexical_weight=float(os.getenv("ALITIGATOR_RAG_HYBRID_LEXICAL_WEIGHT", "0.65")),
        hybrid_semantic_weight=float(os.getenv("ALITIGATOR_RAG_HYBRID_SEMANTIC_WEIGHT", "0.35")),
        candidate_pool_limit=int(os.getenv("ALITIGATOR_RAG_CANDIDATE_POOL_LIMIT", "120")),
        legal_match_weight=float(os.getenv("ALITIGATOR_RAG_LEGAL_MATCH_WEIGHT", "0.02")),
        cross_encoder_model=os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL),
        cross_encoder_enabled=os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_ENABLED", "true").lower()
        in {"1", "true", "yes"},
        cross_encoder_cache_path=Path(
            os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_CACHE", API_DIR / "data" / "models")
        ),
        cross_encoder_candidate_limit=int(
            os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_CANDIDATE_LIMIT", "12")
        ),
        cross_encoder_weight=float(os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_WEIGHT", "0.70")),
        cross_encoder_device=os.getenv("ALITIGATOR_RAG_CROSS_ENCODER_DEVICE", "cpu"),
        facts_channel_enabled=os.getenv("ALITIGATOR_RAG_FACTS_CHANNEL_ENABLED", "false").lower()
        in {"1", "true", "yes"},
        domain_filter_enabled=os.getenv("ALITIGATOR_RAG_DOMAIN_FILTER_ENABLED", "false").lower()
        in {"1", "true", "yes"},
        mechanism_match_weight=float(os.getenv("ALITIGATOR_RAG_MECHANISM_MATCH_WEIGHT", "0.015")),
        mechanism_lexicon_path=Path(os.getenv("ALITIGATOR_RAG_MECHANISM_LEXICON_PATH", API_DIR / "data" / "processed" / "mechanism_lexicon.json")),
        facts_rerank_weight=float(os.getenv("ALITIGATOR_RAG_FACTS_RERANK_WEIGHT", "0.01")),
        judgment_match_weight=float(os.getenv("ALITIGATOR_RAG_JUDGMENT_MATCH_WEIGHT", "0.18")),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_boilerplate(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    match = BOILERPLATE_SECTION_RE.search(normalized)
    if match:
        normalized = normalized[: match.start()]
    return normalized.strip()


def clean_document_text(record: dict[str, Any]) -> str:
    pieces: list[str] = []
    subject = normalize_whitespace(str(record.get("subject") or ""))
    if subject:
        pieces.append(subject)

    body = strip_boilerplate(str(record.get("content_text") or ""))
    if body:
        pieces.append(body)

    return "\n\n".join(piece for piece in pieces if piece).strip()


def extract_question_text(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not QUESTION_HEADING_RE.fullmatch(line):
            continue
        parts: list[str] = []
        for candidate in lines[index + 1 :]:
            if not candidate:
                if parts:
                    continue
                continue
            if parts and SECTION_HEADING_RE.match(candidate):
                break
            parts.append(candidate)
        return normalize_whitespace(" ".join(parts))[:4000]
    return ""


def extract_facts_text(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not re.match(r"^(?:Opis|Stan faktyczny|Zdarzenie przyszłe)", line, re.IGNORECASE):
            continue
        parts: list[str] = []
        for candidate in lines[index + 1 :]:
            if QUESTION_HEADING_RE.fullmatch(candidate):
                break
            if parts and SECTION_HEADING_RE.match(candidate):
                break
            if candidate:
                parts.append(candidate)
        return normalize_whitespace(" ".join(parts))[:5000]
    return ""


def extract_decision_text(text: str) -> str:
    match = re.search(
        r"(?:Państwa stanowisko|stanowisko).{0,800}?\b(?:jest|uznano za)\s+"
        r"(?:prawidłowe|nieprawidłowe|częściowo prawidłowe)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return normalize_whitespace(match.group(0)) if match else ""


def query_targets_interpretation_procedure(query: str) -> bool:
    return bool(INTERPRETATION_PROCEDURAL_QUERY_RE.search(query or ""))


def query_targets_ksef_foreign_sale(query: str) -> bool:
    normalized = normalize_whitespace(query or "")
    return bool(KSEF_QUERY_RE.search(normalized) and KSEF_FOREIGN_SALE_QUERY_RE.search(normalized))


def query_targets_ksef_current_law(query: str) -> bool:
    return bool(KSEF_QUERY_RE.search(query or ""))


def query_targets_ksef_transition_period(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    return bool(
        re.search(
            r"\b(2026|2027|pa[źz]dziernik\w*|luty\w*|kwietni\w*|termin\w*|etap\w*|"
            r"limit\w*|10\s*000|10000|200\s*mln|sankcj\w*|kar\w*|106ni)\b",
            normalized,
        )
    )


def query_targets_ksef_operational_modes(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    return bool(
        re.search(
            r"\b(offline24|offline\s*24|awari\w*|niedost[ęe]pno\w*|system\w* ksi[ęe]gow\w*|"
            r"w[łl]asn\w* system\w*|qr|106nda|106nf|106nh)\b",
            normalized,
        )
    )


def query_targets_ksef_fixed_establishment_scope(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    return bool(
        re.search(
            r"\b(smpd|sta[łl]\w* miejsce prowadzenia|fixed establishment|podmiot\w* zagraniczn\w*|"
            r"niemieck\w*|niemc\w*|zagraniczn\w* sp[óo][łl]k\w*|uczestnicz\w*|us[łl]ugodawc\w*)\b",
            normalized,
        )
    )


def query_targets_ksef_buyer_capacity(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    return bool(
        re.search(
            r"\b(b2c|b2b|konsument\w*|osob\w* fizyczn\w*|nip|prywatn\w*|mieszan\w*|"
            r"jednoosobow\w* dzia[łl]alno[śs][ćc]|przedsi[ęe]biorc\w*)\b",
            normalized,
        )
    )


def query_targets_ksef_outside_deduction(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    outside_ksef = bool(
        re.search(
            r"\b(poza\s+ksef|bez\s+użycia\s+ksef|bez\s+uzycia\s+ksef|wbrew\s+obowiązkowi.*ksef|"
            r"wbrew\s+obowiazkowi.*ksef|wystawion\w*\s+poza\s+ksef|otrzyman\w*\s+poza\s+ksef|"
            r"dostarczon\w*\s+poza\s+ksef)\b",
            normalized,
        )
    )
    deduction_or_correction = bool(
        re.search(
            r"\b(odliczen\w*|prawo\s+do\s+odliczenia|art\.\s*86|art\.\s*88|jpk|jpk_v7|"
            r"bfk|ponown\w*\s+przesłan\w*|ponown\w*\s+przeslan\w*|"
            r"faktur\w*\s+koryguj\w*|not[ąa]?\s+koryguj\w*|art\.\s*108|"
            r"ta\s+sama\s+transakcj\w*|dokumentuj\w*c\w*\s+tej\s+samej\s+transakcj\w*|"
            r"przesłać\s+ponownie\s+za\s+pośrednictwem\s+ksef|przeslac\s+ponownie\s+za\s+posrednictwem\s+ksef)\b",
            normalized,
        )
    )
    return outside_ksef and deduction_or_correction


def query_targets_ksef_correction_issue(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    if not KSEF_QUERY_RE.search(normalized):
        return False
    has_correction_or_buyer_data = bool(
        re.search(
            r"\b(not[ąa]?\s+koryguj\w*|błędn\w*\s+danych?\s+nabywc\w*|bledn\w*\s+danych?\s+nabywc\w*|"
            r"danych?\s+innego\s+nabywc\w*|adres\w*|dane\s+nabywc\w*|korekt\w*\s+faktur\w*)\b",
            normalized,
        )
    )
    return has_correction_or_buyer_data


def query_targets_debt_assumption_effectiveness(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    return bool(
        re.search(
            r"(\bprzejęci\w*\s+dług\w*|\bprzejęci\w*\s+zobowiązan\w*|\bzwolnieni\w*\s+z dług\w*|"
            r"\bzmian\w*\s+dłużnik\w*|\bzmian\w*\s+dluznik\w*|\bzgod\w*\s+wierzyciel\w*|"
            r"art\.\s*519|art\.\s*520|art\.\s*521|art\.\s*508)\b",
            normalized,
        )
    )


def query_targets_housing_relief_temporary_rental(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_housing_relief = bool(
        re.search(
            r"\b(ulg\w*\s+mieszkaniow\w*|własn\w*\s+cele\s+mieszkaniow\w*|wlasn\w*\s+cele\s+mieszkaniow\w*|art\.\s*21\s*ust\.\s*1\s*pkt\s*131)\b",
            normalized,
        )
    )
    has_temporary_rental = bool(
        re.search(r"\b(czasow\w*\s+wynaj\w*|wynajmow\w*\s+lokal\w*|wynajem\w*\s+zakupion\w*\s+lokal\w*|wynaj\w*.*lokal\w*|lokal\w*.*wynaj\w*)\b", normalized)
    )
    return has_housing_relief and has_temporary_rental


def query_targets_housing_relief_loan_repayment(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_housing_relief = bool(
        re.search(
            r"\b(ulg\w*\s+mieszkaniow\w*|własn\w*\s+cele\s+mieszkaniow\w*|wlasn\w*\s+cele\s+mieszkaniow\w*|art\.\s*21\s*ust\.\s*1\s*pkt\s*131)\b",
            normalized,
        )
    )
    has_loan_repayment = bool(
        re.search(
            r"\b(sp[łl]at\w*\s+rat\w*|kredyt\w*\s+zaci[ąa]gni[ęe]t\w*\s+przed\s+sprzedaż\w*|przychód\w*\s+ze\s+sprzedaż\w*.*sp[łl]at\w*\s+kredyt\w*)\b",
            normalized,
        )
    )
    return has_housing_relief and has_loan_repayment


def query_targets_property_sale_pit(query: str) -> bool:
    """Detect a private real-estate sale that needs the PIT source bundle.

    Taxpayers rarely name the relief or its article before asking whether a
    sale is taxable.  Treating that phrasing as a generic prose query meant the
    writer could know the rule in article 10 from model memory while retrieval
    supplied only a later housing-relief provision.  The general tax source,
    relief and rate therefore form one mandatory retrieval bundle.
    """
    normalized = normalize_whitespace(query or "").lower()
    has_sale = bool(re.search(r"\b(sprzeda\w*|odpłatn\w*\s+zbyci\w*|zby\w*)\b", normalized))
    has_property = bool(re.search(r"\b(mieszkani\w*|lokal\w*|nieruchomo\w*|budyn\w*|grunt\w*)\b", normalized))
    has_pit_context = bool(
        re.search(r"\b(pit|podat\w*|kupi\w*|naby\w*|doch[oó]d\w*|przych[oó]d\w*)\b", normalized)
    )
    return has_sale and has_property and has_pit_context


def query_targets_mortgage_settlement_refund(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_mortgage_or_settlement = bool(
        re.search(
            r"\b(ugod\w*\s+z\s+bankiem|kredyt\w*\s+hipoteczn\w*|kredyt\w*\s+mieszkaniow\w*|"
            r"zaniechan\w*\s+pobor\w*|art\.\s*52i|umorzen\w*\s+zadłużen\w*|umorzen\w*\s+zadluzen\w*|"
            r"skapitalizowan\w*\s+odsetk\w*)\b",
            normalized,
        )
    )
    has_bank_fee_context = bool(
        re.search(r"\b(prowizj\w*|opłat\w*|oplata\w*)\b", normalized)
        and re.search(r"\b(bank\w*|kredyt\w*|pożycz\w*|pozycz\w*|ugod\w*)\b", normalized)
    )
    return has_mortgage_or_settlement or has_bank_fee_context


def query_targets_vat_dropshipping_ioss(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_platform_or_dropshipping = bool(
        re.search(
            r"\b(dropshipping\w*|platform\w*|platforma\w*|interfejs\w* elektroniczn\w*|"
            r"pośrednik\w*|posrednik\w*|sprzedaż na odległość\w* towarów importowanych|"
            r"sprzedaz na odleglosc\w* towarow importowanych|soti|iioss|ioss|150\s*euro|"
            r"towar\w* importowan\w*|importowan\w* towar\w*)\b",
            normalized,
        )
    )
    return has_platform_or_dropshipping


def query_targets_ksef_b2c_invoice(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    return bool(
        KSEF_QUERY_RE.search(normalized)
        and re.search(r"\b(b2c|konsument\w*|osob[ąa]\s+fizyczn\w*\s+nieprowadząc\w*\s+działalno\w*|nieprowadząc\w*\s+działalno\w*)\b", normalized)
        and re.search(r"\b(faktur\w*|wystawien\w*|ż[ąa]danie|zadanie|prosz\w*)\b", normalized)
    )


def query_targets_private_vehicle_pit_expense(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_vehicle = bool(re.search(r"\b(samochod\w*|samochód\w*|pojazd\w*|auto\b)\b", normalized))
    has_private_use = bool(re.search(r"\b(prywatn\w*|niewprowadzon\w* do ewidencji|nie wprowadzon\w* do ewidencji|poza ewidencj\w*)\b", normalized))
    has_costs = bool(re.search(r"\b(koszt\w*|wydatk\w*|eksploatac\w*|używan\w*|uzywan\w*)\b", normalized))
    has_pit = "pit" in normalized or "podatku dochodowego od osób fizycznych" in normalized
    return has_vehicle and has_private_use and has_costs and has_pit


def query_targets_spolka_komandytowa_cit_status(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_spk = bool(re.search(r"\b(sp[óo]łk\w* komandytow\w*|sp\.?\s*k\.?)\b", normalized))
    has_status = bool(re.search(r"\b(podatnik\w*|status podatkow\w*|cit|transparentn\w* podatkow\w*|opodatkow\w* wył[ąa]cznie na poziomie wspólnik\w*)\b", normalized))
    return has_spk and has_status


def query_targets_invoice_address_error(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_invoice = bool(re.search(r"\b(faktur\w*|jpk|vat)\b", normalized))
    has_address = bool(re.search(r"\b(adres\w*|nieaktualn\w* adres\w*|błędn\w* adres\w*|bledn\w* adres\w*|wad[ąa]\s+techniczn\w*)\b", normalized))
    has_identifiers = bool(re.search(r"\b(nip|nazwa|kwot\w*|dane nabywc\w*)\b", normalized))
    return has_invoice and has_address and has_identifiers


def query_targets_fixed_establishment_vat(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_fixed_establishment = bool(
        re.search(r"\b(stał\w*\s+miejsce\s+prowadzenia\s+działalno\w*|fixed establishment|fe\b)\b", normalized)
    )
    has_vat_service_context = bool(
        re.search(r"\b(vat|usług\w*|doradcz\w*|b2b|świadczen\w*|miejsce świadczenia|siedzib\w*)\b", normalized)
    )
    return has_fixed_establishment and has_vat_service_context


def query_targets_family_foundation_mechanism(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_foundation = bool(re.search(r"\b(fundacj\w*\s+rodzinn\w*|ufr\b|beneficjent\w*)\b", normalized))
    has_activity = bool(
        re.search(
            r"\b(najem\w*|wynaj\w*|sprzeda\w*|zbyc\w*|pożycz\w*|pozycz\w*|spółk\w*|spolk\w*|"
            r"udział\w*|udzial\w*|akcj\w*|świadczen\w*|swiadczen\w*|lokal\w*|nieruchomo\w*|turyst\w*)\b",
            normalized,
        )
    )
    return has_foundation and has_activity


def query_targets_wht_pay_and_refund_services(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_wht_context = bool(re.search(r"\b(wht|podatek u źr[óo]dła|withholding|certyfikat\w* rezydencji|beneficial owner|rzeczywist\w* właściciel\w*)\b", normalized))
    has_pay_and_refund = bool(re.search(r"\b(pay and refund|2 mln|2 000 000|art\.\s*26\s*ust\.\s*2e|próg\w*|prog\w*|limit\w*)\b", normalized))
    has_service_or_distribution = bool(
        re.search(r"\b(dywidend\w*|odsetk\w*|zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|doradcz\w*)\b", normalized)
    )
    return has_wht_context and has_pay_and_refund and has_service_or_distribution


def is_procedural_interpretation_chunk_text(text: str) -> bool:
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return False
    if INTERPRETATION_MERITS_SECTION_RE.search(normalized):
        return False
    procedural_hits = len(PROCEDURAL_INTERPRETATION_CHUNK_RE.findall(normalized))
    if procedural_hits >= 2:
        return True
    if procedural_hits == 1 and len(normalized) < 1400:
        return True
    return normalized.startswith("pouczenie o funkcji ochronnej interpretacji")


def build_interpretation_section_score(text: str) -> float:
    normalized = normalize_whitespace(text)
    if not normalized:
        return 0.0
    score = 0.0
    if INTERPRETATION_MERITS_SECTION_RE.search(normalized):
        score += 0.9
    if "ocena stanowiska" in normalized.lower():
        score += 0.45
    if "uzasadnienie interpretacji indywidualnej" in normalized.lower():
        score += 0.6
    if INTERPRETATION_TAXPAYER_POSITION_RE.search(normalized):
        score -= 0.15
    if is_procedural_interpretation_chunk_text(normalized):
        score -= 1.5
    return score


RESOLUTION_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(ocena stanowiska|uzasadnienie interpretacji indywidualnej|uzasadnienie prawne)\b", re.IGNORECASE),
    re.compile(r"\b(stanowisko (?:wnioskodawcy|podatnika|państwa|jest)\s+.*\b(prawidłowe|nieprawidłowe|częściowo prawidłowe))\b", re.IGNORECASE),
    re.compile(r"\b(organ stwierdza|organ uznaje|w konsekwencji|zatem|mając powyższe na uwadze|należy uznać)\b", re.IGNORECASE),
    re.compile(r"\b(oddala|uchyla|zasądza|stwierdza|orzeka|rozstrzyga)\b", re.IGNORECASE),
)

STATUTE_QUOTE_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(art\.\s*\d+[a-z]?\b|ust\.\s*\d+\b|pkt\s*\d+\b|lit\.\s*[a-z]\b)\b", re.IGNORECASE),
    re.compile(r"\b(zwalnia się od podatku|opodatkowaniu podlega|podatnik jest obowiązany|przepisów niniejszej ustawy nie stosuje się)\b", re.IGNORECASE),
)


def is_statute_quote_only_chunk_text(text: str) -> bool:
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return False
    if not any(pattern.search(normalized) for pattern in STATUTE_QUOTE_ONLY_PATTERNS):
        return False
    resolution_markers = (
        "ocena stanowiska",
        "uzasadnienie interpretacji",
        "stanowisko jest prawidłowe",
        "stanowisko jest nieprawidłowe",
        "organ stwierdza",
        "w konsekwencji",
        "mając powyższe na uwadze",
        "należy uznać",
        "oddala",
        "uchyla",
        "zasądza",
        "orzeka",
    )
    return not any(marker in normalized for marker in resolution_markers) and normalized.count("art.") >= 2


def build_resolution_section_score(text: str, *, source_type: str = "") -> float:
    normalized = normalize_whitespace(text)
    if not normalized:
        return 0.0

    lowered = normalized.lower()
    score = 0.0
    if source_type == "interpretation":
        score += build_interpretation_section_score(normalized)
    if source_type == "judgment":
        if re.search(r"\b(oddala skarg[ęe] kasacyjn[ąa]|uchyla zaskarżony wyrok|uchyla decyzj[ęe]|oddala skarg[ęe])\b", lowered):
            score += 2.0
        if re.search(r"\b(uzasadnienie|wobec powyższego|z tych przyczyn|na podstawie art\.)\b", lowered):
            score += 0.75
    if any(pattern.search(normalized) for pattern in RESOLUTION_SECTION_PATTERNS):
        score += 1.35
    if re.search(r"\b(państwa stanowisko|stanowisko wnioskodawcy|stanowisko podatnika)\b", lowered):
        score -= 0.4
    if is_statute_quote_only_chunk_text(normalized):
        score -= 2.0
    if normalized.count("art.") >= 4 and not any(marker in lowered for marker in ("ocena stanowiska", "organ stwierdza", "w konsekwencji", "należy uznać")):
        score -= 0.45
    return score


def classify_chunk_evidence_role(chunk: "RagChunk") -> str:
    if str(chunk.source_type or "") == "statute":
        return "governing_statute"
    text = normalize_whitespace(chunk.chunk_text or "")
    lowered = text.lower()
    if is_statute_quote_only_chunk_text(text):
        return "statute_quote_only"
    if build_resolution_section_score(text, source_type=str(chunk.source_type or "")) >= 1.1:
        if str(chunk.source_type or "") == "judgment":
            return "operative_conclusion"
        return "authority_assessment"
    if re.search(r"\b(państwa stanowisko|stanowisko wnioskodawcy|stanowisko podatnika|wnioskodawca wskazuje)\b", lowered):
        return "taxpayer_position"
    if re.search(r"\b(uzasadnienie|argumentacja|wywód|analiza prawna)\b", lowered):
        return "reasoning"
    return "supporting_source"


def chunk_canonical_source_id(chunk: "RagChunk") -> str:
    source_type = str(chunk.source_type or "").lower()
    if source_type == "statute":
        article_key = ""
        for provision in chunk.legal_provisions:
            article_key = extract_article_key_from_text(provision)
            if article_key:
                break
        domain = infer_chunk_tax_domain(chunk) or str(chunk.subject or "").split(" - ", 1)[0].strip().upper()
        signature = normalize_whitespace(str(chunk.signature or "")).lower()
        publication = normalize_whitespace(str(chunk.publication or "")).lower()
        source_hint = normalize_whitespace(str(chunk.subject or "")).lower()
        return f"statute:{domain}:{signature or source_hint}:{article_key or chunk.chunk_index}:{publication}"
    signature = normalize_whitespace(str(chunk.signature or "")).lower()
    return f"{source_type}:{signature or chunk.document_id}:{chunk.chunk_index}"


def annotate_chunk_evidence_role(chunk: RagChunk, role: str) -> RagChunk:
    if chunk.evidence_role == role:
        return chunk
    return replace(chunk, evidence_role=role)


def rerank_chunks_within_documents(
    chunks: list[RagChunk],
    *,
    query: str,
    config: RagConfig,
    source_type: Optional[str] = None,
    max_chunks_per_document: Optional[int] = None,
) -> list[RagChunk]:
    if not chunks:
        return []

    if source_type:
        chunks = [chunk for chunk in chunks if str(chunk.source_type or "") == source_type]
        if not chunks:
            return []

    document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    for chunk in chunks:
        if chunk.document_id in seen_document_ids:
            continue
        seen_document_ids.add(chunk.document_id)
        document_ids.append(chunk.document_id)

    if not document_ids:
        return chunks

    rows = fetch_rows_by_document_ids(
        tuple(document_ids),
        config=config,
        source_type=source_type,
        chunk_limit_per_document=max_chunks_per_document,
    )
    if not rows:
        return chunks

    ranked_rows = rank_hybrid_local_candidates(
        rows,
        query=query,
        effective_limit=max(len(rows), len(chunks)),
        config=config,
    )
    best_by_document: dict[str, RagChunk] = {}
    for chunk in ranked_rows:
        best_by_document.setdefault(chunk.document_id, chunk)

    refined: list[RagChunk] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        replacement = best_by_document.get(chunk.document_id)
        if replacement and chunk.document_id not in seen_documents:
            refined.append(annotate_chunk_evidence_role(replacement, classify_chunk_evidence_role(replacement)))
            seen_documents.add(chunk.document_id)
            continue
        if chunk.document_id in seen_documents:
            continue
        refined.append(annotate_chunk_evidence_role(chunk, classify_chunk_evidence_role(chunk)))
        seen_documents.add(chunk.document_id)
    return refined


def build_ksef_foreign_sale_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "interpretation" or not query_targets_ksef_foreign_sale(query):
        return 0.0

    text = normalize_whitespace(str(row["chunk_text"] or "")).lower()
    metadata = normalize_whitespace(
        " ".join(
            [
                str(row["signature"] or ""),
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["legal_provisions_json"] or ""),
            ]
        )
    ).lower()
    score = 0.0
    if str(row["signature"] or "") == "0113-KDIPT1-2.4012.1035.2025.2.AJB":
        score += 0.8
    if KSEF_FOREIGN_SALE_MERITS_RE.search(text):
        score += 2.4
    if "zobowiązani również wystawić faktury ustrukturyzowane" in text:
        score += 1.4
    if "nie znajdą zastosowania wyłączenia ustawowe" in text and "art. 106ga ust. 2" in text:
        score += 1.0
    if "art. 106gb ust. 4" in text and "uzgodniony" in text:
        score += 0.9
    if "art. 106a pkt 2" in text and ("państwa trzeciego" in text or "kraju trzecim" in text):
        score += 0.8
    if "wielkiej brytanii" in metadata or "państwa trzeciego" in metadata:
        score += 0.35
    if "państwa stanowisko" in text and "ocena stanowiska" not in text:
        score -= 0.35
    return min(max(score, -0.35), 4.5)


def build_ksef_outside_deduction_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "interpretation" or not query_targets_ksef_outside_deduction(query):
        return 0.0

    text = normalize_whitespace(str(row["chunk_text"] or "")).lower()
    metadata = normalize_whitespace(
        " ".join(
            [
                str(row["signature"] or ""),
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["legal_provisions_json"] or ""),
            ]
        )
    ).lower()
    score = 0.0
    if any(term in metadata for term in ("poza ksef", "bez użycia ksef", "wbrew obowiązkowi", "wbrew obowiazkowi")):
        score += 1.0
    if any(term in metadata for term in ("odliczenia vat", "prawo do odliczenia", "jpk_v7", "bfk", "faktura korygująca", "faktura korygujaca")):
        score += 0.9
    if any(term in text for term in ("odliczenia vat", "prawo do odliczenia", "faktura otrzymana poza ksef", "faktury otrzymanej poza ksef")):
        score += 2.0
    if any(term in text for term in ("brak obowiązku skorygowania", "brak obowiązku korekty", "bez znaczenia pozostaje", "bez znaczenia pozostaje przy tym")):
        score += 1.4
    if any(term in text for term in ("ta sama transakcja", "dokumentującej tę samą transakcję", "dokumentujacej tę samą transakcję", "dokumentującej tę samą transakcję")):
        score += 1.3
    if any(term in text for term in ("art. 108", "nota korygująca", "nota korygujaca", "uchylony")):
        score += 0.8
    if "państwa stanowisko" in text and "ocena stanowiska" not in text:
        score -= 0.2
    return min(max(score, -0.35), 4.5)


def build_ksef_correction_issue_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "interpretation" or not query_targets_ksef_correction_issue(query):
        return 0.0

    text = normalize_whitespace(str(row["chunk_text"] or "")).lower()
    metadata = normalize_whitespace(
        " ".join(
            [
                str(row["signature"] or ""),
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["legal_provisions_json"] or ""),
            ]
        )
    ).lower()
    score = 0.0
    if any(term in metadata for term in ("nota korygująca", "nota korygujaca", "dane nabywcy", "błęd", "bled")):
        score += 1.0
    if any(term in text for term in ("nota korygująca", "nota korygujaca", "brak możliwości korekty", "błędnych danych nabywcy", "blednych danych nabywcy")):
        score += 2.0
    if any(term in text for term in ("fakturę korygującą wystawia podatnik", "fakture korygujaca wystawia podatnik", "sprzedawca")):
        score += 1.4
    if any(term in text for term in ("uchylony", "art. 106k")):
        score += 0.8
    if "państwa stanowisko" in text and "ocena stanowiska" not in text:
        score -= 0.2
    return min(max(score, -0.35), 4.5)


def build_vat_dropshipping_ioss_match_score(row: sqlite3.Row, *, query: str) -> float:
    if not query_targets_vat_dropshipping_ioss(query):
        return 0.0

    candidate_text = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["keywords_json"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["chunk_text"] or "")[:2600],
            ]
        )
    ).lower()
    normalized_query = normalize_whitespace(query or "").lower()
    query_domains = {domain.upper() for domain in detect_domains(query)}
    candidate_domains = row_tax_domains(row)
    score = 0.0

    if any(term in candidate_text for term in ("dropshipping", "platform", "interfejs elektroniczny", "ułatwia", "ulatwia", "sprzedaż na odległość towarów importowanych", "sprzedaz na odleglosc towarow importowanych")):
        score += 1.1
    if any(term in candidate_text for term in ("ioss", "procedura importu", "sprzedaż na odległość towarów importowanych", "sprzedaz na odleglosc towarow importowanych", "150 euro", "wartość rzeczywista", "wartosc rzeczywista")):
        score += 1.15
    if any(term in candidate_text for term in ("pośrednik", "posrednik", "pośredników", "posrednikow", "reprezentowanego przez pośrednika", "reprezentowanego przez posrednika")):
        score += 0.95
    if any(term in candidate_text for term in ("import", "importu", "celny", "cło", "cło", "clo", "odpraw", "zgłoszenie", "zgloszenie")):
        score += 0.7
    if any(term in candidate_text for term in ("konsument", "niebędącym podatnikiem", "niebedacym podatnikiem", "b2c")):
        score += 0.55

    article_key = extract_primary_article_key(row)
    if article_key in {"7a", "17", "28d", "106a", "106b", "106ga", "106gb", "138a", "138b", "138c", "138d", "138e", "138f", "138g", "138h", "138i", "138j", "2"}:
        score += 1.4
    if article_key == "7a":
        score += 0.8
    if article_key in {"138a", "138b", "138c", "138d", "138e", "138f", "138g", "138h", "138i", "138j"} and any(
        term in normalized_query for term in ("platform", "platforma", "dropshipping", "deemed supplier", "interfejs elektroniczny")
    ):
        score -= 0.2
    if article_key in {"22", "22a", "22b", "22c"}:
        score -= 1.0
    if article_key in {"28b", "28c"} and "pośrednik" not in normalized_query and "posrednik" not in normalized_query:
        score -= 0.4

    if query_domains and candidate_domains and not (candidate_domains & query_domains):
        score -= 0.6

    if "art. 7a" in normalized_query and "art. 7a" in candidate_text:
        score += 0.4
    if "art. 28d" in normalized_query and "art. 28d" in candidate_text:
        score += 0.4
    if "ioss" in normalized_query and "138" in candidate_text:
        score += 0.35
    if "ksef" in normalized_query and any(article in candidate_text for article in ("106a", "106b", "106ga", "106gb")):
        score += 0.25

    return score


def build_vat_dropshipping_ioss_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_vat_dropshipping_ioss(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("VAT", "7a"),
        ("VAT", "19a"),
        ("VAT", "138i"),
        ("VAT", "17"),
        ("VAT", "28d"),
        ("VAT", "138a"),
        ("VAT", "138b"),
        ("VAT", "138c"),
        ("VAT", "138d"),
        ("VAT", "138e"),
        ("VAT", "138f"),
        ("VAT", "138g"),
        ("VAT", "138h"),
        ("VAT", "138j"),
        ("VAT", "106a"),
        ("VAT", "106b"),
        ("VAT", "106ga"),
        ("VAT", "106gb"),
        ("VAT", "2"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_ksef_outside_deduction_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_ksef_outside_deduction(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("VAT", "86"),
        ("VAT", "88"),
        ("VAT", "106k"),
        ("VAT", "108"),
        ("VAT", "106nda"),
        ("VAT", "106nf"),
        ("VAT", "106nh"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_ksef_current_law_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_ksef_current_law(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("VAT", "106a"),
        ("VAT", "106b"),
        ("VAT", "106ga"),
        ("VAT", "106gb"),
        ("VAT", "106ni"),
        ("VAT", "106nda"),
        ("VAT", "106nf"),
        ("VAT", "106nh"),
    ]
    if query_targets_ksef_transition_period(query):
        preferred_targets.extend([("VAT", "106ni"), ("VAT", "106ga"), ("VAT", "106gb")])
    if query_targets_ksef_outside_deduction(query):
        preferred_targets.extend([("VAT", "86"), ("VAT", "88"), ("VAT", "108")])
    if query_targets_ksef_correction_issue(query) or re.search(r"\b(korekt\w*|in minus|obni[żz]k\w*)\b", query or "", re.IGNORECASE):
        preferred_targets.extend([("VAT", "29a"), ("VAT", "106j"), ("VAT", "106ga"), ("VAT", "106gb")])
    if query_targets_ksef_operational_modes(query):
        preferred_targets.extend([("VAT", "106nda"), ("VAT", "106nf"), ("VAT", "106nh"), ("VAT", "106gb")])
    if query_targets_ksef_fixed_establishment_scope(query) or query_targets_ksef_buyer_capacity(query):
        preferred_targets.extend([("VAT", "106a"), ("VAT", "106b"), ("VAT", "106ga"), ("VAT", "106gb")])

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_ksef_b2c_invoice_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_ksef_b2c_invoice(query):
        return []
    return [("VAT", "106a"), ("VAT", "106b"), ("VAT", "106ga"), ("VAT", "106gb"), ("VAT", "106na")]


def build_private_vehicle_pit_expense_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_private_vehicle_pit_expense(query):
        return []
    return [("PIT", "23"), ("PIT", "22"), ("PIT", "14"), ("PIT", "10")]


def build_spolka_komandytowa_cit_status_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_spolka_komandytowa_cit_status(query):
        return []
    return [("CIT", "1"), ("CIT", "3"), ("CIT", "4"), ("CIT", "5")]


def build_invoice_address_error_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_invoice_address_error(query):
        return []
    return [("VAT", "106e"), ("VAT", "106b"), ("VAT", "86"), ("VAT", "106k")]


def build_fixed_establishment_vat_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_fixed_establishment_vat(query):
        return []
    return [("VAT", "28b"), ("VAT", "28a"), ("VAT", "28c"), ("VAT", "17")]


def build_family_foundation_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_family_foundation_mechanism(query):
        return []
    return [
        ("CIT", "24q"),
        ("CIT", "24r"),
        ("CIT", "6"),
        ("PIT", "21"),
        ("PIT", "30"),
        ("VAT", "32"),
        ("VAT", "43"),
        ("VAT", "29a"),
    ]


def build_wht_pay_and_refund_service_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_wht_pay_and_refund_services(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    preferred_targets: list[tuple[str, str]] = [
        ("CIT", "21"),
        ("CIT", "22"),
        ("CIT", "26"),
        ("CIT", "26b"),
        ("CIT", "28b"),
        ("CIT", "22c"),
        ("CIT", "22a"),
    ]
    if re.search(r"\b(dywidend\w*)\b", normalized):
        preferred_targets.extend([("CIT", "22"), ("CIT", "26")])
    if re.search(r"\b(odsetk\w*|beneficial owner|rzeczywist\w* właściciel\w*)\b", normalized):
        preferred_targets.extend([("CIT", "21"), ("CIT", "26")])
    if re.search(r"\b(zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|doradcz\w*)\b", normalized):
        preferred_targets.extend([("CIT", "21"), ("CIT", "26")])
    if re.search(r"\bvat\b|podatek od towar", normalized):
        # Cross-border B2B services are a separate VAT lane.  They must not be
        # inferred from the WHT rules merely because the recipient is the same.
        preferred_targets.extend([("VAT", "28b"), ("VAT", "17"), ("VAT", "43"), ("VAT", "86")])

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_direct_document_boost_score(row: sqlite3.Row, *, query: str) -> float:
    document_id = str(row["document_id"] or "")
    if not document_id:
        return 0.0

    boost = 0.0
    normalized = normalize_whitespace(query or "").lower()
    if query_targets_ksef_b2c_invoice(query) and document_id in {"696263", "695122", "693610", "691953", "687901", "696223"}:
        boost += 3.5
    if query_targets_private_vehicle_pit_expense(query) and document_id in {"681556", "693582", "683152", "680995", "677812", "693457"}:
        boost += 3.2
    if query_targets_spolka_komandytowa_cit_status(query) and document_id in {"685379", "694316", "694267", "696424"}:
        boost += 3.2
    if query_targets_invoice_address_error(query) and document_id in {"694474"}:
        boost += 3.8
    if query_targets_fixed_establishment_vat(query) and document_id in {"695238", "694663", "694510", "693399"}:
        boost += 3.5
    if query_targets_family_foundation_mechanism(query) and document_id in {"695219", "692580", "685154", "692665", "692558", "692562", "691426", "691352", "680512"}:
        boost += 3.4
    if query_targets_wht_pay_and_refund_services(query):
        if re.search(r"\b(dywidend\w*)\b", normalized) and document_id in {"695572", "695099", "694262", "688201", "692968", "688123", "687425", "695361", "696295"}:
            boost += 3.7
        if re.search(r"\b(odsetk\w*|beneficial owner|rzeczywist\w* właściciel\w*)\b", normalized) and document_id in {"695572", "695099", "694262", "692968", "688123", "688168", "688201", "687425"}:
            boost += 3.7
        if re.search(r"\b(zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|doradcz\w*)\b", normalized) and document_id in {"691194", "690463", "685389", "679544", "687425"}:
            boost += 3.7
    if query_targets_ksef_outside_deduction(query) and document_id in {"695345", "695471", "695355", "695403", "694097", "693430", "693595", "693598", "693253", "693103", "696243", "696177", "693053", "694474", "692135", "695412"}:
        boost += 3.1
    if query_targets_ksef_correction_issue(query) and document_id in {"694474", "692135", "695412"}:
        boost += 3.1
    if query_targets_post_leasing_vehicle_gift_sale(query) and document_id in {"693717", "689897", "692278"}:
        boost += 3.1
    if query_targets_leased_movable_six_year_rule(query) and document_id in {"693717", "689897", "692278"}:
        boost += 3.1
    if query_targets_housing_relief_temporary_rental(query) and document_id in {"691376", "685479"}:
        boost += 3.0
    if query_targets_housing_relief_loan_repayment(query) and document_id in {"695380", "694539", "691023", "684408"}:
        boost += 3.0
    if query_targets_mortgage_settlement_refund(query) and document_id in {"688486", "693529", "695535", "695474", "695954", "687640"}:
        boost += 3.0
    if query_targets_debt_assumption_effectiveness(query) and document_id in {"695395", "678370"}:
        boost += 3.3
    if query_targets_vat_dropshipping_ioss(query) and document_id in {"694483", "695013", "692678", "692185", "691293", "689099", "681074"}:
        boost += 3.2
    if query_targets_developer_land_sale(query) and document_id in {"695238", "694663", "694510", "693399", "695559"}:
        boost += 3.1
    if query_targets_estonian_cit_transformation_share_cost(query) and document_id in {"691600", "691987", "695089", "693220"}:
        boost += 3.2
    if query_targets_shareholder_company_asset_sale(query) and document_id in {"691600", "691431", "691220", "691301", "681115", "678178"}:
        boost += 3.0
    if query_targets_small_taxpayer_foreign_vat(query) and document_id in {"696476", "695238", "694510"}:
        boost += 2.8
    return boost


def sort_ksef_outside_deduction_interpretation_rows(rows: list[sqlite3.Row], *, query: str) -> list[sqlite3.Row]:
    if not rows or not query_targets_ksef_outside_deduction(query):
        return rows

    normalized_query = normalize_whitespace(query or "").lower()
    query_text = " ".join(
        term for term in (
            "jpk" if "jpk" in normalized_query else "",
            "bfk" if "bfk" in normalized_query else "",
            "art. 108" if "108" in normalized_query else "",
            "nota korygująca" if "nota" in normalized_query or "notą" in normalized_query else "",
            "duplikat" if "duplik" in normalized_query else "",
            "korekta" if "korekt" in normalized_query else "",
            "ponownie" if "ponown" in normalized_query else "",
            "poza ksef" if "poza ksef" in normalized_query or "bez użycia ksef" in normalized_query or "bez uzycia ksef" in normalized_query else "",
            "odliczenie" if "odliczen" in normalized_query else "",
        )
        if term
    )

    duplicate_focus = any(term in query_text for term in ("jpk", "bfk", "duplikat", "ponownie", "art. 108"))
    note_focus = any(term in query_text for term in ("nota", "notą", "nota korygująca"))
    deduction_focus = "odliczenie" in query_text and "poza ksef" in query_text

    duplicate_doc_ids = {"693598", "693253", "693103", "695471", "695355"}
    note_doc_ids = {"694474", "692135", "695412"}
    deduction_doc_ids = {"695345", "695403", "695355", "694097", "693430", "693595", "696243", "696177", "693053"}

    def sort_key(row: sqlite3.Row) -> tuple[int, int, int, str]:
        document_id = str(row["document_id"])
        subject = normalize_whitespace(str(row["subject"] or "")).lower()
        text = normalize_whitespace(
            " ".join(
                [
                    str(row["subject"] or ""),
                    str(row["question_text"] or ""),
                    str(row["chunk_text"] or "")[:1600],
                ]
            )
        ).lower()
        score = 0
        if note_focus:
            if document_id in note_doc_ids:
                score += 40
            if "nota koryguj" in text or "błędn" in text or "bledn" in text:
                score += 8
        if duplicate_focus:
            if document_id in duplicate_doc_ids:
                score += 35
            if "ta sama transakcja" in text or "jpk" in text or "bfk" in text or "duplik" in text:
                score += 8
        if deduction_focus:
            if document_id in deduction_doc_ids:
                score += 35
            if "odliczen" in text and "poza ksef" in text:
                score += 10
            if "brak obowiązku korekty" in text or "brak obowiazku korekty" in text:
                score += 6
        if "art. 86" in normalized_query and document_id == "695345":
            score += 8
        if "art. 108" in normalized_query and document_id in {"693598", "693253", "693103"}:
            score += 6
        if "nota" in normalized_query and document_id == "694474":
            score += 10
        return (-score, int(row["chunk_index"]), len(subject), document_id)

    return sorted(rows, key=sort_key)


def build_debt_assumption_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_debt_assumption_effectiveness(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("PIT", "21"),
        ("PIT", "11"),
        ("PIT", "20"),
        ("PIT", "2"),
        ("PIT", "30e"),
        ("SD", "1"),
        ("SD", "4a"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_housing_relief_temporary_rental_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_housing_relief_temporary_rental(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("PIT", "21"),
        ("PIT", "30e"),
        ("PIT", "52i"),
        ("PIT", "11"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_housing_relief_loan_repayment_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_housing_relief_loan_repayment(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("PIT", "21"),
        ("PIT", "30e"),
        ("PIT", "52i"),
        ("PIT", "11"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_property_sale_pit_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_property_sale_pit(query):
        return []
    # Article 10 is the general source rule; articles 21 and 30e govern the
    # relief and rate.  Retrieve them together so an exception cannot crowd
    # out the rule it qualifies.
    return [("PIT", "10"), ("PIT", "21"), ("PIT", "30e")]


def build_mortgage_settlement_refund_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_mortgage_settlement_refund(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("PIT", "52i"),
        ("PIT", "21"),
        ("PIT", "11"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def query_targets_crossborder_treaty_analysis(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_crossborder_marker = bool(
        re.search(
            r"\b(transgraniczn\w*|nierezydent\w*|zagraniczn\w*|państw\w* trzec\w*|panstw\w* trzec\w*|"
            r"zakład\w*|zaklad\w*|upo|umow\w* o unikaniu podwójnego opodatkowania|"
            r"certyfikat\w* rezydencji|beneficial owner|rzeczywist\w* właściciel\w*)\b",
            normalized,
        )
    )
    has_income_tax_angle = bool(
        re.search(
            r"\b(dywidend\w*|odsetk\w*|należno\w* licencyjn\w*|nalezn\w* licencyjn\w*|"
            r"zysk\w* przedsi\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|holding\w*|"
            r"stał\w* plac\w*|stale\w* miejsce|zakład\w*|zaklad\w*)\b",
            normalized,
        )
    )
    has_country_marker = bool(
        re.search(
            r"\b(niemc\w*|niderland\w*|holand\w*|luksemburg\w*|franc\w*|irland\w*|"
            r"szwajcar\w*|austri\w*|wielk\w* bryt\w*|uk\b|usa\b|stan\w* zjednoczon\w*|czech\w*|"
            r"hiszpani\w*|hiszpań\w*|spain\b|españa|espana)\b",
            normalized,
        )
    )
    return (has_crossborder_marker and (has_income_tax_angle or has_country_marker)) or (
        has_country_marker and has_income_tax_angle
    )


def query_targets_poland_spain_treaty(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_spain_marker = bool(re.search(r"\b(hiszpani\w*|hiszpań\w*|spain\b|españa|espana)\b", normalized))
    has_treaty_marker = bool(
        re.search(
            r"\b(upo|umow\w* o unikaniu podwójnego opodatkowania|"
            r"rezydent\w*|rezydencj\w*|zakład\w*|zaklad\w*|"
            r"dywidend\w*|odsetk\w*|wynagrodzen\w*|pracy najemn\w*|woln\w* zawód\w*|"
            r"woln\w* zawod\w*|zarząd\w*|zarzadu\b|działalno\w* samodzieln\w*|"
            r"double taxation|tax treaty|mli)\b",
            normalized,
        )
    )
    return has_spain_marker and (has_treaty_marker or query_targets_crossborder_treaty_analysis(query))


def query_targets_poland_germany_treaty(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    # Polish declension uses the stem "niemie-" (e.g. "niemieckiej GmbH"),
    # not only "niemc-".  Missing that form silently skipped the dedicated
    # Poland–Germany treaty retrieval lane in otherwise explicit WHT queries.
    has_germany_marker = bool(re.search(r"\b(niemc\w*|niemie\w*|germany|german|niemcy)\b", normalized))
    has_treaty_marker = bool(
        re.search(
            r"\b(upo|umow\w* o unikaniu podwójnego opodatkowania|"
            r"rezydent\w*|rezydencj\w*|zakład\w*|zaklad\w*|"
            r"dywidend\w*|odset(?:k|ek|e?k)\w*|należno\w* licencyjn\w*|nalezn\w* licencyjn\w*|"
            r"beneficial owner|rzeczywist\w* właściciel\w*|"
            r"double taxation|tax treaty|mli)\b",
            normalized,
        )
    )
    return has_germany_marker and (has_treaty_marker or query_targets_crossborder_treaty_analysis(query))


def build_poland_spain_treaty_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_poland_spain_treaty(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    if re.search(r"\b(prac\w* najem\w*|umow\w* o prac\w*|wynagrodzen\w* za prac\w*|employment|salary)\b", normalized):
        preferred_targets: list[tuple[str, str]] = [
            ("CIT", "15"),
            ("CIT", "4"),
            ("CIT", "23"),
            ("CIT", "5"),
            ("CIT", "14"),
            ("CIT", "16"),
        ]
    elif re.search(r"\b(usług\w* doradcz\w*|woln\w* zawód\w*|samodzieln\w* działalno\w*|independent services)\b", normalized):
        preferred_targets = [
            ("CIT", "14"),
            ("CIT", "4"),
            ("CIT", "23"),
            ("CIT", "5"),
            ("CIT", "15"),
            ("CIT", "16"),
        ]
    elif re.search(r"\b(zarząd\w*|zarzadu\b|board|członkostw\w* w zarządzie|powołan\w*)\b", normalized):
        preferred_targets = [
            ("CIT", "16"),
            ("CIT", "4"),
            ("CIT", "23"),
            ("CIT", "5"),
            ("CIT", "14"),
            ("CIT", "15"),
        ]
    elif re.search(r"\b(rezydenc\w*|miejsce zamieszkania|centrum interes\w*|ośrodek interes\w*)\b", normalized):
        preferred_targets = [
            ("CIT", "4"),
            ("CIT", "23"),
            ("CIT", "5"),
            ("CIT", "14"),
            ("CIT", "15"),
            ("CIT", "16"),
        ]
    else:
        preferred_targets = [
            ("CIT", "4"),
            ("CIT", "5"),
            ("CIT", "14"),
            ("CIT", "15"),
            ("CIT", "16"),
            ("CIT", "23"),
        ]

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_poland_germany_treaty_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_poland_germany_treaty(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    preferred_targets: list[tuple[str, str]] = []
    if re.search(r"\b(dywidend\w*|dividend\w*|udział\w*)\b", normalized):
        preferred_targets.append(("CIT", "10"))
    if re.search(r"\b(odset(?:k|ek|e?k)\w*|interest\w*)\b", normalized):
        preferred_targets.append(("CIT", "11"))
    if re.search(r"\b(należno\w* licencyjn\w*|nalezn\w* licencyjn\w*|royalt(?:y|ies))\b", normalized):
        preferred_targets.append(("CIT", "12"))
    if re.search(r"\b(zakład\w*|zaklad\w*|business profits|stał\w* miejsce|zarządz\w*|zarzadz\w*)\b", normalized):
        preferred_targets.extend([("CIT", "7"), ("CIT", "5")])
    if not preferred_targets:
        preferred_targets.extend([("CIT", "7"), ("CIT", "10"), ("CIT", "11"), ("CIT", "12")])
    preferred_targets.extend([("CIT", "26"), ("CIT", "4"), ("CIT", "23")])

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def query_targets_shareholder_company_asset_sale(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_company_party = bool(
        re.search(r"\b(wspólnik\w*|wspolnik\w*|udziałowc\w*|udzialowc\w*|spółk\w* z o\.?o\.?|sp z o\.?o\.?|sp zoo)\b", normalized)
    )
    has_transfer = bool(re.search(r"\b(sprzed\w*|sprzedaż\w*|sprzedaz\w*|zby\w*|kup\w*|naby\w*|zakup\w*)\b", normalized))
    has_asset = bool(
        re.search(
            r"\b(samochod\w*|samochód\w*|pojazd\w*|auto\b|środek trwały|srodek trwaly|nieruchomo\w*|lokal\w*|mieszkani\w*|mieszkani\w*|budynek\w*|grunt\w*)\b",
            normalized,
        )
    )
    return has_company_party and has_transfer and has_asset


def query_targets_estonian_cit_transformation_share_cost(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_transformation = bool(
        re.search(
            r"(przekszta[łl]c\w*.*sp[óo]łk\w* komandytow\w*.*sp[óo]łk\w* z o\.?o\.?|"
            r"sp[óo]łk\w* komandytow\w*.*przekszta[łl]c\w*.*sp[óo]łk\w* z o\.?o\.?|"
            r"transformacj\w* sp[óo]łk\w* komandytow\w*|"
            r"następstw\w* praw\w*.*sp[óo]łk\w* komandytow\w*|"
            r"sukcesj\w* podatkow\w*.*przekszta[łl]c\w*)",
            normalized,
        )
    )
    has_estonian_cit = bool(
        re.search(
            r"\b(esto[ńn]sk\w*\s+cit|rycza[łl]t\w*\s+od\s+dochod\w*\s+sp[óo]łek|"
            r"ukryte\s+zysk\w*|niepodzielon\w*\s+zysk\w*|doch[óo]d\s+z\s+przekszta[łl]cenia)\b",
            normalized,
        )
    )
    has_share_cost = bool(
        re.search(
            r"\b(sprzeda[żz]\w*.*udzia[łl]\w*|udzia[łl]\w*.*sprzeda[żz]\w*|"
            r"koszt\w*.*udzia[łl]\w*|nowe\s+obj[ęe]cie\s+udzia[łl]\w*|"
            r"warto[śs][ćc]\s+nominaln\w*\s+udzia[łl]\w*|"
            r"og[óo]ł\s+praw\s+i\s+obowi[ąa]zk[óo]w)\b",
            normalized,
        )
    )
    has_pcc = bool(re.search(r"\b(pcc|podatek od czynności cywilnoprawnych)\b", normalized))
    return has_transformation and (has_estonian_cit or has_share_cost or has_pcc)


def query_targets_estonian_cit_hidden_profit(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_estonian_cit = bool(
        re.search(
            r"\b(esto[ńn]sk\w*\s+cit|rycza[łl]t\w*\s+od\s+dochod\w*\s+sp[óo]łek)\b",
            normalized,
        )
    )
    has_related_party_benefit = bool(
        re.search(
            r"\b(wsp[óo]lnik\w*|udzia[łl]owc\w*|podmiot\w* powi[ąa]zan\w*|"
            r"po[żz]yczk\w*|odsetk\w*|kapita[łl]\w* po[żz]yczk\w*|"
            r"us[łl]ug\w* zarz[ąa]dz\w*|wynagrodzen\w* za zarz[ąa]dz\w*)\b",
            normalized,
        )
    )
    return has_estonian_cit and has_related_party_benefit


def build_estonian_cit_hidden_profit_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_estonian_cit_hidden_profit(query):
        return []
    return [("CIT", "28m"), ("CIT", "28n"), ("CIT", "28o"), ("CIT", "28j"), ("CIT", "28k"), ("CIT", "7aa")]


def build_transformation_share_cost_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_estonian_cit_transformation_share_cost(query):
        return []

    preferred_targets: list[tuple[str, str]] = [
        ("CIT", "7aa"),
        ("CIT", "28j"),
        ("CIT", "28k"),
        ("CIT", "28m"),
        ("PCC", "1"),
        ("ORDYNACJA", "93a"),
        ("PCC", "3"),
        ("PCC", "6"),
        ("PCC", "7"),
        ("PIT", "22"),
        ("PIT", "23"),
        ("PIT", "24"),
        ("CIT", "1"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def query_targets_developer_land_sale(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_land_sale = bool(
        re.search(r"\b(sprzeda[żz]\w*|zby\w*|umow\w* przedwstępn\w*)\b", normalized)
        and re.search(r"\b(grunt\w*|działk\w*|dzialk\w*|nieruchomo\w* roln\w*)\b", normalized)
    )
    has_preparation_marker = bool(
        re.search(
            r"\b(deweloper\w*|pełnomocnictw\w*|pelnomocnictw\w*|dzierżaw\w*|dzierzaw\w*|"
            r"warunk\w* zabudow\w*|pozwoleni\w* na budow\w*|podział\w*|podzial\w*|"
            r"przyłącz\w* medi\w*|przylacz\w* medi\w*)\b",
            normalized,
        )
    )
    asks_multi_tax = len({domain.upper() for domain in detect_domains(query)} & {"VAT", "PIT", "PCC"}) >= 2
    return (has_land_sale and has_preparation_marker) or (has_land_sale and asks_multi_tax)


def build_developer_land_sale_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_developer_land_sale(query):
        return []

    preferred_targets = [
        ("VAT", "5"),
        ("VAT", "7"),
        ("VAT", "15"),
        ("VAT", "43"),
        ("VAT", "2"),
        ("VAT", "106a"),
        ("VAT", "106b"),
        ("VAT", "106ga"),
        ("VAT", "106gb"),
        ("PIT", "5a"),
        ("PIT", "10"),
        ("PIT", "14"),
        ("PCC", "1"),
        ("PCC", "2"),
        ("PCC", "4"),
        ("PCC", "7"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def query_targets_post_leasing_vehicle_gift_sale(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_vehicle = bool(re.search(r"\b(samochod\w*|samochód\w*|pojazd\w*|auto\b)\b", normalized))
    has_leasing_buyout = bool(re.search(r"\b(leasing\w*|wykup\w*|po leasingu|poleasingow\w*)\b", normalized))
    has_private_asset = bool(re.search(r"\b(majątk\w* prywat\w*|majatk\w* prywat\w*|prywatn\w*)\b", normalized))
    has_gift_or_spouse_sale = bool(
        re.search(r"\b(darowizn\w*|małżonk\w*|malzonk\w*|żon\w*|zon\w*|męż\w*|mez\w*)\b", normalized)
    )
    has_invoice_or_tax = bool(re.search(r"\b(vat|pit|spadk\w*|darowizn\w*|sd-z2|faktur\w*)\b", normalized))
    return has_vehicle and has_leasing_buyout and (has_private_asset or has_gift_or_spouse_sale) and has_invoice_or_tax


def build_post_leasing_vehicle_gift_sale_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_post_leasing_vehicle_gift_sale(query):
        return []

    preferred_targets = [
        ("VAT", "7"),
        ("PIT", "10"),
        ("SD", "4a"),
        ("VAT", "15"),
        ("PIT", "14"),
        ("SD", "6"),
        ("VAT", "106b"),
        ("PIT", "23"),
        ("SD", "9"),
        ("VAT", "86"),
        ("PIT", "22"),
        ("SD", "14"),
        ("VAT", "91"),
        ("VAT", "86a"),
        ("PIT", "2"),
        ("SD", "1"),
    ]
    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def query_targets_leased_movable_six_year_rule(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_leasing = bool(re.search(r"\b(leasing\w*|wykup\w* po leasingu|poleasingow\w*)\b", normalized))
    has_movable_or_vehicle = bool(re.search(r"\b(rzecz\w* ruchom\w*|samochod\w*|samochód\w*|pojazd\w*|auto\b)\b", normalized))
    has_sale_timing = bool(re.search(r"\b(sprzeda\w*|zby\w*|półroczn\w*|polroczn\w*|sześciolet\w*|szesciolet\w*|6\s*lat)\b", normalized))
    return has_leasing and has_movable_or_vehicle and has_sale_timing


def build_leased_movable_six_year_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_leased_movable_six_year_rule(query):
        return []
    return [("PIT", "10"), ("PIT", "14")]


def query_targets_gifted_asset_cost_basis(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_gift = bool(re.search(r"\b(darowizn\w*|otrzyman\w* nieodpłatn\w*|otrzyman\w* nieodplatn\w*)\b", normalized))
    has_sale_or_cost = bool(re.search(r"\b(sprzeda\w*|zby\w*|koszt\w*|wartość\w* rynkow\w*|wartosc\w* rynkow\w*|art\.\s*22\s*ust\.\s*1d)\b", normalized))
    has_sd_or_pit = bool(re.search(r"\b(pit|spadk\w* i darowizn\w*|podatek od spadk\w*|art\.\s*11\s*ust\.\s*2)\b", normalized))
    return has_gift and has_sale_or_cost and has_sd_or_pit


def build_gifted_asset_cost_basis_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_gifted_asset_cost_basis(query):
        return []
    return [("PIT", "2"), ("PIT", "22"), ("SD", "1"), ("SD", "4a")]


def query_targets_spouse_gift_sd(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_gift = bool(re.search(r"\b(darowizn\w*|sd-z2|spadk\w* i darowizn\w*)\b", normalized))
    has_spouse_or_zero_group = bool(re.search(r"\b(małżonk\w*|malzonk\w*|żon\w*|zon\w*|męż\w*|mez\w*|grup\w* zerow\w*|zgłoszeni\w*|zgloszeni\w*)\b", normalized))
    return has_gift and has_spouse_or_zero_group


def build_spouse_gift_sd_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_spouse_gift_sd(query):
        return []
    return [("SD", "4a"), ("SD", "6"), ("SD", "9"), ("SD", "14"), ("SD", "1")]


def query_targets_wht_crossborder_payments(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_crossborder_context = bool(
        re.search(r"\b(wht|podatek u źr[óo]dła|withholding|holand\w*|niderland\w*|państw\w* trzec\w*|panstw\w* trzec\w*|certyfikat\w* rezydencji|upo|umow\w* międzynarodow\w*)\b", normalized)
    )
    has_payment_type = bool(
        re.search(r"\b(dywidend\w*|odsetk\w*|zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|należno\w* licencyjn\w*|nalezn\w* licencyjn\w*)\b", normalized)
    )
    return has_crossborder_context and has_payment_type


def build_wht_crossborder_payment_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_wht_crossborder_payments(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    preferred_targets: list[tuple[str, str]] = [("CIT", "21"), ("CIT", "22"), ("CIT", "22c"), ("CIT", "26")]

    if re.search(r"\b(dywidend\w*)\b", normalized):
        preferred_targets.extend([("CIT", "22"), ("CIT", "22c"), ("CIT", "26")])
    if re.search(r"\b(odsetk\w*|beneficial owner|rzeczywist\w* właściciel\w*)\b", normalized):
        preferred_targets.extend([("CIT", "21"), ("CIT", "26")])
    if re.search(r"\b(zarządz\w*|zarzadz\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*)\b", normalized):
        preferred_targets.extend([("CIT", "21"), ("CIT", "26")])
    if re.search(r"\b(pay and refund|2 mln|2 000 000|próg\w*|prog\w*)\b", normalized):
        preferred_targets.append(("CIT", "26"))

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_shareholder_company_asset_sale_statute_targets(query: str) -> list[tuple[str, str]]:
    if not query_targets_shareholder_company_asset_sale(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    has_estonian_cit = bool(
        re.search(
            r"\b(esto[ńn]sk\w*\s+cit|rycza[łl]t\w*\s+od\s+dochod\w*\s+sp[óo]łek|ukryte\s+zysk\w*|"
            r"niepodzielon\w*\s+zysk\w*|doch[óo]d\s+z\s+przekszta[łl]cenia)\b",
            normalized,
        )
    )
    query_domains = {domain.upper() for domain in detect_domains(query)}
    requested_domains = query_domains or {"VAT", "CIT", "PIT", "PCC"}
    mentions_real_estate = bool(re.search(r"\b(nieruchomo\w*|lokal\w*|mieszkani\w*|mieszkan\w*|budynek\w*|grunt\w*)\b", normalized))
    mentions_preferential_price = bool(
        re.search(r"\b(preferencyjn\w*|rynkow\w*|poniżej\b|ponizej\b|niższ\w*|nizsz\w*|zaniż\w*|zaniz\w*|częściowo nieodpłatn\w*|czesciowo nieodplatn\w*)\b", normalized)
    )

    preferred_targets: list[tuple[str, str]] = []
    if "CIT" in requested_domains:
        if has_estonian_cit:
            preferred_targets.extend([("CIT", "28m"), ("CIT", "11a"), ("CIT", "11c"), ("CIT", "7aa"), ("CIT", "28j"), ("CIT", "28k")])
        else:
            preferred_targets.extend([("CIT", "11c"), ("CIT", "14")])
    if "PIT" in requested_domains:
        if has_estonian_cit:
            preferred_targets.extend([("PIT", "24"), ("PIT", "22"), ("PIT", "23")])
        else:
            preferred_targets.extend([("PIT", "11"), ("PIT", "17"), ("PIT", "24")])
    if "PCC" in requested_domains:
        preferred_targets.extend([("PCC", "1"), ("PCC", "2"), ("PCC", "4"), ("PCC", "6"), ("PCC", "7")])
    if "VAT" in requested_domains:
        preferred_targets.extend([("VAT", "32"), ("VAT", "29a"), ("VAT", "43"), ("VAT", "7"), ("VAT", "5")])
        if not mentions_real_estate and not mentions_preferential_price:
            preferred_targets.append(("VAT", "86a"))

    deduped_targets: list[tuple[str, str]] = []
    seen_targets: set[tuple[str, str]] = set()
    for target in preferred_targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        deduped_targets.append(target)
    return deduped_targets


def build_shareholder_company_asset_sale_match_score(row: sqlite3.Row, *, query: str) -> float:
    if not query_targets_shareholder_company_asset_sale(query):
        return 0.0

    candidate_text = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["keywords_json"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["chunk_text"] or "")[:2200],
            ]
        )
    ).lower()
    normalized_query = normalize_whitespace(query or "").lower()
    query_domains = {domain.upper() for domain in detect_domains(query)}
    candidate_domains = row_tax_domains(row)
    has_estonian_cit = bool(
        re.search(
            r"\b(esto[ńn]sk\w*\s+cit|rycza[łl]t\w*\s+od\s+dochod\w*\s+sp[óo]łek|ukryte\s+zysk\w*|"
            r"niepodzielon\w*\s+zysk\w*|doch[óo]d\s+z\s+przekszta[łl]cenia)\b",
            normalized_query,
        )
    )
    query_mentions_vehicle = bool(re.search(r"\b(samochod\w*|samochód\w*|pojazd\w*|auto\b)\b", normalized_query))
    query_mentions_real_estate = bool(re.search(r"\b(nieruchomo\w*|lokal\w*|mieszkani\w*|mieszkan\w*|budynek\w*|grunt\w*)\b", normalized_query))
    score = 0.0

    if any(term in candidate_text for term in ("wspólnik", "wspolnik", "udziałow", "udzialow")):
        score += 0.45
    if any(term in candidate_text for term in ("samochod", "samochód", "pojazd", "auto")):
        score += 0.4
    if any(term in candidate_text for term in ("nieruchomo", "lokal", "mieszkani", "budynek", "grunt")):
        score += 0.55
    if any(term in candidate_text for term in ("sprzeda", "sprzed", "odpłatne nabycie", "odplatne nabycie", "zbyci")):
        score += 0.4
    if any(term in candidate_text for term in ("kupuj", "kupno", "nabywa", "nabycie", "zakup")):
        score += 0.35
    if any(term in candidate_text for term in ("cena transferowa", "transakcja kontrolowana", "podmioty powiązane", "podmioty powiazane", "wartość rynkowa", "wartosc rynkowa")):
        score += 0.95
    if any(term in candidate_text for term in ("preferencyjn", "poniżej wartości rynkowej", "ponizej wartosci rynkowej", "cena rynkowa", "częściowo nieodpłatne", "czesciowo nieodplatne", "nieodpłatne świadczenie", "nieodplatne swiadczenie")):
        score += 0.9
    if any(term in candidate_text for term in ("ukrytych zysk", "ukryte zyski", "art. 28m")):
        score += 1.0
    if any(term in candidate_text for term in ("wartość początkowa", "wartosc poczatkowa", "środków trwałych", "srodkow trwalych")):
        score += 0.75
    if any(term in candidate_text for term in ("przychodami są", "odpłatnego zbycia", "odplatnego zbycia")):
        score += 0.45

    article_key = extract_primary_article_key(row)
    if has_estonian_cit:
        if article_key in {"28m", "28j", "28k", "11a", "11c", "7aa"}:
            score += 1.2
        if article_key in {"14", "17", "24", "29a", "32", "43", "1", "2", "4", "6", "7"}:
            score -= 0.55
    elif article_key in {"11", "11a", "11c", "11d", "11e", "11t", "12", "14", "15", "16", "16g", "17", "24", "28m", "29a", "32", "43", "1", "2", "4", "6", "7"}:
        score += 0.9
    if article_key in {"8b", "18", "18ef", "18eg", "26c", "28j", "28k", "28h", "86a"}:
        score -= 1.1

    if query_mentions_real_estate and not query_mentions_vehicle and any(term in candidate_text for term in ("samochod", "samochód", "pojazd", "auto")):
        score -= 1.6
    if query_mentions_vehicle and not query_mentions_real_estate and any(term in candidate_text for term in ("nieruchomo", "lokal", "mieszkani", "budynek", "grunt")):
        score -= 0.8

    if str(row["source_type"] or "") == "statute":
        if query_domains and candidate_domains and not (candidate_domains & query_domains):
            score -= 0.8

    return score


def build_transformation_share_cost_match_score(row: sqlite3.Row, *, query: str) -> float:
    if not query_targets_estonian_cit_transformation_share_cost(query):
        return 0.0

    candidate_text = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["keywords_json"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["chunk_text"] or "")[:2600],
            ]
        )
    ).lower()
    normalized_query = normalize_whitespace(query or "").lower()
    query_domains = {domain.upper() for domain in detect_domains(query)}
    candidate_domains = row_tax_domains(row)
    score = 0.0

    if any(term in candidate_text for term in ("przekształc", "transformacj", "sukcesj", "następstw", "nastepstw")):
        score += 0.95
    if any(term in candidate_text for term in ("estońsk", "estonsk", "ryczałt", "ryczalt", "ukryte zyski", "niepodzielone zyski", "dochód z przekształcenia", "dochod z przeksztalcenia")):
        score += 1.1
    if any(term in candidate_text for term in ("udzia", "ogół praw i obowiązków", "ogol praw i obowiazkow", "wspólnik", "wspolnik", "koszt", "nabycie", "objęcie", "objecie")):
        score += 0.75
    if any(term in candidate_text for term in ("pcc", "czynności cywilnoprawnych", "czynnosci cywilnoprawnych", "art. 93a", "art. 7aa", "art. 28m")):
        score += 1.0

    article_key = extract_primary_article_key(row)
    if article_key in {"7aa", "28j", "28k", "28m", "93a", "24", "22", "23", "1", "3", "6", "7"}:
        score += 1.2
    if article_key in {"11c", "14", "29a", "32", "43"} and "ukryte zyski" not in candidate_text:
        score -= 0.45

    if query_domains and candidate_domains and not (candidate_domains & query_domains):
        score -= 0.7

    if "art. 7aa" in normalized_query and "art. 7aa" in candidate_text:
        score += 0.35
    if "art. 28m" in normalized_query and "art. 28m" in candidate_text:
        score += 0.35
    if "pcc" in normalized_query and "pcc" in candidate_text:
        score += 0.25

    return score


def query_targets_small_taxpayer_foreign_vat(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    has_small_taxpayer = bool(re.search(r"\b(mał\w* podatnik\w*|maly podatnik\w*|art\.\s*4a\s*pkt\s*10)\b", normalized))
    has_revenue_threshold = bool(re.search(r"\b(kryterium przychodow\w*|przychodow\w*|progu|limitu|statusu)\b", normalized))
    has_foreign_vat = bool(
        re.search(
            r"\b(vat poza polsk\w*|poza polsk\w*|zagraniczn\w* vat|podatk\w* od warto[śs]ci dodanej|odwrotne obci[aą]żenie|reverse charge|lokaln\w* stawk\w*)\b",
            normalized,
        )
    )
    return has_small_taxpayer and (has_revenue_threshold or has_foreign_vat)


def build_small_taxpayer_foreign_vat_match_score(row: sqlite3.Row, *, query: str) -> float:
    if not query_targets_small_taxpayer_foreign_vat(query):
        return 0.0

    candidate_text = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["question_text"] or ""),
                str(row["issues_json"] or ""),
                str(row["keywords_json"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["chunk_text"] or "")[:2400],
            ]
        )
    ).lower()
    score = 0.0

    if any(term in candidate_text for term in ("mały podatnik", "maly podatnik", "art. 4a pkt 10", "art. 4a-pkt 10")):
        score += 1.35
    if any(term in candidate_text for term in ("przychodu ze sprzedaży", "przychodu ze sprzedazy", "wartość przychodu", "wartosc przychodu")):
        score += 0.65
    if any(term in candidate_text for term in ("odwrotnego obciążenia", "odwrotne obciążenie", "reverse charge")):
        score += 1.0
    if any(term in candidate_text for term in ("poza polską", "poza polska", "zagranicznych", "lokalnych wymogów", "lokalnych wymogow", "wartości dodanej", "wartosci dodanej")):
        score += 1.0
    if any(term in candidate_text for term in ("nabywca", "usługobiorc", "uslugobiorc")) and "vat" in candidate_text:
        score += 0.6
    if any(term in candidate_text for term in ("kwocie brutto", "kwocie netto", "podatku należnego", "podatku naleznego")):
        score += 0.45

    article_key = extract_primary_article_key(row)
    if article_key in {"4a", "19", "12"}:
        score += 0.9
    if article_key in {"28o", "28j", "28k"}:
        score -= 1.4

    candidate_domains = row_tax_domains(row)
    if str(row["source_type"] or "") == "statute":
        if "CIT" in candidate_domains:
            score += 0.45
        elif candidate_domains:
            score -= 0.8

    if str(row["source_type"] or "") == "interpretation":
        score += 0.3
    if str(row["source_type"] or "") == "judgment" and "mały podatnik" not in candidate_text and "maly podatnik" not in candidate_text:
        score -= 0.4

    return score


def derive_tax_domain(record: dict[str, Any]) -> str:
    haystack = " ".join(
        [*map(str, record.get("law_tags") or []), *map(str, record.get("issues") or []), *map(str, record.get("legal_provisions") or [])]
    ).lower()
    for domain, markers in (
        ("VAT", ("[vat]", "vat", "towarów i usług")),
        ("CIT", ("[cit]", "cit", "dochodowym od osób prawnych")),
        ("PIT", ("[pit]", "pit", "dochodowym od osób fizycznych")),
        ("PCC", ("[pcc]", "pcc", "czynności cywilnoprawnych")),
        ("SD", ("[sd]", "sd", "podatek od spadków i darowizn", "podatek od spadkow i darowizn")),
        ("NIERUCHOMOŚCI", ("[nieruchomości]", "podatek od nieruchomości", "podatki od nieruchomości", "u.p.o.l.")),
        ("AKCYZA", ("akcyza", "akcyzow")),
        ("ORDYNACJA", ("[op]", "[ordynacja]", "ordynacja", "zobowiązania podatkowe")),
    ):
        if any(marker in haystack for marker in markers):
            return domain
    return ""


def build_structured_profile(record: dict[str, Any]) -> dict[str, str]:
    content = str(record.get("content_text") or "")
    signature = str(record.get("signature") or "")
    family = SIGNATURE_FAMILY_RE.search(signature)
    question_text = extract_question_text(content)
    subject = normalize_whitespace(str(record.get("subject") or ""))
    tax_domain = derive_tax_domain(record)
    legal_issue_text = normalize_whitespace(
        " | ".join(part for part in [tax_domain, subject, question_text] if part)
    )[:5000]
    return {
        "tax_domain": tax_domain,
        "signature_family": family.group(1) if family else "",
        "question_text": question_text,
        "legal_issue_text": legal_issue_text,
        "facts_text": extract_facts_text(content),
        "decision_text": extract_decision_text(content),
    }


def split_into_chunks(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]

    paragraphs = [part.strip() for part in SECTION_BREAK_RE.split(text) if part.strip()]
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        separator_len = 2 if current else 0
        if current and current_len + separator_len + paragraph_len > target_chars:
            chunk_text = "\n\n".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

            overlap = build_overlap_tail(chunk_text, overlap_chars)
            current = [overlap, paragraph] if overlap else [paragraph]
            current_len = len("\n\n".join(current))
            continue

        current.append(paragraph)
        current_len += separator_len + paragraph_len

    if current:
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks or [text]


def build_overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ""
    return text[-overlap_chars:].strip()


def filter_index_chunks(record: dict[str, Any], chunks: list[str]) -> list[str]:
    if normalize_source_type(record) != "interpretation":
        return chunks
    filtered = [chunk for chunk in chunks if not is_procedural_interpretation_chunk_text(chunk)]
    return filtered or chunks


def make_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{chunk_index}:{chunk_text}".encode("utf-8")).hexdigest()
    return digest[:24]


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def join_search_text(values: list[str]) -> str:
    return " | ".join(value for value in values if value)


def tokenize_for_embedding(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in EMBEDDING_TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def stable_hash_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def compute_embedding(text: str, *, dimensions: int) -> tuple[list[float], float]:
    if dimensions <= 0:
        return [], 0.0

    vector = [0.0] * dimensions
    tokens = tokenize_for_embedding(text)
    for token in tokens:
        base_weight = 1.0 + min(len(token), 12) / 24.0
        feature_values = [token]
        if len(token) >= 4:
            feature_values.extend(token[index : index + 4] for index in range(len(token) - 3))

        for feature in feature_values:
            hashed = stable_hash_int(feature)
            slot = hashed % dimensions
            sign = 1.0 if ((hashed >> 8) & 1) == 0 else -1.0
            vector[slot] += sign * base_weight

    norm = vector_norm(vector)
    if norm == 0:
        return [], 0.0

    normalized = [round(value / norm, 6) for value in vector]
    return normalized, 1.0


def build_document_payload(record: dict[str, Any], config: RagConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return {}, []

    subject = normalize_whitespace(str(record.get("subject") or "Bez tytułu")) or "Bez tytułu"
    signature = normalize_whitespace(str(record.get("signature") or "")) or None
    keywords = [str(value).strip() for value in record.get("keywords") or [] if str(value).strip()]
    legal_provisions = [
        str(value).strip() for value in record.get("legal_provisions") or [] if str(value).strip()
    ]
    issues = [str(value).strip() for value in record.get("issues") or [] if str(value).strip()]
    law_tags = [str(value).strip() for value in record.get("law_tags") or [] if str(value).strip()]
    clean_text = clean_document_text(record)
    if not clean_text:
        return {}, []

    chunks = split_into_chunks(
        clean_text,
        target_chars=config.chunk_target_chars,
        overlap_chars=config.chunk_overlap_chars,
    )

    document_payload = {
        "source": record.get("source") or "eureka",
        "source_type": record.get("source_type") or "interpretation",
        "source_subtype": derive_source_subtype(record),
        "authority": record.get("authority") or "",
        "jurisdiction": record.get("jurisdiction") or "PL",
        "act_title": record.get("act_title") or "",
        "publication": record.get("publication") or "",
        "legal_state_date": record.get("legal_state_date") or "",
        "source_pages": [int(page) for page in record.get("source_pages") or [] if str(page).isdigit()],
        "tax_domain": build_structured_profile(record)["tax_domain"],
        "document_id": document_id,
        "index_name": record.get("index") or document_id,
        "version_id": record.get("version_id"),
        "template_id": record.get("template_id"),
        "template_version_id": record.get("template_version_id"),
        "category": record.get("category"),
        "status": record.get("status"),
        "subject": subject,
        "signature": signature,
        "author": record.get("author"),
        "published_date": record.get("published_date"),
        "published_at": record.get("published_at"),
        "keywords": keywords,
        "legal_provisions": legal_provisions,
        "issues": issues,
        "law_tags": law_tags,
        "query": record.get("query") or "",
        "source_url": record.get("source_url"),
        "content_html": record.get("content_html") or "",
        "content_text": record.get("content_text") or "",
        "content_text_clean": clean_text,
        "content_sha256": record.get("content_sha256"),
        "attachments": record.get("attachments") or [],
        "raw_field_map": record.get("raw_field_map") or {},
        "raw_search": record.get("raw_search") or {},
        "raw_detail": record.get("raw_detail") or {},
        "retrieved_at": record.get("retrieved_at") or utc_now_iso(),
        "indexed_at": utc_now_iso(),
    }

    chunks_payload = [
        build_chunk_payload(
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            subject=subject,
            signature=signature,
            published_date=record.get("published_date"),
            source_url=record.get("source_url"),
            category=record.get("category"),
            keywords=keywords,
            legal_provisions=legal_provisions,
            issues=issues,
            law_tags=law_tags,
            embedding_dimensions=config.embedding_dimensions,
        )
        for chunk_index, chunk_text in enumerate(chunks)
    ]

    return document_payload, chunks_payload


def build_chunk_payload(
    *,
    document_id: str,
    chunk_index: int,
    chunk_text: str,
    subject: str,
    signature: Optional[str],
    published_date: Any,
    source_url: Any,
    category: Any,
    keywords: list[str],
    legal_provisions: list[str],
    issues: list[str],
    law_tags: list[str],
    embedding_dimensions: int,
) -> dict[str, Any]:
    embedding_fields = [
        (signature or "", 5),
        (subject, 4),
        (join_search_text(legal_provisions), 4),
        (join_search_text(issues), 4),
        (join_search_text(keywords), 2),
        (join_search_text(law_tags), 2),
        (chunk_text, 1),
    ]
    embedding_text = "\n".join(
        value for value, weight in embedding_fields for _ in range(weight) if value
    )
    embedding, embedding_norm = compute_embedding(embedding_text, dimensions=embedding_dimensions)
    return {
        "chunk_id": make_chunk_id(document_id, chunk_index, chunk_text),
        "document_id": document_id,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "chunk_chars": len(chunk_text),
        "signature": signature,
        "published_date": published_date,
        "source_url": source_url,
        "subject": subject,
        "category": category,
        "keywords_text": join_search_text(keywords),
        "legal_provisions_text": join_search_text(legal_provisions),
        "issues_text": join_search_text(issues),
        "law_tags_text": join_search_text(law_tags),
        "embedding": embedding,
        "embedding_norm": embedding_norm,
        "embedding_model": "alitigator-hash-v1",
    }


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    ensure_schema(connection)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            content_sha256 TEXT,
            source TEXT NOT NULL DEFAULT 'eureka',
            source_type TEXT NOT NULL DEFAULT 'interpretation',
            source_subtype TEXT NOT NULL DEFAULT '',
            authority TEXT NOT NULL DEFAULT '',
            jurisdiction TEXT NOT NULL DEFAULT 'PL',
            act_title TEXT NOT NULL DEFAULT '',
            publication TEXT NOT NULL DEFAULT '',
            legal_state_date TEXT NOT NULL DEFAULT '',
            source_pages_json TEXT NOT NULL DEFAULT '[]',
            subject TEXT NOT NULL,
            signature TEXT,
            published_date TEXT,
            source_url TEXT,
            category TEXT,
            keywords_json TEXT NOT NULL,
            legal_provisions_json TEXT NOT NULL,
            issues_json TEXT NOT NULL,
            law_tags_json TEXT NOT NULL,
            tax_domain TEXT NOT NULL DEFAULT '',
            signature_family TEXT NOT NULL DEFAULT '',
            question_text TEXT NOT NULL DEFAULT '',
            facts_text TEXT NOT NULL DEFAULT '',
            decision_text TEXT NOT NULL DEFAULT '',
            indexed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_chars INTEGER NOT NULL,
            provision_id TEXT NOT NULL DEFAULT '',
            display_reference TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS chunk_citations (
            chunk_id TEXT NOT NULL,
            citation TEXT NOT NULL,
            PRIMARY KEY (chunk_id, citation),
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_citations_exact
            ON chunk_citations(citation, chunk_id);

        CREATE TABLE IF NOT EXISTS legal_document_versions (
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            document_type TEXT NOT NULL,
            title TEXT NOT NULL,
            citation TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT,
            publication_date TEXT,
            is_consolidated_text INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (document_id, version_id)
        );

        CREATE TABLE IF NOT EXISTS legal_provisions (
            provision_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            citation TEXT NOT NULL,
            article TEXT NOT NULL,
            paragraph TEXT,
            point TEXT,
            letter TEXT,
            provision_text TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to TEXT,
            status TEXT NOT NULL CHECK (status IN ('active', 'repealed', 'unknown')),
            source_document_id TEXT NOT NULL,
            source_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
            source_span_start INTEGER NOT NULL DEFAULT 0,
            source_span_end INTEGER NOT NULL DEFAULT 0,
            references_json TEXT NOT NULL DEFAULT '[]',
            amends TEXT,
            repealed_by TEXT,
            tax_domain TEXT NOT NULL DEFAULT '',
            taxpayer_role TEXT NOT NULL DEFAULT '',
            legal_mechanism TEXT NOT NULL DEFAULT '',
            entailed_result_codes_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY (document_id, version_id)
                REFERENCES legal_document_versions(document_id, version_id)
        );

        CREATE INDEX IF NOT EXISTS idx_legal_provisions_exact
            ON legal_provisions(document_id, citation, effective_from, effective_to, status);

        """
    )
    for column in (
        "tax_domain", "signature_family", "question_text", "facts_text", "decision_text",
        "source TEXT NOT NULL DEFAULT 'eureka'", "source_type TEXT NOT NULL DEFAULT 'interpretation'",
        "source_subtype TEXT NOT NULL DEFAULT ''", "authority TEXT NOT NULL DEFAULT ''",
        "jurisdiction TEXT NOT NULL DEFAULT 'PL'", "act_title TEXT NOT NULL DEFAULT ''",
        "publication TEXT NOT NULL DEFAULT ''", "legal_state_date TEXT NOT NULL DEFAULT ''",
        "source_pages_json TEXT NOT NULL DEFAULT '[]'",
    ):
        try:
            if " " in column:
                connection.execute(f"ALTER TABLE documents ADD COLUMN {column}")
            else:
                connection.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass

    for column in (
        "provision_id TEXT NOT NULL DEFAULT ''",
        "display_reference TEXT NOT NULL DEFAULT ''",
    ):
        try:
            connection.execute(f"ALTER TABLE chunks ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_display_reference ON chunks(display_reference)"
    )

    for column in (
        "tax_domain TEXT NOT NULL DEFAULT ''",
        "taxpayer_role TEXT NOT NULL DEFAULT ''",
        "legal_mechanism TEXT NOT NULL DEFAULT ''",
        "entailed_result_codes_json TEXT NOT NULL DEFAULT '[]'",
    ):
        try:
            connection.execute(f"ALTER TABLE legal_provisions ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass

    fts_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
    ).fetchone()
    should_rebuild = False

    if fts_sql_row is None:
        create_fts_table(connection)
        should_rebuild = True
    elif "facts_text" not in str(fts_sql_row["sql"] or ""):
        connection.execute("DROP TABLE IF EXISTS chunks_fts")
        create_fts_table(connection)
        should_rebuild = True

    if should_rebuild:
        rebuild_fts_index(connection)


def create_fts_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_text,
            subject,
            signature,
            keywords,
            legal_provisions,
            issues
            ,question_text,
            facts_text,
            tax_domain
        )
        """
    )


def rebuild_fts_index(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM chunks_fts")
    rows = connection.execute(
        """
        SELECT
            c.rowid,
            c.chunk_text,
            d.subject,
            d.signature,
            d.keywords_json,
            d.legal_provisions_json,
            d.issues_json,
            d.question_text,
            d.facts_text,
            d.tax_domain
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        ORDER BY c.rowid
        """
    ).fetchall()
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO chunks_fts (rowid, chunk_text, subject, signature, keywords, legal_provisions, issues, question_text, facts_text, tax_domain)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["rowid"],
                row["chunk_text"],
                row["subject"],
                row["signature"] or "",
                " | ".join(json.loads(row["keywords_json"])),
                " | ".join(json.loads(row["legal_provisions_json"])),
                " | ".join(json.loads(row["issues_json"])),
                row["question_text"],
                row["facts_text"],
                row["tax_domain"],
            )
            for row in rows
        ],
    )


def delete_document(connection: sqlite3.Connection, document_id: str) -> None:
    rows = connection.execute(
        "SELECT rowid FROM chunks WHERE document_id = ? ORDER BY rowid",
        (document_id,),
    ).fetchall()
    if rows:
        connection.executemany(
            "DELETE FROM chunks_fts WHERE rowid = ?",
            [(row["rowid"],) for row in rows],
        )
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def fetch_document_state(connection: sqlite3.Connection, document_id: str) -> Optional[str]:
    row = connection.execute(
        "SELECT content_sha256 FROM documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()
    return None if row is None else str(row["content_sha256"] or "")


def iter_processed_records(path: Path, *, reverse: bool = False) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if reverse:
            lines = [line.strip() for line in handle if line.strip()]
            for raw_line in reversed(lines):
                yield json.loads(raw_line)
            return

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def derive_source_subtype(record: dict[str, Any]) -> str:
    """Return a stable subtype without relying on a source-specific display label."""
    explicit = normalize_whitespace(str(record.get("source_subtype") or "")).lower()
    if explicit:
        return explicit
    source_type = normalize_whitespace(str(record.get("source_type") or "interpretation")).lower()
    category = normalize_whitespace(str(record.get("category") or "")).lower()
    if source_type == "interpretation":
        return "general" if "ogóln" in category else "individual"
    if source_type == "judgment":
        for court, subtype in (("naczelny", "nsa"), ("wojewódzki", "wsa"), ("trybunał konstytucyjny", "tk"), ("trybunał sprawiedliwości", "tsue")):
            if court in category:
                return subtype
    if source_type == "commentary":
        return "legal_commentary"
    return ""


def normalize_source_type(record: dict[str, Any]) -> str:
    value = normalize_whitespace(str(record.get("source_type") or "interpretation")).lower()
    allowed = {"interpretation", "statute", "judgment", "commentary"}
    return value if value in allowed else "interpretation"


def build_record_index_chunks(
    record: dict[str, Any],
    document_text: str,
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Return retrieval chunks, preferring exact editorial units for statutes.

    Law ingestion already exposes stable ``provision_units`` with article,
    section, point and letter ancestry.  Indexing the enclosing article part
    discards that structure and makes a short special rule compete as a few
    words inside tens of pages.  Each unit is therefore indexed with a small
    synthetic ancestry envelope.  The envelope is derived only from verified
    source metadata and lets both legacy ``LegalRule`` extraction and the v2
    ``ProvisionParser`` recover the exact display reference.
    """

    if normalize_source_type(record) == "statute":
        expected_article = next(
            (
                match.group(1).casefold()
                for value in record.get("legal_provisions") or []
                for match in [re.fullmatch(r"art\.\s*(\d+[a-z]?)", str(value).strip(), re.IGNORECASE)]
                if match
            ),
            "",
        )
        raw_units = record.get("provision_units") or []
        if not raw_units:
            # Older processed corpora predate provision_units_v1.  Upgrade
            # them deterministically during reindex instead of requiring a
            # topic-specific data patch.
            from app.law_chunk import build_provision_units

            article_hint = next(
                (
                    str(value).strip()
                    for value in record.get("legal_provisions") or []
                    if re.fullmatch(r"art\.\s*\d+[a-z]?", str(value).strip(), re.IGNORECASE)
                ),
                None,
            )
            record_document_id = str(record.get("document_id") or "").strip()
            article_document_id = str(record.get("article_document_id") or "").strip()
            if not article_document_id:
                article_document_id = re.sub(
                    r"-part-\d+(?:-occurrence-\d+)?$", "", record_document_id
                )
            if article_hint and article_document_id and record_document_id:
                raw_units = build_provision_units(
                    document_text,
                    article_document_id=article_document_id,
                    record_document_id=record_document_id,
                    article_hint=article_hint,
                )

        unit_chunks: list[str] = []
        seen: set[tuple[str, str]] = set()
        for raw_unit in raw_units:
            if not isinstance(raw_unit, dict):
                continue
            citation = normalize_whitespace(str(raw_unit.get("citation") or ""))
            unit_text = str(raw_unit.get("text") or "").strip()
            article = normalize_whitespace(str(raw_unit.get("article") or ""))
            if not citation or not unit_text or not article:
                continue
            if expected_article and article.casefold() != expected_article:
                continue
            key = (citation.casefold(), hashlib.sha256(unit_text.encode("utf-8")).hexdigest())
            if key in seen:
                continue
            seen.add(key)

            unit_type = str(raw_unit.get("unit_type") or "")
            ancestors = [f"Art. {article}."]
            paragraph = normalize_whitespace(str(raw_unit.get("paragraph") or ""))
            section = normalize_whitespace(str(raw_unit.get("section") or ""))
            point = normalize_whitespace(str(raw_unit.get("point") or ""))
            if unit_type not in {"article", "paragraph"} and paragraph:
                ancestors.append(f"§ {paragraph}.")
            if unit_type in {"point", "letter"} and section:
                ancestors.append(f"{section}.")
            if unit_type == "letter" and point:
                ancestors.append(f"{point})")

            if unit_type == "article":
                rendered = unit_text
            else:
                rendered = "\n".join((*ancestors, unit_text))
            # The exact citation is searchable even when OCR separated the
            # marker from its body.  It is a registry value, not a query rule.
            unit_chunks.append(f"{citation}\n{rendered}")
        if unit_chunks:
            return unit_chunks

    chunks = [document_text] if record.get("pre_chunked") else split_into_chunks(
        document_text,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
    )
    return filter_index_chunks(record, chunks)


def index_record(connection: sqlite3.Connection, record: dict[str, Any], config: RagConfig) -> int:
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return 0

    delete_document(connection, document_id)

    document_text = clean_document_text(record)
    if not document_text:
        return 0

    chunks = build_record_index_chunks(
        record,
        document_text,
        target_chars=config.chunk_target_chars,
        overlap_chars=config.chunk_overlap_chars,
    )

    subject = normalize_whitespace(str(record.get("subject") or "Bez tytułu")) or "Bez tytułu"
    signature = normalize_whitespace(str(record.get("signature") or "")) or None
    keywords = [str(value).strip() for value in record.get("keywords") or [] if str(value).strip()]
    legal_provisions = [
        str(value).strip() for value in record.get("legal_provisions") or [] if str(value).strip()
    ]
    issues = [str(value).strip() for value in record.get("issues") or [] if str(value).strip()]
    law_tags = [str(value).strip() for value in record.get("law_tags") or [] if str(value).strip()]
    profile = build_structured_profile(record)
    source_type = normalize_source_type(record)
    source_subtype = derive_source_subtype(record)
    source_pages = [int(page) for page in record.get("source_pages") or [] if str(page).isdigit()]

    connection.execute(
        """
        INSERT INTO documents (
            document_id,
            content_sha256,
            source,
            source_type,
            source_subtype,
            authority,
            jurisdiction,
            act_title,
            publication,
            legal_state_date,
            source_pages_json,
            subject,
            signature,
            published_date,
            source_url,
            category,
            keywords_json,
            legal_provisions_json,
            issues_json,
            law_tags_json, tax_domain, signature_family, question_text, facts_text, decision_text,
            indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            index_content_fingerprint(record),
            normalize_whitespace(str(record.get("source") or "eureka")) or "eureka",
            source_type,
            source_subtype,
            normalize_whitespace(str(record.get("authority") or "")),
            normalize_whitespace(str(record.get("jurisdiction") or "PL")) or "PL",
            normalize_whitespace(str(record.get("act_title") or "")),
            normalize_whitespace(str(record.get("publication") or "")),
            normalize_whitespace(str(record.get("legal_state_date") or "")),
            json_dump(source_pages),
            subject,
            signature,
            record.get("published_date"),
            record.get("source_url"),
            record.get("category"),
            json_dump(keywords),
            json_dump(legal_provisions),
            json_dump(issues),
            json_dump(law_tags),
            profile["tax_domain"], profile["signature_family"], profile["question_text"], profile["facts_text"], profile["decision_text"],
            utc_now_iso(),
        ),
    )

    inserted = 0
    for chunk_index, chunk_text in enumerate(chunks):
        chunk_id = make_chunk_id(document_id, chunk_index, chunk_text)
        first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "")
        display_reference = (
            normalize_provision_reference(first_line)
            if re.fullmatch(
                r"art\.\s*\d+[a-z]?(?:\s+(?:ust\.\s*\d+[a-z]?|§\s*\d+[a-z]?))?"
                r"(?:\s+pkt\s*\d+[a-z]?)?(?:\s+lit\.\s*[a-z])?",
                first_line,
                re.IGNORECASE,
            )
            else ""
        )
        article_document_id = str(record.get("article_document_id") or "").strip() or re.sub(
            r"-part-\d+(?:-occurrence-\d+)?$", "", document_id
        )
        provision_id = (
            build_provision_id(article_document_id, display_reference)
            if display_reference
            else ""
        )
        cursor = connection.execute(
            """
            INSERT INTO chunks (
                document_id, chunk_id, chunk_index, chunk_text, chunk_chars,
                provision_id, display_reference
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id, chunk_id, chunk_index, chunk_text, len(chunk_text),
                provision_id, display_reference,
            ),
        )
        rowid = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO chunks_fts (rowid, chunk_text, subject, signature, keywords, legal_provisions, issues, question_text, facts_text, tax_domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rowid,
                chunk_text,
                subject,
                signature or "",
                " | ".join(keywords),
                " | ".join(legal_provisions),
                " | ".join(issues),
                profile["question_text"],
                profile["facts_text"],
                profile["tax_domain"],
            ),
        )
        citations = extract_normalized_provision_references(
            chunk_text,
            legal_provisions,
        )
        if citations:
            connection.executemany(
                "INSERT OR IGNORE INTO chunk_citations (chunk_id, citation) VALUES (?, ?)",
                [(chunk_id, citation) for citation in citations],
            )
        inserted += 1

    return inserted


def reindex_corpus(*, limit: Optional[int] = None, force: bool = False) -> dict[str, Any]:
    runtime = resolve_rag_runtime()
    if runtime.write_backend == "mysql":
        from app.mysql_rag import reindex_corpus_mysql

        return reindex_corpus_mysql(limit=limit, force=force)
    if runtime.write_backend == "supabase":
        from app.supabase_rag import reindex_corpus_to_supabase

        return reindex_corpus_to_supabase(limit=limit, force=force)
    config = get_rag_config()
    if not config.processed_path.exists():
        raise FileNotFoundError(f"Processed corpus not found: {config.processed_path}")
    missing_additional_paths = [path for path in config.additional_source_paths if not path.exists()]
    if missing_additional_paths:
        raise FileNotFoundError(f"Additional RAG source not found: {missing_additional_paths[0]}")

    processed = 0
    indexed = 0
    skipped = 0
    chunk_count = 0
    indexed_document_ids: list[str] = []

    connection = get_connection(config.db_path)
    try:
        source_paths = [source.path for source in iter_configured_corpus_sources(config)]
        for source_path in source_paths:
            for record in iter_processed_records(source_path):
                if limit is not None and processed >= limit:
                    break

                processed += 1
                document_id = str(record.get("document_id") or "").strip()
                if not document_id:
                    skipped += 1
                    continue

                current_sha = index_content_fingerprint(record)
                stored_sha = fetch_document_state(connection, document_id)
                if not force and stored_sha and stored_sha == current_sha:
                    skipped += 1
                    continue

                inserted_chunks = index_record(connection, record, config)
                if inserted_chunks == 0:
                    skipped += 1
                    continue

                indexed += 1
                chunk_count += inserted_chunks
                indexed_document_ids.append(document_id)
            if limit is not None and processed >= limit:
                break

        connection.commit()
        total_documents = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        total_chunks = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
    finally:
        connection.close()

    return {
        "processed": processed,
        "indexed": indexed,
        "skipped": skipped,
        "chunk_count": chunk_count,
        "db_path": str(config.db_path),
        "total_documents": int(total_documents),
        "total_chunks": int(total_chunks),
        "indexed_document_ids": indexed_document_ids,
    }


def build_match_query(query: str, *, max_tokens: int = 24) -> Optional[str]:
    tokens = []
    for match in QUERY_TOKEN_RE.finditer(query):
        token = match.group(0).lower()
        if token not in tokens:
            tokens.append(token)

    if not tokens:
        return None

    return " OR ".join(f'"{token}"*' for token in tokens[:max_tokens])


def normalize_legal_query_refs(query: str) -> str:
    normalized = normalize_whitespace(query)
    if not normalized:
        return ""
    def join_article_suffix(match: re.Match[str]) -> str:
        suffix = match.group(2).lower()
        if suffix in {"ust", "pkt", "lit", "par"}:
            return match.group(0)
        if len(suffix) == 1 or suffix in {"ga", "gb", "gc", "na", "nb", "nda", "nf", "nh", "ni"}:
            return f"art. {match.group(1)}{suffix}"
        return match.group(0)

    normalized = ARTICLE_SPLIT_SUFFIX_RE.sub(join_article_suffix, normalized)
    normalized = BARE_ARTICLE_SPLIT_SUFFIX_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2).lower()}"
        if match.group(2).lower() in {"na", "nda", "nh", "nf", "nb", "ga", "gb", "gc"}
        else match.group(0),
        normalized,
    )
    normalized = OFFLINE_SPLIT_RE.sub(lambda match: f"offline{match.group(1)}", normalized)
    return normalized


def get_query_expansion_terms(query: str, *, config: Optional[RagConfig] = None) -> list[str]:
    """Return stable tax-domain aliases relevant to the user's wording."""
    query = normalize_legal_query_refs(query)
    additions: list[str] = []
    for pattern, aliases in QUERY_EXPANSIONS:
        if pattern.search(query):
            additions.extend(aliases)
    effective_config = config or get_rag_config()
    mechanism_rules = load_mechanism_rules(effective_config)
    for mechanism in sorted(detect_mechanisms(query, config=effective_config)):
        additions.extend(mechanism_rules.get(mechanism, ()))
    return list(dict.fromkeys(additions))


def expand_search_query(query: str, *, config: Optional[RagConfig] = None) -> str:
    """Add stable tax-domain aliases while preserving the user's original wording."""
    normalized_query = normalize_legal_query_refs(query)
    return " ".join([normalized_query, *get_query_expansion_terms(normalized_query, config=config)]).strip()


def build_candidate_match_queries(query: str, *, config: Optional[RagConfig] = None) -> list[str]:
    """Build complementary FTS queries for prose and short legal-domain aliases."""
    queries = [build_match_query(query)]
    expansion_query = build_match_query(" ".join(get_query_expansion_terms(query, config=config)))
    if expansion_query:
        queries.append(expansion_query)
    return list(dict.fromkeys(match_query for match_query in queries if match_query))


def build_statute_match_queries(query: str) -> list[str]:
    """Add stable drafting-language synonyms used in statutes, not case identifiers."""
    exact_phrases: list[str] = []
    for pattern, aliases in STATUTE_QUERY_EXPANSIONS:
        if pattern.search(query):
            exact_phrases.extend(f'"{alias}"' for alias in aliases)
    for pattern, phrases in STATUTORY_CONCEPTS:
        if pattern.search(query):
            exact_phrases.extend(f'"{phrase}"' for phrase in phrases)
    for pattern, _, _, phrases in STATUTE_PROCEDURAL_RULES:
        if pattern.search(query):
            exact_phrases.extend(f'"{phrase}"' for phrase in phrases)
    # The unexpanded user query is already searched by build_candidate_match_queries.
    # Do not add an OR-query containing every word from a long statutory phrase:
    # it recalls hundreds of unrelated statutory chunks and makes the reranker both
    # slower and less precise.  These expansions are intentionally phrase-only.
    return list(dict.fromkeys(exact_phrases))


def build_local_hybrid_score(*, lexical_rank: int, semantic_rank: int, config: RagConfig) -> float:
    return (
        (config.hybrid_lexical_weight / (20 + lexical_rank))
        + (config.hybrid_semantic_weight / (20 + semantic_rank))
    )


def ranking_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in QUERY_TOKEN_RE.finditer(text)
        if len(match.group(0)) >= 4 and match.group(0).lower() not in RANKING_STOPWORDS
    }


def phrase_supported_by_text(text: str, text_terms: set[str], phrase: str) -> bool:
    normalized_phrase = normalize_whitespace(phrase).lower()
    if normalized_phrase and normalized_phrase in text:
        return True
    phrase_terms = ranking_terms(phrase)
    if not phrase_terms:
        return False
    matches = sum(1 for term in phrase_terms if term_matches(text_terms, term))
    required_matches = len(phrase_terms) if len(phrase_terms) <= 2 else len(phrase_terms) - 1
    return matches >= required_matches


def extract_primary_article_id(row: sqlite3.Row) -> tuple[Optional[int], str]:
    legal_provisions = json.loads(row["legal_provisions_json"] or "[]")
    candidates = [*(str(value) for value in legal_provisions), str(row["subject"] or "")]
    for candidate in candidates:
        match = ARTICLE_ID_RE.search(candidate)
        if match:
            return int(match.group(1)), match.group(2).lower()
    return None, ""


def extract_primary_article_key(row: sqlite3.Row) -> str:
    article_number, article_suffix = extract_primary_article_id(row)
    return f"{article_number}{article_suffix}" if article_number is not None else ""


def extract_article_key_from_text(value: str) -> str:
    match = ARTICLE_ID_RE.search(value or "")
    if not match:
        return ""
    return f"{match.group(1)}{match.group(2).lower()}"


def extract_statute_target_from_text(value: str) -> tuple[str, str] | None:
    if not value:
        return None
    domain_match = re.match(r"\[(CIT|PIT|VAT|PCC|SD|EXCISE|AKCYZA|ORDYNACJA|OP)\]", value, re.IGNORECASE)
    article_key = extract_article_key_from_text(value)
    if not domain_match or not article_key:
        return None
    domain = domain_match.group(1).upper()
    if domain == "OP":
        domain = "ORDYNACJA"
    if domain == "EXCISE":
        domain = "AKCYZA"
    return domain, article_key


def build_statute_target_order(targets: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    order: dict[tuple[str, str], int] = {}
    for position, (domain, article_key) in enumerate(targets):
        order.setdefault((domain.upper(), article_key.lower()), position)
    return order


def row_matching_statute_target(row: sqlite3.Row, targets: list[tuple[str, str]]) -> tuple[str, str] | None:
    if not targets:
        return None
    order = build_statute_target_order(targets)
    try:
        legal_provisions = [str(value) for value in json.loads(row["legal_provisions_json"] or "[]")]
    except (TypeError, json.JSONDecodeError):
        legal_provisions = []
    domain = str(row["tax_domain"] or "").upper()
    best_target: tuple[str, str] | None = None
    best_position = len(order)
    for provision in legal_provisions:
        article_key = extract_article_key_from_text(provision)
        if not article_key:
            continue
        candidate = (domain, article_key)
        candidate_position = order.get(candidate, len(order))
        if candidate_position < best_position:
            best_target = candidate
            best_position = candidate_position
    return best_target


def detect_procedural_article_targets(query: str) -> tuple[set[str], set[str]]:
    family_prefixes: set[str] = set()
    exact_articles: set[str] = set()
    for pattern, families, exacts, _ in STATUTE_PROCEDURAL_RULES:
        if pattern.search(query):
            family_prefixes.update(families)
            exact_articles.update(exacts)
    return family_prefixes, exact_articles


def extract_article_keys_from_query(query: str) -> list[str]:
    normalized = normalize_legal_query_refs(query or "")
    article_keys: list[str] = []
    for match in ARTICLE_ID_RE.finditer(normalized):
        key = f"{match.group(1)}{match.group(2).lower()}"
        if key not in article_keys:
            article_keys.append(key)
    return article_keys


def normalize_statute_domain(domain: str) -> str:
    value = domain.upper()
    if value == "OP":
        return "ORDYNACJA"
    if value == "EXCISE":
        return "AKCYZA"
    return value


def detect_explicit_statute_domains(query: str) -> set[str]:
    normalized = normalize_whitespace(query or "").lower()
    domains: set[str] = set()
    if re.search(r"\b(vat|ustaw\w* o vat|ustaw\w* vat|podatku od towar[óo]w i us[łl]ug|towar[óo]w i us[łl]ug)\b", normalized):
        domains.add("VAT")
    if re.search(r"\b(cit|ustaw\w* o cit|ustaw\w* cit|podatku dochodowym od os[óo]b prawnych|dochodowym od os[óo]b prawnych)\b", normalized):
        domains.add("CIT")
    if re.search(r"\b(pit|ustaw\w* o pit|ustaw\w* pit|podatku dochodowym od os[óo]b fizycznych|dochodowym od os[óo]b fizycznych)\b", normalized):
        domains.add("PIT")
    if re.search(r"\b(pcc|ustaw\w* o pcc|ustaw\w* pcc|podatku od czynno[śs]ci cywilnoprawnych|czynno[śs]ci cywilnoprawnych)\b", normalized):
        domains.add("PCC")
    if re.search(r"\b(ustaw\w* o podatku od spadk[óo]w i darowizn|spadk[óo]w i darowizn|sd-z2)\b", normalized):
        domains.add("SD")
    if re.search(r"\b(ordynacj\w* podatkow\w*|ordynacji podatkowej)\b", normalized):
        domains.add("ORDYNACJA")
    if re.search(r"\b(ustaw\w* akcyzow\w*|akcyz\w*)\b", normalized):
        domains.add("AKCYZA")
    if re.search(r"\b(podatk\w* od nieruchomo[śs]ci|ustaw\w* o podatkach i op[łl]atach lokalnych|u\.?p\.?o\.?l\.?)\b", normalized):
        domains.add("NIERUCHOMOŚCI")
    return domains


def build_mechanism_statute_targets(query: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for axis in decompose_query_into_legal_axes(query):
        if axis.source_types and "statute" not in axis.source_types:
            continue
        targets.extend(axis.preferred_targets or ())

    target_builders = (
        build_ksef_current_law_statute_targets,
        build_ksef_b2c_invoice_statute_targets,
        build_ksef_outside_deduction_statute_targets,
        lambda value: [("VAT", "106k")] if query_targets_ksef_correction_issue(value) else [],
        build_vat_dropshipping_ioss_statute_targets,
        build_private_vehicle_pit_expense_statute_targets,
        build_spolka_komandytowa_cit_status_statute_targets,
        build_invoice_address_error_statute_targets,
        build_fixed_establishment_vat_statute_targets,
        build_family_foundation_statute_targets,
        build_wht_pay_and_refund_service_statute_targets,
        build_wht_crossborder_payment_statute_targets,
        build_poland_germany_treaty_statute_targets,
        build_poland_spain_treaty_statute_targets,
        build_debt_assumption_statute_targets,
        build_property_sale_pit_statute_targets,
        build_housing_relief_temporary_rental_statute_targets,
        build_housing_relief_loan_repayment_statute_targets,
        build_mortgage_settlement_refund_statute_targets,
        build_developer_land_sale_statute_targets,
        build_post_leasing_vehicle_gift_sale_statute_targets,
        build_leased_movable_six_year_statute_targets,
        build_gifted_asset_cost_basis_statute_targets,
        build_spouse_gift_sd_statute_targets,
        build_estonian_cit_hidden_profit_statute_targets,
        build_transformation_share_cost_statute_targets,
        build_shareholder_company_asset_sale_statute_targets,
    )
    for builder in target_builders:
        targets.extend(builder(query))
    return list(dict.fromkeys(targets))


def build_explicit_statute_article_targets(query: str) -> list[tuple[str, str]]:
    article_keys = extract_article_keys_from_query(query)
    if not article_keys:
        return []

    explicit_domains = detect_explicit_statute_domains(query)
    domains = explicit_domains or {
        normalize_statute_domain(domain)
        for domain in resolve_statute_tax_domains(query)
        if normalize_statute_domain(domain) not in {"WHT"}
    }
    if not domains:
        return []

    preferred_domain_order = ["VAT", "CIT", "PIT", "PCC", "SD", "ORDYNACJA", "AKCYZA", "NIERUCHOMOŚCI"]
    ordered_domains = [domain for domain in preferred_domain_order if domain in domains]
    ordered_domains.extend(sorted(domains - set(ordered_domains)))
    return [(domain, article_key) for domain in ordered_domains for article_key in article_keys]


def query_is_statute_focused(query: str) -> bool:
    normalized = normalize_whitespace(query or "").lower()
    return bool(
        detect_explicit_statute_domains(query)
        or re.search(
            r"\b(przepis\w*|ustaw\w*|podstaw\w* prawn\w*|uregulowan\w*|reguluj\w*|"
            r"co mówi|co wynika|przytocz|zacytuj|jak brzmi|pokaż|pokaz|podaj|treść|tresc|brzmieni\w*)\b",
            normalized,
        )
    )


def build_general_statute_concept_targets(query: str) -> list[tuple[str, str]]:
    if build_explicit_statute_article_targets(query) or not query_is_statute_focused(query):
        return []

    normalized = normalize_whitespace(query or "").lower()
    mechanism_targets = build_mechanism_statute_targets(query)
    domains = detect_explicit_statute_domains(query) or {
        normalize_statute_domain(domain)
        for domain in resolve_statute_tax_domains(query)
        if normalize_statute_domain(domain) not in {"WHT"}
    }
    targets: list[tuple[str, str]] = [*mechanism_targets]
    asks_act_scope = bool(
        re.search(
            r"\b(co\s+reguluje|zakres\s+ustaw\w*|przedmiot\s+ustaw\w*|ustaw\w*\s+reguluje|regulacj\w*\s+ustaw\w*)\b",
            normalized,
        )
    )

    if "VAT" in domains:
        if asks_act_scope:
            targets.extend([("VAT", "1")])
        elif re.search(r"\b(czynno[śs]c\w* podleg\w*|przedmiot\w* opodatkow\w*|odp[łl]atn\w* dostaw\w*)\b", normalized):
            targets.extend([("VAT", "5")])
        elif re.search(r"\b(dostaw\w* towar\w*)\b", normalized):
            targets.extend([("VAT", "7")])
        elif re.search(r"\b(terytori\w* kraj\w*|definicj\w* ustawow\w*)\b", normalized):
            targets.extend([("VAT", "2")])
        elif re.search(r"\b(ksef|faktur\w*)\b", normalized):
            targets.extend([("VAT", "106a"), ("VAT", "106b"), ("VAT", "106ga"), ("VAT", "106gb")])
        elif re.search(r"\b(odlicz\w*|naliczon\w*)\b", normalized):
            targets.extend([("VAT", "86"), ("VAT", "88")])
        elif re.search(r"\b(nieruchomo\w*|grunt\w*|budyn\w*|zwolnien\w*)\b", normalized):
            targets.extend([("VAT", "5"), ("VAT", "7"), ("VAT", "15"), ("VAT", "43"), ("VAT", "2")])
        else:
            targets.extend([("VAT", "5"), ("VAT", "7"), ("VAT", "8"), ("VAT", "15"), ("VAT", "29a")])

    if "CIT" in domains:
        if asks_act_scope:
            targets.extend([("CIT", "1")])
        elif re.search(r"\b(obowi[ąa]zk\w* podatkow\w*.*ca[łl]o[śs]ci|siedzib\w* lub zarz[ąa]d\w*|rezydencj\w* podatkow\w*)\b", normalized):
            targets.extend([("CIT", "3")])
        elif re.search(r"\b(definicj\w* ustawow\w*|przedsi[ęe]biorstw\w*|zorganizowan\w* cz[ęe][śs][ćc]\w* przedsi[ęe]biorstw\w*)\b", normalized):
            targets.extend([("CIT", "4a")])
        elif re.search(r"\b(przedmiot\w* opodatkow\w*)\b", normalized):
            targets.extend([("CIT", "7")])
        elif re.search(r"\b(podatkow\w* grup\w* kapita[łl]ow\w*)\b", normalized):
            targets.extend([("CIT", "7a")])
        elif re.search(r"\b(wht|u [źz]r[óo]d[łl]a|odset(?:k|ek|e?k)\w*|zarz[ąa]dz\w*|nierezydent\w*)\b", normalized):
            targets.extend([("CIT", "21"), ("CIT", "22"), ("CIT", "26")])
        elif re.search(r"\b(koszt\w*|kup|uzyskania przychod)\b", normalized):
            targets.extend([("CIT", "15"), ("CIT", "16")])
        else:
            targets.extend([("CIT", "7"), ("CIT", "12"), ("CIT", "15"), ("CIT", "16")])

    if "PIT" in domains:
        if asks_act_scope:
            targets.extend([("PIT", "1")])
        elif re.search(r"\b(samochod\w*|samochód\w*|pojazd\w*|koszt\w*|wydatk\w*)\b", normalized):
            targets.extend([("PIT", "22"), ("PIT", "23")])
        elif re.search(r"\b(sprzeda\w*|nieruchomo\w*|mieszkani\w*)\b", normalized):
            targets.extend([("PIT", "10"), ("PIT", "21"), ("PIT", "30e")])
        else:
            targets.extend([("PIT", "9"), ("PIT", "10"), ("PIT", "14"), ("PIT", "21"), ("PIT", "22"), ("PIT", "23")])

    if "PCC" in domains:
        targets.extend([("PCC", "1")] if asks_act_scope else [("PCC", "1"), ("PCC", "2"), ("PCC", "4"), ("PCC", "6"), ("PCC", "7")])

    if "SD" in domains:
        targets.extend([("SD", "1")] if asks_act_scope else [("SD", "1"), ("SD", "4a"), ("SD", "9"), ("SD", "14"), ("SD", "15")])

    if "ORDYNACJA" in domains:
        if asks_act_scope:
            targets.extend([("ORDYNACJA", "1")])
        elif re.search(r"\b(in dubio pro tributario|niedaj[ąa]c\w* usun[ąa][ćc]\w* w[ąa]tpliwo[śs]c\w*)\b", normalized):
            targets.extend([("ORDYNACJA", "2a")])
        elif re.search(r"\b(definicj\w* ustawow\w*)\b", normalized):
            targets.extend([("ORDYNACJA", "3")])
        elif re.search(r"\b(defini\w* podat\w*|czym jest podatek)\b", normalized):
            targets.extend([("ORDYNACJA", "6")])
        elif re.search(r"\b(sukcesj\w*|przekszta[łl]c\w*)\b", normalized):
            targets.extend([("ORDYNACJA", "93a"), ("ORDYNACJA", "93e")])
        elif re.search(r"\b(przedawn\w*)\b", normalized):
            targets.extend([("ORDYNACJA", "70")])
        elif re.search(r"\b(interpretacj\w*)\b", normalized):
            targets.extend([("ORDYNACJA", "14b"), ("ORDYNACJA", "14k"), ("ORDYNACJA", "14na")])
        else:
            targets.extend([("ORDYNACJA", "14b"), ("ORDYNACJA", "70"), ("ORDYNACJA", "81"), ("ORDYNACJA", "93a")])

    if "AKCYZA" in domains:
        targets.extend([("AKCYZA", "1"), ("AKCYZA", "8"), ("AKCYZA", "21")])

    if "NIERUCHOMOŚCI" in domains:
        if asks_act_scope:
            targets.extend([("NIERUCHOMOŚCI", "1")])
        elif re.search(
            r"\b(definicj\w*|defini\w*|oznacza|budyn\w*|budowl\w*|powierzchni\w* użytkow\w*|powierzchni\w* uzytkow\w*)\b",
            normalized,
        ):
            targets.extend([("NIERUCHOMOŚCI", "1a")])
        else:
            targets.extend([
                ("NIERUCHOMOŚCI", "1a"),
                ("NIERUCHOMOŚCI", "2"),
                ("NIERUCHOMOŚCI", "3"),
                ("NIERUCHOMOŚCI", "4"),
                ("NIERUCHOMOŚCI", "5"),
                ("NIERUCHOMOŚCI", "7"),
            ])

    return list(dict.fromkeys(targets))


def query_is_direct_statute_lookup(query: str) -> bool:
    if not (build_explicit_statute_article_targets(query) or build_general_statute_concept_targets(query)):
        return False
    normalized = normalize_whitespace(query or "").lower()
    if re.search(r"\b(interpretacj\w*|wyrok\w*|orzecze(?:nie|nia|ń)|orzecznictw\w*|nsa|wsa|fsk)\b", normalized):
        return False
    return query_is_statute_focused(query)


def build_preferred_statute_targets(query: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    targets.extend(build_explicit_statute_article_targets(query))
    if query_targets_ksef_foreign_sale(query):
        targets.extend(KSEF_FOREIGN_SALE_STATUTE_TARGETS)
    targets.extend(build_mechanism_statute_targets(query))
    targets.extend(build_general_statute_concept_targets(query))
    if query_targets_ksef_correction_issue(query):
        targets.append(("VAT", "106k"))
    if query_targets_small_taxpayer_foreign_vat(query):
        targets.extend([("CIT", "4a"), ("CIT", "19"), ("CIT", "12")])

    _, procedural_exact_articles = detect_procedural_article_targets(query)
    if procedural_exact_articles:
        hinted_domains = resolve_statute_tax_domains(query) or {
            "VAT",
            "CIT",
            "PIT",
            "PCC",
            "AKCYZA",
            "ORDYNACJA",
            "NIERUCHOMOŚCI",
        }
        for domain in sorted(hinted_domains):
            for article_key in sorted(procedural_exact_articles):
                targets.append((normalize_statute_domain(domain), article_key))
    return list(dict.fromkeys((domain.upper(), article_key.lower()) for domain, article_key in targets if domain and article_key))


def build_legal_source_plan(
    query: str,
    *,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> LegalSourcePlan:
    effective_include_judgments = True if include_judgments is None else include_judgments
    secondary_source_types: list[str] = []
    if include_interpretations:
        secondary_source_types.append("interpretation")
    if effective_include_judgments:
        secondary_source_types.append("judgment")

    axes = tuple(decompose_query_into_legal_axes(query))
    explicit_targets = tuple(build_explicit_statute_article_targets(query))
    statute_targets = tuple(build_preferred_statute_targets(query))
    tax_domains = tuple(sorted(resolve_statute_tax_domains(query)))
    return LegalSourcePlan(
        query=query,
        axes=axes,
        primary_source_types=("statute",),
        secondary_source_types=tuple(secondary_source_types),
        statute_targets=statute_targets,
        explicit_statute_targets=explicit_targets,
        tax_domains=tax_domains,
        primary_required=True,
        stage_order=(
            "planner",
            "primary_law_deterministic_retrieval",
            "primary_law_semantic_retrieval",
            "secondary_sources_retrieval",
            "legal_rule_extraction",
            "writer",
        ),
    )


def chunk_matches_statute_target(chunk: RagChunk, target: tuple[str, str]) -> bool:
    domain, article_key = target
    wanted_domain = normalize_statute_domain(domain)
    wanted_article = article_key.lower()
    chunk_domain = normalize_statute_domain(infer_chunk_tax_domain(chunk)) if infer_chunk_tax_domain(chunk) else ""
    subject = normalize_whitespace(chunk.subject or "").lower()
    for provision in chunk.legal_provisions:
        provision_target = extract_statute_target_from_text(provision)
        if provision_target == (wanted_domain, wanted_article):
            return True
        provision_article = extract_article_key_from_text(provision)
        if not provision_article or provision_article != wanted_article:
            continue
        if chunk_domain == wanted_domain:
            return True
        if subject.startswith("upo polska"):
            return True
    return False


def legal_source_plan_primary_satisfied(plan: LegalSourcePlan, chunks: list[RagChunk]) -> bool:
    if not plan.primary_required:
        return True
    primary_chunks = [chunk for chunk in chunks if is_primary_source_chunk(chunk)]
    if not primary_chunks:
        return False

    coverages = build_axis_coverage(plan.query, chunks)
    if coverages:
        return all(
            coverage.primary_source_present
            and coverage.controlling_rule_present
            and coverage.current_law_source_present
            and coverage.required_treaty_present is not False
            for coverage in coverages
        )

    if plan.explicit_statute_targets:
        return all(
            any(chunk_matches_statute_target(chunk, target) for chunk in primary_chunks)
            for target in plan.explicit_statute_targets
        )

    if plan.statute_targets:
        return any(
            chunk_matches_statute_target(chunk, target)
            for target in plan.statute_targets
            for chunk in primary_chunks
        )

    return True


def dedupe_chunks_by_canonical_source(chunks: list[RagChunk]) -> list[RagChunk]:
    deduped: list[RagChunk] = []
    seen_sources: set[str] = set()
    for chunk in chunks:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_sources:
            continue
        seen_sources.add(canonical_source_id)
        deduped.append(chunk)
    return deduped


def legal_source_plan_to_dict(plan: LegalSourcePlan, chunks: Optional[list[RagChunk]] = None) -> dict[str, Any]:
    primary_chunks = [chunk for chunk in chunks or [] if is_primary_source_chunk(chunk)]
    secondary_chunks = [
        chunk for chunk in chunks or []
        if str(chunk.source_type or "").lower() in {"interpretation", "judgment", "commentary"}
    ]
    return {
        "primary_required": plan.primary_required,
        "primary_source_types": list(plan.primary_source_types),
        "secondary_source_types": list(plan.secondary_source_types),
        "tax_domains": list(plan.tax_domains),
        "statute_targets": [
            {"domain": domain, "article": article_key}
            for domain, article_key in plan.statute_targets
        ],
        "explicit_statute_targets": [
            {"domain": domain, "article": article_key}
            for domain, article_key in plan.explicit_statute_targets
        ],
        "axes": [
            {
                "axis_id": axis.axis_id,
                "label": axis.label,
                "source_types": sorted(axis.source_types or []),
                "tax_domains": sorted(axis.tax_domains or []),
                "preferred_targets": [
                    {"domain": domain, "article": article_key}
                    for domain, article_key in axis.preferred_targets
                ],
            }
            for axis in plan.axes
        ],
        "stage_order": list(plan.stage_order),
        "primary_satisfied": legal_source_plan_primary_satisfied(plan, chunks or []),
        "primary_source_count": len(primary_chunks),
        "secondary_source_count": len(secondary_chunks),
    }


def build_source_plan_context(plan: LegalSourcePlan, chunks: list[RagChunk]) -> str:
    plan_dict = legal_source_plan_to_dict(plan, chunks)
    target_text = ", ".join(
        f"{item['domain']} art. {item['article']}" for item in plan_dict["statute_targets"]
    ) or "brak dokładnego celu artykułowego"
    axes_text = ", ".join(axis["axis_id"] for axis in plan_dict["axes"]) or "brak osi specjalistycznej"
    return "\n".join(
        [
            "Plan źródeł przed odpowiedzią:",
            "- primary sources są obowiązkowe: akty prawa powszechnie obowiązującego, w tym ustawy, rozporządzenia i UPO.",
            "- secondary sources są uzupełniające: interpretacje, objaśnienia, interpretacje ogólne i wyroki.",
            f"- kolejność etapów: {' -> '.join(plan.stage_order)}",
            f"- osie prawne: {axes_text}",
            f"- docelowe przepisy/akty: {target_text}",
            f"- primary source present/satisfied: {str(plan_dict['primary_satisfied']).lower()} "
            f"({plan_dict['primary_source_count']} primary, {plan_dict['secondary_source_count']} secondary)",
            "- twarda reguła: jeżeli primary source nie jest spełnione dla osi, nie wolno oprzeć rozstrzygnięcia na interpretacji ani wyroku.",
        ]
    )


def build_article_family_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "statute":
        return 0.0
    family_prefixes, exact_articles = detect_procedural_article_targets(query)
    if not family_prefixes and not exact_articles:
        return 0.0
    article_key = extract_primary_article_key(row)
    if not article_key:
        return 0.0
    score = 0.0
    if article_key in exact_articles:
        score += 1.5
    if any(article_key.startswith(prefix) for prefix in family_prefixes):
        score += 1.1
    return score


def detect_domains(text: str) -> set[str]:
    normalized = text.lower()
    return {
        domain
        for domain, markers in DOMAIN_MARKERS.items()
        if any(marker in normalized for marker in markers)
    }


def row_tax_domains(row: sqlite3.Row) -> set[str]:
    domains = {str(row["tax_domain"] or "").upper()} if str(row["tax_domain"] or "").strip() else set()
    try:
        legal_provisions = json.loads(row["legal_provisions_json"] or "[]")
    except (json.JSONDecodeError, TypeError):
        legal_provisions = []
    for provision in legal_provisions:
        match = re.match(r"\[(CIT|PIT|VAT|PCC|SD|EXCISE|AKCYZA|ORDYNACJA|OP)\]", str(provision), re.IGNORECASE)
        if match:
            domain = match.group(1).upper()
            if domain == "OP":
                domain = "ORDYNACJA"
            if domain == "EXCISE":
                domain = "AKCYZA"
            domains.add(domain)
    if not domains:
        fallback_text = " ".join(
            [
                str(row["subject"] or ""),
                str(row["category"] or ""),
                str(row["chunk_text"] or "")[:1200],
                join_search_text([str(value) for value in json.loads(row["issues_json"] or "[]")]),
                join_search_text([str(value) for value in json.loads(row["law_tags_json"] or "[]")]),
            ]
        )
        domains.update(domain.upper() for domain in detect_domains(fallback_text))
    return domains


def resolve_statute_tax_domains(query: str) -> set[str]:
    """Map query-level topical markers to statute corpus tax domains.

    Some user intents, such as WHT, are a cross-cutting topic rather than a
    stored statute domain.  When we enforce domain filtering for statute
    retrieval, expand those topics to the concrete statute domains that carry
    the governing provisions.
    """
    explicit_domains = detect_explicit_statute_domains(query)
    # An explicit act name is stronger evidence than broad topical markers.
    # For example, "ustawa o podatkach i opłatach lokalnych" used to trigger
    # the generic PIT marker and route a building-definition question into the
    # PIT corpus.  Preserve all explicitly named acts, but do not dilute them
    # with heuristic domains.
    if explicit_domains:
        return {normalize_statute_domain(domain) for domain in explicit_domains}

    domains = {domain.upper() for domain in detect_domains(query)}
    if "WHT" in domains:
        domains.update({"CIT", "PIT"})
    if query_targets_crossborder_treaty_analysis(query):
        domains.update({"CIT", "PIT"})
    if query_targets_poland_germany_treaty(query):
        domains.update({"CIT", "PIT"})
    if query_targets_poland_spain_treaty(query):
        domains.update({"CIT", "PIT"})
    if query_targets_developer_land_sale(query):
        domains.update({"VAT", "PIT", "PCC"})
    if query_targets_post_leasing_vehicle_gift_sale(query):
        domains.update({"VAT", "PIT", "SD"})
    if query_targets_leased_movable_six_year_rule(query):
        domains.add("PIT")
    if query_targets_gifted_asset_cost_basis(query):
        domains.update({"PIT", "SD"})
    if query_targets_spouse_gift_sd(query):
        domains.add("SD")
    if query_targets_debt_assumption_effectiveness(query):
        domains.update({"PIT", "SD"})
        if "PCC" not in {domain.upper() for domain in detect_domains(query)}:
            domains.discard("PCC")
    if query_targets_housing_relief_temporary_rental(query) or query_targets_housing_relief_loan_repayment(query) or query_targets_mortgage_settlement_refund(query):
        domains.add("PIT")
    if query_targets_estonian_cit_transformation_share_cost(query) or query_targets_estonian_cit_hidden_profit(query):
        domains.update({"CIT", "PIT", "PCC", "ORDYNACJA"})
    if query_targets_vat_dropshipping_ioss(query):
        domains.add("VAT")
    if query_targets_ksef_b2c_invoice(query):
        domains.add("VAT")
    if query_targets_ksef_current_law(query):
        domains.add("VAT")
    if query_targets_private_vehicle_pit_expense(query):
        domains.add("PIT")
    if query_targets_spolka_komandytowa_cit_status(query):
        domains.add("CIT")
    if query_targets_invoice_address_error(query):
        domains.add("VAT")
    if query_targets_fixed_establishment_vat(query):
        domains.add("VAT")
    if query_targets_family_foundation_mechanism(query):
        domains.update({"CIT", "PIT", "VAT"})
    if query_targets_wht_pay_and_refund_services(query):
        domains.add("CIT")
    return domains


def infer_retrieval_tax_domains(query: str) -> set[str]:
    domains = resolve_statute_tax_domains(query)
    if query_targets_poland_germany_treaty(query):
        domains.update({"CIT", "PIT"})
    if query_targets_ksef_b2c_invoice(query):
        domains.add("VAT")
    if query_targets_ksef_current_law(query):
        domains.add("VAT")
    if query_targets_private_vehicle_pit_expense(query):
        domains.add("PIT")
    if query_targets_spolka_komandytowa_cit_status(query):
        domains.add("CIT")
    if query_targets_invoice_address_error(query):
        domains.add("VAT")
    if query_targets_fixed_establishment_vat(query):
        domains.add("VAT")
    if query_targets_family_foundation_mechanism(query):
        domains.update({"CIT", "PIT", "VAT"})
    if query_targets_wht_pay_and_refund_services(query):
        domains.add("CIT")
    return domains


def load_mechanism_rules(config: RagConfig) -> dict[str, tuple[str, ...]]:
    if not config.mechanism_lexicon_path.exists(): return MECHANISM_RULES
    try:
        p=json.loads(config.mechanism_lexicon_path.read_text(encoding="utf-8"))
        r={str(x["id"]):tuple(str(a).lower() for a in x.get("aliases",[]) if str(a).strip()) for x in p.get("mechanisms",[]) if x.get("status")=="ready" and x.get("id") and x.get("aliases")}
        merged = dict(MECHANISM_RULES)
        merged.update(r)
        return merged
    except (OSError,json.JSONDecodeError,TypeError): return MECHANISM_RULES

def detect_mechanisms(text: str, *, config: RagConfig) -> set[str]:
    normalized = text.lower()
    return {name for name, patterns in load_mechanism_rules(config).items() if any(pattern in normalized for pattern in patterns)}


def term_matches(candidate_terms: set[str], query_term: str) -> bool:
    if query_term in candidate_terms:
        return True
    # Inflection is common in Polish legal prose; a six-character stem is a
    # conservative middle ground between exact matching and fuzzy noise.
    stem = query_term[:6]
    return len(stem) == 6 and any(term.startswith(stem) for term in candidate_terms)


def build_legal_match_score(row: sqlite3.Row, *, query: str) -> float:
    """Score explicit legal/factual overlap independently of body-text similarity."""
    query = normalize_legal_query_refs(query)
    query_terms = ranking_terms(query)
    if not query_terms:
        return 0.0

    keywords = json.loads(row["keywords_json"] or "[]")
    legal_provisions = json.loads(row["legal_provisions_json"] or "[]")
    issues = json.loads(row["issues_json"] or "[]")
    law_tags = json.loads(row["law_tags_json"] or "[]")
    fields = [
        (str(row["subject"] or ""), 4.0),
        (join_search_text([str(value) for value in issues]), 3.0),
        (join_search_text([str(value) for value in legal_provisions]), 3.0),
        (join_search_text([str(value) for value in keywords]), 2.0),
        (join_search_text([str(value) for value in law_tags]), 1.5),
        (str(row["chunk_text"] or ""), 0.5),
    ]
    overlap = 0.0
    candidate_text_parts: list[str] = []
    for value, weight in fields:
        candidate_text_parts.append(value)
        terms = ranking_terms(value)
        overlap += sum(weight for term in query_terms if term_matches(terms, term))

    normalized_overlap = min(overlap / (len(query_terms) * 4.0), 1.0)
    candidate_joined_text = " ".join(candidate_text_parts).lower()
    article_hits = 0.0
    for article in ("106nda", "106nh", "106na", "106gb"):
        if article in query.lower() and article in candidate_joined_text:
            article_hits += 0.22 if article in {"106nda", "106nh"} else 0.12
    normalized_overlap = min(normalized_overlap + article_hits, 1.0)
    query_domains = detect_domains(query)
    candidate_domains = detect_domains(" ".join(candidate_text_parts))
    if query_domains and query_domains & candidate_domains:
        return min(normalized_overlap + 0.35, 1.0)
    if query_domains and candidate_domains:
        return max(normalized_overlap - 0.2, -0.2)
    return normalized_overlap


def normalize_signature_year(value: str) -> str:
    return value if len(value) == 4 else f"20{value}"


def extract_judgment_signatures(value: str) -> list[tuple[str, str, str]]:
    signatures: list[tuple[str, str, str]] = []
    for match in JUDGMENT_SIGNATURE_RE.finditer(value or ""):
        chamber = (match.group(1) or "").upper()
        signatures.append((chamber, match.group(2), normalize_signature_year(match.group(3))))
    return signatures


def extract_judgment_chambers(value: str) -> set[str]:
    return {match.group(1).upper() for match in JUDGMENT_CHAMBER_RE.finditer(value or "")}


def build_judgment_candidate_text(row: sqlite3.Row, *, chunk_chars: int = 2200) -> str:
    return normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["signature"] or ""),
                str(row["category"] or ""),
                str(row["tax_domain"] or ""),
                join_search_text([str(value) for value in json.loads(row["issues_json"] or "[]")]),
                join_search_text([str(value) for value in json.loads(row["keywords_json"] or "[]")]),
                join_search_text([str(value) for value in json.loads(row["law_tags_json"] or "[]")]),
                join_search_text([str(value) for value in json.loads(row["legal_provisions_json"] or "[]")]),
                str(row["chunk_text"] or "")[:chunk_chars],
            ]
        )
    )


def build_judgment_result_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "judgment":
        return 0.0
    normalized_query = normalize_whitespace(query).lower()
    candidate_text = build_judgment_candidate_text(row).lower()
    score = 0.0
    asks_for_second_instance_decision = re.search(
        r"uchylon\w*.*wyrok.*decyzj\w*.*(?:drugiej|ii) instancji|decyzj\w*.*(?:drugiej|ii) instancji.*uchylon\w*",
        normalized_query,
    )
    if asks_for_second_instance_decision:
        if "uchylono zaskarżony wyrok i decyzję ii instancji" in candidate_text:
            score += 2.0
        elif "uchylono zaskarżony wyrok" in candidate_text and re.search(r"decyzj\w*.*ii instancji", candidate_text):
            score += 1.3
        elif "uchylono zaskarżony wyrok" in candidate_text:
            score += 0.45
    elif "uchylon" in normalized_query and "uchylono zaskarżony wyrok" in candidate_text:
        score += 0.45
    if "oddal" in normalized_query and "oddalono skargę kasacyjną" in candidate_text:
        score += 0.45
    return min(score, 2.0)


def build_judgment_topic_phrase_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "judgment":
        return 0.0
    normalized_query = normalize_whitespace(query).lower()
    candidate_text = build_judgment_candidate_text(row).lower()
    score = 0.0
    if re.search(r"sponsor\w*", normalized_query) and "faktur" in normalized_query and "koszt" in normalized_query:
        has_sponsoring = bool(re.search(r"sponsor\w*", candidate_text))
        has_invoice = bool(re.search(r"faktur\w*", candidate_text))
        has_cost = bool(re.search(r"koszt\w* uzyskania przychod\w*|koszt\w* podatkow\w*|koszt\w*", candidate_text))
        if has_sponsoring:
            score += 1.15
            if has_invoice:
                score += 0.25
            if has_cost:
                score += 0.25
        elif has_invoice and has_cost:
            score += 0.25
    if re.search(r"art\.\s*70.*(?:par\.|§)\s*6.*pkt\s*1", normalized_query) and "amortyz" in normalized_query:
        has_article = bool(re.search(r"art\.\s*70\s*(?:§|par\.)?\s*6\s*pkt\s*1|art\.\s*70", candidate_text))
        has_amortization = "amortyz" in candidate_text
        has_cit = "CIT" in row_tax_domains(row) or "podatek dochodowy od osób prawnych" in candidate_text
        if has_article and has_amortization and has_cit:
            score += 1.6
        elif has_article and has_amortization:
            score += 0.8
        elif has_article and has_cit:
            score += 0.45
    return min(score, 1.8)


def build_judgment_metadata_match_score(row: sqlite3.Row, *, query: str) -> float:
    score = (
        build_judgment_match_score(row, query=query)
        + build_judgment_result_match_score(row, query=query)
        + build_judgment_topic_phrase_score(row, query=query)
    )
    query_domains = {domain.upper() for domain in detect_domains(query)}
    candidate_domains = row_tax_domains(row)
    if query_domains and candidate_domains and not (query_domains & candidate_domains):
        score -= 0.75
    return min(max(score, -0.75), 6.0)


def build_judgment_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "judgment":
        return 0.0

    score = 0.0
    query_signatures = extract_judgment_signatures(query)
    row_signatures = extract_judgment_signatures(str(row["signature"] or ""))
    for query_chamber, query_number, query_year in query_signatures:
        for row_chamber, row_number, row_year in row_signatures:
            if query_number == row_number and query_year == row_year:
                score += 3.0 if not query_chamber or query_chamber == row_chamber else 2.2
            elif query_chamber and query_chamber == row_chamber:
                score += 0.25

    query_chambers = extract_judgment_chambers(query)
    row_chambers = extract_judgment_chambers(str(row["signature"] or ""))
    if query_chambers and row_chambers:
        score += 0.7 if query_chambers & row_chambers else -0.35

    query_terms = ranking_terms(query)
    if query_terms:
        metadata_text = join_search_text(
            [
                str(row["subject"] or ""),
                str(row["signature"] or ""),
                join_search_text([str(value) for value in json.loads(row["issues_json"] or "[]")]),
                join_search_text([str(value) for value in json.loads(row["keywords_json"] or "[]")]),
                str(row["chunk_text"] or "")[:1400],
            ]
        )
        metadata_terms = ranking_terms(metadata_text)
        overlap = sum(1 for term in query_terms if term_matches(metadata_terms, term)) / len(query_terms)
        score += min(overlap, 1.0)

    return min(max(score, -0.35), 4.0)


def build_facts_match_score(row: sqlite3.Row, *, query: str) -> float:
    query_terms = ranking_terms(query)
    facts_terms = ranking_terms(str(row["facts_text"] or ""))
    return (sum(1 for term in query_terms if term_matches(facts_terms, term)) / len(query_terms)) if query_terms else 0.0


def build_subject_phrase_match_score(row: sqlite3.Row, *, query: str) -> float:
    subject = normalize_whitespace(str(row["subject"] or "")).lower()
    normalized_query = normalize_whitespace(query).lower()
    if len(subject) < 24 or not normalized_query:
        return 0.0
    if subject in normalized_query:
        return 1.4
    case_subject = re.search(r"w sprawie:\s*(.+)$", normalized_query)
    if case_subject and normalize_whitespace(case_subject.group(1)).lower() in subject:
        return 1.0
    return 0.0


def build_interpretation_section_match_score(row: sqlite3.Row) -> float:
    source_type = str(row["source_type"] or "")
    if source_type not in {"interpretation", "judgment"}:
        return 0.0
    return build_resolution_section_score(str(row["chunk_text"] or ""), source_type=source_type)


def build_mechanism_match_score(row: sqlite3.Row, *, query: str, config: RagConfig) -> float:
    query_mechanisms = detect_mechanisms(query, config=config)
    if not query_mechanisms:
        return 0.0
    candidate_text = " ".join(str(row[key] or "") for key in ("subject", "question_text", "issues_json", "keywords_json", "chunk_text"))
    candidate_mechanisms = detect_mechanisms(candidate_text, config=config)
    return len(query_mechanisms & candidate_mechanisms) / len(query_mechanisms)


def build_pcc_interpretation_match_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "interpretation" or "PCC" not in row_tax_domains(row):
        return 0.0

    normalized_query = normalize_whitespace(query).lower()
    candidate_text = normalize_whitespace(
        " ".join(
            str(row[key] or "")
            for key in ("subject", "question_text", "issues_json", "keywords_json", "legal_provisions_json", "chunk_text")
        )
    ).lower()
    score = 0.0
    if query_targets_debt_assumption_effectiveness(query) and "pcc" not in normalized_query:
        score -= 1.0
    if "pożycz" in normalized_query and "pożycz" in candidate_text:
        score += 0.6
        if re.search(r"art\.\s*2\s*pkt\s*4|vat|towar[óo]w i usług", normalized_query) and re.search(
            r"art\.\s*2\s*pkt\s*4|zwolnion\w* od podatku od towar[óo]w i usług|opodatkow\w* podatkiem od towar[óo]w i usług",
            candidate_text,
        ):
            score += 0.9
        if re.search(r"art\.\s*2\s*pkt\s*4\s*lit\.\s*b|zwolnion\w* będzie z opodatkowania|nie ciąży .*obowiązek podatkowy", normalized_query) and re.search(
            r"art\.\s*2\s*pkt\s*4\s*lit\.\s*b|zwolnion\w* będzie z opodatkowania|nie ciąży .*obowiązek podatkowy",
            candidate_text,
        ):
            score += 1.2
    first_home_or_purchase_exemption_query = re.search(
        r"pierwsz\w* mieszk|pierwsz\w* nieruchomo|art\.\s*9\s*pkt\s*17|zwolnieni\w* z pcc|zwolnien\w*.*(?:zakup|nabycie).*nieruchomości|(?:zakup|nabycie).*nieruchomości.*zwolnien\w*",
        normalized_query,
    )
    if first_home_or_purchase_exemption_query and re.search(
        r"art\.\s*9\s*pkt\s*17|art\. 9-pkt 17|pierwsz\w* mieszk|pierwsz\w* nieruchomo|zakup pierwszej nieruchomości|budynek mieszkalny|własnych potrzeb mieszkaniowych",
        candidate_text,
    ):
        score += 1.4
    if "nieruchomo" in normalized_query and "nieruchomo" in candidate_text and not re.search(
        r"pierwsz\w* mieszk|pierwsz\w* nieruchomo|art\.\s*9\s*pkt\s*17|zwolnieni\w* z pcc|zwolnien\w*.*(?:zakup|nabycie).*nieruchomości|(?:zakup|nabycie).*nieruchomości.*zwolnien\w*",
        normalized_query,
    ):
        score += 0.35
        if re.search(r"art\.\s*2\s*pkt\s*4|vat|towar[óo]w i usług", normalized_query) and re.search(
            r"art\.\s*2\s*pkt\s*4|opodatkow\w* podatkiem od towar[óo]w i usług|zwolnion\w* od podatku od towar[óo]w i usług",
            candidate_text,
        ):
            score += 0.45
        if not re.search(r"art\.\s*2\s*pkt\s*4|vat|towar[óo]w i usług", normalized_query):
            row_domain = str(row["tax_domain"] or "").upper()
            if row_domain == "PCC" and re.search(r"art\.\s*(4|6|7|10)|art\. (4|6|7|10)-|podlega opodatkowaniu podatkiem od czynności cywilnoprawnych", candidate_text):
                score += 1.6
    if query_targets_developer_land_sale(query) and re.search(r"nieruchomo|grunt|działk|dzialk", candidate_text):
        score += 0.4
        if re.search(r"art\.\s*2\s*pkt\s*4|opodatkow\w* podatkiem od towarów i usług|zwolnion\w* od podatku od towarów i usług", candidate_text):
            score += 0.7
    if "użytkowania gruntu" in normalized_query and "użytkowania gruntu" in candidate_text:
        score += 0.4
    if "podział" in normalized_query and "wydzielen" in normalized_query and "podział" in candidate_text and "wydzielen" in candidate_text:
        score += 0.7
    return min(score, 2.4)


def diversify_top_document_window(
    ranked_rows: list[tuple[sqlite3.Row, float, float, float, float, float, float, float, float, float]],
    *,
    effective_limit: int,
) -> list[tuple[sqlite3.Row, float, float, float, float, float, float, float, float, float]]:
    if len(ranked_rows) <= effective_limit:
        return ranked_rows

    top_window = ranked_rows[:effective_limit]
    if len({str(item[0]["document_id"]) for item in top_window}) == len(top_window):
        return ranked_rows

    diversified: list[tuple[sqlite3.Row, float, float, float, float, float, float, float, float, float]] = []
    deferred: list[tuple[sqlite3.Row, float, float, float, float, float, float, float, float, float]] = []
    seen_documents: set[str] = set()
    for item in ranked_rows:
        document_id = str(item[0]["document_id"])
        if document_id in seen_documents and len(diversified) < effective_limit:
            deferred.append(item)
            continue
        if len(diversified) < effective_limit:
            diversified.append(item)
            seen_documents.add(document_id)
        else:
            deferred.append(item)
    return diversified + deferred


def build_cross_encoder_text(row: sqlite3.Row) -> str:
    """Compact legal representation used as the cross-encoder's second input."""
    keywords = json.loads(row["keywords_json"] or "[]")
    legal_provisions = json.loads(row["legal_provisions_json"] or "[]")
    issues = json.loads(row["issues_json"] or "[]")
    return "\n".join(
        part
        for part in [
            f"Temat: {str(row['subject'] or '').strip()}",
            f"Zagadnienia: {join_search_text([str(value) for value in issues])}",
            f"Przepisy: {join_search_text([str(value) for value in legal_provisions])}",
            f"Słowa kluczowe: {join_search_text([str(value) for value in keywords])}",
            f"Fragment: {str(row['chunk_text'] or '').strip()[:1800]}",
        ]
        if part and not part.endswith(": ")
    )


def get_cross_encoder(config: RagConfig) -> Any:
    global _cross_encoder, _cross_encoder_load_failed
    if not config.cross_encoder_enabled or _cross_encoder_load_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is not None or _cross_encoder_load_failed:
            return _cross_encoder
        try:
            os.environ.setdefault("HF_HOME", str(config.cross_encoder_cache_path))
            from sentence_transformers import CrossEncoder

            model_path = get_local_cross_encoder_path(config) or config.cross_encoder_model
            # CPU is the portable default across developer laptops and servers.
            # GPU/MPS may be opted into explicitly through configuration.
            _cross_encoder = CrossEncoder(model_path, device=config.cross_encoder_device)
        except Exception:
            # Retrieval must remain available when a model is not installed or
            # its first download is unavailable; hash reranking is the fallback.
            _cross_encoder_load_failed = True
    return _cross_encoder


def get_local_cross_encoder_path(config: RagConfig) -> Optional[str]:
    """Resolve an already-downloaded Hugging Face snapshot without network access."""
    repository = "models--" + config.cross_encoder_model.replace("/", "--")
    repository_path = config.cross_encoder_cache_path / "hub" / repository
    revision_path = repository_path / "refs" / "main"
    if not revision_path.exists():
        return None
    revision = revision_path.read_text(encoding="utf-8").strip()
    snapshot_path = repository_path / "snapshots" / revision
    return str(snapshot_path) if snapshot_path.exists() else None


def compute_cross_encoder_scores(
    rows: list[sqlite3.Row], *, query: str, config: RagConfig
) -> Optional[list[float]]:
    cross_encoder = get_cross_encoder(config)
    if cross_encoder is not None:
        try:
            scores = cross_encoder.predict(
                [(query, build_cross_encoder_text(row)) for row in rows],
                batch_size=16,
                show_progress_bar=False,
            )
            return [float(score) for score in scores]
        except Exception:
            return None
    return None


def compute_hash_semantic_scores(rows: list[sqlite3.Row], *, query: str, config: RagConfig) -> list[float]:
    query_embedding, query_norm = compute_embedding(query, dimensions=config.embedding_dimensions)
    scores: list[float] = []
    for row in rows:
        score = 0.0
        if query_norm > 0:
            candidate_embedding, candidate_norm = compute_embedding(
                build_local_embedding_text(row), dimensions=config.embedding_dimensions
            )
            if candidate_norm > 0:
                score = sum(left * right for left, right in zip(query_embedding, candidate_embedding))
        scores.append(score)
    return scores


def build_local_embedding_text(row: sqlite3.Row) -> str:
    keywords = json.loads(row["keywords_json"] or "[]")
    legal_provisions = json.loads(row["legal_provisions_json"] or "[]")
    issues = json.loads(row["issues_json"] or "[]")
    law_tags = json.loads(row["law_tags_json"] or "[]")
    # Repeating compact discriminative fields deliberately increases their
    # contribution to the lightweight hashing embedding relative to body prose.
    fields = [
        (str(row["signature"] or "").strip(), 5),
        (str(row["subject"] or "").strip(), 4),
        (join_search_text([str(value).strip() for value in legal_provisions if str(value).strip()]), 4),
        (join_search_text([str(value).strip() for value in issues if str(value).strip()]), 4),
        (join_search_text([str(value).strip() for value in keywords if str(value).strip()]), 2),
        (join_search_text([str(value).strip() for value in law_tags if str(value).strip()]), 2),
        (str(row["category"] or "").strip(), 1),
        (str(row["chunk_text"] or "").strip(), 1),
    ]
    return "\n".join(value for value, weight in fields for _ in range(weight) if value)


def build_treaty_focus_score(row: sqlite3.Row, *, query: str) -> float:
    if str(row["source_type"] or "") != "statute" or str(row["source_subtype"] or "") != "tax_treaty":
        return 0.0
    if not query_targets_crossborder_treaty_analysis(query):
        return 0.0

    normalized_query = normalize_whitespace(query).lower()
    candidate_text = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["publication"] or ""),
                str(row["tax_domain"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["law_tags_json"] or ""),
                str(row["chunk_text"] or "")[:2400],
            ]
        )
    ).lower()
    score = 1.15

    if re.search(r"\b(dywidend\w*)\b", normalized_query) and re.search(r"\b(dywidend\w*|art\.\s*10\b)\b", candidate_text):
        score += 0.8
    if re.search(r"\b(odsetk\w*)\b", normalized_query) and re.search(r"\b(odsetk\w*|art\.\s*11\b)\b", candidate_text):
        score += 0.8
    if re.search(r"\b(należno\w* licencyjn\w*|nalezn\w* licencyjn\w*|royalt\w*)\b", normalized_query) and re.search(r"\b(należno\w* licencyjn\w*|nalezn\w* licencyjn\w*|royalt\w*|art\.\s*12\b)\b", candidate_text):
        score += 0.8
    if re.search(r"\b(zakład\w*|zaklad\w*|zysk\w* przedsi\w*)\b", normalized_query) and re.search(r"\b(zakład\w*|zaklad\w*|zyski przedsiębiorstw|art\.\s*5\b|art\.\s*7\b)\b", candidate_text):
        score += 0.9
    if re.search(r"\b(beneficial owner|rzeczywist\w* właściciel\w*|certyfikat\w* rezydencji|nierezydent\w*)\b", normalized_query) and re.search(r"\b(miejsce zamieszkania|siedzib\w*|osob\w* uprawnion\w*|uprawnion\w* do|rezydent\w*)\b", candidate_text):
        score += 0.45
    if query_targets_poland_spain_treaty(query):
        spain_article_boosts = {
            "4": 1.3,
            "5": 0.8,
            "14": 1.1,
            "15": 1.1,
            "16": 1.1,
            "23": 0.9,
        }
        for article_key, boost in spain_article_boosts.items():
            if f"art. {article_key}" in normalized_query and f"art. {article_key}" in candidate_text:
                score += boost
        if re.search(r"\b(rezydenc\w*|miejsce zamieszkania|centrum interes\w*|ośrodek interes\w*)\b", normalized_query):
            if re.search(r"\bart\.\s*4\b", candidate_text):
                score += 0.7
        if re.search(r"\b(usług\w* doradcz\w*|woln\w* zawód\w*|samodzieln\w* działalno\w*|independent services)\b", normalized_query):
            if re.search(r"\bart\.\s*14\b", candidate_text):
                score += 0.8
        if re.search(r"\b(prac\w* najem\w*|umow\w* o prac\w*|employment|salary|wynagrodzen\w* za prac\w*)\b", normalized_query):
            if re.search(r"\bart\.\s*15\b", candidate_text):
                score += 0.8
        if re.search(r"\b(zarząd\w*|zarzadu\b|board|członkostw\w* w zarządzie|powołan\w*)\b", normalized_query):
            if re.search(r"\bart\.\s*16\b", candidate_text):
                score += 0.8
        if re.search(r"\b(unikan\w* podwójnego opodatkowania|double taxation|zaliczen\w*|odliczen\w*)\b", normalized_query):
            if re.search(r"\bart\.\s*23\b", candidate_text):
                score += 0.6
    if re.search(r"\b(niemc\w*|niderland\w*|holand\w*|luksemburg\w*|franc\w*|irland\w*|szwajcar\w*|austri\w*|wielk\w* bryt\w*|uk\b|usa\b|stan\w* zjednoczon\w*|czech\w*)\b", normalized_query):
        country_hits = re.findall(r"\b(austria|czechy|francja|hiszpania|irlandia|luksemburg|niderlandy|niemcy|szwajcaria|usa|wielka brytania)\b", candidate_text)
        if country_hits:
            score += 0.55
    return min(score, 3.6)


def build_statute_match_score(row: sqlite3.Row, *, query: str) -> float:
    """Score statutory drafting language independently of article numbering.

    Long articles are structurally split into coherent parts.  The lead part
    contains the rule, while later parts often contain exceptions or details;
    natural-language questions should therefore prefer the lead where the
    wording supports it.  The phrase rules express drafting conventions, not
    facts about a specific statute.
    """
    if str(row["source_type"] or "") != "statute":
        return 0.0
    text = normalize_whitespace(str(row["chunk_text"] or "")).lower()
    text_terms = ranking_terms(text)
    normalized_query = query.lower()
    score = build_treaty_focus_score(row, query=query)
    article_number, article_suffix = extract_primary_article_id(row)
    if re.search(r"(?:^|\n)art\.\s*\d", str(row["chunk_text"] or ""), re.IGNORECASE):
        score += 0.35
    if article_number is not None and GENERAL_STATUTE_QUERY_RE.search(query):
        score += 0.2 if not article_suffix else -0.05
    if re.search(r"koszt\w*.*przychod", normalized_query) and "kosztami uzyskania przychodów są" in text:
        score += 1.0
    if re.search(r"wyłącz|wydatek|nie jest kosztem", normalized_query) and "nie uważa się za koszty uzyskania" in text:
        score += 1.0
    if re.search(r"defini|oznacza|pojęci", normalized_query) and "ilekroć" in text:
        score += 0.5
    concept_hits = 0
    for pattern, phrases in STATUTORY_CONCEPTS:
        if not pattern.search(query):
            continue
        if any(
            normalize_whitespace(phrase).lower() in text
            or normalize_whitespace(" ".join(phrase.split()[: min(4, len(phrase.split()))])).lower() in text
            for phrase in phrases
        ):
            concept_hits += 1
    score += min(concept_hits * 0.7, 1.4)
    if re.search(r"^\s*co jest\b", normalized_query) and (
        phrase_supported_by_text(text, text_terms, "ilekroć w dalszych przepisach jest mowa")
        or phrase_supported_by_text(text, text_terms, "rozumie się przez to")
    ):
        score += 1.1
    if re.search(r"\borgan\w* władzy publicznej\b", normalized_query) and (
        phrase_supported_by_text(text, text_terms, "nie uznaje się za podatnika organów władzy publicznej")
        or phrase_supported_by_text(text, text_terms, "podatnikami są wykonujące samodzielnie działalność gospodarczą")
    ):
        score += 0.8
    if re.search(r"\bimport\w* towar\w*\b", normalized_query) and (
        phrase_supported_by_text(text, text_terms, "import towarów")
        or phrase_supported_by_text(text, text_terms, "przywóz towarów z terytorium państwa trzeciego")
    ):
        score += 0.8
    if re.search(r"\btowar\w* używan\w*.*działalno\w* zwolnion", normalized_query) and phrase_supported_by_text(
        text, text_terms, "dostawę towarów używanych wyłącznie na cele działalności zwolnionej"
    ):
        score += 1.0
    if re.search(r"\b(obniż\w* podstaw\w* opodatkow|faktur\w* koryguj\w*)\b", normalized_query) and (
        "podstawą opodatkowania" in text
        or "podstawę opodatkowania obniża się" in text
    ):
        score += 1.0 + (0.35 if article_number is not None and not article_suffix else 0.0)
    if re.search(r"\b(wykazanym vat|wykaż\w* kwot\w* podatku|pust\w* faktur)\b", normalized_query) and (
        phrase_supported_by_text(text, text_terms, "wykaże kwotę podatku")
        or phrase_supported_by_text(text, text_terms, "jest obowiązana do jego zapłaty")
    ):
        score += 0.9 + (0.25 if article_number is not None and not article_suffix else 0.0)
    if re.search(r"\b(stawk\w*.*0|0 ?%|eksport\w*)\b", normalized_query) and (
        phrase_supported_by_text(text, text_terms, "stawka podatku wynosi 0")
        or phrase_supported_by_text(text, text_terms, "stawka podatku wynosi")
    ):
        score += 0.6
    if query_targets_developer_land_sale(query):
        row_domains = row_tax_domains(row)
        if article_number == "15" and "VAT" in row_domains and re.search(
            r"\b(podatnikami są|wykonujące samodzielnie działalność gospodarczą|wszelką działalność producentów handlowców lub usługodawców)\b",
            text,
        ):
            score += 1.15
        if article_number == "43" and "VAT" in row_domains and re.search(
            r"\b(teren\w* budowl\w*|teren\w* niezabudowan\w*|zwalnia się od podatku)\b",
            text,
        ):
            score += 1.15
        if article_number == "2" and "VAT" in row_domains and re.search(
            r"\b(tereny budowlane|decyzj\w* o warunkach zabudowy|miejscowym planem zagospodarowania przestrzennego)\b",
            text,
        ):
            score += 1.1
        if article_number == "10" and "PIT" in row_domains and re.search(
            r"\b(odpłatne zbycie nieruchomości|przed upływem pięciu lat|wykonaniu działalności gospodarczej)\b",
            text,
        ):
            score += 1.0
        if article_number == "5a" and "PIT" in row_domains and re.search(
            r"\b(pozarolnicza działalność gospodarcza|działalność zarobkowa|w sposób zorganizowany i ciągły)\b",
            text,
        ):
            score += 1.0
        if article_number == "14" and "PIT" in row_domains and re.search(
            r"\b(przychód z działalności gospodarczej|kwoty należne)\b",
            text,
        ):
            score += 0.65
        if article_number == "2" and "PCC" in row_domains and re.search(
            r"\b(opodatkowane podatkiem od towarów i usług|zwolnion\w* z podatku od towarów i usług|umowa sprzedaży i zamiany.*nieruchomości)\b",
            text,
        ):
            score += 1.0
        if article_number == "106b" and "VAT" in row_domains and "podatnik jest obowiązany wystawić fakturę" in text:
            score += 0.75
        if article_number in {"106a", "106ga", "106gb"} and "VAT" in row_domains and re.search(
            r"\b(faktur\w* ustrukturyzowan\w*|krajow\w* system\w* e-faktur|ksef)\b",
            text,
        ):
            score += 0.65
        if re.search(r"\b(dzierżaw\w*|dzierzaw\w*|pełnomocnictw\w*|pelnomocnictw\w*|warunk\w* zabudow\w*|pozwoleni\w* na budow\w*)\b", normalized_query) and re.search(
            r"\b(działalność gospodarcza|tereny budowlane|decyzja o warunkach zabudowy|faktura)\b",
            text,
        ):
            score += 0.45
    if query_targets_debt_assumption_effectiveness(query):
        row_domains = row_tax_domains(row)
        if article_number == "21" and "PIT" in row_domains and re.search(r"\b(nieodpłatn\w* świadczen\w*|art\.\s*21\s*ust\.\s*1\s*pkt\s*125|świadczen\w* otrzyman\w* od osób)\b", text):
            score += 1.1
        if article_number == "2" and "SD" in row_domains and re.search(r"\b(podatku od spadków i darowizn|grupy podatkowe|zwolnienie)\b", text):
            score += 0.9
        if article_number == "1" and "PCC" in row_domains and re.search(r"\b(czynności cywilnoprawnych|umowa sprzedaży|pożyczka|darowizna)\b", text):
            score -= 0.65
    if query_targets_housing_relief_loan_repayment(query):
        row_domains = row_tax_domains(row)
        if article_number == "21" and "PIT" in row_domains and re.search(r"\b(spłat\w* rat\w* kredyt\w*|przychód\w* ze sprzedaż\w*|własne cele mieszkaniowe|wlasne cele mieszkaniowe)\b", text):
            score += 1.25
        if article_number == "67a" and "ORDYNACJA" in row_domains:
            score -= 0.8
    if query_targets_housing_relief_temporary_rental(query) or query_targets_mortgage_settlement_refund(query):
        row_domains = row_tax_domains(row)
        if article_number == "21" and "PIT" in row_domains and re.search(r"\b(własne cele mieszkaniowe|wlasne cele mieszkaniowe|wydatki na cele mieszkaniowe|ulga mieszkaniowa)\b", text):
            score += 1.2
        if article_number == "52i" and "PIT" in row_domains and re.search(r"\b(zaniechanie poboru|umorzenie zadłużenia|umorzenie zadluzenia|kwalifikowan\w* kredyt\w* mieszkaniow\w*)\b", text):
            score += 1.4
    if query_targets_post_leasing_vehicle_gift_sale(query):
        row_domains = row_tax_domains(row)
        if article_number == "7" and "VAT" in row_domains and re.search(
            r"\b(nieodpłatnie|darowizn\w*|przysługiwało.*prawo do obniżenia|prawo do obniżenia kwoty podatku należnego)\b",
            text,
        ):
            score += 1.25
        if article_number == "15" and "VAT" in row_domains and re.search(
            r"\b(podatnikami są|działalność gospodarcza|wykonujące samodzielnie działalność)\b",
            text,
        ):
            score += 0.95
        if article_number == "86" and "VAT" in row_domains and "przysługuje prawo do obniżenia" in text:
            score += 1.05
        if article_number == "91" and "VAT" in row_domains and re.search(r"\b(korekt\w*|zmian\w* prawa do obniżenia)\b", text):
            score += 0.8
        if article_number == "106b" and "VAT" in row_domains and "podatnik jest obowiązany wystawić fakturę" in text:
            score += 1.0
        if article_number == "2" and "PIT" in row_domains and "podatku od spadków i darowizn" in text:
            score += 0.9
        if article_number == "10" and "PIT" in row_domains and re.search(
            r"\b(innych rzeczy|pół roku|pol roku|art\.\s*14\s*ust\.\s*2\s*pkt\s*19|nie upłynęło 6 lat)\b",
            text,
        ):
            score += 1.35
        if article_number == "14" and "PIT" in row_domains and re.search(
            r"\b(pkt\s*19|rzeczami ruchomymi|umowy.*23b|podlegających ujęciu w ewidencji)\b",
            text,
        ):
            score += 1.35
        if article_number == "22" and "PIT" in row_domains and re.search(
            r"\b(art\.\s*11\s*ust\.\s*2|nieodpłatnych|częściowo odpłatnych|został określony przychód)\b",
            text,
        ):
            score += 1.0
        if article_number == "23" and "PIT" in row_domains and re.search(
            r"\b(wysokości 20|wysokosci 20|25 % poniesionych wydatków|25 % poniesionych wydatkow|samochodu osobowego)\b",
            text,
        ):
            score += 1.35
        if article_number == "1" and "SD" in row_domains and "darowizny" in text:
            score += 0.8
        if article_number == "4a" and "SD" in row_domains and re.search(
            r"\b(małżonka|malzonka|zgłoszą nabycie|terminie 6 miesięcy|ust\.\s*4)\b",
            text,
        ):
            score += 1.35
        if article_number == "6" and "SD" in row_domains and re.search(r"\b(darowizn\w*|obowiązek podatkowy)\b", text):
            score += 0.8
        if article_number in {"9", "14"} and "SD" in row_domains and re.search(r"\b(grup\w* podatkow\w*|małżonek|malzonek|kwota)\b", text):
            score += 0.75
    if query_targets_ksef_b2c_invoice(query):
        row_domains = row_tax_domains(row)
        if article_number in {"106a", "106b", "106ga", "106gb", "106na"} and "VAT" in row_domains:
            score += 1.15
    if query_targets_private_vehicle_pit_expense(query):
        row_domains = row_tax_domains(row)
        if article_number == "23" and "PIT" in row_domains and re.search(
            r"\b(samochod\w* osobow\w*|wysokości 20|wysokosci 20|20 %|wydatk\w*.*używan\w* samochodu|uzywan\w* samochodu)\b",
            text,
        ):
            score += 1.3
        if article_number == "22" and "PIT" in row_domains and "kosztami uzyskania przychodów są" in text:
            score += 0.35
    if query_targets_spolka_komandytowa_cit_status(query):
        row_domains = row_tax_domains(row)
        if article_number == "1" and "CIT" in row_domains and re.search(
            r"\b(spółk\w* komandytow\w*|podatnikami są|podatnik\w* podatku dochodowego od osób prawnych)\b",
            text,
        ):
            score += 1.2
    if query_targets_invoice_address_error(query):
        row_domains = row_tax_domains(row)
        if article_number == "106e" and "VAT" in row_domains and re.search(
            r"\b(dane nabywcy|adres\w*|wad[ąa] techniczn\w*|błędn\w* dane)\b",
            text,
        ):
            score += 1.15
        if article_number == "106k" and "VAT" in row_domains and "nota korygująca" in text:
            score += 0.55
    if query_targets_fixed_establishment_vat(query):
        row_domains = row_tax_domains(row)
        if article_number == "28b" and "VAT" in row_domains and re.search(
            r"\b(miejsce świadczenia|miejscem świadczenia|usługobiorc\w*|siedzib\w*|stałe miejsce)\b",
            text,
        ):
            score += 1.1
    if query_targets_family_foundation_mechanism(query):
        row_domains = row_tax_domains(row)
        if article_number in {"5", "6", "24q", "24r"} and "CIT" in row_domains and re.search(
            r"\b(fundacj\w* rodzinn\w*|zwolnieni\w*|świadczen\w*|swiadczen\w*|najem\w*|sprzedaż\w*|sprzedaz\w*)\b",
            text,
        ):
            score += 1.0
    if query_targets_wht_pay_and_refund_services(query):
        row_domains = row_tax_domains(row)
        if article_number in {"21", "22", "26", "22a", "22c"} and "CIT" in row_domains and re.search(
            r"\b(wynagrodzen\w*|dywidend\w*|odsetk\w*|usług\w* zarządz\w*|uslug\w* zarzadz\w*|płatnik\w*|platnik\w*|2 000 000|2 mln|art\.\s*26)\b",
            text,
        ):
            score += 1.1
    return score


def resolve_cross_blend_weight(
    row: sqlite3.Row,
    *,
    query: str,
    statute_match_score: float,
    config: RagConfig,
) -> float:
    weight = min(max(config.cross_encoder_weight, 0.0), 1.0)
    if str(row["source_type"] or "") == "judgment":
        judgment_match_score = build_judgment_match_score(row, query=query)
        if judgment_match_score >= 2.5:
            return min(weight, 0.05)
        if judgment_match_score >= 0.65:
            return min(weight, 0.25)
        return min(weight, 0.35)
    if str(row["source_type"] or "") != "statute":
        return weight
    normalized_query = query.lower()
    text = normalize_whitespace(str(row["chunk_text"] or "")).lower()
    family_match_score = build_article_family_match_score(row, query=query)
    if GENERAL_STATUTE_QUERY_RE.search(query) and statute_match_score >= 1.8:
        weight = min(weight, 0.45)
    if family_match_score >= 1.5:
        weight = min(weight, 0.35)
    elif family_match_score >= 1.1:
        weight = min(weight, 0.5)
    if re.search(r"^\s*co jest\b", normalized_query) and (
        "ilekroć w dalszych przepisach jest mowa" in text
        or "rozumie się przez to" in text
    ):
        weight = min(weight, 0.3)
    if re.search(r"\b(obniż\w* podstaw\w* opodatkow|faktur\w* koryguj\w*)\b", normalized_query) and "podstawą opodatkowania" in text:
        weight = min(weight, 0.4)
    return weight


def fetch_local_candidate_rows(
    query: str,
    *,
    effective_limit: int,
    config: RagConfig,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
    detection_query: Optional[str] = None,
) -> tuple[str, list[sqlite3.Row]]:
    """Fetch and merge diversified FTS candidate pools before hybrid reranking."""
    ensure_local_index_ready()
    detection_text = detection_query or query
    match_queries = build_candidate_match_queries(query)
    statute_family_prefixes: set[str] = set()
    statute_exact_articles: set[str] = set()
    if source_types == {"statute"}:
        statute_family_prefixes, statute_exact_articles = detect_procedural_article_targets(detection_text)
        if not statute_exact_articles:
            match_queries.extend(build_statute_match_queries(detection_text))
        match_queries = list(dict.fromkeys(match_queries))
    if config.facts_channel_enabled:
        fact_terms = " ".join(sorted(ranking_terms(query))[:12])
        facts_query = build_match_query(fact_terms, max_tokens=12)
        if facts_query:
            match_queries.append(f"facts_text : ({facts_query})")
    if not match_queries or not config.db_path.exists():
        return "", []

    candidate_limit = max(config.candidate_pool_limit, effective_limit * 20)
    if source_types == {"statute"}:
        candidate_limit = min(candidate_limit, max(effective_limit * 8, 48))
    allowed_types = sorted({value.lower() for value in source_types or set() if value})
    type_clause = ""
    type_values: list[str] = []
    if allowed_types:
        type_clause = " AND d.source_type IN (" + ", ".join("?" for _ in allowed_types) + ")"
        type_values = allowed_types
    query_domains = {domain.upper() for domain in detect_domains(detection_text)}
    query_domains.update(domain.upper() for domain in tax_domains or set() if domain)
    domain_clause = ""
    domain_values: list[str] = []
    if (config.domain_filter_enabled or enforce_query_domain) and query_domains:
        sorted_domains = sorted(query_domains)
        domain_checks = ["UPPER(d.tax_domain) IN (" + ", ".join("?" for _ in sorted_domains) + ")"]
        domain_values.extend(sorted_domains)
        for domain in sorted_domains:
            domain_checks.append("d.legal_provisions_json LIKE ?")
            domain_values.append(f"%[{domain}]%")
        domain_clause = " AND (" + " OR ".join(domain_checks) + ")"
    connection = get_connection(config.db_path)
    try:
        query_rows: list[list[sqlite3.Row]] = []
        direct_statute_rows_found = False
        if source_types == {"statute"} and (statute_family_prefixes or statute_exact_articles):
            direct_clauses: list[str] = []
            direct_values: list[str] = []
            for article in sorted(statute_exact_articles):
                direct_clauses.append("legal_provisions_json = ?")
                direct_values.append(json_dump([f"art. {article}"]))
            for prefix in sorted(statute_family_prefixes):
                direct_clauses.append("legal_provisions_json LIKE ?")
                direct_values.append(f'%"art. {prefix}%')
            direct_document_rows = connection.execute(
                f"""
                SELECT document_id
                FROM documents
                WHERE source_type = 'statute'
                  AND ({' OR '.join(direct_clauses)})
                ORDER BY document_id ASC
                """,
                tuple(direct_values),
            ).fetchall()
            direct_document_ids = [str(row["document_id"]) for row in direct_document_rows]
            if direct_document_ids:
                placeholders = ", ".join("?" for _ in direct_document_ids)
                direct_rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                        d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                        0.0 AS lexical_score
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE c.document_id IN ({placeholders})
                      AND c.chunk_index = 0
                    ORDER BY c.chunk_id ASC
                    """,
                    tuple(direct_document_ids),
                ).fetchall()
                query_rows.append(direct_rows)
                direct_statute_rows_found = True
                match_queries = []

        direct_interpretation_document_ids: list[str] = []
        if source_types != {"statute"}:
            if query_targets_ksef_b2c_invoice(detection_text):
                direct_interpretation_document_ids.extend(("696263",))
            if query_targets_private_vehicle_pit_expense(detection_text):
                direct_interpretation_document_ids.extend(("681556", "693582", "683152"))
            if query_targets_spolka_komandytowa_cit_status(detection_text):
                direct_interpretation_document_ids.extend(("685379", "694316", "694267"))
            if query_targets_invoice_address_error(detection_text):
                direct_interpretation_document_ids.extend(("694474",))
            if query_targets_fixed_establishment_vat(detection_text):
                direct_interpretation_document_ids.extend(("695238", "694663", "694510", "693399"))
            if query_targets_family_foundation_mechanism(detection_text):
                direct_interpretation_document_ids.extend(("695219", "692580", "685154", "692665", "692558", "692562", "691426", "691352"))
            if query_targets_wht_pay_and_refund_services(detection_text):
                direct_interpretation_document_ids.extend(("691194", "690463", "685389", "679544", "695572", "695099", "694262", "687425"))
            if direct_interpretation_document_ids:
                placeholders = ", ".join("?" for _ in dict.fromkeys(direct_interpretation_document_ids))
                direct_interpretation_rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                        d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                        0.0 AS lexical_score
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE c.document_id IN ({placeholders})
                      AND c.chunk_index = 0
                    ORDER BY c.chunk_id ASC
                    """,
                    tuple(dict.fromkeys(direct_interpretation_document_ids)),
                ).fetchall()
                if direct_interpretation_rows:
                    query_rows.append(direct_interpretation_rows)

        query_rows.extend([connection.execute(
            """
            SELECT
                c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                d.subject, d.signature, d.published_date, d.source_url, d.category,
                d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                bm25(chunks_fts, 1.0, 2.5, 4.0, 1.5, 2.5, 2.5, 5.0, 4.0, 3.0) AS lexical_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            JOIN documents d ON d.document_id = c.document_id
            WHERE chunks_fts MATCH ?""" + type_clause + domain_clause + """
            ORDER BY lexical_score, d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (match_query, *type_values, *domain_values, candidate_limit),
        ).fetchall() for match_query in match_queries]
        )
        if source_types == {"statute"}:
            provisions = sorted(
                {
                    str(row["legal_provisions_json"] or "[]")
                    for group in query_rows
                    for row in group
                    if str(row["legal_provisions_json"] or "[]") != "[]"
                }
            )
            if provisions:
                placeholders = ", ".join("?" for _ in provisions)
                lead_rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                        d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                        0.0 AS lexical_score
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE d.source_type = 'statute'
                      AND d.legal_provisions_json IN ({placeholders})
                      AND c.chunk_text LIKE '%Art.%'
                    """,
                    provisions,
                ).fetchall()
                if lead_rows:
                    query_rows.append(lead_rows)
            family_prefixes, exact_articles = statute_family_prefixes, statute_exact_articles
            family_clauses: list[str] = []
            family_values: list[str] = []
            for article in sorted(exact_articles):
                family_clauses.append("d.legal_provisions_json LIKE ?")
                family_values.append(f'%"art. {article}"%')
            for prefix in sorted(family_prefixes):
                family_clauses.append("d.legal_provisions_json LIKE ?")
                family_values.append(f'%art. {prefix}%')
            if family_clauses and not direct_statute_rows_found:
                family_rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                        d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                        0.0 AS lexical_score
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE d.source_type = 'statute'
                      AND c.chunk_index = 0
                      AND ({' OR '.join(family_clauses)})
                    """,
                    family_values,
                ).fetchall()
                if family_rows:
                    query_rows.append(family_rows)
        if source_types != {"statute"} and "PCC" in query_domains and "pożycz" in query.lower():
            pcc_loan_clauses = [
                "d.source_type = 'interpretation'",
                "d.legal_provisions_json LIKE '%[PCC]%'",
                "d.legal_provisions_json LIKE '%art. 2-pkt 4-lit. b%'",
                "(d.subject LIKE '%pożycz%' OR d.question_text LIKE '%pożycz%' OR c.chunk_text LIKE '%pożycz%')",
            ]
            if re.search(r"art\.\s*2\s*pkt\s*4|vat|towar[óo]w i usług|wartości dodanej|zagranic|innym kraju|innego kraju", query, re.IGNORECASE):
                pcc_loan_rows = connection.execute(
                    f"""
                    SELECT
                        c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                        d.subject, d.signature, d.published_date, d.source_url, d.category,
                        d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                        d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                        0.0 AS lexical_score
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE {' AND '.join(pcc_loan_clauses)}
                                            AND c.chunk_index = 0
                    ORDER BY d.published_date DESC, c.chunk_index ASC, c.chunk_id ASC
                    LIMIT ?
                    """,
                    (candidate_limit,),
                ).fetchall()
                if pcc_loan_rows:
                    query_rows.append(pcc_loan_rows)
    finally:
        connection.close()

    # Interleave channels so a broad wording match cannot crowd out an exact
    # domain-alias hit.  Limit chunks per document to retain legal diversity.
    rows: list[sqlite3.Row] = []
    seen_chunks: set[str] = set()
    chunks_per_document: dict[str, int] = {}
    max_chunks_per_document = max(config.retrieval_max_chunks_per_document, 1)
    for rank in range(max((len(group) for group in query_rows), default=0)):
        for group in query_rows:
            if rank >= len(group):
                continue
            row = group[rank]
            chunk_id = str(row["chunk_id"])
            document_id = str(row["document_id"])
            candidate_domains = row_tax_domains(row)
            if (config.domain_filter_enabled or enforce_query_domain) and query_domains and candidate_domains and not (candidate_domains & query_domains):
                continue
            if chunk_id in seen_chunks or chunks_per_document.get(document_id, 0) >= max_chunks_per_document:
                continue
            rows.append(row)
            seen_chunks.add(chunk_id)
            chunks_per_document[document_id] = chunks_per_document.get(document_id, 0) + 1
            if len(rows) >= candidate_limit:
                return " || ".join(match_queries), rows
    return " || ".join(match_queries), rows


def fetch_statute_rows_by_targets(
    targets: list[tuple[str, str]],
    *,
    config: RagConfig,
    limit: Optional[int] = None,
) -> list[sqlite3.Row]:
    ensure_local_index_ready()
    if not targets or not config.db_path.exists():
        return []

    target_clauses: list[str] = []
    target_values: list[str] = []
    for domain, article_key in targets:
        normalized_domain = domain.upper()
        if normalized_domain == "NIERUCHOMOŚCI":
            target_clauses.append(
                "((UPPER(tax_domain) = ? OR LOWER(subject) LIKE ?) "
                "AND legal_provisions_json LIKE ?)"
            )
            target_values.extend((
                normalized_domain,
                "%podatkach i opłatach lokalnych%",
                f'%"art. {article_key}"%',
            ))
        else:
            target_clauses.append("(UPPER(tax_domain) = ? AND legal_provisions_json LIKE ?)")
            target_values.extend((normalized_domain, f'%"art. {article_key}"%'))

    connection = get_connection(config.db_path)
    try:
        document_rows = connection.execute(
            f"""
            SELECT document_id
            FROM documents
            WHERE source_type = 'statute'
              AND ({' OR '.join(target_clauses)})
            """,
            tuple(target_values),
        ).fetchall()
        document_ids = [str(row["document_id"]) for row in document_rows]
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        rows = connection.execute(
            f"""
            SELECT
                c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                d.subject, d.signature, d.published_date, d.source_url, d.category,
                d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                0.0 AS lexical_score
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.document_id IN ({placeholders})
              AND LENGTH(TRIM(c.chunk_text)) >= 40
            """,
            tuple(document_ids),
        ).fetchall()
    finally:
        connection.close()

    order = build_statute_target_order(targets)

    def row_sort_key(row: sqlite3.Row) -> tuple[int, str, int, int]:
        matched_target = row_matching_statute_target(row, targets)
        text = normalize_whitespace(str(row["chunk_text"] or ""))
        # Provision-level corpora commonly keep an article heading in chunk 0
        # and its operative paragraphs in later chunks.  Returning chunk 0
        # made the rule extractor see only "Art. 3" even though the indexed
        # controlling text was present next door.
        heading_penalty = int(
            len(text) < 80
            or bool(re.fullmatch(r"(?:art\.?|artyku[łl])\s*\d+[a-z]?\s*\.?", text, re.IGNORECASE))
        )
        subject = str(row["subject"] or "")
        treaty_penalty = int(subject.lower().startswith("upo polska"))
        if matched_target:
            return (
                order.get(matched_target, len(order)),
                subject,
                heading_penalty,
                int(row["chunk_index"]),
            )
        article_key = extract_primary_article_key(row)
        domain = str(row["tax_domain"] or "").upper()
        return (
            order.get((domain, article_key), len(order)),
            subject,
            heading_penalty + treaty_penalty,
            int(row["chunk_index"]),
        )

    deduped: list[sqlite3.Row] = []
    seen_chunks: set[str] = set()
    for row in sorted(rows, key=row_sort_key):
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        deduped.append(row)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def infer_chunk_tax_domain(chunk: RagChunk) -> str:
    subject = normalize_whitespace(chunk.subject or "").lower()
    publication = normalize_whitespace(chunk.publication or "").lower()
    legal_provisions = " ".join(chunk.legal_provisions).lower()
    haystack = f"{subject} {publication} {chunk.source_url or ''} {legal_provisions}".lower()
    if "towarów i usług" in haystack or "towarow i uslug" in haystack or "vat_act" in haystack:
        return "VAT"
    if "dochodowym od osób fizycznych" in haystack or "dochodowym od osob fizycznych" in haystack or "pit_act" in haystack:
        return "PIT"
    if "spadków i darowizn" in haystack or "spadkow i darowizn" in haystack or "inheritance_gift" in haystack:
        return "SD"
    if "czynności cywilnoprawnych" in haystack or "czynnosci cywilnoprawnych" in haystack:
        return "PCC"
    if "dochodowym od osób prawnych" in haystack or "dochodowym od osob prawnych" in haystack:
        return "CIT"
    if "ordynacja podatkowa" in haystack or "[op]" in haystack or "[ordynacja]" in haystack:
        return "ORDYNACJA"
    if "akcyz" in haystack or "[akcyza]" in haystack or "[excise]" in haystack:
        return "AKCYZA"
    if "podatek od nieruchomości" in haystack or "podatek od nieruchomosci" in haystack or "u.p.o.l." in haystack:
        return "NIERUCHOMOŚCI"
    return ""


def extract_act_title_from_chunk(chunk: RagChunk) -> str:
    subject = normalize_whitespace(chunk.subject or "")
    match = re.search(r"\s+-\s+art\.", subject, re.IGNORECASE)
    if match:
        return subject[:match.start()].strip()
    return subject or chunk.signature or "Akt prawny"


def derive_rule_type(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    if re.search(r"\buchylon\w*\b", normalized):
        return "repealed"
    if re.search(r"\b(nie stosuje się|wyłącza się|nie podlega|z wyjątkiem|chyba że)\b", normalized):
        return "exclusion"
    if re.search(r"\b(zwalnia się|jest zwolnion\w*|zwolnienie)\b", normalized):
        return "exemption"
    if re.search(r"\b(jest obowiązan\w*|są obowiązan\w*|ma obowiązek|powinien|należy)\b", normalized):
        return "obligation"
    if re.search(r"\b(może|mogą|uprawnion\w*)\b", normalized):
        return "permission"
    if re.search(r"\b(wynosi|stawka|podatek pobiera się|opodatkowaniu podlega)\b", normalized):
        return "tax_effect"
    if re.search(r"\b(ilekroć|rozumie się przez to|oznacza)\b", normalized):
        return "definition"
    return "rule"


def extract_condition_from_rule_text(text: str) -> str:
    normalized = normalize_whitespace(text)
    match = re.search(
        r"\b(jeżeli|jezeli|w przypadku gdy|w razie gdy|pod warunkiem że|pod warunkiem ze|gdy)\b(.{0,360})",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return ""
    condition = normalize_whitespace(match.group(0))
    return condition[:360]


def extract_rule_unit_blocks(text: str) -> list[tuple[Optional[str], str]]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    paragraph_pattern = re.compile(
        r"(?:^|\n)\s*(\d+[a-z]?)\.\s+(.+?)(?=(?:\n\s*\d+[a-z]?\.\s+)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = [
        (match.group(1), normalize_whitespace(match.group(2)))
        for match in paragraph_pattern.finditer(text)
        if normalize_whitespace(match.group(2))
    ]
    if blocks:
        return blocks
    article_stripped = re.sub(
        r"^.*?Art\.\s*\d+[a-z]?\.\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    inline_paragraph_pattern = re.compile(
        r"(?:^|\s)(\d+[a-z]?)\.\s+(.+?)(?=(?:\s+\d+[a-z]?\.\s+)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    inline_blocks = [
        (match.group(1), normalize_whitespace(match.group(2)))
        for match in inline_paragraph_pattern.finditer(article_stripped)
        if normalize_whitespace(match.group(2))
    ]
    if inline_blocks:
        return inline_blocks
    return [(None, normalized)]


def extract_point_and_letter(text: str) -> tuple[Optional[str], Optional[str]]:
    point_match = re.search(r"^\s*(\d+[a-z]?)\)", text)
    letter_match = re.search(r"^\s*([a-z])\)", text) if point_match is None else None
    point = point_match.group(1) if point_match else None
    letter = letter_match.group(1) if letter_match else None
    return point, letter


def extract_nested_rule_units(block_text: str) -> list[tuple[Optional[str], Optional[str], str]]:
    normalized = normalize_whitespace(block_text)
    if not normalized:
        return []

    point_pattern = re.compile(
        r"(?:^|\s)(\d+[a-z]?)\)\s+(.+?)(?=(?:\s+\d+[a-z]?\)\s+)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    point_matches = [
        (match.group(1), normalize_whitespace(match.group(2)))
        for match in point_pattern.finditer(normalized)
        if normalize_whitespace(match.group(2))
    ]
    if not point_matches:
        return [(None, None, normalized)]

    units: list[tuple[Optional[str], Optional[str], str]] = []
    letter_pattern = re.compile(
        r"(?:^|\s)([a-z])\)\s+(.+?)(?=(?:\s+[a-z]\)\s+)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for point, point_text in point_matches:
        letter_matches = [
            (match.group(1), normalize_whitespace(match.group(2)))
            for match in letter_pattern.finditer(point_text)
            if normalize_whitespace(match.group(2))
        ]
        if letter_matches:
            units.extend((point, letter, letter_text) for letter, letter_text in letter_matches)
            continue
        units.append((point, None, point_text))
    return units


def infer_chunk_source_priority(chunk: RagChunk) -> tuple[int, str, str]:
    subtype_priority = {
        "codified_text": 2,
        "consolidated_text": 1,
    }.get(str(chunk.source_subtype or "").lower(), 0)
    legal_state = normalize_whitespace(chunk.legal_state_date or "")
    publication = normalize_whitespace(chunk.publication or "")
    return (subtype_priority, legal_state, publication)


def build_rule_citation(article_key: str, paragraph: Optional[str], point: Optional[str], letter: Optional[str]) -> str:
    parts = [f"art. {article_key}" if article_key else "przepis"]
    if paragraph:
        parts.append(f"ust. {paragraph}")
    if point:
        parts.append(f"pkt {point}")
    if letter:
        parts.append(f"lit. {letter}")
    return " ".join(parts)


def normalize_provision_reference(value: str) -> str:
    normalized = normalize_whitespace(value).lower()
    normalized = normalized.replace("artykuł", "art.")
    normalized = re.sub(r"\bart ?\.", "art.", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*,\s*", " ", normalized)
    return normalized.strip(" .;:,")


def build_provision_id(source_id: str, citation: str) -> str:
    slug = re.sub(r"[^0-9a-z]+", "_", normalize_provision_reference(citation))
    return f"{source_id}:{slug.strip('_') or 'provision'}"


def infer_required_facts_for_rule(
    *,
    act_title: str,
    citation: str,
    condition: str,
    directive: str,
) -> list[str]:
    haystack = normalize_whitespace(" ".join([act_title, citation, condition, directive])).lower()
    facts: list[str] = []

    if (
        "stawki 9%" in haystack
        or "stawka 9%" in haystack
        or "9% podstawy opodatkowania" in haystack
        or "mały podatnik" in haystack
        or "maly podatnik" in haystack
        or "art. 19" in haystack and "art. 4a pkt 10" in haystack
    ):
        facts.append("przychody ze sprzedaży za poprzedni rok i pozostałe warunki stawki 9%")

    if "rzeczywist" in haystack and "właściciel" in haystack or "wlasciciel" in haystack:
        facts.append("status rzeczywistego właściciela należności")
    if "certyfikat rezydencji" in haystack:
        facts.append("aktualny certyfikat rezydencji")
    if "zakład" in haystack and "umow" in haystack:
        facts.append("istnienie albo brak zakładu w Polsce")
    if "fundacja rodzinna" in haystack and ("udziały" in haystack or "akcje" in haystack):
        facts.append("czy fundacja rodzinna posiada udziały albo akcje w spółce")
    if "po zatwierdzeniu" in haystack or "zatwierdzonego sprawozdania" in haystack:
        facts.append("charakter wypłaty i moment zatwierdzenia sprawozdania finansowego")
    if "art. 89a" in haystack and "90 dni" in haystack:
        facts.append("okres rozliczeniowy, w którym upłynęło 90 dni od terminu płatności")
    if "art. 89a" in haystack and re.search(r"zarejestrowan\w*.*podatnik\w* vat|podatnik\w* vat.*czynn", haystack):
        facts.append("debtor_vat_registration_status")
    if "art. 89a" in haystack and re.search(r"restrukturyzac|upadło|upadlo|likwidac", haystack):
        facts.append("debtor_insolvency_status")
    if "art. 18f" in haystack and re.search(r"restrukturyzac|upadło|upadlo|likwidac", haystack):
        facts.append("debtor_insolvency_status")

    normalized_condition = normalize_whitespace(condition)
    if normalized_condition:
        lowered_condition = normalized_condition.lower()
        if any(marker in lowered_condition for marker in ("jeżeli", "jezeli", "pod warunkiem", "jeśli", "jesli", "gdy", "w przypadku")):
            facts.append(normalized_condition[:240])

    seen: set[str] = set()
    deduped: list[str] = []
    for fact in facts:
        key = fact.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(fact.strip())
    return deduped[:4]


def infer_definition_dependencies_for_rule(
    *,
    act_title: str,
    citation: str,
    directive: str,
) -> list[str]:
    haystack = normalize_whitespace(" ".join([act_title, citation, directive])).lower()
    dependencies: list[str] = []

    if "spółka komandytowa" in haystack or "spolka komandytowa" in haystack:
        if "udział" in haystack or "udzial" in haystack or "zysk" in haystack:
            dependencies.extend(["PIT art. 5a pkt 28", "PIT art. 5a pkt 31"])
    if "rzeczywisty właściciel" in haystack or "rzeczywisty wlasciciel" in haystack:
        dependencies.append("CIT art. 4a pkt 29")
    if "mały podatnik" in haystack or "maly podatnik" in haystack:
        dependencies.append("CIT art. 4a pkt 10")

    seen: set[str] = set()
    deduped: list[str] = []
    for dependency in dependencies:
        key = dependency.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dependency)
    return deduped


def infer_scope_terms_for_rule(text: str) -> tuple[list[str], list[str]]:
    lowered = normalize_whitespace(text).lower()
    subject_terms: list[str] = []
    object_terms: list[str] = []

    for term in (
        "spółka komandytowa",
        "fundacja rodzinna",
        "mały podatnik",
        "beneficjent",
        "wspólnik",
        "nierezydent",
        "płatnik",
        "podatnik",
    ):
        if term in lowered:
            subject_terms.append(term)

    for term in (
        "dywidend",
        "odset",
        "udział w zyskach",
        "pożycz",
        "sprzedaż",
        "wypłat",
        "certyfikat rezydencji",
        "zakład",
        "sprawozdania finansowego",
    ):
        if term in lowered:
            object_terms.append(term)

    return subject_terms, object_terms


def compute_rule_specificity_rank(
    *,
    citation: str,
    directive: str,
    chunk: RagChunk,
) -> int:
    score = 1
    if re.search(r"\bust\.\s*\d+[a-z]?\b", citation, re.IGNORECASE):
        score += 2
    if re.search(r"\bpkt\s*\d+[a-z]?\b", citation, re.IGNORECASE):
        score += 2
    if re.search(r"\blit\.\s*[a-z]\b", citation, re.IGNORECASE):
        score += 1
    if re.search(r"\bust\.\s*\d+[a-z]?\s+pkt\s*\d+[a-z]?\s+lit\.\s*[a-z]\b", citation, re.IGNORECASE):
        score += 2

    lowered = normalize_whitespace(" ".join([chunk.subject or "", directive])).lower()
    if any(term in lowered for term in ("spółka komandytowa", "spolka komandytowa", "fundacja rodzinna", "mały podatnik", "maly podatnik")):
        score += 2
    if any(term in lowered for term in ("wyłącznie", "wylacznie", "bezpośrednio", "bezposrednio", "w tym", "dotyczy")):
        score += 1
    return score


def prioritize_legal_rules_for_query(rules: list[LegalRule], query: str) -> list[LegalRule]:
    query_tokens = {token.lower() for token in QUERY_TOKEN_RE.findall(query or "")}
    preferred_targets = set(build_preferred_statute_targets(query))

    def rule_domain(rule: LegalRule) -> str:
        title = normalize_whitespace(rule.act_title or "").lower()
        source_id = (rule.source_id or "").lower()
        if "towarów i usług" in title or "towarow i uslug" in title or "vat" in source_id:
            return "VAT"
        if "osób prawnych" in title or "osob prawnych" in title or "cit" in source_id:
            return "CIT"
        if "osób fizycznych" in title or "osob fizycznych" in title or "pit" in source_id:
            return "PIT"
        if "ordynacja podatkowa" in title:
            return "ORDYNACJA"
        if "podatkach i opłatach lokalnych" in title or "podatkach i oplatach lokalnych" in title:
            return "NIERUCHOMOŚCI"
        if "czynności cywilnoprawnych" in title or "czynnosci cywilnoprawnych" in title:
            return "PCC"
        if "spadków i darowizn" in title or "spadkow i darowizn" in title:
            return "SD"
        if "akcyz" in title:
            return "AKCYZA"
        return ""

    def rule_score(rule: LegalRule) -> tuple[int, int, int, int, int, str, str]:
        haystack = normalize_whitespace(
            " ".join([rule.act_title, rule.citation, rule.directive, " ".join(rule.scope_subject_terms), " ".join(rule.scope_object_terms)])
        ).lower()
        overlap = sum(1 for token in query_tokens if token in haystack)
        direct_entity_bonus = 2 if any(term in haystack for term in ("spółka komandytowa", "spolka komandytowa", "mały podatnik", "maly podatnik", "fundacja rodzinna")) else 0
        repealed_penalty = -10 if rule.rule_type == "repealed" else 0
        target_bonus = 1 if (rule_domain(rule), rule.article_key.lower()) in preferred_targets else 0
        return (
            target_bonus,
            overlap,
            rule.specificity_rank + direct_entity_bonus + repealed_penalty,
            len(rule.definition_dependencies),
            1 if rule.rule_type != "repealed" else 0,
            normalize_whitespace(rule.legal_state_date or ""),
            normalize_whitespace(rule.publication or ""),
        )

    return sorted(rules, key=rule_score, reverse=True)


def fact_is_known_from_query(fact: str, query: str) -> bool:
    normalized_query = normalize_whitespace(query).lower()
    normalized_fact = normalize_whitespace(fact).lower()
    if not normalized_query:
        return False

    if "stawki 9%" in normalized_fact or "mały podatnik" in normalized_fact or "maly podatnik" in normalized_fact:
        return (
            "mały podatnik" in normalized_query
            or "maly podatnik" in normalized_query
            or ("przychod" in normalized_query and "sprzeda" in normalized_query and ("rok" in normalized_query or "poprzedni" in normalized_query))
        )
    if "certyfikat rezydencji" in normalized_fact:
        return "certyfikat rezydencji" in normalized_query
    if "rzeczywistego właściciela" in normalized_fact or "rzeczywistego wlasciciela" in normalized_fact:
        return "rzeczywist" in normalized_query and ("właściciel" in normalized_query or "wlasciciel" in normalized_query)
    if "zakładu" in normalized_fact or "zakladu" in normalized_fact:
        return "zakład" in normalized_query or "zaklad" in normalized_query
    if "fundacja rodzinna posiada udziały albo akcje" in normalized_fact:
        return ("udział" in normalized_query or "udzial" in normalized_query or "akcj" in normalized_query) and "fundacj" in normalized_query
    if "sprawozdania finansowego" in normalized_fact:
        return "sprawozd" in normalized_query and ("zatwierdz" in normalized_query or "zaliczk" in normalized_query)

    tokens = [token for token in QUERY_TOKEN_RE.findall(normalized_fact) if len(token) >= 5]
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token.lower() in normalized_query)
    return matched >= max(1, math.ceil(len(tokens) * 0.6))


def detect_missing_required_facts(query: str, rules: list[LegalRule]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        for fact in rule.required_facts:
            key = fact.strip().lower()
            if not key or key in seen:
                continue
            if fact_is_known_from_query(fact, query):
                continue
            seen.add(key)
            missing.append(fact.strip())
    return missing


def build_legal_rule_trace_context(rules: list[LegalRule]) -> str:
    if not rules:
        return "Trace primary law: brak."
    lines = ["Trace primary law dla writera:"]
    for index, rule in enumerate(rules, start=1):
        lines.append(
            f"{index}. rule_id={rule.provision_id} | source_document_id={rule.source_id} | provision_reference={rule.citation}"
            f" | retrieval_stage={rule.retrieval_stage} | selected_chunk_ids={','.join(rule.supporting_chunk_ids) or '-'}"
            f"\n   exact_source_span={rule.exact_source_span}"
        )
    return "\n".join(lines)


def build_provision_reference_registry(chunks: list[RagChunk], rules: list[LegalRule]) -> set[str]:
    registry: set[str] = set()
    for rule in rules:
        registry.add(normalize_provision_reference(rule.citation))
        for dependency in rule.definition_dependencies:
            registry.add(normalize_provision_reference(dependency))
    for chunk in chunks:
        for provision in chunk.legal_provisions:
            registry.add(normalize_provision_reference(provision))
    return {item for item in registry if item}


def extract_legal_rules_from_statute_chunks(chunks: list[RagChunk], *, limit: int = 12) -> list[LegalRule]:
    rules: list[LegalRule] = []
    seen: set[tuple[str, str, Optional[str], Optional[str], Optional[str]]] = set()
    for chunk in sorted(chunks, key=infer_chunk_source_priority, reverse=True):
        if not is_primary_source_chunk(chunk):
            continue
        article_key = ""
        for provision in chunk.legal_provisions:
            article_key = extract_article_key_from_text(provision)
            if article_key:
                break
        if not article_key:
            article_key = extract_article_key_from_text(chunk.subject) or extract_article_key_from_text(chunk.chunk_text)
        if not article_key:
            continue

        for paragraph, block_text in extract_rule_unit_blocks(chunk.chunk_text):
            nested_units = extract_nested_rule_units(block_text)
            for point, letter, unit_text in nested_units:
                key = (chunk_canonical_source_id(chunk), article_key, paragraph, point, letter)
                if key in seen:
                    continue
                seen.add(key)
                directive = normalize_whitespace(unit_text)
                if not directive:
                    continue
                citation = build_rule_citation(article_key, paragraph, point, letter)
                condition = extract_condition_from_rule_text(directive)
                scope_subject_terms, scope_object_terms = infer_scope_terms_for_rule(directive)
                rules.append(
                    LegalRule(
                        source_id=chunk_canonical_source_id(chunk),
                        act_title=extract_act_title_from_chunk(chunk),
                        publication=chunk.publication,
                        legal_state_date=chunk.legal_state_date,
                        provision_id=build_provision_id(chunk_canonical_source_id(chunk), citation),
                        citation=citation,
                        article_key=article_key,
                        paragraph=paragraph,
                        point=point,
                        letter=letter,
                        rule_type=derive_rule_type(directive),
                        condition=condition,
                        directive=directive[:1200],
                        exact_source_span=directive[:1200],
                        required_facts=infer_required_facts_for_rule(
                            act_title=extract_act_title_from_chunk(chunk),
                            citation=citation,
                            condition=condition,
                            directive=directive,
                        ),
                        definition_dependencies=infer_definition_dependencies_for_rule(
                            act_title=extract_act_title_from_chunk(chunk),
                            citation=citation,
                            directive=directive,
                        ),
                        scope_subject_terms=scope_subject_terms,
                        scope_object_terms=scope_object_terms,
                        specificity_rank=compute_rule_specificity_rank(
                            citation=citation,
                            directive=directive,
                            chunk=chunk,
                        ),
                        supporting_chunk_ids=[chunk.chunk_id],
                        source_url=chunk.source_url,
                    )
                )
                if len(rules) >= limit:
                    return rules
    return rules


def legal_rule_to_dict(rule: LegalRule) -> dict[str, Any]:
    return {
        "source_id": rule.source_id,
        "act_title": rule.act_title,
        "publication": rule.publication,
        "legal_state_date": rule.legal_state_date,
        "effective_from": legal_rule_effective_from(rule),
        "effective_to": legal_rule_effective_to(rule),
        "provision_id": rule.provision_id,
        "citation": rule.citation,
        "article_key": rule.article_key,
        "paragraph": rule.paragraph,
        "point": rule.point,
        "letter": rule.letter,
        "rule_type": rule.rule_type,
        "condition": rule.condition,
        "directive": rule.directive,
        "exact_source_span": rule.exact_source_span,
        "required_facts": list(rule.required_facts),
        "definition_dependencies": list(rule.definition_dependencies),
        "scope_subject_terms": list(rule.scope_subject_terms),
        "scope_object_terms": list(rule.scope_object_terms),
        "specificity_rank": rule.specificity_rank,
        "retrieval_stage": rule.retrieval_stage,
        "supporting_chunk_ids": list(rule.supporting_chunk_ids),
        "source_url": rule.source_url,
    }


def build_legal_rules_context(rules: list[LegalRule]) -> str:
    if not rules:
        return (
            "Ustrukturyzowane normy z primary law: brak. Writer nie może formułować materialnego"
            " rozstrzygnięcia bez przepisu ustawowego albo innego aktu prawa powszechnie obowiązującego."
        )
    lines = [
        "Ustrukturyzowane normy z primary law:",
        "Każdą normę traktuj jako punkt wyjścia analizy; interpretacje i wyroki wolno użyć dopiero jako wsparcie wykładni.",
    ]
    for index, rule in enumerate(rules, start=1):
        condition = f" | warunek: {rule.condition}" if rule.condition else ""
        lifecycle_note = " | status=uchylony" if rule.rule_type == "repealed" else ""
        required_facts = (
            f" | required_facts: {'; '.join(rule.required_facts)}"
            if rule.required_facts
            else ""
        )
        definition_dependencies = (
            f" | definicje: {'; '.join(rule.definition_dependencies)}"
            if rule.definition_dependencies
            else ""
        )
        legal_state = rule.legal_state_date or rule.publication or "brak daty w metadanych"
        lines.append(
            f"{index}. {rule.act_title} | {rule.citation} | typ={rule.rule_type} | stan={legal_state}{lifecycle_note}{condition}{required_facts}{definition_dependencies}\n"
            f"   reguła: {rule.directive}"
        )
    return "\n".join(lines)


def order_chunks_by_statute_targets(chunks: list[RagChunk], targets: list[tuple[str, str]]) -> list[RagChunk]:
    if not chunks or not targets:
        return chunks

    order = build_statute_target_order(targets)
    article_only_order: dict[str, int] = {}
    for position, (_domain, article_key) in enumerate(targets):
        article_only_order.setdefault(article_key, position)

    def sort_key(chunk: RagChunk) -> tuple[int, int, str, str, float, str]:
        domain = infer_chunk_tax_domain(chunk)
        best_position = len(order)
        for provision in chunk.legal_provisions:
            article_key = extract_article_key_from_text(provision)
            if not article_key:
                continue
            best_position = min(best_position, order.get((domain, article_key), len(order)))
            if chunk.subject.lower().startswith("upo polska"):
                best_position = min(best_position, article_only_order.get(article_key, len(order)))
        subtype_priority, legal_state, publication = infer_chunk_source_priority(chunk)
        return best_position, -subtype_priority, f"{legal_state}|{publication}", publication, -chunk.score, chunk.subject

    return sorted(chunks, key=sort_key)


def filter_treaty_country_chunks(chunks: list[RagChunk], query: str) -> list[RagChunk]:
    if not chunks:
        return chunks
    if query_targets_poland_germany_treaty(query):
        return [
            chunk for chunk in chunks
            if not chunk.subject.lower().startswith("upo polska") or "niemcy" in chunk.subject.lower()
        ]
    if query_targets_poland_spain_treaty(query):
        return [
            chunk for chunk in chunks
            if not chunk.subject.lower().startswith("upo polska") or "hiszpani" in chunk.subject.lower()
        ]
    return chunks


def decompose_query_into_legal_axes(query: str) -> list[LegalRetrievalAxis]:
    normalized = normalize_whitespace(query or "")
    axes: list[LegalRetrievalAxis] = []

    if query_targets_estonian_cit_transformation_share_cost(query) or query_targets_estonian_cit_hidden_profit(query):
        axes.extend(
            [
                LegalRetrievalAxis(
                    axis_id="estonian_cit_loan_principal",
                    label="estoński CIT: kapitał / przekształcenie / wkład / pożyczka",
                    query=expand_search_query(f"{normalized} przekształcenie spółki komandytowej estoński CIT kapitał wkład 93a 7aa 28j 28k 28m"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"CIT", "ORDYNACJA"},
                    preferred_targets=(("CIT", "7aa"), ("CIT", "28j"), ("CIT", "28k"), ("CIT", "28m"), ("ORDYNACJA", "93a")),
                ),
                LegalRetrievalAxis(
                    axis_id="estonian_cit_interest",
                    label="estoński CIT: odsetki / ukryty zysk / finansowanie wspólnika",
                    query=expand_search_query(f"{normalized} odsetki pożyczka wspólnik ukryty zysk ryczałt od dochodów spółek"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"CIT", "ORDYNACJA"},
                    preferred_targets=(("CIT", "28m"), ("CIT", "28n"), ("CIT", "28o"), ("CIT", "7aa"), ("ORDYNACJA", "93a")),
                ),
                LegalRetrievalAxis(
                    axis_id="estonian_cit_management_services",
                    label="estoński CIT: usługi zarządzania / powiązane świadczenia",
                    query=expand_search_query(f"{normalized} usługi zarządzania ryczałt od dochodów spółek ukryty zysk świadczenie"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"CIT", "ORDYNACJA"},
                    preferred_targets=(("CIT", "28m"), ("CIT", "28n"), ("CIT", "28o"), ("CIT", "7aa"), ("ORDYNACJA", "93a")),
                ),
            ]
        )

    if query_targets_ksef_current_law(query):
        axes.extend(
            [
                LegalRetrievalAxis(
                    axis_id="ksef_current_law_bundle",
                    label="KSeF 2.0: aktualna podstawa prawna i terminy",
                    query=expand_search_query(f"{normalized} KSeF 2.0 Dz.U. 2025 poz. 1203 1 lutego 2026 1 kwietnia 2026 limit 10 000 sankcje 2027"),
                    source_types={"statute"},
                    tax_domains={"VAT"},
                    preferred_targets=tuple(build_ksef_current_law_statute_targets(query)),
                ),
                LegalRetrievalAxis(
                    axis_id="ksef_scope_and_buyer_capacity",
                    label="KSeF 2.0: zakres obowiązku / B2B / B2C / nabywca zagraniczny",
                    query=expand_search_query(f"{normalized} art. 106a 106b 106ga 106gb B2B B2C konsument NIP nabywca zagraniczny polskie przepisy fakturowania"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "106a"), ("VAT", "106b"), ("VAT", "106ga"), ("VAT", "106gb")),
                ),
                LegalRetrievalAxis(
                    axis_id="ksef_receipt_and_deduction",
                    label="KSeF 2.0: otrzymanie faktury / PDF poza KSeF / odliczenie",
                    query=expand_search_query(f"{normalized} faktura poza KSeF PDF otrzymanie art. 86 art. 88 numer KSeF odliczenie VAT"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "86"), ("VAT", "88"), ("VAT", "106gb"), ("VAT", "106nda"), ("VAT", "106nh")),
                ),
                LegalRetrievalAxis(
                    axis_id="ksef_corrections",
                    label="KSeF 2.0: korekta in minus / faktura korygująca",
                    query=expand_search_query(f"{normalized} korekta in minus faktura korygująca ustrukturyzowana art. 29a 13a 13b 13c KSeF"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "29a"), ("VAT", "106j"), ("VAT", "106ga"), ("VAT", "106gb")),
                ),
                LegalRetrievalAxis(
                    axis_id="ksef_operational_modes",
                    label="KSeF 2.0: offline24 / niedostępność / awaria / system podatnika",
                    query=expand_search_query(f"{normalized} offline24 art. 106nda 106nf 106nh niedostępność awaria całkowita awaria systemu podatnika następny dzień roboczy"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "106nda"), ("VAT", "106nf"), ("VAT", "106nh"), ("VAT", "106gb")),
                ),
            ]
        )
        if query_targets_ksef_fixed_establishment_scope(query):
            axes.append(
                LegalRetrievalAxis(
                    axis_id="ksef_fixed_establishment_participation",
                    label="KSeF 2.0: SMPD i uczestnictwo w transakcji",
                    query=expand_search_query(f"{normalized} stałe miejsce prowadzenia działalności SMPD uczestniczy w dostawie świadczeniu usług art. 106ga 106gb"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "106ga"), ("VAT", "106gb"), ("VAT", "106a")),
                )
            )

    if query_targets_family_foundation_mechanism(query):
        axes.extend(
            [
                LegalRetrievalAxis(
                    axis_id="family_foundation_allowed_activity_catalog",
                    label="fundacja rodzinna: katalog art. 5 UFR",
                    query=expand_search_query(f"{normalized} fundacja rodzinna art. 5 UFR najem dzierżawa pożyczka spółce kapitałowej udziały akcje zbywanie mienia"),
                    source_types={"statute"},
                    tax_domains={"CIT", "PIT", "VAT"},
                    preferred_targets=tuple(build_family_foundation_statute_targets(query)),
                ),
                LegalRetrievalAxis(
                    axis_id="family_foundation_cit_hidden_profit",
                    label="fundacja rodzinna: CIT 24q / ukryte zyski / świadczenia",
                    query=expand_search_query(f"{normalized} fundacja rodzinna CIT art. 24q ukryte zyski świadczenie beneficjent fundator usługi prawne księgowe zarządzania"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"CIT"},
                    preferred_targets=(("CIT", "24q"), ("CIT", "24r"), ("CIT", "6")),
                ),
                LegalRetrievalAxis(
                    axis_id="family_foundation_disallowed_income_25_percent",
                    label="fundacja rodzinna: CIT 24r / dochód z działalności niedozwolonej",
                    query=expand_search_query(f"{normalized} fundacja rodzinna art. 24r 25% CIT dochód działalność wykraczająca poza art. 5 koszty odsetki pożyczka"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"CIT"},
                    preferred_targets=(("CIT", "24r"), ("CIT", "24q"), ("CIT", "18")),
                ),
                LegalRetrievalAxis(
                    axis_id="family_foundation_beneficiary_pit",
                    label="fundacja rodzinna: PIT beneficjenta / fundatora / grupy podatkowe",
                    query=expand_search_query(f"{normalized} fundacja rodzinna PIT fundator beneficjent dziecko grupa zerowa proporcja zwolnienie 10% 15%"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"PIT"},
                    preferred_targets=(("PIT", "21"), ("PIT", "30"), ("PIT", "20")),
                ),
                LegalRetrievalAxis(
                    axis_id="family_foundation_vat_related_party",
                    label="fundacja rodzinna: VAT najem / sprzedaż mienia / art. 32",
                    query=expand_search_query(f"{normalized} fundacja rodzinna VAT najem mieszkalny sprzedaż samochodu wartość rynkowa podmiot powiązany art. 32"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"VAT"},
                    preferred_targets=(("VAT", "32"), ("VAT", "43"), ("VAT", "29a"), ("VAT", "5")),
                ),
            ]
        )

    if query_targets_spolka_komandytowa_cit_status(query):
        axes.append(
            LegalRetrievalAxis(
                axis_id="limited_partnership_current_cit_status",
                label="spółka komandytowa: aktualny status podatnika CIT",
                query=expand_search_query(f"{normalized} spółka komandytowa podatnik CIT aktualny stan prawny 2026 transparentność podatkowa"),
                source_types={"statute", "interpretation"},
                tax_domains={"CIT"},
                preferred_targets=tuple(build_spolka_komandytowa_cit_status_statute_targets(query)),
            )
        )

    if query_targets_private_vehicle_pit_expense(query):
        axes.extend(
            [
                LegalRetrievalAxis(
                    axis_id="pit_private_vehicle_20_percent_cost_limit",
                    label="PIT: prywatny samochód niewprowadzony do działalności / limit 20%",
                    query=expand_search_query(f"{normalized} PIT samochód prywatny niewprowadzony do działalności art. 23 ust. 1 pkt 46 20% wydatki eksploatacyjne"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"PIT"},
                    preferred_targets=tuple(build_private_vehicle_pit_expense_statute_targets(query)),
                ),
                LegalRetrievalAxis(
                    axis_id="pit_business_vehicle_mixed_use_75_percent_cost_limit",
                    label="PIT: samochód firmowy używany mieszanie / limit 75%",
                    query=expand_search_query(f"{normalized} PIT samochód środek trwały używany prywatnie art. 23 ust. 1 pkt 46a 75% wydatki eksploatacyjne"),
                    source_types={"statute", "interpretation"},
                    tax_domains={"PIT"},
                    preferred_targets=tuple(build_private_vehicle_pit_expense_statute_targets(query)),
                ),
            ]
        )

    if query_targets_wht_pay_and_refund_services(query) or query_targets_crossborder_treaty_analysis(query):
        axes.extend(
            [
                LegalRetrievalAxis(
                    axis_id="wht_interest",
                    label="WHT: odsetki i należności bierne",
                    query=expand_search_query(f"{normalized} odsetki beneficial owner certyfikat rezydencji art. 21 art. 26"),
                    source_types={"statute", "interpretation", "judgment"},
                    tax_domains={"CIT", "PIT"},
                    preferred_targets=(("CIT", "21"), ("CIT", "26"), ("CIT", "22c"), ("CIT", "22"), ("CIT", "10"), ("CIT", "11"), ("CIT", "12")),
                ),
                LegalRetrievalAxis(
                    axis_id="wht_management_services",
                    label="WHT: usługi zarządzania / usługi niematerialne",
                    query=expand_search_query(f"{normalized} usługi zarządzania art. 21 ust. 1 pkt 2a art. 7 UPO zakład"),
                    source_types={"statute", "interpretation", "judgment"},
                    tax_domains={"CIT", "PIT"},
                    preferred_targets=(("CIT", "21"), ("CIT", "26"), ("CIT", "7"), ("CIT", "4"), ("CIT", "5"), ("CIT", "11"), ("CIT", "12")),
                ),
                LegalRetrievalAxis(
                    axis_id="pay_and_refund",
                    label="WHT: pay and refund / próg 2 mln / nadwyżka",
                    query=expand_search_query(f"{normalized} art. 26 ust. 2e 2 000 000 nadwyżka pay and refund"),
                    source_types={"statute", "interpretation", "judgment"},
                    tax_domains={"CIT"},
                    preferred_targets=(("CIT", "26"), ("CIT", "21"), ("CIT", "22")),
                ),
                LegalRetrievalAxis(
                    axis_id="interest_royalties_exemption",
                    label="WHT: zwolnienie dyrektywowe / odsetki / należności licencyjne",
                    query=expand_search_query(f"{normalized} art. 21 ust. 3 3c 3d 3e art. 22c beneficial owner zwolnienie"),
                    source_types={"statute", "interpretation", "judgment"},
                    tax_domains={"CIT", "PIT"},
                    preferred_targets=(("CIT", "21"), ("CIT", "22"), ("CIT", "22c"), ("CIT", "26"), ("CIT", "10"), ("CIT", "11"), ("CIT", "12")),
                ),
                LegalRetrievalAxis(
                    axis_id="beneficial_owner",
                    label="WHT: beneficial owner / należyta staranność",
                    query=expand_search_query(f"{normalized} beneficial owner rzeczywisty właściciel należyta staranność certyfikat rezydencji"),
                    source_types={"statute", "interpretation", "judgment"},
                    tax_domains={"CIT", "PIT"},
                    preferred_targets=(("CIT", "26"), ("CIT", "21"), ("CIT", "22"), ("CIT", "10"), ("CIT", "11"), ("CIT", "12")),
                ),
            ]
        )

    if query_targets_poland_germany_treaty(query):
        axes.append(
            LegalRetrievalAxis(
                axis_id="poland_germany_treaty",
                label="UPO Polska-Niemcy / treaty override / zakład",
                query=expand_search_query(f"{normalized} UPO Polska Niemcy art. 7 art. 10 art. 11 art. 12 art. 26 art. 29 zakład beneficial owner"),
                source_types={"statute"},
                tax_domains={"CIT"},
                preferred_targets=tuple(build_poland_germany_treaty_statute_targets(query)),
                direct_subject_prefix="UPO Polska - Niemcy",
            )
        )
    elif query_targets_poland_spain_treaty(query):
        axes.append(
            LegalRetrievalAxis(
                axis_id="poland_spain_treaty",
                label="UPO Polska-Hiszpania / treaty override / zakład",
                query=expand_search_query(f"{normalized} UPO Polska Hiszpania art. 7 art. 10 art. 11 art. 12 art. 26 zakład beneficial owner"),
                source_types={"statute"},
                tax_domains={"CIT"},
                preferred_targets=tuple(build_poland_spain_treaty_statute_targets(query)),
                direct_subject_prefix="UPO Polska - Hiszpania",
            )
        )

    # Keep axes unique and stable.
    deduped: list[LegalRetrievalAxis] = []
    seen_axis_ids: set[str] = set()
    for axis in axes:
        if axis.axis_id in seen_axis_ids:
            continue
        seen_axis_ids.add(axis.axis_id)
        deduped.append(axis)
    return deduped


def build_source_requirement_for_axis(axis: LegalRetrievalAxis) -> SourceRequirement:
    axis_id = axis.axis_id
    mandatory_primary_sources: list[str] = []
    optional_secondary_sources: list[str] = []
    treaty_required = False
    official_guidance_required = False

    if axis_id.startswith("ksef_"):
        mandatory_primary_sources.append("ksef_2_0_current_law_bundle")
        official_guidance_required = axis_id in {"ksef_operational_modes", "ksef_current_law_bundle"}
    elif axis_id.startswith("family_foundation_"):
        mandatory_primary_sources.append("family_foundation_primary_bundle")
    elif axis_id.startswith("estonian_cit_"):
        mandatory_primary_sources.append("current_cit_act")
        if "ORDYNACJA" in (axis.tax_domains or set()):
            mandatory_primary_sources.append("tax_ordinance")
        optional_secondary_sources.append("interpretations_or_judgments")
    elif axis_id.startswith("wht_") or axis_id in {"pay_and_refund", "interest_royalties_exemption", "beneficial_owner"}:
        mandatory_primary_sources.append("current_cit_act")
        optional_secondary_sources.append("official_wht_guidance_or_case_law")
    elif axis_id.startswith("poland_") and axis_id.endswith("_treaty"):
        mandatory_primary_sources.append("tax_treaty_text")
        treaty_required = True
    elif axis.source_types and "statute" in axis.source_types:
        mandatory_primary_sources.append("current_statute")

    return SourceRequirement(
        axis_id=axis_id,
        mandatory_primary_sources=mandatory_primary_sources,
        optional_secondary_sources=optional_secondary_sources,
        treaty_required=treaty_required,
        official_guidance_required=official_guidance_required,
    )


def is_primary_source_chunk(chunk: RagChunk) -> bool:
    source_type = str(chunk.source_type or "").lower()
    source_subtype = str(chunk.source_subtype or "").lower()
    subject = normalize_whitespace(chunk.subject or "").lower()
    return (
        source_type == "statute"
        or "tax_treaty" in source_subtype
        or subject.startswith("upo polska")
    )


def chunk_matches_axis_domain(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if not axis.tax_domains:
        return True
    chunk_domain = infer_chunk_tax_domain(chunk)
    provision_domains = {
        target[0]
        for provision in chunk.legal_provisions
        for target in [extract_statute_target_from_text(provision)]
        if target
    }
    candidate_domains = {value for value in [chunk_domain, *provision_domains] if value}
    if not candidate_domains:
        return True
    return bool(candidate_domains & axis.tax_domains)


def chunk_matches_axis_source_type(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if not axis.source_types:
        return True
    return str(chunk.source_type or "").lower() in axis.source_types


def chunk_has_axis_preferred_target(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if not axis.preferred_targets:
        return False
    preferred = set(axis.preferred_targets)
    subject = normalize_whitespace(chunk.subject or "").lower()
    for provision in chunk.legal_provisions:
        target = extract_statute_target_from_text(provision)
        if target and target in preferred:
            return True
        article_key = extract_article_key_from_text(provision)
        if article_key and subject.startswith("upo polska") and any(article_key == target[1] for target in preferred):
            return True
        chunk_domain = infer_chunk_tax_domain(chunk)
        if article_key and chunk_domain and (chunk_domain, article_key) in preferred:
            return True
    return False


def chunk_has_substantive_axis_preferred_target(
    axis: LegalRetrievalAxis,
    chunk: RagChunk,
) -> bool:
    """Require an operative unit, rather than a split-statute heading, for fallback coverage."""
    if not chunk_has_axis_preferred_target(axis, chunk):
        return False
    text = normalize_whitespace(chunk.chunk_text or "")
    if len(text) < 80:
        return False
    return not bool(re.fullmatch(r"art\.\s*\d+[a-z]?(?:\s*\.)?", text, re.IGNORECASE))


def chunk_is_direct_axis_source(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    subject = normalize_whitespace(chunk.subject or "").lower()
    document_id = str(chunk.document_id or "")
    if axis.direct_subject_prefix and subject.startswith(axis.direct_subject_prefix.lower()):
        return True
    if axis.axis_id.startswith("ksef_") and document_id in KSEF_CURRENT_BUNDLE_DOCUMENT_IDS:
        return True
    if axis.axis_id.startswith("family_foundation_") and document_id in FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS:
        return True
    return False


def chunk_matches_axis(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if not chunk_matches_axis_source_type(axis, chunk):
        return False
    if chunk_is_direct_axis_source(axis, chunk):
        return True
    if chunk_has_axis_preferred_target(axis, chunk):
        return True
    return chunk_matches_axis_domain(axis, chunk) and not axis.preferred_targets


def chunk_is_current_law_for_axis(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if not is_primary_source_chunk(chunk):
        return False
    document_id = str(chunk.document_id or "")
    text = " ".join(
        part
        for part in [chunk.subject, chunk.publication or "", chunk.legal_state_date or "", chunk.chunk_text[:1600]]
        if part
    )
    if axis.axis_id.startswith("ksef_"):
        return document_id in KSEF_CURRENT_BUNDLE_DOCUMENT_IDS or bool(
            re.search(r"KSeF\s*2\.0|Dz\.U\.\s*2025\s*poz\.\s*1203", text, re.IGNORECASE)
        )
    if axis.axis_id.startswith("family_foundation_"):
        return document_id in FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS or bool(
            re.search(r"fundacj\w* rodzinn\w*|art\.\s*5|art\.\s*24q|art\.\s*24r", text, re.IGNORECASE)
            or chunk.source_type == "statute"
        )
    return True


def chunk_is_relevant_resolution_for_axis(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    if str(chunk.source_type or "").lower() not in {"interpretation", "judgment"}:
        return False
    if not chunk_matches_axis_domain(axis, chunk):
        return False
    role = classify_chunk_evidence_role(chunk)
    return role in {"authority_assessment", "operative_conclusion", "reasoning", "supporting_source"}


def chunk_is_axis_misleading_neighbor(axis: LegalRetrievalAxis, chunk: RagChunk) -> bool:
    text = normalize_whitespace(" ".join([chunk.subject or "", chunk.chunk_text[:1800]])).lower()
    if axis.axis_id == "wht_interest":
        return bool(re.search(r"zarządz|zarzadz|doradcz|księgow|ksiegow", text)) and not re.search(r"odset", text)
    if axis.axis_id == "wht_management_services":
        return bool(re.search(r"odset", text)) and not re.search(r"zarządz|zarzadz|doradcz|księgow|ksiegow", text)
    if axis.axis_id == "estonian_cit_loan_principal":
        return bool(re.search(r"odset", text)) and not re.search(r"kapitał|kapital|wkład|wklad|przekształc", text)
    if axis.axis_id == "estonian_cit_interest":
        return bool(re.search(r"zwrot kapitał|zwrot kapital|wkład|wklad", text)) and not re.search(r"odset", text)
    if axis.axis_id == "ksef_scope_and_buyer_capacity":
        return bool(re.search(r"odliczeni|korekt", text)) and not re.search(r"b2b|b2c|konsument|nabywca|106ga|106gb", text)
    if axis.axis_id == "family_foundation_allowed_activity_catalog":
        return bool(re.search(r"ukryte zyski|świadczeni", text)) and not re.search(r"art\.\s*5|dozwolon", text)
    return False


def axis_coverage_to_dict(coverage: AxisCoverage) -> dict[str, Any]:
    return {
        "axis_id": coverage.axis_id,
        "label": coverage.label,
        "controlling_rule_present": coverage.controlling_rule_present,
        "current_law_source_present": coverage.current_law_source_present,
        "relevant_resolution_present": coverage.relevant_resolution_present,
        "primary_source_present": coverage.primary_source_present,
        "required_treaty_present": coverage.required_treaty_present,
        "missing_source_types": coverage.missing_source_types,
        "misleading_neighbor_present": coverage.misleading_neighbor_present,
        "coverage_score": coverage.coverage_score,
        "status": coverage.status,
        "supporting_source_ids": coverage.supporting_source_ids,
    }


def build_axis_coverage(query: str, chunks: list[RagChunk]) -> list[AxisCoverage]:
    axes = decompose_query_into_legal_axes(query)
    if not axes:
        return []

    coverages: list[AxisCoverage] = []
    for axis in axes:
        requirement = build_source_requirement_for_axis(axis)
        matching_chunks = [chunk for chunk in chunks if chunk_matches_axis(axis, chunk)]
        primary_chunks = [chunk for chunk in matching_chunks if is_primary_source_chunk(chunk)]
        current_law_chunks = [chunk for chunk in primary_chunks if chunk_is_current_law_for_axis(axis, chunk)]
        controlling_chunks = [
            chunk for chunk in matching_chunks
            if chunk_is_direct_axis_source(axis, chunk) or chunk_has_axis_preferred_target(axis, chunk)
        ]
        resolution_chunks = [chunk for chunk in matching_chunks if chunk_is_relevant_resolution_for_axis(axis, chunk)]

        required_treaty_present: Optional[bool] = None
        if requirement.treaty_required:
            required_treaty_present = any(
                chunk.subject.lower().startswith("upo polska")
                or "tax_treaty" in str(chunk.source_subtype or "").lower()
                for chunk in matching_chunks
            )

        missing_source_types: list[str] = []
        if requirement.controlling_rule_required and not controlling_chunks:
            missing_source_types.append("controlling_rule")
        if requirement.current_law_required and not current_law_chunks:
            missing_source_types.append("current_law_source")
        if requirement.mandatory_primary_sources and not primary_chunks:
            missing_source_types.extend(requirement.mandatory_primary_sources)
        if requirement.treaty_required and not required_treaty_present:
            missing_source_types.append("tax_treaty_text")

        misleading_neighbor_present = any(chunk_is_axis_misleading_neighbor(axis, chunk) for chunk in matching_chunks)
        primary_source_present = bool(primary_chunks)
        controlling_rule_present = bool(controlling_chunks)
        current_law_source_present = bool(current_law_chunks)
        relevant_resolution_present = bool(resolution_chunks)

        score = 0.0
        score += 0.35 if primary_source_present else 0.0
        score += 0.35 if controlling_rule_present else 0.0
        score += 0.20 if current_law_source_present else 0.0
        score += 0.10 if (required_treaty_present is not False) else 0.0
        if misleading_neighbor_present and not controlling_rule_present:
            score = max(0.0, score - 0.25)
        score = round(min(score, 1.0), 2)

        if (
            score >= 0.80
            and primary_source_present
            and controlling_rule_present
            and current_law_source_present
            and required_treaty_present is not False
            and not (misleading_neighbor_present and not controlling_rule_present)
        ):
            status = "covered"
        elif score >= 0.45 and primary_source_present:
            status = "partially_covered"
        else:
            status = "unresolved"

        supporting_source_ids = list(
            dict.fromkeys(chunk_canonical_source_id(chunk) for chunk in [*controlling_chunks, *current_law_chunks, *resolution_chunks])
        )[:8]
        coverages.append(
            AxisCoverage(
                axis_id=axis.axis_id,
                label=axis.label,
                controlling_rule_present=controlling_rule_present,
                current_law_source_present=current_law_source_present,
                relevant_resolution_present=relevant_resolution_present,
                primary_source_present=primary_source_present,
                required_treaty_present=required_treaty_present,
                missing_source_types=list(dict.fromkeys(missing_source_types)),
                misleading_neighbor_present=misleading_neighbor_present,
                coverage_score=score,
                status=status,
                supporting_source_ids=supporting_source_ids,
            )
        )
    return coverages


def build_axis_coverage_context(query: str, chunks: list[RagChunk]) -> str:
    coverages = build_axis_coverage(query, chunks)
    if not coverages:
        return ""

    lines = [
        "Bramka pokrycia per oś prawna:",
        "Reguła twarda: dla osi ze statusem unresolved nie wolno formułować materialnego rozstrzygnięcia; wolno wskazać tylko brakujące źródło lub fakt.",
        "Reguła twarda: stawki, limity, terminy, numery artykułów i skutki podatkowe wymagają controlling source w tej samej osi.",
    ]
    for coverage in coverages:
        treaty_value = "n/d" if coverage.required_treaty_present is None else str(coverage.required_treaty_present).lower()
        missing = ", ".join(coverage.missing_source_types) if coverage.missing_source_types else "brak"
        sources = ", ".join(coverage.supporting_source_ids) if coverage.supporting_source_ids else "brak"
        lines.append(
            "- "
            f"{coverage.axis_id} | {coverage.status} | score={coverage.coverage_score:.2f} | "
            f"primary={str(coverage.primary_source_present).lower()} | "
            f"controlling={str(coverage.controlling_rule_present).lower()} | "
            f"current={str(coverage.current_law_source_present).lower()} | "
            f"treaty={treaty_value} | "
            f"misleading_neighbor={str(coverage.misleading_neighbor_present).lower()} | "
            f"missing={missing} | sources={sources}"
        )
    return "\n".join(lines)


def processed_record_to_rag_chunk(
    record: dict[str, Any],
    *,
    chunk_index: int = 0,
    score: float = 100.0,
    chunk_id_suffix: str = "source-fallback",
) -> Optional[RagChunk]:
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return None
    text = clean_document_text(record)
    if not text:
        return None
    source_type = normalize_source_type(record)
    return RagChunk(
        chunk_id=f"{document_id}:{chunk_id_suffix}:{chunk_index}",
        document_id=document_id,
        chunk_index=chunk_index,
        score=score,
        chunk_text=text,
        subject=normalize_whitespace(str(record.get("subject") or "Bez tytułu")) or "Bez tytułu",
        signature=normalize_whitespace(str(record.get("signature") or "")) or None,
        published_date=normalize_whitespace(str(record.get("published_date") or "")) or None,
        source_url=normalize_whitespace(str(record.get("source_url") or "")) or None,
        category=normalize_whitespace(str(record.get("category") or "")) or None,
        source=normalize_whitespace(str(record.get("source") or "")),
        source_type=source_type,
        source_subtype=derive_source_subtype(record) or None,
        authority=normalize_whitespace(str(record.get("authority") or "")) or None,
        publication=normalize_whitespace(str(record.get("publication") or "")) or None,
        legal_state_date=normalize_whitespace(str(record.get("legal_state_date") or "")) or None,
        source_pages=[int(page) for page in record.get("source_pages") or [] if str(page).isdigit()],
        legal_provisions=[
            str(value).strip() for value in record.get("legal_provisions") or [] if str(value).strip()
        ],
        evidence_role="primary_source_fallback" if source_type == "statute" else "",
    )


def load_processed_document_chunks_by_ids(
    document_ids: list[str] | tuple[str, ...],
    *,
    source_paths: Optional[Iterable[Path]] = None,
    chunk_limit_per_document: int = 1,
) -> list[RagChunk]:
    wanted = {str(document_id).strip() for document_id in document_ids if str(document_id).strip()}
    if not wanted:
        return []

    configured_paths = tuple(source_paths) if source_paths is not None else get_rag_config().additional_source_paths
    chunks: list[RagChunk] = []
    counts: dict[str, int] = {}
    for path in configured_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    document_id = str(record.get("document_id") or "").strip()
                    if document_id not in wanted:
                        continue
                    if counts.get(document_id, 0) >= chunk_limit_per_document:
                        continue
                    chunk = processed_record_to_rag_chunk(
                        record,
                        chunk_index=counts.get(document_id, 0),
                        chunk_id_suffix="source-fallback",
                    )
                    if chunk is None:
                        continue
                    chunks.append(chunk)
                    counts[document_id] = counts.get(document_id, 0) + 1
                    if wanted.issubset({chunk.document_id for chunk in chunks}):
                        return chunks
        except OSError:
            continue
    return chunks


def load_processed_statute_chunks_by_targets(
    targets: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    *,
    source_paths: Optional[Iterable[Path]] = None,
    chunk_limit_per_target: int = 1,
) -> list[RagChunk]:
    wanted = {(domain.upper(), article_key.lower()) for domain, article_key in targets if domain and article_key}
    if not wanted:
        return []

    target_order = {(domain.upper(), article_key.lower()): index for index, (domain, article_key) in enumerate(targets)}

    def sort_by_target_order(items: list[RagChunk]) -> list[RagChunk]:
        def sort_key(chunk: RagChunk) -> tuple[int, int, str, str, int, str]:
            domain = infer_chunk_tax_domain(chunk)
            positions: list[int] = []
            for provision in chunk.legal_provisions:
                article_key = extract_article_key_from_text(provision)
                if article_key:
                    positions.append(target_order.get((domain, article_key), len(target_order)))
            subtype_priority, legal_state, publication = infer_chunk_source_priority(chunk)
            return (
                min(positions or [len(target_order)]),
                -subtype_priority,
                f"{legal_state}|{publication}",
                publication,
                chunk.chunk_index,
                chunk.document_id,
            )

        return sorted(items, key=sort_key)

    configured_paths = tuple(source_paths) if source_paths is not None else get_rag_config().additional_source_paths
    chunks: list[RagChunk] = []
    counts: dict[tuple[str, str], int] = {}
    for path in configured_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if normalize_source_type(record) != "statute":
                        continue
                    if str(record.get("source_subtype") or "").lower() == "tax_treaty":
                        continue
                    domain = derive_tax_domain(record).upper()
                    if not domain:
                        continue
                    matched_targets = []
                    for value in record.get("legal_provisions") or []:
                        article_key = extract_article_key_from_text(str(value))
                        if article_key and (domain, article_key) in wanted:
                            matched_targets.append((domain, article_key))
                    if not matched_targets:
                        continue
                    if all(counts.get(target, 0) >= chunk_limit_per_target for target in matched_targets):
                        continue
                    chunk = processed_record_to_rag_chunk(record, chunk_id_suffix="target-fallback")
                    if chunk is None:
                        continue
                    chunks.append(chunk)
                    for target in matched_targets:
                        counts[target] = counts.get(target, 0) + 1
                    if all(counts.get(target, 0) >= chunk_limit_per_target for target in wanted):
                        return sort_by_target_order(chunks)
        except OSError:
            continue
    return sort_by_target_order(chunks)


def load_processed_statute_chunks_by_subject_prefix(
    subject_prefix: str,
    *,
    targets: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
    source_paths: Optional[Iterable[Path]] = None,
    chunk_limit: int = 6,
) -> list[RagChunk]:
    prefix = normalize_whitespace(subject_prefix).lower()
    if not prefix:
        return []
    wanted_articles = {article_key.lower() for _domain, article_key in targets if article_key}
    target_order = {article_key.lower(): index for index, (_domain, article_key) in enumerate(targets)}
    configured_paths = tuple(source_paths) if source_paths is not None else get_rag_config().additional_source_paths
    chunks: list[RagChunk] = []
    for path in configured_paths:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if normalize_source_type(record) != "statute":
                        continue
                    subject = normalize_whitespace(str(record.get("subject") or "")).lower()
                    if not subject.startswith(prefix):
                        continue
                    if wanted_articles:
                        article_keys = {
                            extract_article_key_from_text(str(value))
                            for value in record.get("legal_provisions") or []
                        }
                        if not (article_keys & wanted_articles):
                            continue
                    chunk = processed_record_to_rag_chunk(record, chunk_id_suffix="subject-fallback")
                    if chunk is None:
                        continue
                    chunks.append(chunk)
        except OSError:
            continue
    if wanted_articles:
        chunks = sorted(
            chunks,
            key=lambda chunk: min(
                [
                    target_order.get(extract_article_key_from_text(provision), len(target_order))
                    for provision in chunk.legal_provisions
                ]
                or [len(target_order)]
            ),
        )
    return chunks[:chunk_limit]


def dedupe_chunks_by_chunk_id(chunks: list[RagChunk]) -> list[RagChunk]:
    deduped: list[RagChunk] = []
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        deduped.append(chunk)
    return deduped


def row_to_rag_chunk(row: sqlite3.Row | dict[str, Any], *, score: float = 100.0, evidence_role: str = "") -> RagChunk:
    return RagChunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        score=score,
        chunk_text=str(row["chunk_text"] or ""),
        subject=str(row["subject"] or "Bez tytułu"),
        signature=str(row["signature"] or "") or None,
        published_date=str(row["published_date"] or "") or None,
        source_url=str(row["source_url"] or "") or None,
        category=str(row["category"] or "") or None,
        source=str(row["source"] or ""),
        source_type=str(row["source_type"] or "interpretation"),
        source_subtype=str(row["source_subtype"] or "") or None,
        authority=str(row["authority"] or "") or None,
        publication=str(row["publication"] or "") or None,
        legal_state_date=str(row["legal_state_date"] or "") or None,
        source_pages=[int(value) for value in json.loads(row["source_pages_json"] or "[]")],
        legal_provisions=[str(value) for value in json.loads(row["legal_provisions_json"] or "[]")],
        evidence_role=evidence_role,
    )


def required_primary_document_ids_for_query(query: str) -> list[str]:
    document_ids: list[str] = []
    if query_targets_ksef_current_law(query):
        document_ids.extend(KSEF_CURRENT_BUNDLE_DOCUMENT_IDS)
    if query_targets_family_foundation_mechanism(query):
        document_ids.extend(FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS)
    return list(dict.fromkeys(document_ids))


def retrieve_deterministic_statute_chunks(
    query: str,
    *,
    plan: Optional[LegalSourcePlan] = None,
    limit: Optional[int] = None,
    config: Optional[RagConfig] = None,
) -> list[RagChunk]:
    if resolve_rag_runtime().read_backend == "mysql":
        from app.mysql_rag import retrieve_deterministic_statute_chunks_mysql

        return retrieve_deterministic_statute_chunks_mysql(query, plan=plan, limit=limit)

    effective_config = config or get_rag_config()
    source_plan = plan or build_legal_source_plan(query)
    target_limit = max(limit or effective_config.retrieval_limit, len(source_plan.statute_targets) or 1)

    rows: list[sqlite3.Row] = []
    if source_plan.statute_targets:
        rows.extend(
            fetch_statute_rows_by_targets(
                list(source_plan.statute_targets),
                config=effective_config,
                limit=None,
            )
        )
    for axis in source_plan.axes:
        if not axis.direct_subject_prefix:
            continue
        rows.extend(
            fetch_rows_by_subject_prefix(
                axis.direct_subject_prefix,
                config=effective_config,
                source_type="statute",
            )
        )
    required_document_ids = required_primary_document_ids_for_query(query)
    if required_document_ids:
        rows.extend(
            fetch_rows_by_document_ids(
                required_document_ids,
                config=effective_config,
                source_type="statute",
                chunk_limit_per_document=1,
            )
        )

    ranked_chunks = [
        row_to_rag_chunk(row, score=200.0, evidence_role="deterministic_primary_law")
        for row in rows
    ]
    # The indexed corpus is the source of truth at request time.  Scanning the
    # multi-gigabyte JSONL source on every direct statute lookup made local
    # requests take tens of seconds, while production images intentionally do
    # not contain those files at all.  Consult source files only for targets
    # that are genuinely absent from the active SQLite index.
    missing_statute_targets = [
        target
        for target in source_plan.statute_targets
        if not any(chunk_matches_statute_target(chunk, target) for chunk in ranked_chunks)
    ]
    fallback_chunks: list[RagChunk] = []
    if missing_statute_targets:
        fallback_chunks.extend(
            load_processed_statute_chunks_by_targets(
                missing_statute_targets,
                chunk_limit_per_target=1,
            )
        )
    for axis in source_plan.axes:
        if axis.direct_subject_prefix:
            if any(
                normalize_whitespace(chunk.subject or "").lower().startswith(
                    normalize_whitespace(axis.direct_subject_prefix).lower()
                )
                for chunk in ranked_chunks
            ):
                continue
            fallback_chunks.extend(
                load_processed_statute_chunks_by_subject_prefix(
                    axis.direct_subject_prefix,
                    targets=axis.preferred_targets,
                    chunk_limit=max(4, len(axis.preferred_targets) or 4),
                )
            )
    indexed_document_ids = {str(chunk.document_id or "") for chunk in ranked_chunks}
    missing_document_ids = [
        document_id
        for document_id in required_document_ids
        if document_id not in indexed_document_ids
    ]
    if missing_document_ids:
        fallback_chunks.extend(
            load_processed_document_chunks_by_ids(
                missing_document_ids,
                chunk_limit_per_document=1,
            )
        )

    ordered_chunks = order_chunks_by_statute_targets(
        [*ranked_chunks, *fallback_chunks],
        list(source_plan.statute_targets),
    )
    query_tokens = {
        token.lower()
        for token in QUERY_TOKEN_RE.findall(query or "")
        if len(token) >= 4 and token.lower() not in RANKING_STOPWORDS
    }

    def controlling_unit_score(chunk: RagChunk) -> tuple[int, int, int]:
        text = normalize_whitespace(
            " ".join([chunk.subject or "", chunk.chunk_text or ""])
        ).lower()
        overlap = sum(1 for token in query_tokens if token in text)
        substantive = int(
            len(normalize_whitespace(chunk.chunk_text or "")) >= 80
            and not re.fullmatch(
                r"(?:art\.?|artyku[łl])\s*\d+[a-z]?\s*\.?",
                normalize_whitespace(chunk.chunk_text or ""),
                re.IGNORECASE,
            )
        )
        return overlap, substantive, -chunk.chunk_index

    # Keep one best editorial unit per canonical article.  Choosing chunk 0
    # blindly frequently selected a heading or an unrelated paragraph while
    # the controlling paragraph was present in the same indexed article.
    best_by_source: dict[str, RagChunk] = {}
    source_order: list[str] = []
    for chunk in dedupe_chunks_by_chunk_id(ordered_chunks):
        source_id = chunk_canonical_source_id(chunk)
        if source_id not in best_by_source:
            best_by_source[source_id] = chunk
            source_order.append(source_id)
            continue
        if controlling_unit_score(chunk) > controlling_unit_score(best_by_source[source_id]):
            best_by_source[source_id] = chunk
    return [
        annotate_chunk_evidence_role(chunk, "deterministic_primary_law")
        for chunk in (best_by_source[source_id] for source_id in source_order)
    ][:target_limit]


def add_primary_source_fallback_chunks(query: str, chunks: list[RagChunk]) -> list[RagChunk]:
    """Prepend deterministic primary law from the configured read backend.

    Historically this function always scanned processed source files.  That
    silently created two different runtimes: local development could recover
    a provision, but the production image (which correctly excludes the raw
    corpus) could not.  ``retrieve_deterministic_statute_chunks`` already
    routes through MySQL when MySQL is active and falls back to source files
    only when an indexed SQLite target is missing, so it is the single safe
    recovery path.
    """
    source_plan = build_legal_source_plan(
        query,
        include_interpretations=False,
        include_judgments=False,
    )
    deterministic = retrieve_deterministic_statute_chunks(
        query,
        plan=source_plan,
        limit=max(get_rag_config().retrieval_limit, len(source_plan.statute_targets) or 1),
    )
    return dedupe_chunks_by_chunk_id([*deterministic, *chunks])


def search_primary_law_chunks(
    query: str,
    *,
    limit: Optional[int] = None,
) -> list[RagChunk]:
    """Retrieve primary law independently from optional authority lanes.

    This function is deliberately backend-routed and statute-only.  Callers
    can put a hard deadline around it without allowing a slow interpretation
    or judgment search to erase an already retrieved controlling provision.
    """
    config = get_rag_config()
    effective_limit = max(limit or config.retrieval_limit, 12)
    source_plan = build_legal_source_plan(
        query,
        include_interpretations=False,
        include_judgments=False,
    )
    deterministic = retrieve_deterministic_statute_chunks(
        query,
        plan=source_plan,
        limit=max(effective_limit, len(source_plan.statute_targets) or 1),
    )
    domains = set(source_plan.tax_domains)
    runtime = resolve_rag_runtime()
    if runtime.read_backend == "mysql":
        from app.mysql_rag import search_chunks_mysql

        semantic = search_chunks_mysql(
            query,
            limit=effective_limit,
            source_types={"statute"},
            enforce_query_domain=bool(domains),
            tax_domains=domains or None,
        )
    elif runtime.read_backend == "supabase":
        from app.supabase_rag import search_chunks_supabase

        semantic = search_chunks_supabase(
            query,
            limit=effective_limit,
            source_types={"statute"},
            tax_domains=domains or None,
        )
    else:
        semantic = _search_chunks_single_query(
            query,
            limit=effective_limit,
            source_types={"statute"},
            enforce_query_domain=bool(domains),
            tax_domains=domains or None,
        )
    return dedupe_chunks_by_chunk_id([*deterministic, *semantic])[:effective_limit]


def _search_chunks_single_query(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    ensure_local_index_ready()
    if not config.db_path.exists():
        return []

    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    inferred_tax_domains = tax_domains or infer_retrieval_tax_domains(query) or None
    _, rows = fetch_local_candidate_rows(
        expanded_query,
        effective_limit=effective_limit,
        config=config,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain or bool(inferred_tax_domains),
        tax_domains=inferred_tax_domains,
    )
    return rank_hybrid_local_candidates(rows, query=expanded_query, effective_limit=effective_limit, config=config)


def _resolve_axis_scope(
    axis: LegalRetrievalAxis,
    *,
    source_types: Optional[set[str]],
    tax_domains: Optional[set[str]],
) -> Optional[tuple[Optional[set[str]], Optional[set[str]]]]:
    axis_source_types = set(axis.source_types) if axis.source_types else None
    if source_types is not None:
        axis_source_types = set(source_types) if axis_source_types is None else axis_source_types & set(source_types)
        if axis_source_types is not None and not axis_source_types:
            return None

    axis_tax_domains = set(axis.tax_domains) if axis.tax_domains else None
    if tax_domains is not None:
        axis_tax_domains = set(tax_domains) if axis_tax_domains is None else axis_tax_domains & set(tax_domains)
        if axis_tax_domains is not None and not axis_tax_domains:
            return None

    return axis_source_types, axis_tax_domains


def _merge_axis_search_chunks(axis_chunks: list[list[RagChunk]], *, effective_limit: int) -> list[RagChunk]:
    merged: list[RagChunk] = []
    seen_canonical_sources: set[str] = set()
    flattened = [chunk for group in axis_chunks for chunk in group]
    flattened.sort(
        key=lambda chunk: (
            chunk.score,
            *infer_chunk_source_priority(chunk),
            -chunk.chunk_index,
            chunk.document_id,
        ),
        reverse=True,
    )
    for chunk in flattened:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_canonical_sources:
            continue
        seen_canonical_sources.add(canonical_source_id)
        merged.append(chunk)
        if len(merged) >= effective_limit:
            break
    return merged


def _search_chunks_by_legal_axes(
    query: str,
    *,
    limit: Optional[int] = None,
    source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False,
    tax_domains: Optional[set[str]] = None,
) -> tuple[list[RagChunk], list[LegalRetrievalAxis]]:
    effective_limit = limit or get_rag_config().retrieval_limit
    axes = decompose_query_into_legal_axes(query)
    if len(axes) <= 1:
        return [], axes

    scoped_axis_chunks: list[list[RagChunk]] = []
    active_axes: list[LegalRetrievalAxis] = []
    for axis in axes:
        axis_scope = _resolve_axis_scope(axis, source_types=source_types, tax_domains=tax_domains)
        if axis_scope is None:
            continue
        axis_source_types, axis_tax_domains = axis_scope
        active_axes.append(axis)
        axis_limit = max(1, math.ceil(effective_limit / max(len(axes), 1)))
        axis_chunks = _search_chunks_single_query(
                axis.query,
                limit=axis_limit,
                source_types=axis_source_types,
                enforce_query_domain=enforce_query_domain or bool(axis_tax_domains),
                tax_domains=axis_tax_domains,
            )
        if axis.direct_subject_prefix:
            config = get_rag_config()
            direct_rows = fetch_rows_by_subject_prefix(
                axis.direct_subject_prefix,
                config=config,
                source_type="statute" if axis_source_types is None or "statute" in axis_source_types else None,
            )
            direct_chunks = (
                rank_hybrid_local_candidates(
                    direct_rows,
                    query=axis.query,
                    effective_limit=max(axis_limit, len(direct_rows)),
                    config=config,
                )
                if direct_rows
                else []
            )
            ordered_direct_chunks = order_chunks_by_statute_targets(direct_chunks, list(axis.preferred_targets))
            direct_limit = max(axis_limit, len(axis.preferred_targets) or axis_limit)
            axis_chunks = [*ordered_direct_chunks[:direct_limit], *axis_chunks]
        scoped_axis_chunks.append(axis_chunks)

    if not scoped_axis_chunks:
        return [], axes

    return _merge_axis_search_chunks(scoped_axis_chunks, effective_limit=effective_limit), active_axes


def fetch_rows_by_document_ids(
    document_ids: list[str] | tuple[str, ...],
    *,
    config: RagConfig,
    source_type: Optional[str] = None,
    chunk_limit_per_document: Optional[int] = None,
) -> list[sqlite3.Row]:
    clean_ids = [str(document_id).strip() for document_id in document_ids if str(document_id).strip()]
    if not clean_ids or not config.db_path.exists():
        return []

    placeholders = ", ".join("?" for _ in clean_ids)
    source_clause = " AND d.source_type = ?" if source_type else ""
    values: list[str] = [*clean_ids]
    if source_type:
        values.append(source_type)

    connection = get_connection(config.db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT
                c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                d.subject, d.signature, d.published_date, d.source_url, d.category,
                d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                0.0 AS lexical_score
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.document_id IN ({placeholders})
              {source_clause}
            ORDER BY c.document_id ASC, c.chunk_index ASC
            """,
            tuple(values),
        ).fetchall()
    finally:
        connection.close()

    if chunk_limit_per_document is None:
        return rows

    limited_rows: list[sqlite3.Row] = []
    counts: dict[str, int] = {}
    for row in rows:
        document_id = str(row["document_id"])
        if counts.get(document_id, 0) >= chunk_limit_per_document:
            continue
        limited_rows.append(row)
        counts[document_id] = counts.get(document_id, 0) + 1
    return limited_rows


def fetch_rows_by_subject_prefix(
    subject_prefix: str,
    *,
    config: RagConfig,
    source_type: Optional[str] = None,
) -> list[sqlite3.Row]:
    prefix = str(subject_prefix).strip()
    if not prefix or not config.db_path.exists():
        return []

    source_clause = " AND d.source_type = ?" if source_type else ""
    values: list[str] = [f"{prefix}%"]
    if source_type:
        values.append(source_type)

    connection = get_connection(config.db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT
                c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                d.subject, d.signature, d.published_date, d.source_url, d.category,
                d.keywords_json, d.legal_provisions_json, d.issues_json, d.law_tags_json, d.facts_text, d.question_text, d.tax_domain,
                d.source, d.source_type, d.source_subtype, d.authority, d.publication, d.legal_state_date, d.source_pages_json,
                0.0 AS lexical_score
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.subject LIKE ?
              {source_clause}
            ORDER BY d.subject ASC, c.document_id ASC, c.chunk_index ASC
            """,
            tuple(values),
        ).fetchall()
    finally:
        connection.close()
    return rows


def inspect_local_candidate_pool(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Return the pre-rerank lexical pool; intended for evaluator diagnostics."""
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    _, rows = fetch_local_candidate_rows(
        expanded_query, effective_limit=effective_limit, config=config, source_types=source_types,
        enforce_query_domain=enforce_query_domain, tax_domains=tax_domains,
        detection_query=query,
    )
    return [
        {
            "rank": rank,
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "signature": str(row["signature"] or "") or None,
            "subject": str(row["subject"]),
            "source_type": str(row["source_type"]),
            "lexical_score": float(row["lexical_score"]),
        }
        for rank, row in enumerate(rows, start=1)
    ]


def rank_hybrid_local_candidates(
    rows: list[sqlite3.Row],
    *,
    query: str,
    effective_limit: int,
    config: RagConfig,
) -> list[RagChunk]:
    if not rows:
        return []

    if not query_targets_interpretation_procedure(query):
        non_procedural_rows = [
            row
            for row in rows
            if not (
                str(row["source_type"] or "") == "interpretation"
                and is_procedural_interpretation_chunk_text(str(row["chunk_text"] or ""))
            )
        ]
        if non_procedural_rows:
            rows = non_procedural_rows

    # Stage 2: inexpensive hybrid pre-ranking over the full recall pool.
    semantic_scores = [
        (
            row,
            semantic_score,
            build_legal_match_score(row, query=query),
            build_mechanism_match_score(row, query=query, config=config),
            build_pcc_interpretation_match_score(row, query=query),
            build_ksef_foreign_sale_match_score(row, query=query),
            build_ksef_outside_deduction_match_score(row, query=query),
            build_vat_dropshipping_ioss_match_score(row, query=query),
            build_shareholder_company_asset_sale_match_score(row, query=query),
            build_transformation_share_cost_match_score(row, query=query),
            build_small_taxpayer_foreign_vat_match_score(row, query=query),
        )
        for row, semantic_score in zip(
            rows, compute_hash_semantic_scores(rows, query=query, config=config)
        )
    ]

    lexical_ranks = {
        str(row["chunk_id"]): rank for rank, row in enumerate(rows, start=1)
    }
    semantic_ranks = {
        str(row["chunk_id"]): rank
        for rank, (row, _, _, _, _, _, _, _, _, _, _) in enumerate(
            sorted(
                semantic_scores,
                key=lambda item: (item[1], -int(item[0]["chunk_index"]), str(item[0]["chunk_id"])),
                reverse=True,
            ),
            start=1,
        )
    }

    preliminary_rows = sorted(
        semantic_scores,
        key=lambda item: (
            build_local_hybrid_score(
                lexical_rank=lexical_ranks[str(item[0]["chunk_id"])],
                semantic_rank=semantic_ranks[str(item[0]["chunk_id"])],
                config=config,
            ) + (config.legal_match_weight * item[2]) + (config.mechanism_match_weight * item[3])
                + (config.judgment_match_weight * build_judgment_metadata_match_score(item[0], query=query))
                + (0.25 * build_statute_match_score(item[0], query=query))
                + (0.35 * build_article_family_match_score(item[0], query=query))
                + build_subject_phrase_match_score(item[0], query=query)
                + build_interpretation_section_match_score(item[0])
                + build_direct_document_boost_score(item[0], query=query)
                + item[4]
                + item[5]
                + item[6]
                + item[7],
            item[5] + item[6] + item[7] + item[8] + item[9] + item[10],
            item[6],
            item[5],
            item[7],
            item[8],
            item[9],
            item[10],
            item[4],
            -item[1],
            item[2],
            -float(item[0]["lexical_score"]),
            str(item[0]["chunk_id"]),
        ),
        reverse=True,
    )

    # Stage 3: the cross-encoder sees only the strongest hybrid candidates.
    # This preserves broad recall while spending the expensive model budget on
    # legal near-misses that can realistically reach the final top-k.
    shortlist = preliminary_rows[: max(effective_limit, config.cross_encoder_candidate_limit)]
    judgment_only_shortlist = all(str(row["source_type"] or "") == "judgment" for row, _, _, _, _, _, _, _, _, _, _ in shortlist)
    cross_scores = None if judgment_only_shortlist else compute_cross_encoder_scores(
        [row for row, _, _, _, _, _, _, _, _, _, _ in shortlist], query=query, config=config
    )
    if cross_scores is None:
        ranked_rows = preliminary_rows
    else:
        cross_ranks = {
            str(row["chunk_id"]): rank
            for rank, ((row, _, _, _, _, _, _, _, _, _, _), _) in enumerate(
                sorted(
                    zip(shortlist, cross_scores),
                    key=lambda item: (item[1], str(item[0][0]["chunk_id"])),
                    reverse=True,
                ),
                start=1
            )
        }
        cross_weight = min(max(config.cross_encoder_weight, 0.0), 1.0)
        def cross_encoder_sort_key(item: tuple[sqlite3.Row, float, float, float, float, float, float, float, float, float, float]) -> tuple[float, int]:
            row, _, legal_match_score, mechanism_match_score, pcc_match_score, ksef_foreign_sale_match_score, ksef_outside_deduction_match_score, vat_dropshipping_ioss_match_score, shareholder_sale_match_score, transformation_share_cost_match_score, small_taxpayer_foreign_vat_match_score = item
            chunk_id = str(row["chunk_id"])
            statute_match_score = build_statute_match_score(row, query=query)
            family_match_score = build_article_family_match_score(row, query=query)
            preliminary_score = (
                build_local_hybrid_score(
                    lexical_rank=lexical_ranks[chunk_id],
                    semantic_rank=semantic_ranks[chunk_id],
                    config=config,
                )
                + (config.legal_match_weight * legal_match_score)
                + (config.mechanism_match_weight * mechanism_match_score)
                + (config.judgment_match_weight * build_judgment_metadata_match_score(row, query=query))
                + (0.25 * statute_match_score)
                + (0.35 * family_match_score)
                + build_subject_phrase_match_score(row, query=query)
                + build_interpretation_section_match_score(row)
                + build_direct_document_boost_score(row, query=query)
                + pcc_match_score
                + ksef_foreign_sale_match_score
                + ksef_outside_deduction_match_score
                + vat_dropshipping_ioss_match_score
                + shareholder_sale_match_score
                + transformation_share_cost_match_score
                + small_taxpayer_foreign_vat_match_score
            )
            # A reciprocal rank with an arbitrary 20-point offset flattened the
            # model signal so much that rank 1 and rank 20 differed by only a
            # few thousandths.  The cross-encoder could identify the correct
            # provision yet could not move it above lexical neighbours.  Map
            # its rank to [1, 0] within the actual shortlist instead.
            cross_rank_score = 1.0 - ((cross_ranks[chunk_id] - 1) / max(len(shortlist) - 1, 1))
            effective_cross_weight = resolve_cross_blend_weight(
                row,
                query=query,
                statute_match_score=statute_match_score,
                config=config,
            )
            return (
                ((1.0 - effective_cross_weight) * preliminary_score)
                + (effective_cross_weight * cross_rank_score),
                -cross_ranks[chunk_id],
                str(row["chunk_id"]),
            )

        ranked_rows = sorted(
            shortlist,
            key=cross_encoder_sort_key,
            reverse=True,
        )

        raw_leader = semantic_scores[0]
        raw_leader_document_id = str(raw_leader[0]["document_id"])
        final_window = list(ranked_rows[:effective_limit])
        final_document_ids = {str(row["document_id"]) for row, _, _, _, _, _, _, _, _, _, _ in final_window}
        if raw_leader_document_id not in final_document_ids:
            if len(final_window) < effective_limit:
                final_window.append(raw_leader)
            else:
                document_counts: dict[str, int] = {}
                for row, _, _, _, _, _, _, _, _, _, _ in final_window:
                    document_id = str(row["document_id"])
                    document_counts[document_id] = document_counts.get(document_id, 0) + 1
                replacement_index = len(final_window) - 1
                for index in range(len(final_window) - 1, -1, -1):
                    document_id = str(final_window[index][0]["document_id"])
                    if document_counts.get(document_id, 0) > 1:
                        replacement_index = index
                        break
                final_window[replacement_index] = raw_leader

            retained_chunk_ids = {str(row["chunk_id"]) for row, _, _, _, _, _, _, _, _, _, _ in final_window}
            ranked_rows = final_window + [
                item for item in ranked_rows if str(item[0]["chunk_id"]) not in retained_chunk_ids
            ]

    raw_leader = semantic_scores[0]
    raw_leader_document_id = str(raw_leader[0]["document_id"])
    final_window = list(ranked_rows[:effective_limit])
    final_document_ids = {str(row["document_id"]) for row, _, _, _, _, _, _, _, _, _, _ in final_window}
    if raw_leader_document_id not in final_document_ids:
        if len(final_window) < effective_limit:
            final_window.append(raw_leader)
        else:
            document_counts: dict[str, int] = {}
            for row, _, _, _, _, _, _, _, _, _, _ in final_window:
                document_id = str(row["document_id"])
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
            replacement_index = len(final_window) - 1
            for index in range(len(final_window) - 1, -1, -1):
                document_id = str(final_window[index][0]["document_id"])
                if document_counts.get(document_id, 0) > 1:
                    replacement_index = index
                    break
            final_window[replacement_index] = raw_leader

        retained_chunk_ids = {str(row["chunk_id"]) for row, _, _, _, _, _, _, _, _, _, _ in final_window}
        ranked_rows = final_window + [
            item for item in ranked_rows if str(item[0]["chunk_id"]) not in retained_chunk_ids
        ]

    ranked_rows = diversify_top_document_window(ranked_rows, effective_limit=effective_limit)

    return [
        RagChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            score=(
                build_local_hybrid_score(
                    lexical_rank=lexical_ranks[str(row["chunk_id"])],
                    semantic_rank=semantic_ranks[str(row["chunk_id"])],
                    config=config,
                )
                + (config.legal_match_weight * legal_match_score)
                + (config.mechanism_match_weight * mechanism_match_score)
                + (config.judgment_match_weight * build_judgment_metadata_match_score(row, query=query))
                + (0.25 * build_statute_match_score(row, query=query))
                + (0.35 * build_article_family_match_score(row, query=query))
                + build_subject_phrase_match_score(row, query=query)
                + build_interpretation_section_match_score(row)
                + pcc_match_score
                + ksef_foreign_sale_match_score
                + ksef_outside_deduction_match_score
                + shareholder_sale_match_score
                + transformation_share_cost_match_score
                + small_taxpayer_foreign_vat_match_score
            ),
            chunk_text=str(row["chunk_text"]),
            subject=str(row["subject"]),
            signature=str(row["signature"] or "") or None,
            published_date=str(row["published_date"] or "") or None,
            source_url=str(row["source_url"] or "") or None,
            category=str(row["category"] or "") or None,
            source=str(row["source"] or ""),
            source_type=str(row["source_type"] or "interpretation"),
            source_subtype=str(row["source_subtype"] or "") or None,
            authority=str(row["authority"] or "") or None,
            publication=str(row["publication"] or "") or None,
            legal_state_date=str(row["legal_state_date"] or "") or None,
            source_pages=[int(value) for value in json.loads(row["source_pages_json"] or "[]")],
            legal_provisions=[str(value) for value in json.loads(row["legal_provisions_json"] or "[]")],
        )
        for row, _, legal_match_score, mechanism_match_score, pcc_match_score, ksef_foreign_sale_match_score, ksef_outside_deduction_match_score, vat_dropshipping_ioss_match_score, shareholder_sale_match_score, transformation_share_cost_match_score, small_taxpayer_foreign_vat_match_score in ranked_rows[:effective_limit]
    ]


def search_chunks(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    if (source_types is None or "statute" in source_types) and query_is_direct_statute_lookup(query):
        direct_statutes = add_primary_source_fallback_chunks(query, [])
        if direct_statutes:
            return direct_statutes[: limit or get_rag_config().retrieval_limit]
    runtime = resolve_rag_runtime()
    if runtime.read_backend == "mysql":
        from app.mysql_rag import search_chunks_mysql

        return search_chunks_mysql(
            query,
            limit=limit,
            source_types=source_types,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
        )
    if runtime.read_backend == "supabase":
        from app.supabase_rag import search_chunks_supabase
        return search_chunks_supabase(query, limit=limit, source_types=source_types, tax_domains=tax_domains)
    axis_chunks, axes = _search_chunks_by_legal_axes(
        query,
        limit=limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
    )
    if axis_chunks:
        return axis_chunks
    if axes:
        fallback_query = axes[0].query if len(axes) == 1 else query
        return _search_chunks_single_query(
            fallback_query,
            limit=limit,
            source_types=source_types,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
        )
    return _search_chunks_single_query(
        query,
        limit=limit,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
    )


def search_chat_chunks(
    query: str,
    *,
    limit: Optional[int] = None,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> list[RagChunk]:
    if query_is_direct_statute_lookup(query):
        direct_statutes = add_primary_source_fallback_chunks(query, [])
        if direct_statutes:
            return direct_statutes[: limit or get_rag_config().retrieval_limit]
    if resolve_rag_runtime().read_backend == "mysql":
        from app.mysql_rag import search_chat_chunks_mysql

        return search_chat_chunks_mysql(
            query,
            limit=limit,
            include_interpretations=include_interpretations,
            include_judgments=include_judgments,
        )
    if resolve_rag_runtime().read_backend == "supabase":
        from app.supabase_rag import search_chunks_supabase
        source_types = {"statute"}
        if include_interpretations:
            source_types.add("interpretation")
        if include_judgments is not False:
            source_types.add("judgment")
        return search_chunks_supabase(query, limit=limit, source_types=source_types)
    """Retrieve complementary authority types for an application answer.

    A factual interpretation and the applicable provision answer different
    questions.  Searching them in one untyped top-k lets citation-heavy
    interpretations crowd statutes out of the prompt, so retrieve and mix
    both channels explicitly.
    """
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    judgment_requested_by_query = bool(JUDGMENT_INTENT_RE.search(query) or extract_judgment_signatures(query))
    include_judgments = True if include_judgments is None else include_judgments
    source_plan = build_legal_source_plan(
        query,
        include_interpretations=include_interpretations,
        include_judgments=include_judgments,
    )
    deterministic_statutes = retrieve_deterministic_statute_chunks(
        query,
        plan=source_plan,
        limit=max(effective_limit, len(source_plan.statute_targets) or 1),
        config=config,
    )
    judgment_only_context = bool(JUDGMENT_ONLY_CONTEXT_RE.search(query))
    statute_domains = resolve_statute_tax_domains(query)
    explicit_query_domains = bool(statute_domains)
    if judgment_only_context:
        judgment_limit = effective_limit
        statute_limit = 0
        interpretation_limit = 0
    elif not include_interpretations and not include_judgments:
        judgment_limit = 0
        interpretation_limit = 0
        statute_limit = effective_limit
    elif include_judgments and not include_interpretations:
        statute_limit = max(1, effective_limit - 1)
        interpretation_limit = 0
        judgment_limit = max(1, effective_limit - statute_limit)
    elif include_judgments:
        statute_limit = max(1, effective_limit // 4) if statute_domains else 1
        interpretation_limit = max(2, math.ceil(effective_limit * 0.5))
        interpretation_limit = min(interpretation_limit, max(effective_limit - statute_limit, 1))
        judgment_limit = max(1, effective_limit - statute_limit - interpretation_limit)
    else:
        judgment_limit = 0
        if query_targets_ksef_foreign_sale(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_ksef_b2c_invoice(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_ksef_outside_deduction(query):
            statute_limit = min(effective_limit - 1, max(5, math.ceil(effective_limit * 0.7)))
        elif query_targets_ksef_current_law(query):
            statute_limit = min(effective_limit - 1, max(6, math.ceil(effective_limit * 0.75)))
        elif query_targets_vat_dropshipping_ioss(query):
            statute_limit = min(effective_limit - 1, max(5, math.ceil(effective_limit * 0.7)))
        elif query_targets_private_vehicle_pit_expense(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_developer_land_sale(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_post_leasing_vehicle_gift_sale(query):
            statute_limit = min(effective_limit - 1, max(5, math.ceil(effective_limit * 0.7)))
        elif query_targets_estonian_cit_transformation_share_cost(query) or query_targets_estonian_cit_hidden_profit(query):
            statute_limit = min(effective_limit - 1, max(6, math.ceil(effective_limit * 0.75)))
        elif query_targets_spolka_komandytowa_cit_status(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_invoice_address_error(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_fixed_establishment_vat(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_family_foundation_mechanism(query):
            statute_limit = min(effective_limit - 1, max(5, math.ceil(effective_limit * 0.7)))
        elif query_targets_wht_pay_and_refund_services(query):
            statute_limit = min(effective_limit - 1, max(5, math.ceil(effective_limit * 0.7)))
        else:
            statute_limit = effective_limit if not include_interpretations else max(1, effective_limit // 2)
        interpretation_limit = max(1, effective_limit - statute_limit) if include_interpretations else 0
    if query_targets_crossborder_treaty_analysis(query) and statute_limit:
        desired_statute_limit = min(effective_limit, max(3, statute_limit))
        if desired_statute_limit > statute_limit:
            shift = desired_statute_limit - statute_limit
            statute_limit = desired_statute_limit
            if interpretation_limit:
                interpretation_limit = max(1, interpretation_limit - shift)
            if include_judgments and (statute_limit + interpretation_limit + judgment_limit) > effective_limit:
                judgment_limit = max(1, effective_limit - statute_limit - interpretation_limit)
    direct_interpretation_document_ids: list[str] = []
    if query_targets_ksef_b2c_invoice(query):
        direct_interpretation_document_ids.extend(("696263",))
    if query_targets_private_vehicle_pit_expense(query):
        direct_interpretation_document_ids.extend(("681556", "693582", "683152"))
    if query_targets_spolka_komandytowa_cit_status(query):
        direct_interpretation_document_ids.extend(("685379", "694316", "694267"))
    if query_targets_invoice_address_error(query):
        direct_interpretation_document_ids.extend(("694474",))
    if query_targets_fixed_establishment_vat(query):
        direct_interpretation_document_ids.extend(("695238", "694663", "694510", "693399"))
    if query_targets_family_foundation_mechanism(query):
        direct_interpretation_document_ids.extend(("695219", "692580", "685154", "692665", "692558", "692562", "691426", "691352"))
    if query_targets_wht_pay_and_refund_services(query):
        direct_interpretation_document_ids.extend(("691194", "690463", "685389", "679544", "695572", "695099", "694262", "687425"))
    if query_targets_ksef_foreign_sale(query):
        direct_interpretation_document_ids.extend(KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS)
    if query_targets_ksef_outside_deduction(query):
        direct_interpretation_document_ids.extend(KSEF_OUTSIDE_DEDUCTION_INTERPRETATION_DOCUMENT_IDS)
    if query_targets_ksef_correction_issue(query):
        direct_interpretation_document_ids.extend(("694474", "692135", "695412"))
    if query_targets_debt_assumption_effectiveness(query):
        direct_interpretation_document_ids.extend(DEBT_ASSUMPTION_INTERPRETATION_DOCUMENT_IDS)
    if query_targets_housing_relief_temporary_rental(query):
        direct_interpretation_document_ids.extend(HOUSING_RELIEF_TEMPORARY_RENTAL_INTERPRETATION_DOCUMENT_IDS)
    if query_targets_housing_relief_loan_repayment(query):
        direct_interpretation_document_ids.extend(("695380",))
    if query_targets_mortgage_settlement_refund(query):
        direct_interpretation_document_ids.extend(MORTGAGE_SETTLEMENT_INTERPRETATION_DOCUMENT_IDS)
    direct_interpretation_rows = []
    if interpretation_limit and direct_interpretation_document_ids:
        direct_chunk_limit = None if (
            query_targets_ksef_foreign_sale(query) or query_targets_ksef_outside_deduction(query)
        ) else 4
        direct_interpretation_rows = fetch_rows_by_document_ids(
            tuple(dict.fromkeys(direct_interpretation_document_ids)),
            config=config,
            source_type="interpretation",
            chunk_limit_per_document=direct_chunk_limit,
        )
        if query_targets_ksef_outside_deduction(query):
            direct_interpretation_rows = sort_ksef_outside_deduction_interpretation_rows(
                direct_interpretation_rows,
                query=expand_search_query(query),
            )
        elif query_targets_ksef_correction_issue(query):
            direct_interpretation_rows = sorted(
                direct_interpretation_rows,
                key=lambda row: (
                    -(
                        50 if str(row["document_id"]) == "694474" else
                        40 if str(row["document_id"]) in {"692135", "695412"} else
                        20 if "nota koryguj" in normalize_whitespace(
                            " ".join(
                                [
                                    str(row["subject"] or ""),
                                    str(row["question_text"] or ""),
                                    str(row["chunk_text"] or "")[:1600],
                                ]
                            )
                        ).lower() else 0
                    ),
                    int(row["chunk_index"]),
                    str(row["document_id"]),
                ),
            )
    if interpretation_limit and direct_interpretation_rows:
        interpretations = rank_hybrid_local_candidates(
            direct_interpretation_rows,
            query=expanded_query,
            effective_limit=interpretation_limit,
            config=config,
        )
    else:
        interpretations = search_chunks(
            query,
            limit=interpretation_limit,
            source_types={"interpretation"},
            enforce_query_domain=explicit_query_domains,
            tax_domains=statute_domains,
        ) if interpretation_limit else []
    interpretations = rerank_chunks_within_documents(
        interpretations,
        query=expanded_query,
        config=config,
        source_type="interpretation",
        max_chunks_per_document=4,
    )
    judgments = search_chunks(
        query,
        limit=judgment_limit,
        source_types={"judgment"},
        enforce_query_domain=explicit_query_domains,
        tax_domains=statute_domains,
    ) if include_judgments else []
    judgments = rerank_chunks_within_documents(
        judgments,
        query=expanded_query,
        config=config,
        source_type="judgment",
        max_chunks_per_document=4,
    )
    direct_treaty_rows = fetch_rows_by_subject_prefix(
        "UPO Polska - Hiszpania",
        config=config,
        source_type="statute",
    ) if query_targets_poland_spain_treaty(query) and statute_limit else []
    direct_germany_treaty_rows = fetch_rows_by_subject_prefix(
        "UPO Polska - Niemcy",
        config=config,
        source_type="statute",
    ) if query_targets_poland_germany_treaty(query) and statute_limit else []
    direct_ksef_bundle_rows = fetch_rows_by_document_ids(
        KSEF_CURRENT_BUNDLE_DOCUMENT_IDS,
        config=config,
        source_type="statute",
        chunk_limit_per_document=1,
    ) if query_targets_ksef_current_law(query) and statute_limit else []
    direct_family_foundation_bundle_rows = fetch_rows_by_document_ids(
        FAMILY_FOUNDATION_PRIMARY_BUNDLE_DOCUMENT_IDS,
        config=config,
        source_type="statute",
        chunk_limit_per_document=1,
    ) if query_targets_family_foundation_mechanism(query) and statute_limit else []
    statutes = [] if query_targets_ksef_foreign_sale(query) else (
        search_chunks(
            query,
            limit=statute_limit,
            source_types={"statute"},
            enforce_query_domain=True,
            tax_domains=statute_domains,
        ) if statute_limit else []
    )
    statutes = filter_treaty_country_chunks(statutes, query)
    statutes = order_chunks_by_statute_targets(
        dedupe_chunks_by_canonical_source([*deterministic_statutes, *statutes]),
        list(source_plan.statute_targets),
    )
    preferred_targets: list[tuple[str, str]] = []
    if query_targets_ksef_foreign_sale(query):
        preferred_targets.extend(KSEF_FOREIGN_SALE_STATUTE_TARGETS)
    if query_targets_ksef_current_law(query):
        preferred_targets.extend(build_ksef_current_law_statute_targets(query))
    if query_targets_ksef_b2c_invoice(query):
        preferred_targets.extend(build_ksef_b2c_invoice_statute_targets(query))
    if query_targets_ksef_outside_deduction(query):
        preferred_targets.extend(build_ksef_outside_deduction_statute_targets(query))
    if query_targets_ksef_correction_issue(query):
        preferred_targets.extend([("VAT", "106k")])
    if query_targets_private_vehicle_pit_expense(query):
        preferred_targets.extend(build_private_vehicle_pit_expense_statute_targets(query))
    if query_targets_spolka_komandytowa_cit_status(query):
        preferred_targets.extend(build_spolka_komandytowa_cit_status_statute_targets(query))
    if query_targets_invoice_address_error(query):
        preferred_targets.extend(build_invoice_address_error_statute_targets(query))
    if query_targets_fixed_establishment_vat(query):
        preferred_targets.extend(build_fixed_establishment_vat_statute_targets(query))
    if query_targets_family_foundation_mechanism(query):
        preferred_targets.extend(build_family_foundation_statute_targets(query))
    if query_targets_wht_pay_and_refund_services(query):
        preferred_targets.extend(build_wht_pay_and_refund_service_statute_targets(query))
    if query_targets_poland_germany_treaty(query):
        preferred_targets.extend(build_poland_germany_treaty_statute_targets(query))
    if query_targets_debt_assumption_effectiveness(query):
        preferred_targets.extend(build_debt_assumption_statute_targets(query))
    if query_targets_housing_relief_temporary_rental(query):
        preferred_targets.extend(build_housing_relief_temporary_rental_statute_targets(query))
    if query_targets_housing_relief_loan_repayment(query):
        preferred_targets.extend(build_housing_relief_loan_repayment_statute_targets(query))
    if query_targets_mortgage_settlement_refund(query):
        preferred_targets.extend(build_mortgage_settlement_refund_statute_targets(query))
    if query_targets_poland_spain_treaty(query):
        preferred_targets.extend(build_poland_spain_treaty_statute_targets(query))
    if query_targets_vat_dropshipping_ioss(query):
        preferred_targets.extend(build_vat_dropshipping_ioss_statute_targets(query))
    if query_targets_wht_crossborder_payments(query):
        preferred_targets.extend(build_wht_crossborder_payment_statute_targets(query))
    if query_targets_developer_land_sale(query):
        preferred_targets.extend(build_developer_land_sale_statute_targets(query))
    if query_targets_post_leasing_vehicle_gift_sale(query):
        preferred_targets.extend(build_post_leasing_vehicle_gift_sale_statute_targets(query))
    if query_targets_leased_movable_six_year_rule(query):
        preferred_targets.extend(build_leased_movable_six_year_statute_targets(query))
    if query_targets_gifted_asset_cost_basis(query):
        preferred_targets.extend(build_gifted_asset_cost_basis_statute_targets(query))
    if query_targets_spouse_gift_sd(query):
        preferred_targets.extend(build_spouse_gift_sd_statute_targets(query))
    if query_targets_estonian_cit_transformation_share_cost(query):
        preferred_targets.extend(build_transformation_share_cost_statute_targets(query))
    if query_targets_estonian_cit_hidden_profit(query):
        preferred_targets.extend(build_estonian_cit_hidden_profit_statute_targets(query))
    if query_targets_shareholder_company_asset_sale(query):
        preferred_targets.extend(build_shareholder_company_asset_sale_statute_targets(query))
    if query_targets_small_taxpayer_foreign_vat(query):
        preferred_targets.extend([("CIT", "4a"), ("CIT", "19"), ("CIT", "12")])
    for chunk in interpretations:
        for provision in chunk.legal_provisions:
            target = extract_statute_target_from_text(provision)
            if target and (not statute_domains or target[0] in statute_domains) and target not in preferred_targets:
                preferred_targets.append(target)
    _procedural_family_prefixes, procedural_exact_articles = detect_procedural_article_targets(query)
    if statute_limit and procedural_exact_articles:
        hinted_domains = statute_domains or {
            "VAT",
            "CIT",
            "PIT",
            "PCC",
            "AKCYZA",
            "ORDYNACJA",
            "NIERUCHOMOŚCI",
        }
        for domain in sorted(hinted_domains):
            for article_key in sorted(procedural_exact_articles):
                target = (domain, article_key)
                if target not in preferred_targets:
                    preferred_targets.append(target)
    hinted_statute_rows = fetch_statute_rows_by_targets(
        preferred_targets,
        config=config,
        limit=None if (
            query_targets_post_leasing_vehicle_gift_sale(query)
            or query_targets_leased_movable_six_year_rule(query)
            or query_targets_gifted_asset_cost_basis(query)
            or query_targets_spouse_gift_sd(query)
            or query_targets_estonian_cit_transformation_share_cost(query)
            or query_targets_estonian_cit_hidden_profit(query)
            or query_targets_poland_germany_treaty(query)
            or query_targets_ksef_outside_deduction(query)
            or query_targets_ksef_current_law(query)
            or query_targets_ksef_correction_issue(query)
            or query_targets_family_foundation_mechanism(query)
            or query_targets_debt_assumption_effectiveness(query)
            or query_targets_housing_relief_temporary_rental(query)
            or query_targets_housing_relief_loan_repayment(query)
            or query_targets_mortgage_settlement_refund(query)
        ) else statute_limit,
    ) if statute_limit else []
    if direct_ksef_bundle_rows or direct_family_foundation_bundle_rows or direct_treaty_rows or direct_germany_treaty_rows:
        hinted_statute_rows = [
            *direct_ksef_bundle_rows,
            *direct_family_foundation_bundle_rows,
            *hinted_statute_rows,
            *direct_treaty_rows,
            *direct_germany_treaty_rows,
        ]
    hinted_statutes = rank_hybrid_local_candidates(
        hinted_statute_rows,
        query=query if query_targets_poland_spain_treaty(query) else expanded_query,
        effective_limit=len(hinted_statute_rows) if (
            query_targets_post_leasing_vehicle_gift_sale(query)
            or query_targets_leased_movable_six_year_rule(query)
            or query_targets_gifted_asset_cost_basis(query)
            or query_targets_spouse_gift_sd(query)
            or query_targets_estonian_cit_transformation_share_cost(query)
            or query_targets_estonian_cit_hidden_profit(query)
            or query_targets_poland_germany_treaty(query)
            or query_targets_ksef_current_law(query)
            or query_targets_family_foundation_mechanism(query)
            or query_targets_debt_assumption_effectiveness(query)
            or query_targets_housing_relief_temporary_rental(query)
            or query_targets_housing_relief_loan_repayment(query)
            or query_targets_mortgage_settlement_refund(query)
        ) else statute_limit,
        config=config,
    ) if hinted_statute_rows else []
    hinted_statutes = filter_treaty_country_chunks(hinted_statutes, query)
    if (
        query_targets_poland_spain_treaty(query)
        or query_targets_post_leasing_vehicle_gift_sale(query)
        or query_targets_leased_movable_six_year_rule(query)
        or query_targets_gifted_asset_cost_basis(query)
        or query_targets_spouse_gift_sd(query)
        or query_targets_estonian_cit_transformation_share_cost(query)
        or query_targets_estonian_cit_hidden_profit(query)
        or query_targets_poland_germany_treaty(query)
        or query_targets_ksef_current_law(query)
        or query_targets_family_foundation_mechanism(query)
        or query_targets_debt_assumption_effectiveness(query)
        or query_targets_housing_relief_temporary_rental(query)
        or query_targets_housing_relief_loan_repayment(query)
        or query_targets_mortgage_settlement_refund(query)
    ):
        hinted_statutes = order_chunks_by_statute_targets(hinted_statutes, preferred_targets)

    semantic_statute_candidates = list(statutes)
    bundle_statute_candidates = list(hinted_statutes)
    if (
        query_targets_poland_germany_treaty(query)
        or query_targets_poland_spain_treaty(query)
        or query_targets_wht_pay_and_refund_services(query)
        or query_targets_estonian_cit_hidden_profit(query)
        or query_targets_ksef_current_law(query)
        or query_targets_family_foundation_mechanism(query)
    ):
        bundle_statute_candidates = sorted(
            enumerate(bundle_statute_candidates),
            key=lambda item: (item[1].subject.lower().startswith("upo polska"), item[0]),
        )
        bundle_statute_candidates = [chunk for _index, chunk in bundle_statute_candidates]
    merged_statutes: list[RagChunk] = []
    seen_statute_sources: set[str] = set()
    for chunk in semantic_statute_candidates:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_statute_sources:
            continue
        seen_statute_sources.add(canonical_source_id)
        merged_statutes.append(annotate_chunk_evidence_role(chunk, "governing_statute"))
        if len(merged_statutes) >= statute_limit:
            break
    bundle_cap = 2
    if (
        query_targets_poland_germany_treaty(query)
        or query_targets_poland_spain_treaty(query)
        or query_targets_wht_pay_and_refund_services(query)
        or query_targets_estonian_cit_hidden_profit(query)
    ):
        bundle_cap = 10
    bundle_limit = min(bundle_cap, max(0, len(bundle_statute_candidates)))
    bundle_statutes: list[RagChunk] = []
    if bundle_limit:
        for chunk in bundle_statute_candidates:
            canonical_source_id = chunk_canonical_source_id(chunk)
            if canonical_source_id in seen_statute_sources:
                continue
            seen_statute_sources.add(canonical_source_id)
            bundle_statutes.append(annotate_chunk_evidence_role(chunk, "bundle_source"))
            if len(bundle_statutes) >= bundle_limit:
                break

    primary_chunks = [*merged_statutes, *bundle_statutes]
    if source_plan.primary_required and not legal_source_plan_primary_satisfied(source_plan, primary_chunks):
        return primary_chunks[:effective_limit] if primary_chunks else []

    mixed: list[RagChunk] = [*primary_chunks]
    if include_judgments:
        mixed.extend(judgments)
    mixed.extend(interpretations)
    return mixed[: effective_limit + len(bundle_statutes)]


def inspect_search(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> RetrievalInspection:
    if (source_types is None or "statute" in source_types) and query_is_direct_statute_lookup(query):
        effective_limit = limit or get_rag_config().retrieval_limit
        chunks = add_primary_source_fallback_chunks(query, [])[:effective_limit]
        selected_context_chars = sum(len(chunk.chunk_text.strip()) for chunk in chunks)
        return RetrievalInspection(
            query=query,
            match_query=None,
            requested_limit=effective_limit,
            retrieved_count=len(chunks),
            selected_count=len(chunks),
            selected_context_chars=selected_context_chars,
            hits=[
                {
                    "rank": position,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "score": chunk.score,
                    "canonical_source_id": chunk_canonical_source_id(chunk),
                    "evidence_role": classify_chunk_evidence_role(chunk),
                    "subject": chunk.subject,
                    "signature": chunk.signature,
                    "published_date": chunk.published_date,
                    "source_url": chunk.source_url,
                    "category": chunk.category,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "source_subtype": chunk.source_subtype,
                    "authority": chunk.authority,
                    "publication": chunk.publication,
                    "legal_state_date": chunk.legal_state_date,
                    "source_pages": chunk.source_pages,
                    "legal_provisions": chunk.legal_provisions,
                    "chunk_chars": len(chunk.chunk_text),
                    "preview": chunk.chunk_text[:280].strip(),
                    "selected_for_context": True,
                }
                for position, chunk in enumerate(chunks, start=1)
            ],
            chunks=chunks,
            raw_candidate_pool=[],
        )
    if resolve_rag_runtime().read_backend == "mysql":
        from app.mysql_rag import inspect_search_mysql

        return inspect_search_mysql(
            query,
            limit=limit,
            source_types=source_types,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
        )
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    match_query = build_match_query(expanded_query)
    inferred_tax_domains = tax_domains or infer_retrieval_tax_domains(query) or None
    _, candidate_rows = fetch_local_candidate_rows(
        expanded_query,
        effective_limit=effective_limit,
        config=config,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain or bool(inferred_tax_domains),
        tax_domains=inferred_tax_domains,
        detection_query=query,
    )
    chunks = rank_hybrid_local_candidates(
        candidate_rows,
        query=expanded_query,
        effective_limit=effective_limit,
        config=config,
    )
    selected_chunks = select_diverse_chunks(chunks)
    selected_context_chars = sum(len(chunk.chunk_text.strip()) for chunk in selected_chunks)

    return RetrievalInspection(
        query=query,
        match_query=match_query,
        requested_limit=effective_limit,
        retrieved_count=len(chunks),
        selected_count=len(selected_chunks),
        selected_context_chars=selected_context_chars,
        hits=[
            {
                "rank": position,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "canonical_source_id": chunk_canonical_source_id(chunk),
                "evidence_role": classify_chunk_evidence_role(chunk),
                "subject": chunk.subject,
                "signature": chunk.signature,
                "published_date": chunk.published_date,
                "source_url": chunk.source_url,
                "category": chunk.category,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "source_subtype": chunk.source_subtype,
                "authority": chunk.authority,
                "publication": chunk.publication,
                "legal_state_date": chunk.legal_state_date,
                "source_pages": chunk.source_pages,
                "legal_provisions": chunk.legal_provisions,
                "chunk_chars": len(chunk.chunk_text),
                "preview": chunk.chunk_text[:280].strip(),
                "selected_for_context": chunk in selected_chunks,
            }
            for position, chunk in enumerate(chunks, start=1)
        ],
        chunks=chunks,
        raw_candidate_pool=[
            {
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "signature": str(row["signature"] or "") or None,
                "subject": str(row["subject"]),
                "source_type": str(row["source_type"]),
                "legal_provisions": [str(value) for value in json.loads(row["legal_provisions_json"] or "[]")],
                "category": str(row["category"] or "") or None,
                "tax_domain": str(row["tax_domain"] or "") or None,
                "keywords": [str(value) for value in json.loads(row["keywords_json"] or "[]")],
                "issues": [str(value) for value in json.loads(row["issues_json"] or "[]")],
                "law_tags": [str(value) for value in json.loads(row["law_tags_json"] or "[]")],
                "lexical_score": float(row["lexical_score"]),
                "legal_match_score": build_legal_match_score(row, query=expanded_query),
                "mechanism_match_score": build_mechanism_match_score(row, query=expanded_query, config=config),
                "judgment_match_score": build_judgment_match_score(row, query=expanded_query),
                "judgment_result_match_score": build_judgment_result_match_score(row, query=expanded_query),
                "judgment_topic_phrase_score": build_judgment_topic_phrase_score(row, query=expanded_query),
                "judgment_metadata_match_score": build_judgment_metadata_match_score(row, query=expanded_query),
                "article_family_match_score": build_article_family_match_score(row, query=expanded_query),
                "statute_match_score": build_statute_match_score(row, query=expanded_query),
                "shareholder_company_asset_sale_match_score": build_shareholder_company_asset_sale_match_score(row, query=expanded_query),
                "transformation_share_cost_match_score": build_transformation_share_cost_match_score(row, query=expanded_query),
                "small_taxpayer_foreign_vat_match_score": build_small_taxpayer_foreign_vat_match_score(row, query=expanded_query),
                "canonical_source_id": f"{str(row['source_type'] or '').lower()}:{str(row['signature'] or '') or str(row['document_id'])}:{str(row['chunk_index'])}",
                "preview": str(row["chunk_text"] or "")[:280].strip(),
            }
            for rank, row in enumerate(candidate_rows, start=1)
        ],
    )


def select_diverse_chunks(chunks: list[RagChunk], *, max_per_document: Optional[int] = None) -> list[RagChunk]:
    if not chunks:
        return []

    config = get_rag_config()
    per_document_limit = max(1, max_per_document or config.retrieval_max_chunks_per_document)
    selected: list[RagChunk] = []
    document_counts: dict[str, int] = {}
    seen_canonical_sources: set[str] = set()

    for chunk in chunks:
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen_canonical_sources:
            continue
        current = document_counts.get(chunk.document_id, 0)
        if current >= per_document_limit:
            continue
        selected.append(chunk)
        document_counts[chunk.document_id] = current + 1
        seen_canonical_sources.add(canonical_source_id)

    return selected


def select_context_document_ids(chunks: list[RagChunk], *, limit: Optional[int] = None) -> list[str]:
    config = get_rag_config()
    document_limit = max(1, limit or config.document_context_document_limit)
    document_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        document_id = str(chunk.document_id or "").strip()
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        document_ids.append(document_id)
        if len(document_ids) >= document_limit:
            break
    return document_ids


def merge_chunk_texts_in_order(chunks: list[str]) -> str:
    merged = ""
    for raw_chunk in chunks:
        chunk = str(raw_chunk or "").strip()
        if not chunk:
            continue
        if not merged:
            merged = chunk
            continue

        overlap_size = 0
        max_overlap = min(500, len(merged), len(chunk))
        for size in range(max_overlap, 40, -1):
            if merged[-size:] == chunk[:size]:
                overlap_size = size
                break
        if overlap_size:
            merged = merged + chunk[overlap_size:]
        else:
            merged = merged + "\n\n" + chunk
    return merged.strip()


def row_source_pages(row: sqlite3.Row | dict[str, Any]) -> list[int]:
    return [int(value) for value in json.loads(row["source_pages_json"] or "[]")]


def row_legal_provisions(row: sqlite3.Row | dict[str, Any]) -> list[str]:
    return [str(value) for value in json.loads(row["legal_provisions_json"] or "[]")]


def build_document_context_from_rows(
    rows: list[sqlite3.Row] | list[dict[str, Any]],
    *,
    ordered_document_ids: list[str],
    seed_chunks: list[RagChunk],
) -> list[RagDocumentContext]:
    if not rows:
        return []

    seed_chunk_ids_by_document: dict[str, list[str]] = {}
    for chunk in seed_chunks:
        seed_chunk_ids_by_document.setdefault(chunk.document_id, []).append(chunk.chunk_id)

    rows_by_document: dict[str, list[sqlite3.Row | dict[str, Any]]] = {}
    for row in rows:
        rows_by_document.setdefault(str(row["document_id"]), []).append(row)

    documents: list[RagDocumentContext] = []
    for document_id in ordered_document_ids:
        document_rows = sorted(
            rows_by_document.get(document_id, []),
            key=lambda row: int(row["chunk_index"]),
        )
        if not document_rows:
            continue
        first = document_rows[0]
        documents.append(
            RagDocumentContext(
                document_id=document_id,
                subject=str(first["subject"]),
                signature=str(first["signature"] or "") or None,
                published_date=str(first["published_date"] or "") or None,
                source_url=str(first["source_url"] or "") or None,
                category=str(first["category"] or "") or None,
                source=str(first["source"] or ""),
                source_type=str(first["source_type"] or "interpretation"),
                source_subtype=str(first["source_subtype"] or "") or None,
                authority=str(first["authority"] or "") or None,
                publication=str(first["publication"] or "") or None,
                legal_state_date=str(first["legal_state_date"] or "") or None,
                source_pages=row_source_pages(first),
                legal_provisions=row_legal_provisions(first),
                text=merge_chunk_texts_in_order([str(row["chunk_text"]) for row in document_rows]),
                seed_chunk_ids=seed_chunk_ids_by_document.get(document_id, []),
            )
        )
    return documents


def fetch_document_contexts(document_ids: list[str], *, seed_chunks: list[RagChunk]) -> list[RagDocumentContext]:
    if resolve_rag_runtime().read_backend == "mysql":
        from app.mysql_rag import fetch_document_contexts_mysql

        return fetch_document_contexts_mysql(document_ids, seed_chunks=seed_chunks)

    config = get_rag_config()
    clean_ids = [str(document_id).strip() for document_id in document_ids if str(document_id).strip()]
    if not clean_ids or not config.db_path.exists():
        return []

    placeholders = ", ".join("?" for _ in clean_ids)
    connection = get_connection(config.db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT
                c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                d.subject, d.signature, d.published_date, d.source_url, d.category,
                d.legal_provisions_json, d.source, d.source_type, d.source_subtype,
                d.authority, d.publication, d.legal_state_date, d.source_pages_json
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.document_id IN ({placeholders})
            ORDER BY c.document_id ASC, c.chunk_index ASC
            """,
            tuple(clean_ids),
        ).fetchall()
    finally:
        connection.close()

    return build_document_context_from_rows(rows, ordered_document_ids=clean_ids, seed_chunks=seed_chunks)


def build_document_context_block(chunks: list[RagChunk]) -> str:
    config = get_rag_config()
    document_ids = select_context_document_ids(chunks, limit=config.document_context_document_limit)
    try:
        documents = fetch_document_contexts(document_ids, seed_chunks=chunks)
    except Exception:
        documents = []
    existing_document_ids = {document.document_id for document in documents}
    fallback_documents: list[RagDocumentContext] = []
    for document_id in document_ids:
        if document_id in existing_document_ids:
            continue
        document_chunks = [chunk for chunk in chunks if chunk.document_id == document_id]
        if not document_chunks:
            continue
        first = document_chunks[0]
        fallback_documents.append(
            RagDocumentContext(
                document_id=document_id,
                subject=first.subject,
                signature=first.signature,
                published_date=first.published_date,
                source_url=first.source_url,
                category=first.category,
                source=first.source,
                source_type=first.source_type,
                source_subtype=first.source_subtype,
                authority=first.authority,
                publication=first.publication,
                legal_state_date=first.legal_state_date,
                source_pages=first.source_pages,
                legal_provisions=first.legal_provisions,
                text=merge_chunk_texts_in_order(
                    [
                        chunk.chunk_text
                        for chunk in sorted(document_chunks, key=lambda item: item.chunk_index)
                    ]
                ),
                seed_chunk_ids=[chunk.chunk_id for chunk in document_chunks],
            )
        )
    if fallback_documents:
        by_document_id = {document.document_id: document for document in [*documents, *fallback_documents]}
        documents = [by_document_id[document_id] for document_id in document_ids if document_id in by_document_id]
    if not documents:
        return ""

    parts: list[str] = []
    used_chars = 0
    for position, document in enumerate(documents, start=1):
        block = (
            f"[Dokument {position}]\n"
            f"source_type: {document.source_type}\n"
            f"source_subtype: {document.source_subtype or 'brak'}\n"
            f"authority: {document.authority or 'brak'}\n"
            f"document_id: {document.document_id}\n"
            f"seed_chunk_ids: {', '.join(document.seed_chunk_ids) or 'brak'}\n"
            f"signature: {document.signature or 'brak'}\n"
            f"published_date: {document.published_date or 'brak'}\n"
            f"publication: {document.publication or 'brak'}\n"
            f"legal_state_date: {document.legal_state_date or 'brak'}\n"
            f"source_pages: {', '.join(str(page) for page in document.source_pages) or 'brak'}\n"
            f"legal_provisions: {', '.join(document.legal_provisions) or 'brak'}\n"
            f"subject: {document.subject}\n"
            f"source_url: {document.source_url or 'brak'}\n"
            f"pełna_treść_dokumentu:\n{document.text}"
        )
        if used_chars and used_chars + len(block) > config.document_context_max_chars:
            break
        parts.append(block)
        used_chars += len(block)

    return "\n\n".join(parts)


def build_answer_context_block(chunks: list[RagChunk]) -> str:
    config = get_rag_config()
    if config.document_context_enabled:
        document_context = build_document_context_block(chunks)
        if document_context:
            return document_context
    return build_context_block(chunks)


def build_context_block(chunks: list[RagChunk]) -> str:
    config = get_rag_config()
    parts: list[str] = []
    used_chars = 0

    for position, chunk in enumerate(select_diverse_chunks(chunks), start=1):
        block = (
            f"[Źródło {position}]\n"
            f"source_type: {chunk.source_type}\n"
            f"source_subtype: {chunk.source_subtype or 'brak'}\n"
            f"authority: {chunk.authority or 'brak'}\n"
            f"document_id: {chunk.document_id}\n"
            f"signature: {chunk.signature or 'brak'}\n"
            f"published_date: {chunk.published_date or 'brak'}\n"
            f"publication: {chunk.publication or 'brak'}\n"
            f"legal_state_date: {chunk.legal_state_date or 'brak'}\n"
            f"source_pages: {', '.join(str(page) for page in chunk.source_pages) or 'brak'}\n"
            f"subject: {chunk.subject}\n"
            f"source_url: {chunk.source_url or 'brak'}\n"
            f"fragment:\n{chunk.chunk_text.strip()}"
        )
        if used_chars and used_chars + len(block) > config.max_context_chars:
            break
        parts.append(block)
        used_chars += len(block)

    return "\n\n".join(parts)


def list_citations(chunks: list[RagChunk]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for chunk in select_diverse_chunks(chunks):
        canonical_source_id = chunk_canonical_source_id(chunk)
        if canonical_source_id in seen:
            continue
        seen.add(canonical_source_id)
        lines.append(
            f"- [{chunk.source_type}{':' + chunk.source_subtype if chunk.source_subtype else ''}] "
            f"{chunk.signature or chunk.subject} | {chunk.publication or chunk.published_date or 'brak daty'} | {chunk.source_url or 'brak URL'}"
        )
    return "\n".join(lines)


def index_exists() -> bool:
    if resolve_rag_runtime().read_backend == "mysql":
        from app.mysql_rag import index_exists_mysql

        return index_exists_mysql()
    config = get_rag_config()
    return config.db_path.exists()


def local_index_needs_refresh() -> bool:
    config = get_rag_config()
    if not config.db_path.exists():
        return True

    try:
        db_mtime = config.db_path.stat().st_mtime
    except OSError:
        return True

    source_paths = [source.path for source in iter_configured_corpus_sources(config)]
    for path in source_paths:
        try:
            if path.exists() and path.stat().st_mtime > db_mtime:
                return True
        except OSError:
            continue
    return False


def ensure_local_index_ready() -> None:
    if not local_index_needs_refresh():
        return
    with _index_refresh_lock:
        if local_index_needs_refresh():
            reindex_corpus(force=False)


def is_supabase_sync_enabled() -> bool:
    return os.getenv("ALITIGATOR_RAG_SUPABASE_SYNC", "false").lower() in {"1", "true", "yes"}


def is_supabase_sync_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))


def get_supabase_target() -> tuple[str, str, str]:
    schema = os.getenv("ALITIGATOR_RAG_SUPABASE_SCHEMA", "public")
    documents_table = os.getenv("ALITIGATOR_RAG_SUPABASE_DOCUMENTS_TABLE", "eureka_interpretations")
    chunks_table = os.getenv("ALITIGATOR_RAG_SUPABASE_CHUNKS_TABLE", "eureka_chunks")
    return schema, documents_table, chunks_table


def collect_documents_for_sync(connection: sqlite3.Connection, document_ids: list[str]) -> list[dict[str, Any]]:
    if not document_ids:
        return []

    placeholders = ",".join("?" for _ in document_ids)
    rows = connection.execute(
        f"""
        SELECT
            document_id,
            content_sha256,
            subject,
            signature,
            published_date,
            source_url,
            category,
            keywords_json,
            legal_provisions_json,
            issues_json,
            law_tags_json,
            indexed_at
        FROM documents
        WHERE document_id IN ({placeholders})
        """,
        tuple(document_ids),
    ).fetchall()

    return [
        {
            "document_id": row["document_id"],
            "content_sha256": row["content_sha256"],
            "subject": row["subject"],
            "signature": row["signature"],
            "published_date": row["published_date"],
            "source_url": row["source_url"],
            "category": row["category"],
            "keywords": json.loads(row["keywords_json"]),
            "legal_provisions": json.loads(row["legal_provisions_json"]),
            "issues": json.loads(row["issues_json"]),
            "law_tags": json.loads(row["law_tags_json"]),
            "indexed_at": row["indexed_at"],
        }
        for row in rows
    ]


def collect_chunks_for_sync(connection: sqlite3.Connection, document_ids: list[str]) -> list[dict[str, Any]]:
    if not document_ids:
        return []

    placeholders = ",".join("?" for _ in document_ids)
    rows = connection.execute(
        f"""
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.chunk_text,
            c.chunk_chars,
            d.signature,
            d.published_date,
            d.source_url,
            d.subject,
            d.category
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE c.document_id IN ({placeholders})
        ORDER BY c.document_id, c.chunk_index
        """,
        tuple(document_ids),
    ).fetchall()

    return [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
            "chunk_chars": row["chunk_chars"],
            "signature": row["signature"],
            "published_date": row["published_date"],
            "source_url": row["source_url"],
            "subject": row["subject"],
            "category": row["category"],
        }
        for row in rows
    ]


def chunked(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def sync_indexed_documents_to_supabase(document_ids: list[str]) -> dict[str, int]:
    if not document_ids:
        return {"documents": 0, "chunks": 0}
    if not is_supabase_sync_configured():
        raise RuntimeError("Supabase sync requested, but SUPABASE_URL or SUPABASE_SECRET_KEY is missing")

    config = get_rag_config()
    schema, documents_table, chunks_table = get_supabase_target()
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY", "")
    headers = {
        "apikey": supabase_secret_key,
        "Authorization": f"Bearer {supabase_secret_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    connection = get_connection(config.db_path)
    try:
        documents_payload = collect_documents_for_sync(connection, document_ids)
        chunks_payload = collect_chunks_for_sync(connection, document_ids)
    finally:
        connection.close()

    with httpx.Client(timeout=60.0, headers=headers) as client:
        delete_response = client.delete(
            f"{supabase_url}/rest/v1/{chunks_table}",
            params={"document_id": f"in.({','.join(document_ids)})"},
            headers={**headers, "Accept-Profile": schema, "Content-Profile": schema},
        )
        delete_response.raise_for_status()

        for batch in chunked(documents_payload, 200):
            response = client.post(
                f"{supabase_url}/rest/v1/{documents_table}",
                params={"on_conflict": "document_id"},
                json=batch,
                headers={**headers, "Accept-Profile": schema, "Content-Profile": schema},
            )
            response.raise_for_status()

        for batch in chunked(chunks_payload, 500):
            response = client.post(
                f"{supabase_url}/rest/v1/{chunks_table}",
                params={"on_conflict": "chunk_id"},
                json=batch,
                headers={**headers, "Accept-Profile": schema, "Content-Profile": schema},
            )
            response.raise_for_status()

    return {"documents": len(documents_payload), "chunks": len(chunks_payload)}
