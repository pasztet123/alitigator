from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx


APP_DIR = Path(__file__).resolve().parent
API_DIR = APP_DIR.parent
DEFAULT_PROCESSED_PATH = API_DIR / "data" / "processed" / "eureka_interpretations.jsonl"
DEFAULT_LAW_SOURCE_PATHS = (
    API_DIR / "data" / "laws" / "processed" / "excise_act_DU_2026_412.jsonl",
    API_DIR / "data" / "laws" / "processed" / "vat_act_DU_2025_775.jsonl",
    API_DIR / "data" / "laws" / "processed" / "cit_act_DU_2026_554.jsonl",
    API_DIR / "data" / "laws" / "processed" / "pit_act_DU_2026_592.jsonl",
    API_DIR / "data" / "laws" / "processed" / "pcc_act_DU_2026_191.jsonl",
    API_DIR / "data" / "laws" / "processed" / "inheritance_gift_tax_act_DU_2026_478.jsonl",
    API_DIR / "data" / "laws" / "processed" / "tax_ordinance_DU_2026_622.jsonl",
    API_DIR / "data" / "laws" / "processed" / "local_taxes_act_DU_2025_707.jsonl",
    API_DIR / "data" / "laws" / "processed" / "tax_treaties_core.jsonl",
    API_DIR / "data" / "processed" / "cbosa_nsa_fsk_judgments.jsonl",
)
DEFAULT_RAG_DB_PATH = API_DIR / "data" / "processed" / "eureka_rag.sqlite3"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

WHITESPACE_RE = re.compile(r"\s+")
QUERY_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}")
EMBEDDING_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{2,}")
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
ARTICLE_ID_RE = re.compile(r"\bart\.\s*(\d+)([a-z]*)\b", re.IGNORECASE)
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
    "vat": ("vat", "ksef", "faktur", "odliczen", "sprzedaż", "sprzedaz"),
    "cit": ("cit", "estońsk", "estonsk", "spółk", "spolk", "holding"),
    "pit": ("pit", "ryczałt", "ryczalt", "ulga", "rezydenc"),
    "pcc": ("pcc", "czynności", "czynnosci", "aport", "współwłas", "wspolwlas"),
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


@dataclass(frozen=True)
class RagConfig:
    processed_path: Path
    additional_source_paths: tuple[Path, ...]
    db_path: Path
    chunk_target_chars: int
    chunk_overlap_chars: int
    retrieval_limit: int
    max_context_chars: int
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
            r"szwajcar\w*|austri\w*|wielk\w* bryt\w*|uk\b|usa\b|stan\w* zjednoczon\w*|czech\w*)\b",
            normalized,
        )
    )
    return (has_crossborder_marker and (has_income_tax_angle or has_country_marker)) or (
        has_country_marker and has_income_tax_angle
    )


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
    query_domains = {domain.upper() for domain in detect_domains(query)}
    requested_domains = query_domains or {"VAT", "CIT", "PIT", "PCC"}
    mentions_real_estate = bool(re.search(r"\b(nieruchomo\w*|lokal\w*|mieszkani\w*|mieszkan\w*|budynek\w*|grunt\w*)\b", normalized))
    mentions_preferential_price = bool(
        re.search(r"\b(preferencyjn\w*|rynkow\w*|poniżej\b|ponizej\b|niższ\w*|nizsz\w*|zaniż\w*|zaniz\w*|częściowo nieodpłatn\w*|czesciowo nieodplatn\w*)\b", normalized)
    )

    preferred_targets: list[tuple[str, str]] = []
    if "CIT" in requested_domains:
        preferred_targets.extend([("CIT", "11c"), ("CIT", "14")])
    if "PIT" in requested_domains:
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
    if article_key in {"11", "11a", "11c", "11d", "11e", "11t", "12", "14", "15", "16", "16g", "17", "24", "28m", "29a", "32", "43", "1", "2", "4", "6", "7"}:
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
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

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


def index_record(connection: sqlite3.Connection, record: dict[str, Any], config: RagConfig) -> int:
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return 0

    delete_document(connection, document_id)

    document_text = clean_document_text(record)
    if not document_text:
        return 0

    chunks = [document_text] if record.get("pre_chunked") else split_into_chunks(
        document_text,
        target_chars=config.chunk_target_chars,
        overlap_chars=config.chunk_overlap_chars,
    )
    chunks = filter_index_chunks(record, chunks)

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
            record.get("content_sha256"),
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
        cursor = connection.execute(
            """
            INSERT INTO chunks (document_id, chunk_id, chunk_index, chunk_text, chunk_chars)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, chunk_id, chunk_index, chunk_text, len(chunk_text)),
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
        inserted += 1

    return inserted


def reindex_corpus(*, limit: Optional[int] = None, force: bool = False) -> dict[str, Any]:
    if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}:
        from app.mysql_rag import reindex_corpus_mysql

        return reindex_corpus_mysql(limit=limit, force=force)
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
        source_paths = (config.processed_path, *config.additional_source_paths)
        for source_path in source_paths:
            for record in iter_processed_records(source_path):
                if limit is not None and processed >= limit:
                    break

                processed += 1
                document_id = str(record.get("document_id") or "").strip()
                if not document_id:
                    skipped += 1
                    continue

                current_sha = str(record.get("content_sha256") or "")
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
    normalized = ARTICLE_SPLIT_SUFFIX_RE.sub(lambda match: f"art. {match.group(1)}{match.group(2).lower()}", normalized)
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


def detect_procedural_article_targets(query: str) -> tuple[set[str], set[str]]:
    family_prefixes: set[str] = set()
    exact_articles: set[str] = set()
    for pattern, families, exacts, _ in STATUTE_PROCEDURAL_RULES:
        if pattern.search(query):
            family_prefixes.update(families)
            exact_articles.update(exacts)
    return family_prefixes, exact_articles


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
    domains = {domain.upper() for domain in detect_domains(query)}
    if "WHT" in domains:
        domains.update({"CIT", "PIT"})
    if query_targets_crossborder_treaty_analysis(query):
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
    if str(row["source_type"] or "") != "interpretation":
        return 0.0
    return build_interpretation_section_score(str(row["chunk_text"] or ""))


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
    ranked_rows: list[tuple[sqlite3.Row, float, float, float, float, float, float, float]],
    *,
    effective_limit: int,
) -> list[tuple[sqlite3.Row, float, float, float, float, float, float, float]]:
    if len(ranked_rows) <= effective_limit:
        return ranked_rows

    top_window = ranked_rows[:effective_limit]
    if len({str(item[0]["document_id"]) for item in top_window}) == len(top_window):
        return ranked_rows

    diversified: list[tuple[sqlite3.Row, float, float, float, float, float, float, float]] = []
    deferred: list[tuple[sqlite3.Row, float, float, float, float, float, float, float]] = []
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
    if re.search(r"\b(niemc\w*|niderland\w*|holand\w*|luksemburg\w*|franc\w*|irland\w*|szwajcar\w*|austri\w*|wielk\w* bryt\w*|uk\b|usa\b|stan\w* zjednoczon\w*|czech\w*)\b", normalized_query):
        country_hits = re.findall(r"\b(austria|czechy|francja|irlandia|luksemburg|niderlandy|niemcy|szwajcaria|usa|wielka brytania)\b", candidate_text)
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
                LIMIT ?
                """,
                (*direct_values, candidate_limit),
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

    target_clauses = ["(UPPER(tax_domain) = ? AND legal_provisions_json LIKE ?)" for _ in targets]
    target_values: list[str] = []
    for domain, article_key in targets:
        target_values.extend((domain.upper(), f'%"art. {article_key}"%'))

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
              AND c.chunk_index = 0
            """,
            tuple(document_ids),
        ).fetchall()
    finally:
        connection.close()

    order = {(domain.upper(), article_key): position for position, (domain, article_key) in enumerate(targets)}

    def row_sort_key(row: sqlite3.Row) -> tuple[int, str]:
        article_key = extract_primary_article_key(row)
        domain = str(row["tax_domain"] or "").upper()
        return order.get((domain, article_key), len(order)), str(row["subject"] or "")

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
    haystack = f"{subject} {publication} {chunk.source_url or ''}".lower()
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
    return ""


def order_chunks_by_statute_targets(chunks: list[RagChunk], targets: list[tuple[str, str]]) -> list[RagChunk]:
    if not chunks or not targets:
        return chunks

    order = {(domain.upper(), article_key): position for position, (domain, article_key) in enumerate(targets)}

    def sort_key(chunk: RagChunk) -> tuple[int, float, str]:
        article_key = extract_article_key_from_text(chunk.legal_provisions[0] if chunk.legal_provisions else "")
        domain = infer_chunk_tax_domain(chunk)
        return order.get((domain, article_key), len(order)), -chunk.score, chunk.subject

    return sorted(chunks, key=sort_key)


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
            build_shareholder_company_asset_sale_match_score(row, query=query),
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
        for rank, (row, _, _, _, _, _, _, _) in enumerate(
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
            + item[4]
            + item[5],
            item[6] + item[7],
            item[7],
            item[5],
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
    judgment_only_shortlist = all(str(row["source_type"] or "") == "judgment" for row, _, _, _, _, _, _, _ in shortlist)
    cross_scores = None if judgment_only_shortlist else compute_cross_encoder_scores(
        [row for row, _, _, _, _, _, _, _ in shortlist], query=query, config=config
    )
    if cross_scores is None:
        ranked_rows = preliminary_rows
    else:
        cross_ranks = {
            str(row["chunk_id"]): rank
            for rank, ((row, _, _, _, _, _, _, _), _) in enumerate(
                sorted(
                    zip(shortlist, cross_scores),
                    key=lambda item: (item[1], str(item[0][0]["chunk_id"])),
                    reverse=True,
                ),
                start=1
            )
        }
        cross_weight = min(max(config.cross_encoder_weight, 0.0), 1.0)
        def cross_encoder_sort_key(item: tuple[sqlite3.Row, float, float, float, float, float, float, float]) -> tuple[float, int]:
            row, _, legal_match_score, mechanism_match_score, pcc_match_score, ksef_match_score, shareholder_sale_match_score, small_taxpayer_foreign_vat_match_score = item
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
                + pcc_match_score
                + ksef_match_score
                + shareholder_sale_match_score
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
        final_document_ids = {str(row["document_id"]) for row, _, _, _, _, _, _, _ in final_window}
        if raw_leader_document_id not in final_document_ids:
            if len(final_window) < effective_limit:
                final_window.append(raw_leader)
            else:
                document_counts: dict[str, int] = {}
                for row, _, _, _, _, _, _, _ in final_window:
                    document_id = str(row["document_id"])
                    document_counts[document_id] = document_counts.get(document_id, 0) + 1
                replacement_index = len(final_window) - 1
                for index in range(len(final_window) - 1, -1, -1):
                    document_id = str(final_window[index][0]["document_id"])
                    if document_counts.get(document_id, 0) > 1:
                        replacement_index = index
                        break
                final_window[replacement_index] = raw_leader

            retained_chunk_ids = {str(row["chunk_id"]) for row, _, _, _, _, _, _, _ in final_window}
            ranked_rows = final_window + [
                item for item in ranked_rows if str(item[0]["chunk_id"]) not in retained_chunk_ids
            ]

    raw_leader = semantic_scores[0]
    raw_leader_document_id = str(raw_leader[0]["document_id"])
    final_window = list(ranked_rows[:effective_limit])
    final_document_ids = {str(row["document_id"]) for row, _, _, _, _, _, _, _ in final_window}
    if raw_leader_document_id not in final_document_ids:
        if len(final_window) < effective_limit:
            final_window.append(raw_leader)
        else:
            document_counts: dict[str, int] = {}
            for row, _, _, _, _, _, _, _ in final_window:
                document_id = str(row["document_id"])
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
            replacement_index = len(final_window) - 1
            for index in range(len(final_window) - 1, -1, -1):
                document_id = str(final_window[index][0]["document_id"])
                if document_counts.get(document_id, 0) > 1:
                    replacement_index = index
                    break
            final_window[replacement_index] = raw_leader

        retained_chunk_ids = {str(row["chunk_id"]) for row, _, _, _, _, _, _, _ in final_window}
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
                + ksef_match_score
                + shareholder_sale_match_score
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
        for row, _, legal_match_score, mechanism_match_score, pcc_match_score, ksef_match_score, shareholder_sale_match_score, small_taxpayer_foreign_vat_match_score in ranked_rows[:effective_limit]
    ]


def search_chunks(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}:
        from app.mysql_rag import search_chunks_mysql

        return search_chunks_mysql(
            query,
            limit=limit,
            source_types=source_types,
            enforce_query_domain=enforce_query_domain,
            tax_domains=tax_domains,
        )
    config = get_rag_config()
    ensure_local_index_ready()
    if not config.db_path.exists():
        return []

    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    _, rows = fetch_local_candidate_rows(
        expanded_query, effective_limit=effective_limit, config=config, source_types=source_types,
        enforce_query_domain=enforce_query_domain, tax_domains=tax_domains,
    )
    return rank_hybrid_local_candidates(rows, query=expanded_query, effective_limit=effective_limit, config=config)


def search_chat_chunks(
    query: str,
    *,
    limit: Optional[int] = None,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> list[RagChunk]:
    if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}:
        from app.mysql_rag import search_chat_chunks_mysql

        return search_chat_chunks_mysql(
            query,
            limit=limit,
            include_interpretations=include_interpretations,
            include_judgments=include_judgments,
        )
    """Retrieve complementary authority types for an application answer.

    A factual interpretation and the applicable provision answer different
    questions.  Searching them in one untyped top-k lets citation-heavy
    interpretations crowd statutes out of the prompt, so retrieve and mix
    both channels explicitly.
    """
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    judgment_requested_by_query = bool(JUDGMENT_INTENT_RE.search(query) or extract_judgment_signatures(query))
    include_judgments = judgment_requested_by_query if include_judgments is None else include_judgments
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
        elif query_targets_developer_land_sale(query):
            statute_limit = min(effective_limit - 1, max(4, math.ceil(effective_limit * 0.6)))
        elif query_targets_post_leasing_vehicle_gift_sale(query):
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
    if interpretation_limit and query_targets_ksef_foreign_sale(query):
        interpretation_rows = fetch_rows_by_document_ids(
            KSEF_FOREIGN_SALE_INTERPRETATION_DOCUMENT_IDS,
            config=config,
            source_type="interpretation",
        )
        interpretations = rank_hybrid_local_candidates(
            interpretation_rows,
            query=expand_search_query(query),
            effective_limit=interpretation_limit,
            config=config,
        ) if interpretation_rows else []
    else:
        interpretations = search_chunks(
            query,
            limit=interpretation_limit,
            source_types={"interpretation"},
            enforce_query_domain=explicit_query_domains,
            tax_domains=statute_domains,
        ) if interpretation_limit else []
    judgments = search_chunks(
        query,
        limit=judgment_limit,
        source_types={"judgment"},
        enforce_query_domain=explicit_query_domains,
        tax_domains=statute_domains,
    ) if include_judgments else []
    statutes = [] if query_targets_ksef_foreign_sale(query) else (
        search_chunks(
            query,
            limit=statute_limit,
            source_types={"statute"},
            enforce_query_domain=True,
            tax_domains=statute_domains,
        ) if statute_limit else []
    )
    preferred_targets: list[tuple[str, str]] = []
    if query_targets_ksef_foreign_sale(query):
        preferred_targets.extend(KSEF_FOREIGN_SALE_STATUTE_TARGETS)
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
        ) else statute_limit,
    ) if statute_limit else []
    hinted_statutes = rank_hybrid_local_candidates(
        hinted_statute_rows,
        query=expand_search_query(query),
        effective_limit=len(hinted_statute_rows) if (
            query_targets_post_leasing_vehicle_gift_sale(query)
            or query_targets_leased_movable_six_year_rule(query)
            or query_targets_gifted_asset_cost_basis(query)
            or query_targets_spouse_gift_sd(query)
        ) else statute_limit,
        config=config,
    ) if hinted_statute_rows else []
    if (
        query_targets_post_leasing_vehicle_gift_sale(query)
        or query_targets_leased_movable_six_year_rule(query)
        or query_targets_gifted_asset_cost_basis(query)
        or query_targets_spouse_gift_sd(query)
    ):
        hinted_statutes = order_chunks_by_statute_targets(hinted_statutes, preferred_targets)

    merged_statutes: list[RagChunk] = []
    seen_statute_chunks: set[str] = set()
    prefer_hinted_statutes = (
        query_targets_ksef_foreign_sale(query)
        or query_targets_wht_crossborder_payments(query)
        or query_targets_developer_land_sale(query)
        or query_targets_post_leasing_vehicle_gift_sale(query)
        or query_targets_leased_movable_six_year_rule(query)
        or query_targets_gifted_asset_cost_basis(query)
        or query_targets_spouse_gift_sd(query)
        or query_targets_shareholder_company_asset_sale(query)
        or query_targets_small_taxpayer_foreign_vat(query)
    )
    statute_candidates = [*hinted_statutes, *statutes] if prefer_hinted_statutes else [*statutes, *hinted_statutes]
    if (
        query_targets_post_leasing_vehicle_gift_sale(query)
        or query_targets_leased_movable_six_year_rule(query)
        or query_targets_gifted_asset_cost_basis(query)
        or query_targets_spouse_gift_sd(query)
    ):
        unique_statute_candidates: list[RagChunk] = []
        duplicate_statute_candidates: list[RagChunk] = []
        seen_article_targets: set[tuple[str, str]] = set()
        for chunk in statute_candidates:
            article_key = extract_article_key_from_text(chunk.legal_provisions[0] if chunk.legal_provisions else "")
            article_target = (infer_chunk_tax_domain(chunk), article_key)
            if article_target[0] and article_target[1] and article_target not in seen_article_targets:
                seen_article_targets.add(article_target)
                unique_statute_candidates.append(chunk)
            else:
                duplicate_statute_candidates.append(chunk)
        statute_candidates = [*unique_statute_candidates, *duplicate_statute_candidates]
    for chunk in statute_candidates:
        if chunk.chunk_id in seen_statute_chunks:
            continue
        seen_statute_chunks.add(chunk.chunk_id)
        merged_statutes.append(chunk)
        if len(merged_statutes) >= statute_limit:
            break

    mixed: list[RagChunk] = []
    if include_judgments:
        for position in range(max(len(judgments), len(merged_statutes), len(interpretations))):
            if position < len(judgments):
                mixed.append(judgments[position])
            if position < len(merged_statutes):
                mixed.append(merged_statutes[position])
            if position < len(interpretations):
                mixed.append(interpretations[position])
        return mixed[:effective_limit]

    for position in range(max(len(interpretations), len(merged_statutes))):
        if position < len(merged_statutes):
            mixed.append(merged_statutes[position])
        if position < len(interpretations):
            mixed.append(interpretations[position])
    return mixed[:effective_limit]


def inspect_search(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> RetrievalInspection:
    if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}:
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
    _, candidate_rows = fetch_local_candidate_rows(
        expanded_query,
        effective_limit=effective_limit,
        config=config,
        source_types=source_types,
        enforce_query_domain=enforce_query_domain,
        tax_domains=tax_domains,
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
                "small_taxpayer_foreign_vat_match_score": build_small_taxpayer_foreign_vat_match_score(row, query=expanded_query),
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

    for chunk in chunks:
        current = document_counts.get(chunk.document_id, 0)
        if current >= per_document_limit:
            continue
        selected.append(chunk)
        document_counts[chunk.document_id] = current + 1

    return selected


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
    seen: set[tuple[str, Optional[str], Optional[str], str]] = set()
    for chunk in select_diverse_chunks(chunks):
        key = (chunk.document_id, chunk.signature, chunk.published_date, chunk.source_type)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- [{chunk.source_type}{':' + chunk.source_subtype if chunk.source_subtype else ''}] "
            f"{chunk.signature or chunk.subject} | {chunk.publication or chunk.published_date or 'brak daty'} | {chunk.source_url or 'brak URL'}"
        )
    return "\n".join(lines)


def index_exists() -> bool:
    if os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() in {"mysql", "mariadb"}:
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

    source_paths = (config.processed_path, *config.additional_source_paths)
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
