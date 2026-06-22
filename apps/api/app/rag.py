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
    API_DIR / "data" / "laws" / "processed" / "tax_ordinance_DU_2026_622.jsonl",
)
DEFAULT_RAG_DB_PATH = API_DIR / "data" / "processed" / "eureka_rag.sqlite3"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

WHITESPACE_RE = re.compile(r"\s+")
QUERY_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}")
EMBEDDING_TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{2,}")
SECTION_BREAK_RE = re.compile(r"\n{2,}")
BOILERPLATE_SECTION_RE = re.compile(
    r"\n(?=(?:Pouczenie o funkcji ochronnej interpretacji|Funkcja ochronna interpretacji|"
    r"Prawo do wniesienia skargi|Mają Państwo prawo do zaskarżenia|Skargę do Sądu wnosi się))",
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
GENERAL_STATUTE_QUERY_RE = re.compile(
    r"\b(co jest|co oznacza|co rozumieć|jak ustawa definiuje|jakie zasady|gdzie uregulowano|"
    r"kiedy .* nie jest|czy .* ma obowiązek)\b",
    re.IGNORECASE,
)

# The search corpus uses both abbreviations and their expanded legal names.  Keeping
# these aliases here makes a user's natural query match either form without an LLM.
QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bksef\b|\bkrajow(?:y|ego) system(?:u)? e[ -]?faktur", re.IGNORECASE), ("Krajowy System e-Faktur", "faktura ustrukturyzowana")),
    (re.compile(r"\bwht\b|podatek u źr[óo]dła|withholding", re.IGNORECASE), ("WHT", "podatek u źródła")),
    (re.compile(r"\bpcc\b|podatek od czynności cywilnoprawnych", re.IGNORECASE), ("PCC", "podatek od czynności cywilnoprawnych")),
    (re.compile(r"sp[óo]łk[ai] holdingow", re.IGNORECASE), ("Polska Spółka Holdingowa", "PSH")),
    (re.compile(r"ograniczon(?:y|ego) obowi[ąa]zek podatkow", re.IGNORECASE), ("ograniczony obowiązek podatkowy", "nierezydent", "rezydencja podatkowa")),
    (re.compile(r"skala podatkow|wyb[óo]r formy opodatkowania", re.IGNORECASE), ("skala podatkowa", "forma opodatkowania", "oświadczenie")),
    (re.compile(r"esto[ńn]sk(?:i|iego)?\s+cit|rycza[łl]t(?:em)? od dochod[óo]w sp[óo][łl]ek", re.IGNORECASE), ("estoński CIT", "ryczałt od dochodów spółek")),
    (re.compile(r"\bip\s*box\b", re.IGNORECASE), ("IP Box", "kwalifikowane prawo własności intelektualnej")),
    (re.compile(r"\bexit\s+tax\b", re.IGNORECASE), ("exit tax", "dochody z niezrealizowanych zysków", "podatek od dochodów z niezrealizowanych zysków")),
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
    "wht": ("wht", "źródła", "zrodla", "withholding"),
    "akcyza": ("akcyza", "akcyzow", "skład podatkowy", "sklad podatkowy"),
}
MECHANISM_RULES: dict[str, tuple[str, ...]] = {
    "invoice_outside_ksef": ("poza ksef", "faktura papierowa", "faktura pdf"),
    "input_vat_deduction": ("odliczyć vat", "prawo do odliczenia"),
    "limited_tax_liability": ("ograniczony obowiązek", "183 dni", "centrum interesów"),
    "return_relief": ("ulga na powrót",),
    "termination_of_co_ownership": ("zniesienie współwłasności", "zniesienie wspólwłasności"),
    "equalization_payment": ("spłata", "splata"),
    "thermomodernization_relief": ("termomoderniz",),
    "housing_relief": ("ulga mieszkaniowa",),
    "temporary_rental": ("czasowy wynajem", "wynajmować lokal"),
    "dropshipping": ("dropshipping", "klient jako importer"),
    "land_sale_vat": ("sprzedaż działki", "sprzedaz dzialki"),
    "buyer_power_of_attorney": ("pełnomocnictw", "pelnomocnictw"),
    "private_leased_vehicle_sale": ("samochód leasing", "samochod leasing", "wykup", "majątku prywat"),
    "senior_relief": ("ulga dla pracujących seniorów", "ulga dla senior"),
}
STATUTE_PROCEDURAL_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
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


def derive_tax_domain(record: dict[str, Any]) -> str:
    haystack = " ".join(
        [*map(str, record.get("law_tags") or []), *map(str, record.get("issues") or []), *map(str, record.get("legal_provisions") or [])]
    ).lower()
    for domain, markers in (
        ("VAT", ("[vat]", "vat", "towarów i usług")),
        ("CIT", ("[cit]", "cit", "dochodowym od osób prawnych")),
        ("PIT", ("[pit]", "pit", "dochodowym od osób fizycznych")),
        ("PCC", ("[pcc]", "pcc", "czynności cywilnoprawnych")),
        ("AKCYZA", ("akcyza", "akcyzow")),
        ("ORDYNACJA", ("ordynacja", "zobowiązania podatkowe")),
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


def get_query_expansion_terms(query: str) -> list[str]:
    """Return stable tax-domain aliases relevant to the user's wording."""
    additions: list[str] = []
    for pattern, aliases in QUERY_EXPANSIONS:
        if pattern.search(query):
            additions.extend(aliases)
    return list(dict.fromkeys(additions))


def expand_search_query(query: str) -> str:
    """Add stable tax-domain aliases while preserving the user's original wording."""
    return " ".join([query, *get_query_expansion_terms(query)]).strip()


def build_candidate_match_queries(query: str) -> list[str]:
    """Build complementary FTS queries for prose and short legal-domain aliases."""
    queries = [build_match_query(query)]
    expansion_query = build_match_query(" ".join(get_query_expansion_terms(query)))
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


def load_mechanism_rules(config: RagConfig) -> dict[str, tuple[str, ...]]:
    if not config.mechanism_lexicon_path.exists(): return MECHANISM_RULES
    try:
        p=json.loads(config.mechanism_lexicon_path.read_text(encoding="utf-8"))
        r={str(x["id"]):tuple(str(a).lower() for a in x.get("aliases",[]) if str(a).strip()) for x in p.get("mechanisms",[]) if x.get("status")=="ready" and x.get("id") and x.get("aliases")}
        return r or MECHANISM_RULES
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
    query_domains = detect_domains(query)
    candidate_domains = detect_domains(" ".join(candidate_text_parts))
    if query_domains and query_domains & candidate_domains:
        return min(normalized_overlap + 0.35, 1.0)
    if query_domains and candidate_domains:
        return max(normalized_overlap - 0.2, -0.2)
    return normalized_overlap


def build_facts_match_score(row: sqlite3.Row, *, query: str) -> float:
    query_terms = ranking_terms(query)
    facts_terms = ranking_terms(str(row["facts_text"] or ""))
    return (sum(1 for term in query_terms if term_matches(facts_terms, term)) / len(query_terms)) if query_terms else 0.0


def build_mechanism_match_score(row: sqlite3.Row, *, query: str, config: RagConfig) -> float:
    query_mechanisms = detect_mechanisms(query,config=config)
    if not query_mechanisms:
        return 0.0
    candidate_text = " ".join(str(row[key] or "") for key in ("subject", "question_text", "issues_json", "keywords_json", "chunk_text"))
    candidate_mechanisms = detect_mechanisms(candidate_text,config=config)
    return len(query_mechanisms & candidate_mechanisms) / len(query_mechanisms)


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
    score = 0.0
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
    return score


def resolve_cross_blend_weight(
    row: sqlite3.Row,
    *,
    query: str,
    statute_match_score: float,
    config: RagConfig,
) -> float:
    weight = min(max(config.cross_encoder_weight, 0.0), 1.0)
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
) -> tuple[str, list[sqlite3.Row]]:
    """Fetch and merge diversified FTS candidate pools before hybrid reranking."""
    match_queries = build_candidate_match_queries(query)
    if source_types == {"statute"}:
        match_queries.extend(build_statute_match_queries(query))
        match_queries = list(dict.fromkeys(match_queries))
    if config.facts_channel_enabled:
        fact_terms = " ".join(sorted(ranking_terms(query))[:12])
        facts_query = build_match_query(fact_terms, max_tokens=12)
        if facts_query:
            match_queries.append(f"facts_text : ({facts_query})")
    if not match_queries or not config.db_path.exists():
        return "", []

    candidate_limit = max(config.candidate_pool_limit, effective_limit * 20)
    allowed_types = sorted({value.lower() for value in source_types or set() if value})
    type_clause = ""
    type_values: list[str] = []
    if allowed_types:
        type_clause = " AND d.source_type IN (" + ", ".join("?" for _ in allowed_types) + ")"
        type_values = allowed_types
    connection = get_connection(config.db_path)
    try:
        query_rows = [connection.execute(
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
            WHERE chunks_fts MATCH ?""" + type_clause + """
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, *type_values, candidate_limit),
        ).fetchall() for match_query in match_queries]
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
            family_prefixes, exact_articles = detect_procedural_article_targets(query)
            family_clauses: list[str] = []
            family_values: list[str] = []
            for article in sorted(exact_articles):
                family_clauses.append("d.legal_provisions_json LIKE ?")
                family_values.append(f'%art. {article}%')
            for prefix in sorted(family_prefixes):
                family_clauses.append("d.legal_provisions_json LIKE ?")
                family_values.append(f'%art. {prefix}%')
            if family_clauses:
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
    finally:
        connection.close()

    # Interleave channels so a broad wording match cannot crowd out an exact
    # domain-alias hit.  Limit chunks per document to retain legal diversity.
    rows: list[sqlite3.Row] = []
    seen_chunks: set[str] = set()
    chunks_per_document: dict[str, int] = {}
    max_chunks_per_document = max(config.retrieval_max_chunks_per_document, 1)
    query_domains = {domain.upper() for domain in detect_domains(query)}
    query_domains.update(domain.upper() for domain in tax_domains or set() if domain)
    for rank in range(max((len(group) for group in query_rows), default=0)):
        for group in query_rows:
            if rank >= len(group):
                continue
            row = group[rank]
            chunk_id = str(row["chunk_id"])
            document_id = str(row["document_id"])
            candidate_domain = str(row["tax_domain"] or "").upper()
            if (config.domain_filter_enabled or enforce_query_domain) and query_domains and candidate_domain and candidate_domain not in query_domains:
                continue
            if chunk_id in seen_chunks or chunks_per_document.get(document_id, 0) >= max_chunks_per_document:
                continue
            rows.append(row)
            seen_chunks.add(chunk_id)
            chunks_per_document[document_id] = chunks_per_document.get(document_id, 0) + 1
            if len(rows) >= candidate_limit:
                return " || ".join(match_queries), rows
    return " || ".join(match_queries), rows


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

    # Stage 2: inexpensive hybrid pre-ranking over the full recall pool.
    semantic_scores = [
        (row, semantic_score, build_legal_match_score(row, query=query), build_mechanism_match_score(row, query=query, config=config))
        for row, semantic_score in zip(
            rows, compute_hash_semantic_scores(rows, query=query, config=config)
        )
    ]

    lexical_ranks = {
        str(row["chunk_id"]): rank for rank, row in enumerate(rows, start=1)
    }
    semantic_ranks = {
        str(row["chunk_id"]): rank
        for rank, (row, _, _, _) in enumerate(
            sorted(
                semantic_scores,
                key=lambda item: (item[1], -int(item[0]["chunk_index"])),
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
            + (0.25 * build_statute_match_score(item[0], query=query))
            + (0.35 * build_article_family_match_score(item[0], query=query)),
            -item[1],
            item[2],
            -float(item[0]["lexical_score"]),
        ),
        reverse=True,
    )

    # Stage 3: the cross-encoder sees only the strongest hybrid candidates.
    # This preserves broad recall while spending the expensive model budget on
    # legal near-misses that can realistically reach the final top-k.
    shortlist = preliminary_rows[: max(effective_limit, config.cross_encoder_candidate_limit)]
    cross_scores = compute_cross_encoder_scores(
        [row for row, _, _, _ in shortlist], query=query, config=config
    )
    if cross_scores is None:
        ranked_rows = preliminary_rows
    else:
        cross_ranks = {
            str(row["chunk_id"]): rank
            for rank, ((row, _, _, _), _) in enumerate(
                sorted(zip(shortlist, cross_scores), key=lambda item: item[1], reverse=True), start=1
            )
        }
        cross_weight = min(max(config.cross_encoder_weight, 0.0), 1.0)
        def cross_encoder_sort_key(item: tuple[sqlite3.Row, float, float, float]) -> tuple[float, int]:
            row, _, legal_match_score, mechanism_match_score = item
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
                + (0.25 * statute_match_score)
                + (0.35 * family_match_score)
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
            )

        ranked_rows = sorted(
            shortlist,
            key=cross_encoder_sort_key,
            reverse=True,
        )

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
                + (0.25 * build_statute_match_score(row, query=query))
                + (0.35 * build_article_family_match_score(row, query=query))
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
        for row, _, legal_match_score, mechanism_match_score in ranked_rows[:effective_limit]
    ]


def search_chunks(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> list[RagChunk]:
    config = get_rag_config()
    if not config.db_path.exists():
        return []

    effective_limit = limit or config.retrieval_limit
    expanded_query = expand_search_query(query)
    _, rows = fetch_local_candidate_rows(
        expanded_query, effective_limit=effective_limit, config=config, source_types=source_types,
        enforce_query_domain=enforce_query_domain, tax_domains=tax_domains,
    )
    return rank_hybrid_local_candidates(rows, query=expanded_query, effective_limit=effective_limit, config=config)


def search_chat_chunks(query: str, *, limit: Optional[int] = None) -> list[RagChunk]:
    """Retrieve complementary authority types for an application answer.

    A factual interpretation and the applicable provision answer different
    questions.  Searching them in one untyped top-k lets citation-heavy
    interpretations crowd statutes out of the prompt, so retrieve and mix
    both channels explicitly.
    """
    config = get_rag_config()
    effective_limit = limit or config.retrieval_limit
    statute_limit = max(1, effective_limit // 2)
    interpretation_limit = max(1, effective_limit - statute_limit)
    interpretations = search_chunks(
        query, limit=interpretation_limit, source_types={"interpretation"}
    )
    statutes = search_chunks(
        query,
        limit=statute_limit,
        source_types={"statute"},
        enforce_query_domain=True,
    )

    mixed: list[RagChunk] = []
    for position in range(max(len(interpretations), len(statutes))):
        if position < len(statutes):
            mixed.append(statutes[position])
        if position < len(interpretations):
            mixed.append(interpretations[position])
    return mixed[:effective_limit]


def inspect_search(
    query: str, *, limit: Optional[int] = None, source_types: Optional[set[str]] = None,
    enforce_query_domain: bool = False, tax_domains: Optional[set[str]] = None,
) -> RetrievalInspection:
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
                "lexical_score": float(row["lexical_score"]),
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
    config = get_rag_config()
    return config.db_path.exists()


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
