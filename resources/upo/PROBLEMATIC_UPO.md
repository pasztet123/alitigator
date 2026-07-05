# Problematyczne UPO

Poniższe pliki nie osiągnęły progu jakości potrzebnego do bezpiecznego indeksowania artykułów.

## Wielka Brytania

- Plik: `resources/upo/wielka_brytania/umowa_pl.pdf`
- Źródło: `https://www.podatki.gov.pl/media/uanpfvts/wlk-brytania-konwencja-tekst-polski.pdf`
- Status: `partial_text_only`
- Powód: OCR wyodrębnił tylko 7 artykułów, więc bazowy tekst umowy nie został włączony do JSONL.
- Uwaga: w indeksie jest już dostępny `tekst_syntetyczny_mli_pl.pdf`, więc coverage dla UK nie jest zerowy, ale brakuje czystego bazowego tekstu umowy.
