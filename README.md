# Hypothesis Lab

Cel fazy: sprawdzić dwie rzeczy techniczne z dokumentu projektowego.

1. Czy yfinance pobierze ok. 10 lat dziennych notowań ETF-ów (GLD, IAU, SLV, CPER, USO) z runnera GitHub Actions.
2. Czy aplikacja Streamlit na Community Cloud działa z prywatnego repozytorium i odczyta dane z plików Parquet.

## Zawartość

- `scripts/phase0_probe.py` - pobiera dane, waliduje je, zapisuje Parquet do `data/` i raport `data/probe_report.json`
- `.github/workflows/phase0-probe.yml` - uruchamia skrypt ręcznie w GitHub Actions i zapisuje dane do repozytorium
- `app/streamlit_app.py` - minimalna aplikacja pokazująca wczytane dane
- `requirements.txt` (aplikacja) i `requirements-pipeline.txt` (job pobierający dane)

## Kroki

1. Utwórz na GitHubie **prywatne** repozytorium (np. `hypothesis-lab`) i wgraj do niego całą zawartość tego katalogu, razem z ukrytym folderem `.github`.
   Repozytorium musi być prywatne: dane z Yahoo mają licencję na użytek osobisty i nie wolno ich publikować.
2. W repozytorium: Actions -> **Phase 0 probe** -> **Run workflow**. Jeśli GitHub zgłosi brak uprawnień do zapisu, wejdź w Settings -> Actions -> General -> Workflow permissions i ustaw "Read and write permissions".
3. Po zakończeniu otwórz zakładkę Summary uruchomienia. Zobaczysz tabelę ze statusem każdego symbolu (OK / WARN / FAIL), liczbą wierszy i zakresem dat. Pliki Parquet trafią też do folderu `data/` w repozytorium.
4. Wejdź na https://share.streamlit.io, zaloguj się GitHubem, daj dostęp do prywatnego repozytorium i utwórz aplikację: branch `main`, plik główny `app/streamlit_app.py`. W ustawieniach aplikacji ustaw widoczność na prywatną.
5. Sprawdź, czy aplikacja pokazuje tabelę z danymi i wykres.

## Co zgłosić z powrotem

- tabelę z Summary z kroku 3 (albo plik `data/probe_report.json`),
- czy krok 4 zadziałał z prywatnym repozytorium na darmowym planie,
- ewentualne komunikaty błędów.

## Test lokalny bez internetu

    pip install -r requirements-pipeline.txt -r requirements.txt
    python scripts/phase0_probe.py --synthetic
    DATA_DIR=data_synthetic streamlit run app/streamlit_app.py

Tryb `--synthetic` generuje losowe dane i zapisuje je do `data_synthetic/` (ignorowanego przez git). Służy tylko do sprawdzenia kodu, nie do analiz.

## Uwagi

- Wersje akcji w workflow (`checkout@v4`, `setup-python@v5`, `upload-artifact@v4`) mogą mieć nowsze wydania. Jeśli GitHub zgłosi ostrzeżenie o przestarzałej wersji, podnieś numer.
- Harmonogram (cron) dodamy po pozytywnym wyniku sondy.

---

# Faza 1: silnik hipotez

Biblioteka `hypothesis_lab/` liczy, w ilu przypadkach hipoteza się sprawdza, na tle bazy porównawczej, z miarami niepewności.
Nie zależy od Streamlita, GitHub Actions ani Databricksa.

## Moduły

- `spec.py` - serializowalna specyfikacja hipotezy (`HypothesisSpec`, `CostModel`) ze skrótem do powtarzalności
- `signals.py` - gotowe hipotezy: `mean_reversion`, `momentum`, `shock_rebound` (sygnał z sesji t używa tylko danych do t)
- `engine.py` - zdarzenia, trafność, baza, test dwumianowy, bootstrap blokowy, hold-out z oczyszczeniem, test wrażliwości
- `data.py` - wczytywanie Parquet, ceny skorygowane, znacznik sesji z zerowym wolumenem
- `cli.py` - uruchamianie z wiersza poleceń

## Szybki start

    pip install -e ".[dev]"
    python -m pytest -q

    # GLD, powrót do średniej, horyzont 5 sesji, z kosztami
    python -m hypothesis_lab.cli --symbol GLD --type mean_reversion --param W=20 --param k=2 --horizon 5 \
        --costs --commission-bps 1 --spread-bps 2

    # SLV bez 10%-owych dni w oknie pozycji, wykonanie po następnym otwarciu
    python -m hypothesis_lab.cli --symbol SLV --type shock_rebound --param x_pct=3 --horizon 3 \
        --exclude-extreme 0.10 --exec "open_t+1"

    # demo bez danych
    python -m hypothesis_lab.cli --synthetic --type momentum

Dane są czytane z `data/<SYMBOL>.parquet` (te same pliki, które zapisuje sonda).

## Założenia, które warto znać

- Wykonanie `close_t`: wejście po cenie zamknięcia sesji sygnału, wyjście po N sesjach. `open_t+1`: wejście po otwarciu następnej sesji, wyjście po otwarciu sesji t+1+N.
- Domyślnie jedna otwarta pozycja naraz (`--overlap` to wyłącza, ale wtedy p-wartość zawyża istotność).
- Baza porównawcza to trafność tej samej mieszanki pozycji (long/short) na wszystkich sesjach okresu.
- Hold-out to ostatnie 30% okresu. Zdarzenia próby muszą zakończyć się przed granicą podziału.
- Test wrażliwości liczy ten sam wynik bez 5 największych dni (wg |zwrotu|).

## Testy

`tests/` zawiera: ręcznie policzone przykłady, test wycieku przyszłości (z meta-testem, który sprawdza, że detektor wykrywa znane wycieki), test błądzenia losowego (brak "przewagi" w szumie), testy specyfikacji i statystyki.


---

# Faza 2: codzienny job i interfejs

## Codzienna aktualizacja danych

Workflow `.github/workflows/daily-update.yml` uruchamia `scripts/update_data.py`:

- harmonogram: 22:45 UTC w dni robocze (po zamknięciu giełdy USA w czasie letnim i zimowym) oraz 05:30 UTC wt-sob jako poranne uzupełnienie; zadanie jest idempotentne,
- dane GLD, SLV, CPER, USO (`--symbols` zmienia listę), okno do 10 lat,
- niepełna bieżąca sesja jest odrzucana, ostatnie 5 sesji jest zawsze nadpisywane świeżymi danymi,
- rozbieżność starszych sesji względem nowego pobrania (split, dywidenda) uruchamia pełne odświeżenie symbolu,
- błędna świeczka (np. high < low) nie trafia do pliku, a zadanie kończy się błędem,
- po aktualizacji sprawdzane są: spójność ostatniej sesji między symbolami i aktualność danych (maks. 5 dni),
- wynik zapisuje `data/update_status.json`, a podsumowanie trafia do zakładki Summary uruchomienia,
- alarm: nieudane uruchomienie workflow (GitHub wysyła e-mail, jeśli masz włączone powiadomienia),
- zmiany w `data/` są commitowane do repozytorium, a Streamlit Community Cloud wdraża je automatycznie.

Ręczne uruchomienie: Actions -> Daily data update -> Run workflow (opcja "force_full" robi pełne odświeżenie 10 lat).
Lokalnie: `python scripts/update_data.py` (opcje `--force-full`, `--symbols GLD,SLV`).

Pierwsze uruchomienie po wdrożeniu: usuń `data/IAU.parquet` (wybrano GLD) i uruchom workflow raz z "force_full".
Plik `scripts/phase0_probe.py` jest pozostałością Fazy 0 i można go usunąć.

## Interfejs (Streamlit)

`app/streamlit_app.py`, lokalnie: `streamlit run app/streamlit_app.py` (katalog danych: zmienna `DATA_DIR`).

Panel boczny prowadzi przez wybór: 1. instrument i okres, 2. hipoteza z parametrami, 3. horyzont, kierunek, cel i moment wykonania,
koszty, hold-out i wykluczenia. Wynik pokazuje trafność na tle bazy, przedział ufności, p dwumianowe (także po korekcie na liczbę
prób) i ostrzeżenia, a w zakładkach: tabelę segmentów (cały okres, próba, hold-out), zdarzenia z eksportem CSV, wykresy,
pełną specyfikację z eksportem JSON i historię prób w bieżącej sesji.

Historia prób (licznik do korekty na wielokrotne testowanie) jest przechowywana w sesji przeglądarki i znika po odświeżeniu strony.
