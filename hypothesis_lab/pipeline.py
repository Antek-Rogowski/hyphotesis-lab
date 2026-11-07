"""Codzienna aktualizacja danych: pobranie, walidacja i bezpieczne dopisanie do plików Parquet.

Zasady:
  * źródło danych jest ukryte za interfejsem DataProvider (łatwa wymiana yfinance na licencjonowanego dostawcę),
  * zapis jest idempotentny: ponowne uruchomienie tego samego dnia nic nie zmienia,
  * niepełna bieżąca sesja (pobranie w trakcie notowań) jest odrzucana,
  * ostatnie RECENT_OVERWRITE sesji jest zawsze nadpisywane świeżymi danymi (korekty, uszkodzone świeczki),
  * starsze sesje są porównywane z nowym pobraniem; rozbieżność (split, dywidenda) uruchamia pełne odświeżenie,
  * nowe wiersze przechodzą walidację; błędna świeczka nie trafia do pliku, a zadanie kończy się błędem,
  * po aktualizacji sprawdzamy spójność symboli i aktualność danych oraz zapisujemy update_status.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

STORED_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume", "source", "ingested_at"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

DEFAULT_SYMBOLS = ["GLD", "SLV", "CPER", "USO"]
FRESH_PERIOD = "3mo"          # okno pobierane na co dzień
FULL_PERIOD = "10y"           # okno pełnego odświeżenia
MAX_YEARS = 10                # nie trzymamy danych starszych niż tyle lat od ostatniej sesji
RECENT_OVERWRITE = 5          # tyle ostatnich sesji nadpisujemy świeżymi danymi
REVISION_TOLERANCE = 1e-4     # względna rozbieżność cen/współczynnika korekty uznawana za rewizję
MAX_STALE_DAYS = 5            # starsza ostatnia sesja (dni kalendarzowe) => błąd
SESSION_DONE_MINUTES_ET = 16 * 60 + 30   # 16:00 ET zamknięcie + 30 min bufora
STATUS_FILE = "update_status.json"


class DataQualityError(RuntimeError):
    """Nowe dane nie przeszły walidacji."""


# --------------------------------------------------------------------------
# Dostawca danych
# --------------------------------------------------------------------------
class DataProvider(Protocol):
    name: str

    def fetch_daily(self, symbol: str, period: str) -> pd.DataFrame:
        """Zwraca dzienne notowania w formacie yfinance (Open, High, Low, Close, Adj Close, Volume)."""


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, retries: int = 3, pause: float = 5.0) -> None:
        self.retries = retries
        self.pause = pause

    def fetch_daily(self, symbol: str, period: str) -> pd.DataFrame:
        import yfinance as yf

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False, actions=False)
                if raw is None or raw.empty:
                    raise ValueError("pusta odpowiedź z Yahoo (limit zapytań, blokada IP, błąd sieci lub zły symbol)")
                return raw
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.pause * attempt)
        raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


# --------------------------------------------------------------------------
# Czyszczenie i walidacja
# --------------------------------------------------------------------------
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def standardize(raw: pd.DataFrame, symbol: str, source: str, now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df = df[["open", "high", "low", "close", "adj_close", "volume"]].copy()

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=PRICE_COLUMNS, how="all")      # puste świeczki (np. bieżąca sesja bez danych)

    now = (now_utc or _utc_now()).tz_convert("UTC").tz_localize(None)
    df.insert(0, "symbol", symbol)
    df["source"] = source
    df["ingested_at"] = now
    return df


def drop_partial_session(df: pd.DataFrame, now_utc: pd.Timestamp | None = None) -> tuple[pd.DataFrame, bool]:
    """Odrzuca świeczkę bieżącego dnia, jeśli sesja w Nowym Jorku jeszcze się nie zakończyła."""
    now_utc = now_utc if now_utc is not None else _utc_now()
    now_et = now_utc.tz_convert("America/New_York")
    today_et = pd.Timestamp(now_et.date())
    minutes_et = now_et.hour * 60 + now_et.minute
    if len(df) and df.index.max() == today_et and minutes_et < SESSION_DONE_MINUTES_ET:
        return df.iloc[:-1], True
    return df, False


def check_rows(df: pd.DataFrame) -> list[str]:
    """Błędy blokujące zapis. Zwraca listę opisów (pusta = dane poprawne)."""
    problems: list[str] = []
    if df[PRICE_COLUMNS].isna().any().any():
        problems.append(f"puste ceny w {int(df[PRICE_COLUMNS].isna().any(axis=1).sum())} wierszach")
    if (df[PRICE_COLUMNS] <= 0).any().any():
        problems.append("ceny <= 0")
    if (df["high"] < df["low"]).any():
        problems.append(f"high < low w {int((df['high'] < df['low']).sum())} wierszach")
    if (df["volume"] < 0).any():
        problems.append("ujemny wolumen")
    return problems


def detect_revision(stored: pd.DataFrame, fresh: pd.DataFrame, recent_n: int = RECENT_OVERWRITE) -> str:
    """Porównuje starsze zapisane sesje z nowym pobraniem. Zwraca opis rewizji lub pusty tekst."""
    if len(stored) <= recent_n:
        return ""
    cutoff = stored.index[-recent_n]
    old = stored[stored.index < cutoff]
    common = old.index.intersection(fresh.index)
    if len(common) == 0:
        return ""
    a, b = old.loc[common], fresh.loc[common]
    close_diff = float((a["close"] / b["close"] - 1.0).abs().max())
    factor_diff = float(((a["adj_close"] / a["close"]) / (b["adj_close"] / b["close"]) - 1.0).abs().max())
    if close_diff > REVISION_TOLERANCE:
        return f"zmieniły się ceny zamknięcia historycznych sesji (maks. {close_diff:.2%}), np. split"
    if factor_diff > REVISION_TOLERANCE:
        return f"zmienił się współczynnik korekty ceny skorygowanej (maks. {factor_diff:.2%}), np. dywidenda"
    return ""


# --------------------------------------------------------------------------
# Pliki
# --------------------------------------------------------------------------
def _path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol}.parquet"


def read_stored(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    missing = [c for c in STORED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"zapisany plik {path.name} nie ma kolumn: {missing}")
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    df.index.name = "date"
    return df.sort_index()


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df[STORED_COLUMNS].to_parquet(tmp, engine="pyarrow")
    os.replace(tmp, path)


def _same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Czy dwie ramki mają te same sesje i wartości (pomijamy czas pobrania ingested_at)."""
    if not a.index.equals(b.index):
        return False
    numeric = PRICE_COLUMNS + ["volume"]
    same_numbers = np.array_equal(a[numeric].to_numpy(float), b[numeric].to_numpy(float), equal_nan=True)
    same_labels = bool((a[["symbol", "source"]].to_numpy() == b[["symbol", "source"]].to_numpy()).all())
    return bool(same_numbers and same_labels)


def trim_years(df: pd.DataFrame, max_years: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.index >= df.index.max() - pd.DateOffset(years=max_years)]


def merge_recent(stored: pd.DataFrame, fresh: pd.DataFrame, recent_n: int = RECENT_OVERWRITE) -> pd.DataFrame:
    """Stare sesje zostają, ostatnie recent_n sesji i wszystko nowsze pochodzi ze świeżego pobrania."""
    cutoff = stored.index[-recent_n] if len(stored) > recent_n else stored.index[0]
    kept = stored[stored.index < cutoff]
    newer = fresh[fresh.index >= cutoff]
    return pd.concat([kept, newer]).sort_index()


# --------------------------------------------------------------------------
# Aktualizacja
# --------------------------------------------------------------------------
def _fetch_clean(provider: DataProvider, symbol: str, period: str, now_utc: pd.Timestamp):
    raw = provider.fetch_daily(symbol, period)
    df = standardize(raw, symbol, provider.name, now_utc)
    df, dropped = drop_partial_session(df, now_utc)
    if df.empty:
        raise DataQualityError("brak danych po oczyszczeniu")
    return df, dropped


def update_symbol(symbol: str, provider: DataProvider, data_dir: str | Path = "data",
                  now_utc: pd.Timestamp | None = None, force_full: bool = False,
                  max_years: int = MAX_YEARS) -> dict:
    now_utc = now_utc if now_utc is not None else _utc_now()
    path = _path(data_dir, symbol)
    result: dict = {"symbol": symbol, "status": "ok", "mode": None, "written": False, "warnings": []}
    try:
        stored = read_stored(path)
        mode = "initial" if stored is None else ("full_refresh" if force_full else "incremental")
        fresh, dropped = _fetch_clean(provider, symbol, FRESH_PERIOD if mode == "incremental" else FULL_PERIOD, now_utc)

        if mode == "incremental":
            reason = ""
            if fresh.index.min() > stored.index.max():
                reason = "luka w danych: zapisane notowania kończą się przed oknem pobrania"
            else:
                reason = detect_revision(stored, fresh)
            if reason:
                mode = "full_refresh"
                result["refresh_reason"] = reason
                fresh, dropped = _fetch_clean(provider, symbol, FULL_PERIOD, now_utc)

        merged = merge_recent(stored, fresh) if mode == "incremental" else fresh
        merged = trim_years(merged, max_years)

        to_check = merged[merged.index >= fresh.index.min()] if mode == "incremental" else merged
        problems = check_rows(to_check)
        if problems:
            raise DataQualityError("; ".join(problems))

        unchanged = stored is not None and _same_values(stored, merged)
        if not unchanged:
            write_atomic(merged, path)
            result["written"] = True
        final = stored if unchanged else merged

        result.update(
            mode="unchanged" if unchanged else mode,
            partial_dropped=dropped,
            rows_before=0 if stored is None else len(stored),
            rows_after=len(final),
            new_rows=len(final.index.difference(stored.index)) if stored is not None else len(final),
            first_date=str(final.index.min().date()),
            last_date=str(final.index.max().date()),
        )
        if dropped:
            result["warnings"].append("odrzucono niepełną bieżącą sesję")
        if result.get("refresh_reason"):
            result["warnings"].append(f"pełne odświeżenie: {result['refresh_reason']}")
    except Exception as exc:  # noqa: BLE001 - błąd jednego symbolu nie może zatrzymać pozostałych
        result.update(status="fail", error=f"{type(exc).__name__}: {exc}")
    return result


def assess(results: list[dict], now_utc: pd.Timestamp, max_stale_days: int = MAX_STALE_DAYS) -> dict:
    """Ocena całości: błędy symboli, rozjazd ostatnich sesji, przestarzałe dane."""
    problems: list[str] = []
    for r in results:
        if r["status"] == "fail":
            problems.append(f"{r['symbol']}: {r.get('error', 'błąd')}")

    ok = [r for r in results if r["status"] == "ok"]
    last_dates = {r["symbol"]: r["last_date"] for r in ok}
    latest = max(last_dates.values()) if last_dates else None
    if len(set(last_dates.values())) > 1:
        problems.append("rozjazd ostatnich sesji między symbolami: "
                        + ", ".join(f"{s} {d}" for s, d in sorted(last_dates.items())))
    if latest is not None:
        today_et = pd.Timestamp(now_utc.tz_convert("America/New_York").date())
        age = (today_et - pd.Timestamp(latest)).days
        if age > max_stale_days:
            problems.append(f"dane przestarzałe: ostatnia sesja {latest}, {age} dni temu")

    warned = any(r["warnings"] for r in results)
    overall = "fail" if problems else ("warn" if warned else "ok")
    return {"overall_status": overall, "problems": problems, "latest_session": latest}


def update_all(symbols: list[str], provider: DataProvider, data_dir: str | Path = "data",
               now_utc: pd.Timestamp | None = None, force_full: bool = False,
               max_years: int = MAX_YEARS, pause: float = 2.0) -> dict:
    now_utc = now_utc if now_utc is not None else _utc_now()
    results = []
    for i, symbol in enumerate(symbols):
        results.append(update_symbol(symbol, provider, data_dir, now_utc, force_full, max_years))
        if pause and i < len(symbols) - 1:
            time.sleep(pause)

    status = assess(results, now_utc)
    status.update(
        generated_at=str(now_utc.tz_convert("UTC").tz_localize(None)),
        provider=provider.name,
        symbols={r["symbol"]: r for r in results},
    )
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / STATUS_FILE).write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def render_markdown(status: dict) -> str:
    lines = [
        "## Aktualizacja danych",
        "",
        f"Wynik: **{status['overall_status'].upper()}** | ostatnia sesja: **{status['latest_session']}** | "
        f"źródło: {status['provider']} | {status['generated_at']} UTC",
        "",
        "| Symbol | Status | Tryb | Nowe wiersze | Wiersze | Ostatnia sesja | Uwagi |",
        "|---|---|---|---|---|---|---|",
    ]
    for symbol, r in status["symbols"].items():
        if r["status"] == "fail":
            lines.append(f"| {symbol} | FAIL | - | - | - | - | {r.get('error', '')} |")
        else:
            lines.append(f"| {symbol} | OK | {r['mode']} | {r['new_rows']} | {r['rows_after']} | "
                         f"{r['last_date']} | {'; '.join(r['warnings']) or '-'} |")
    if status["problems"]:
        lines += ["", "### Problemy", ""] + [f"- {p}" for p in status["problems"]]
    return "\n".join(lines) + "\n"
