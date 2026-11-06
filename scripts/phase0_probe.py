#!/usr/bin/env python3
"""Faza 0: sonda danych.

Sprawdza, czy da się pobrać ~10 lat dziennych notowań ETF-ów z yfinance
(docelowo z runnera GitHub Actions), waliduje je, zapisuje do Parquet
i tworzy raport (JSON + podsumowanie w Actions).

Przykłady:
    python scripts/phase0_probe.py                       # prawdziwe dane
    python scripts/phase0_probe.py --tickers GLD,SLV     # wybrane symbole
    python scripts/phase0_probe.py --synthetic           # dane losowe, offline
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["GLD", "IAU", "SLV", "CPER", "USO"]
COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]

MIN_ROWS = 1500          # ok. 6 lat sesji; 10 lat to ok. 2500
EXPECTED_YEARS = 9.5     # poniżej tego raportujemy ostrzeżenie o krótszej historii
MAX_STALE_DAYS = 7       # starsze ostatnie notowanie => ostrzeżenie
GAP_WARN_DAYS = 5        # przerwa dłuższa niż tyle dni kalendarzowych => ostrzeżenie
JUMP_WARN = 0.15         # dzienna zmiana skorygowanej ceny większa niż 15% => ostrzeżenie

STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}


def utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


# --------------------------------------------------------------------------
# Pobieranie
# --------------------------------------------------------------------------
def fetch_yfinance(symbol: str, period: str, retries: int = 3, pause: float = 5.0) -> pd.DataFrame:
    """Pobiera dzienne notowania. Surowe ceny (auto_adjust=False) + Adj Close."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(symbol).history(
                period=period, interval="1d", auto_adjust=False, actions=False
            )
            if raw is None or raw.empty:
                raise ValueError("pusta odpowiedź z Yahoo (sprawdź log: limit zapytań, blokada adresu IP, błąd sieci lub zły symbol)")
            return raw
        except Exception as exc:  # noqa: BLE001 - chcemy złapać wszystko i zaraportować
            last_err = exc
            if attempt < retries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def make_synthetic(symbol: str, years: int = 10) -> pd.DataFrame:
    """Losowe błądzenie w formacie zbliżonym do yfinance. Tylko do testów offline."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    end = utc_now_naive().normalize()
    idx = pd.bdate_range(end=end, periods=int(years * 252))
    rets = rng.normal(0.0002, 0.01, len(idx))
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
    volume = rng.integers(1_000_000, 5_000_000, len(idx))
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=idx,
    )
    return df


# --------------------------------------------------------------------------
# Standaryzacja i walidacja
# --------------------------------------------------------------------------
def standardize(raw: pd.DataFrame, symbol: str, source: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
        notes.append("brak kolumny Adj Close - użyto Close")

    df = df[COLUMNS].copy()

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.insert(0, "symbol", symbol)
    df["source"] = source
    df["ingested_at"] = utc_now_naive()
    return df, notes


def validate(df: pd.DataFrame) -> dict:
    """Zwraca metryki i listy problemów (fail = blokujące, warn = do wyjaśnienia)."""
    fails: list[str] = []
    warns: list[str] = []

    rows = len(df)
    first, last = df.index.min(), df.index.max()
    years = (last - first).days / 365.25
    stale_days = int((utc_now_naive().normalize() - last).days)

    nan_cells = int(df[PRICE_COLUMNS].isna().sum().sum())
    nonpositive = int((df[PRICE_COLUMNS] <= 0).sum().sum())
    high_lt_low = int((df["high"] < df["low"]).sum())
    zero_volume = int((df["volume"] <= 0).sum())

    gaps = df.index.to_series().diff().dt.days.dropna()
    big_gaps = int((gaps > GAP_WARN_DAYS).sum())

    rets = df["adj_close"].pct_change().dropna()
    max_jump = float(rets.abs().max()) if len(rets) else 0.0

    if rows < MIN_ROWS:
        fails.append(f"za mało wierszy ({rows} < {MIN_ROWS})")
    if nonpositive:
        fails.append(f"ceny <= 0: {nonpositive}")
    if high_lt_low:
        fails.append(f"high < low w {high_lt_low} wierszach")

    if years < EXPECTED_YEARS:
        warns.append(f"krótsza historia: {years:.1f} lat")
    if stale_days > MAX_STALE_DAYS:
        warns.append(f"ostatnie notowanie sprzed {stale_days} dni")
    if nan_cells:
        warns.append(f"puste komórki cen: {nan_cells}")
    if big_gaps:
        warns.append(f"przerwy > {GAP_WARN_DAYS} dni: {big_gaps}")
    if max_jump > JUMP_WARN:
        warns.append(f"największa dzienna zmiana {max_jump:.1%}")
    if zero_volume:
        warns.append(f"sesje z zerowym wolumenem: {zero_volume}")

    status = "fail" if fails else ("warn" if warns else "ok")
    return {
        "status": status,
        "rows": rows,
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "years": round(years, 2),
        "stale_days": stale_days,
        "max_abs_daily_return": round(max_jump, 4),
        "fails": fails,
        "warns": warns,
    }


# --------------------------------------------------------------------------
# Raportowanie
# --------------------------------------------------------------------------
def render_markdown(report: dict) -> str:
    lines = [
        "## Faza 0 - sonda danych",
        "",
        f"Źródło: **{report['source']}** | okres: **{report['period']}** | "
        f"wynik ogólny: **{report['overall_status'].upper()}**",
        "",
        "| Symbol | Status | Wiersze | Od | Do | Lata | Uwagi |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        if r["status"] == "fail" and "error" in r:
            lines.append(f"| {r['symbol']} | FAIL | - | - | - | - | {r['error']} |")
            continue
        remarks = "; ".join(r["fails"] + r["warns"] + r.get("notes", [])) or "-"
        lines.append(
            f"| {r['symbol']} | {r['status'].upper()} | {r['rows']} | {r['first_date']} | "
            f"{r['last_date']} | {r['years']} | {remarks} |"
        )
    env = report["environment"]
    lines += [
        "",
        f"Środowisko: Python {env['python']}, pandas {env['pandas']}, "
        f"yfinance {env['yfinance']}, {env['platform']}",
    ]
    return "\n".join(lines) + "\n"


def environment_info() -> dict:
    try:
        import yfinance

        yf_version = getattr(yfinance, "__version__", "?")
    except Exception:  # noqa: BLE001
        yf_version = "niezainstalowane"
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "yfinance": yf_version,
        "platform": platform.platform(),
        "github_actions": bool(os.environ.get("GITHUB_ACTIONS")),
    }


# --------------------------------------------------------------------------
# Główna procedura
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="symbole po przecinku")
    parser.add_argument("--period", default="10y", help="okres dla yfinance, np. 10y")
    parser.add_argument("--out-dir", default=None, help="katalog wyjściowy (domyślnie data lub data_synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="dane losowe zamiast Yahoo (tylko testy offline)")
    parser.add_argument("--pause", type=float, default=2.0, help="przerwa między symbolami w sekundach")
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("Brak symboli do pobrania.", file=sys.stderr)
        return 2

    source = "synthetic" if args.synthetic else "yfinance"
    out_dir = Path(args.out_dir or ("data_synthetic" if args.synthetic else "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, symbol in enumerate(tickers):
        t0 = time.time()
        try:
            raw = make_synthetic(symbol) if args.synthetic else fetch_yfinance(symbol, args.period)
            df, notes = standardize(raw, symbol, source)
            result = validate(df)
            result["notes"] = notes
            df.to_parquet(out_dir / f"{symbol}.parquet", engine="pyarrow")
            result["file"] = f"{symbol}.parquet"
        except Exception as exc:  # noqa: BLE001
            result = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        result["symbol"] = symbol
        result["seconds"] = round(time.time() - t0, 1)
        results.append(result)
        print(f"[{result['status'].upper():4}] {symbol} ({result['seconds']}s)", flush=True)
        if i < len(tickers) - 1 and not args.synthetic:
            time.sleep(args.pause)

    overall = max((r["status"] for r in results), key=lambda s: STATUS_ORDER[s])
    report = {
        "generated_at": str(utc_now_naive()),
        "source": source,
        "period": args.period,
        "overall_status": overall,
        "environment": environment_info(),
        "results": results,
    }

    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report)
    print("\n" + markdown)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(markdown)

    return 1 if overall == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
