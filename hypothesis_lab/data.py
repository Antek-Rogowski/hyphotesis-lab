"""Wczytywanie i przygotowanie notowań z plików Parquet."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def prepare_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje ramkę notowań i dodaje kolumny pomocnicze.

    Dodaje:
      adj_open     - otwarcie skorygowane tym samym współczynnikiem co adj_close/close
                     (potrzebne, gdy wykonanie transakcji następuje po otwarciu),
      zero_volume  - znacznik sesji z zerowym wolumenem (cena może być nieaktualna).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"brakuje kolumn: {missing}")

    out = df[REQUIRED_COLUMNS].astype(float).copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    if out.index.has_duplicates:
        raise ValueError("indeks zawiera zduplikowane daty")
    out = out.sort_index()

    factor = out["adj_close"] / out["close"]
    out["adj_open"] = out["open"] * factor
    out["zero_volume"] = out["volume"] <= 0
    return out


def load_prices(symbol: str, data_dir: str | Path = "data") -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol.strip().upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"brak pliku danych: {path}")
    return prepare_prices(pd.read_parquet(path))


def slice_period(prices: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is None and end is None:
        return prices
    return prices.loc[start:end]


def data_fingerprint(prices: pd.DataFrame) -> str:
    """Skrót zawartości danych: pozwala sprawdzić, że wynik pochodzi z tej samej migawki."""
    hashed = pd.util.hash_pandas_object(prices[REQUIRED_COLUMNS], index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:12]
