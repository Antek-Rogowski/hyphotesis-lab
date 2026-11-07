"""Dane syntetyczne do testów i demonstracji (nie do analiz rynkowych)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import prepare_prices


def make_prices(close, open_=None, volume=None, start: str = "2020-01-02") -> pd.DataFrame:
    """Buduje ramkę notowań z podanych list (ceny skorygowane = surowe). Do testów ręcznych."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    open_arr = close.copy() if open_ is None else np.asarray(open_, dtype=float)
    vol = np.full(n, 1_000_000.0) if volume is None else np.asarray(volume, dtype=float)
    idx = pd.bdate_range(start=start, periods=n)
    raw = pd.DataFrame(
        {
            "open": open_arr,
            "high": np.maximum(open_arr, close),
            "low": np.minimum(open_arr, close),
            "close": close,
            "adj_close": close,
            "volume": vol,
        },
        index=idx,
    )
    return prepare_prices(raw)


def random_walk_prices(seed: int = 0, n: int = 2500, drift: float = 0.0002, vol: float = 0.01,
                       start: str = "2016-11-07") -> pd.DataFrame:
    """Losowe błądzenie: nie ma w nim żadnej struktury do wykrycia."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1.0 + rng.normal(0.0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return make_prices(close, open_, volume, start)
