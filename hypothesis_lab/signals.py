"""Generatory sygnałów.

Zasada: sygnał w sesji t korzysta wyłącznie z danych z sesji <= t (okna kroczące
liczone wstecz). Dzięki temu hipoteza nie może "podejrzeć" przyszłości.
Wartości sygnału: +1 (long), -1 (short), 0 (brak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# nazwa hipotezy -> {parametr: (typ, domyślna, min, max, min_włącznie)}
PARAM_SPECS: dict[str, dict[str, tuple]] = {
    "mean_reversion": {
        "W": (int, 20, 2, 500, True),        # długość okna średniej (sesje)
        "k": (float, 2.0, 0.0, 10.0, False),  # próg w odchyleniach standardowych
    },
    "momentum": {
        "L": (int, 20, 1, 500, True),             # okno zwrotu (sesje)
        "threshold": (float, 0.0, 0.0, 1.0, True),  # minimalny |zwrot|, ułamek
    },
    "shock_rebound": {
        "x_pct": (float, 3.0, 0.0, 50.0, False),  # próg dziennej zmiany w procentach
    },
}


def resolve_params(hypothesis_type: str, params: dict | None = None) -> dict:
    """Uzupełnia domyślne wartości, rzutuje typy i sprawdza zakresy."""
    if hypothesis_type not in PARAM_SPECS:
        raise ValueError(f"nieznany typ hipotezy: {hypothesis_type!r}")
    spec = PARAM_SPECS[hypothesis_type]
    params = dict(params or {})
    unknown = set(params) - set(spec)
    if unknown:
        raise ValueError(f"nieznane parametry dla {hypothesis_type}: {sorted(unknown)}")

    resolved: dict = {}
    for name, (typ, default, lo, hi, lo_inclusive) in spec.items():
        raw = params.get(name, default)
        if typ is int and not float(raw).is_integer():
            raise ValueError(f"parametr {name} musi być liczbą całkowitą (dostałem {raw!r})")
        value = typ(raw)
        above_lo = value >= lo if lo_inclusive else value > lo
        if not (above_lo and value <= hi):
            bound = "[" if lo_inclusive else "("
            raise ValueError(f"parametr {name}={value} poza zakresem {bound}{lo}, {hi}]")
        resolved[name] = value
    return resolved


def _to_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(values.astype("int8"), index=index, name="signal")


def mean_reversion(prices: pd.DataFrame, W: int, k: float) -> pd.Series:
    """Long, gdy cena jest k odchyleń poniżej średniej z W sesji; short, gdy powyżej."""
    c = prices["adj_close"]
    z = (c - c.rolling(W).mean()) / c.rolling(W).std()
    values = np.select([z < -k, z > k], [1, -1], default=0)  # NaN -> brak sygnału
    return _to_series(values, c.index)


def momentum(prices: pd.DataFrame, L: int, threshold: float) -> pd.Series:
    """Long po dodatnim zwrocie z L sesji, short po ujemnym (powyżej progu)."""
    c = prices["adj_close"]
    r = c / c.shift(L) - 1.0
    values = np.select([r > threshold, r < -threshold], [1, -1], default=0)
    return _to_series(values, c.index)


def shock_rebound(prices: pd.DataFrame, x_pct: float) -> pd.Series:
    """Long po dziennym spadku o co najmniej x%, short po dziennym wzroście o co najmniej x%."""
    c = prices["adj_close"]
    r = c / c.shift(1) - 1.0
    x = x_pct / 100.0
    values = np.select([r <= -x, r >= x], [1, -1], default=0)
    return _to_series(values, c.index)


_FUNCTIONS = {
    "mean_reversion": mean_reversion,
    "momentum": momentum,
    "shock_rebound": shock_rebound,
}


def compute_signals(prices: pd.DataFrame, hypothesis_type: str, params: dict | None = None) -> pd.Series:
    resolved = resolve_params(hypothesis_type, params)
    return _FUNCTIONS[hypothesis_type](prices, **resolved)
