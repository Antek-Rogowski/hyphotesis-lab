"""Rejestr hipotez i generatory sygnałów (reguły o stałych parametrach).

Zasada: sygnał w sesji t korzysta wyłącznie z danych z sesji <= t (okna kroczące liczone wstecz)
oraz z kalendarza. Dzięki temu hipoteza nie może "podejrzeć" przyszłości.
Wartości sygnału: +1 (long), -1 (short), 0 (brak).

Nowe hipotezy dodaje się w osobnych plikach dekoratorem @register(...) i importuje je w
hypothesis_lab/__init__.py (patrz classics.py). Interfejs i CLI budują się z rejestru automatycznie.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParamDef:
    label: str                       # etykieta w interfejsie
    typ: type                        # int, float albo bool
    default: float | int | bool
    lo: float | None = None
    hi: float | None = None
    lo_inclusive: bool = True
    step: float | None = None        # krok pola liczbowego (dla float)


@dataclass(frozen=True)
class HypothesisDef:
    key: str
    title: str
    family: str
    description: str
    params: dict[str, ParamDef]
    fn: Callable[..., pd.Series]
    validate: Callable[[dict], None] | None = None   # sprawdzenie zależności między parametrami
    reference: str = ""


REGISTRY: dict[str, HypothesisDef] = {}
PARAM_SPECS: dict[str, dict[str, ParamDef]] = {}     # klucz -> parametry (widok na rejestr)


def register(key: str, title: str, family: str, description: str, params: dict[str, ParamDef],
             validate: Callable[[dict], None] | None = None, reference: str = ""):
    """Dekorator rejestrujący funkcję sygnału jako hipotezę dostępną w silniku, CLI i interfejsie."""
    def decorator(fn: Callable[..., pd.Series]):
        if key in REGISTRY:
            raise ValueError(f"hipoteza {key!r} jest już zarejestrowana")
        REGISTRY[key] = HypothesisDef(key, title, family, description, params, fn, validate, reference)
        PARAM_SPECS[key] = params
        return fn
    return decorator


def _coerce(name: str, d: ParamDef, raw):
    if d.typ is bool:
        if isinstance(raw, (bool, np.bool_)) or raw in (0, 1):
            return bool(raw)
        raise ValueError(f"parametr {name} musi być wartością logiczną (dostałem {raw!r})")
    if d.typ is int:
        if not float(raw).is_integer():
            raise ValueError(f"parametr {name} musi być liczbą całkowitą (dostałem {raw!r})")
        value = int(raw)
    else:
        value = float(raw)
    above_lo = d.lo is None or (value >= d.lo if d.lo_inclusive else value > d.lo)
    if not (above_lo and (d.hi is None or value <= d.hi)):
        bracket = "[" if d.lo_inclusive else "("
        raise ValueError(f"parametr {name}={value} poza zakresem {bracket}{d.lo}, {d.hi}]")
    return value


def resolve_params(hypothesis_type: str, params: dict | None = None) -> dict:
    """Uzupełnia domyślne wartości, rzutuje typy, sprawdza zakresy i zależności między parametrami."""
    if hypothesis_type not in REGISTRY:
        raise ValueError(f"nieznany typ hipotezy: {hypothesis_type!r}")
    definition = REGISTRY[hypothesis_type]
    params = dict(params or {})
    unknown = set(params) - set(definition.params)
    if unknown:
        raise ValueError(f"nieznane parametry dla {hypothesis_type}: {sorted(unknown)}")
    resolved = {name: _coerce(name, d, params.get(name, d.default)) for name, d in definition.params.items()}
    if definition.validate is not None:
        definition.validate(resolved)
    return resolved


def to_signal(values, index: pd.Index) -> pd.Series:
    return pd.Series(np.asarray(values).astype("int8"), index=index, name="signal")


def compute_signals(prices: pd.DataFrame, hypothesis_type: str, params: dict | None = None) -> pd.Series:
    resolved = resolve_params(hypothesis_type, params)
    return REGISTRY[hypothesis_type].fn(prices, **resolved)


# --------------------------------------------------------------------------
# Hipotezy podstawowe
# --------------------------------------------------------------------------
@register(
    "mean_reversion", "Powrót do średniej", "Odwrócenie",
    "Long, gdy cena jest k odchyleń standardowych poniżej średniej z W sesji, short, gdy powyżej "
    "(odpowiednik wstęg Bollingera).",
    {"W": ParamDef("Okno średniej W (sesje)", int, 20, 2, 500),
     "k": ParamDef("Próg k (odchylenia standardowe)", float, 2.0, 0.0, 10.0, False, 0.1)},
)
def mean_reversion(prices: pd.DataFrame, W: int, k: float) -> pd.Series:
    c = prices["adj_close"]
    z = (c - c.rolling(W).mean()) / c.rolling(W).std()
    return to_signal(np.select([z < -k, z > k], [1, -1], default=0), c.index)   # NaN -> brak sygnału


@register(
    "momentum", "Momentum", "Momentum",
    "Long po dodatnim zwrocie z ostatnich L sesji, short po ujemnym (powyżej progu).",
    {"L": ParamDef("Okno zwrotu L (sesje)", int, 20, 1, 500),
     "threshold": ParamDef("Minimalny zwrot (ułamek, 0.01 = 1%)", float, 0.0, 0.0, 1.0, True, 0.005)},
)
def momentum(prices: pd.DataFrame, L: int, threshold: float) -> pd.Series:
    c = prices["adj_close"]
    r = c / c.shift(L) - 1.0
    return to_signal(np.select([r > threshold, r < -threshold], [1, -1], default=0), c.index)


@register(
    "shock_rebound", "Odbicie po gwałtownym ruchu", "Odwrócenie",
    "Long po dziennym spadku o co najmniej X%, short po dziennym wzroście o co najmniej X%.",
    {"x_pct": ParamDef("Próg dziennej zmiany X (%)", float, 3.0, 0.0, 50.0, False, 0.1)},
)
def shock_rebound(prices: pd.DataFrame, x_pct: float) -> pd.Series:
    c = prices["adj_close"]
    r = c / c.shift(1) - 1.0
    x = x_pct / 100.0
    return to_signal(np.select([r <= -x, r >= x], [1, -1], default=0), c.index)
