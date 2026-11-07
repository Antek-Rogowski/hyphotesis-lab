"""Pięć klasycznych hipotez popularnych w środowiskach quantowych (jeden instrument, dane dzienne).

Plik jest niezależny od silnika: hipotezy rejestrują się dekoratorem @register i są widoczne w CLI i interfejsie
po zaimportowaniu tego modułu (robi to hypothesis_lab/__init__.py). Wzór dla własnych plików:

    from .signals import ParamDef, register, to_signal

    @register("moja_hipoteza", "Tytuł", "Rodzina", "Opis.", {"N": ParamDef("Okno N", int, 20, 2, 250)})
    def moja_hipoteza(prices, N):          # prices: ramka z kolumnami adj_close, adj_high, adj_low, adj_open, volume...
        ...                                # korzystaj tylko z danych z sesji <= t
        return to_signal(wartosci, prices.index)   # +1 long, -1 short, 0 brak
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import ParamDef, register, to_signal


# --------------------------------------------------------------------------
# 1. Przecięcie średnich kroczących (podążanie za trendem)
# --------------------------------------------------------------------------
def _check_sma(p: dict) -> None:
    if p["fast"] >= p["slow"]:
        raise ValueError("średnia szybka musi być krótsza niż wolna (fast < slow)")


@register(
    "sma_crossover", "Przecięcie średnich kroczących", "Trend",
    "Long, gdy średnia szybka leży powyżej wolnej, short, gdy poniżej. Opcjonalnie sygnał tylko w dniu "
    "przecięcia (złoty i martwy krzyż).",
    {"fast": ParamDef("Średnia szybka (sesje)", int, 20, 2, 250),
     "slow": ParamDef("Średnia wolna (sesje)", int, 100, 3, 500),
     "only_cross": ParamDef("Sygnał tylko w dniu przecięcia", bool, False)},
    validate=_check_sma,
    reference="Brock, Lakonishok i LeBaron (1992); Faber (2007)",
)
def sma_crossover(prices: pd.DataFrame, fast: int, slow: int, only_cross: bool) -> pd.Series:
    c = prices["adj_close"]
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    state = pd.Series(np.select([f > s, f < s], [1, -1], default=0), index=c.index)   # NaN -> 0
    if only_cross:
        prev = state.shift(1).fillna(0)
        state = state.where((prev != 0) & (state != 0) & (state != prev), 0)
    return to_signal(state.to_numpy(), c.index)


# --------------------------------------------------------------------------
# 2. Wybicie z kanału Donchiana (strategia Żółwi)
# --------------------------------------------------------------------------
@register(
    "donchian_breakout", "Wybicie z kanału Donchiana", "Wybicie",
    "Long, gdy zamknięcie przekracza maksimum z poprzednich N sesji, short, gdy schodzi poniżej ich minimum.",
    {"N": ParamDef("Okno kanału N (sesje)", int, 20, 2, 250)},
    reference="Richard Donchian; strategia Żółwi (lata 80.)",
)
def donchian_breakout(prices: pd.DataFrame, N: int) -> pd.Series:
    c = prices["adj_close"]
    upper = prices["adj_high"].rolling(N).max().shift(1)    # maksimum z N sesji PRZED bieżącą
    lower = prices["adj_low"].rolling(N).min().shift(1)
    return to_signal(np.select([c > upper, c < lower], [1, -1], default=0), c.index)


# --------------------------------------------------------------------------
# 3. RSI Wildera: odwrócenie po wyprzedaniu i wykupieniu
# --------------------------------------------------------------------------
def _check_rsi(p: dict) -> None:
    if p["low"] >= p["high"]:
        raise ValueError("próg wyprzedania musi być niższy niż próg wykupienia (low < high)")


@register(
    "rsi_reversal", "RSI: wyprzedanie i wykupienie", "Odwrócenie",
    "Long, gdy RSI spada poniżej progu wyprzedania, short, gdy rośnie powyżej progu wykupienia. "
    "RSI liczone wygładzaniem Wildera (EWMA z alfa = 1/n).",
    {"n": ParamDef("Okno RSI n (sesje)", int, 14, 2, 100),
     "low": ParamDef("Próg wyprzedania", float, 30.0, 1.0, 49.0, True, 1.0),
     "high": ParamDef("Próg wykupienia", float, 70.0, 51.0, 99.0, True, 1.0)},
    validate=_check_rsi,
    reference="J. Welles Wilder (1978); wariant krótkoterminowy RSI(2): Connors",
)
def rsi_reversal(prices: pd.DataFrame, n: int, low: float, high: float) -> pd.Series:
    c = prices["adj_close"]
    delta = c.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)      # strata = 0 daje RSI = 100
    return to_signal(np.select([rsi < low, rsi > high], [1, -1], default=0), c.index)


# --------------------------------------------------------------------------
# 4. Momentum szeregu czasowego 12-1
# --------------------------------------------------------------------------
def _check_tsmom(p: dict) -> None:
    if p["skip"] >= p["lookback"]:
        raise ValueError("pominięcie ostatnich sesji musi być krótsze niż okno wstecz (skip < lookback)")


@register(
    "tsmom", "Momentum szeregu czasowego 12-1", "Momentum",
    "Long, gdy zwrot z okna wstecz (domyślnie 12 miesięcy) z pominięciem ostatnich sesji (domyślnie miesiąc) "
    "jest dodatni, short, gdy ujemny.",
    {"lookback": ParamDef("Okno wstecz (sesje)", int, 252, 20, 756),
     "skip": ParamDef("Pominięte ostatnie sesje", int, 21, 0, 120)},
    validate=_check_tsmom,
    reference="Moskowitz, Ooi i Pedersen (2012)",
)
def tsmom(prices: pd.DataFrame, lookback: int, skip: int) -> pd.Series:
    c = prices["adj_close"]
    r = c.shift(skip) / c.shift(lookback) - 1.0
    return to_signal(np.select([r > 0, r < 0], [1, -1], default=0), c.index)


# --------------------------------------------------------------------------
# 5. Efekt przełomu miesiąca (anomalia kalendarzowa)
# --------------------------------------------------------------------------
@register(
    "turn_of_month", "Efekt przełomu miesiąca", "Kalendarz",
    "Long na zamknięciu ostatniego dnia roboczego miesiąca (lub kilka dni wcześniej). Ustaw horyzont ok. 4 sesji, "
    "żeby objąć początek kolejnego miesiąca. Kalendarz to dni robocze pon.-pt., bez świąt giełdowych.",
    {"days_before_end": ParamDef("Dni robocze przed końcem miesiąca", int, 0, 0, 10)},
    reference="Ariel (1987); Lakonishok i Smidt (1988)",
)
def turn_of_month(prices: pd.DataFrame, days_before_end: int) -> pd.Series:
    idx = pd.DatetimeIndex(prices.index)
    month_end = idx.to_period("M").to_timestamp(how="end").normalize()
    start = (idx + pd.Timedelta(days=1)).values.astype("datetime64[D]")
    stop = (month_end + pd.Timedelta(days=1)).values.astype("datetime64[D]")
    remaining = np.busday_count(start, stop)          # dni robocze po t do końca miesiąca (wiedza kalendarzowa)
    return to_signal(np.where(remaining == days_before_end, 1, 0), idx)
