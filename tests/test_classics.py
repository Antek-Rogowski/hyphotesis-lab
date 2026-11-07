"""Testy pięciu klasycznych hipotez: ręcznie policzone sygnały, walidacja parametrów, metadane."""
import numpy as np
import pandas as pd
import pytest

from hypothesis_lab import HypothesisSpec, run_hypothesis
from hypothesis_lab.signals import REGISTRY, ParamDef, compute_signals, register, resolve_params, to_signal
from hypothesis_lab.synthetic import make_prices, random_walk_prices

CLASSICS = ["sma_crossover", "donchian_breakout", "rsi_reversal", "tsmom", "turn_of_month"]


def sig(prices, htype, **params):
    return compute_signals(prices, htype, params)


def test_registry_contains_core_and_classic_hypotheses_with_complete_metadata():
    assert set(CLASSICS) <= set(REGISTRY) and len(REGISTRY) >= 8
    for key, d in REGISTRY.items():
        assert d.title and d.family and d.description, key
        assert d.params, key
        for name, p in d.params.items():
            assert p.label, f"{key}.{name} bez etykiety"
            assert p.lo is not None and p.hi is not None or p.typ is bool, f"{key}.{name} bez zakresu"


def test_sma_crossover_state_and_event_modes():
    # spadek, potem wzrost: szybka średnia przecina wolną od dołu, a później od góry
    close = list(np.linspace(100, 80, 25)) + list(np.linspace(80, 130, 30)) + list(np.linspace(130, 70, 30))
    prices = make_prices(close)
    state = sig(prices, "sma_crossover", fast=3, slow=10, only_cross=False)
    event = sig(prices, "sma_crossover", fast=3, slow=10, only_cross=True)
    assert state.iloc[:9].eq(0).all()                       # okres rozgrzewki bez sygnału
    assert state.iloc[15] == -1 and state.iloc[50] == 1 and state.iloc[-1] == -1
    flips = state[(state != state.shift(1)) & (state.shift(1) != 0) & (state != 0)]
    assert list(event[event != 0].index) == list(flips.index)       # zdarzenia dokładnie w dniach przecięcia
    assert (event[event != 0] == state[event != 0]).all()
    assert len(event[event != 0]) == 2


def test_donchian_breakout_uses_only_previous_sessions():
    close = [100.0] * 12 + [105.0, 105.0] + [90.0]
    prices = make_prices(close)
    s = sig(prices, "donchian_breakout", N=5)
    # wybicie w górę w sesji 12 (ponad maksimum z sesji 7..11), kolejna sesja na tym samym poziomie nie jest wybiciem
    assert s.iloc[12] == 1 and s.iloc[13] == 0
    assert s.iloc[14] == -1                                  # zamknięcie poniżej minimum z poprzednich 5 sesji
    assert s.iloc[:5].eq(0).all()


def test_rsi_reversal_on_monotonic_series():
    up = make_prices(np.linspace(100, 160, 60))
    down = make_prices(np.linspace(160, 100, 60))
    assert sig(up, "rsi_reversal", n=14, low=30.0, high=70.0).iloc[-1] == -1        # RSI = 100, wykupienie
    assert sig(down, "rsi_reversal", n=14, low=30.0, high=70.0).iloc[-1] == 1       # RSI = 0, wyprzedanie
    assert sig(up, "rsi_reversal", n=14, low=30.0, high=70.0).iloc[:13].eq(0).all()  # rozgrzewka


def test_rsi_value_matches_independent_wilder_computation():
    prices = random_walk_prices(4, n=300)
    c = prices["adj_close"].to_numpy()
    n, alpha = 14, 1.0 / 14
    d = np.diff(c)
    rsi = np.full(len(c), np.nan)
    for t in range(1, len(c)):                               # rekurencja Wildera zapisana ręcznie
        gain, loss = max(d[t - 1], 0.0), max(-d[t - 1], 0.0)
        if t == 1:
            avg_gain, avg_loss = gain, loss                  # EWMA startuje od pierwszej obserwacji
        else:
            avg_gain = (1 - alpha) * avg_gain + alpha * gain
            avg_loss = (1 - alpha) * avg_loss + alpha * loss
        rsi[t] = 100 - 100 / (1 + avg_gain / avg_loss)
    expected = np.where(np.isnan(rsi), 0, np.select([rsi < 30, rsi > 70], [1, -1], 0))
    got = sig(prices, "rsi_reversal", n=n, low=30.0, high=70.0).to_numpy()
    assert (got[n + 1:] == expected[n + 1:]).all()           # po rozgrzewce sygnały są identyczne
    assert (got != 0).sum() > 10                             # test nie jest pusty


def test_tsmom_matches_hand_computation():
    close = [100 + 8 * np.sin(i / 4.0) + 0.3 * i for i in range(70)]
    prices = make_prices(close)
    s = sig(prices, "tsmom", lookback=20, skip=3)
    c = np.array(close, float)
    for t in range(20, len(c)):
        r = c[t - 3] / c[t - 20] - 1                         # zwrot z okna wstecz z pominięciem 3 ostatnich sesji
        assert s.iloc[t] == (1 if r > 0 else -1 if r < 0 else 0), t
    assert s.iloc[:20].eq(0).all()
    assert set(np.unique(s.iloc[20:])) == {-1, 1}            # w danych występują oba kierunki


def test_turn_of_month_signals_last_business_day_and_offsets():
    prices = make_prices(np.linspace(100, 110, 90), start="2024-01-01")   # sesje pon.-pt. bez świąt
    last = sig(prices, "turn_of_month", days_before_end=0)
    on = [d.strftime("%Y-%m-%d") for d in last[last == 1].index]
    assert on[:2] == ["2024-01-31", "2024-02-29"]                        # środa i czwartek
    assert (last[last == 1].index.to_period("M").duplicated() == False).all()   # jeden sygnał na miesiąc
    two = sig(prices, "turn_of_month", days_before_end=2)
    assert [d.strftime("%Y-%m-%d") for d in two[two == 1].index][:2] == ["2024-01-29", "2024-02-27"]
    assert set(np.unique(last)) <= {0, 1}                                # tylko long


@pytest.mark.parametrize("htype,params", [
    ("sma_crossover", {"fast": 50, "slow": 50}),
    ("sma_crossover", {"fast": 60, "slow": 50}),
    ("rsi_reversal", {"low": 60.0}),
    ("rsi_reversal", {"low": 30.0, "high": 20.0}),
    ("tsmom", {"lookback": 30, "skip": 30}),
    ("tsmom", {"lookback": 30, "skip": 60}),
    ("turn_of_month", {"days_before_end": 11}),
    ("donchian_breakout", {"N": 1}),
    ("sma_crossover", {"only_cross": 3}),
])
def test_invalid_parameters_are_rejected(htype, params):
    with pytest.raises(ValueError):
        resolve_params(htype, params)


def test_bool_parameter_accepts_zero_one_and_bool():
    assert resolve_params("sma_crossover", {"only_cross": 1})["only_cross"] is True
    assert resolve_params("sma_crossover", {"only_cross": False})["only_cross"] is False
    assert resolve_params("sma_crossover")["only_cross"] is False       # wartość domyślna


@pytest.mark.parametrize("htype", CLASSICS)
def test_each_classic_runs_end_to_end_and_is_serialisable(htype):
    prices = random_walk_prices(21, n=2000)
    spec = HypothesisSpec(symbol="SYN", hypothesis_type=htype, horizon=4, n_boot=100)
    res = run_hypothesis(prices, spec)
    full = res.summaries["full"]
    assert full["n_trades"] > 10
    assert HypothesisSpec.from_json(spec.to_json()).spec_hash() == spec.spec_hash()
    import json
    json.dumps(res.to_dict())


def test_new_hypothesis_can_be_registered_from_another_file_and_used_immediately():
    @register("test_always_long_mondays", "Poniedziałki", "Kalendarz", "Long w każdy poniedziałek.",
              {"dummy": ParamDef("Parametr", int, 1, 0, 5)})
    def mondays(prices, dummy):
        return to_signal((prices.index.weekday == 0).astype(int), prices.index)

    try:
        prices = random_walk_prices(2, n=600)
        s = compute_signals(prices, "test_always_long_mondays")
        assert set(s[s == 1].index.weekday) == {0}
        res = run_hypothesis(prices, HypothesisSpec(symbol="SYN", hypothesis_type="test_always_long_mondays",
                                                    horizon=2, n_boot=50, sensitivity_top_days=0))
        assert res.summaries["full"]["n_trades"] > 10
        with pytest.raises(ValueError):
            register("test_always_long_mondays", "x", "x", "x", {})(mondays)   # duplikat jest odrzucany
    finally:
        from hypothesis_lab.signals import PARAM_SPECS
        REGISTRY.pop("test_always_long_mondays", None)
        PARAM_SPECS.pop("test_always_long_mondays", None)
