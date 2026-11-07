"""Testy silnika na małych, ręcznie policzonych przykładach."""
import numpy as np
import pandas as pd
import pytest

from hypothesis_lab import CostModel, HypothesisSpec, evaluate_signals
from hypothesis_lab.synthetic import make_prices, random_walk_prices


def spec(**kw):
    base = dict(symbol="T", hypothesis_type="momentum", horizon=2, holdout_frac=0.0,
                n_boot=50, sensitivity_top_days=0)
    base.update(kw)
    return HypothesisSpec(**base)


def signal_at(prices, mapping):
    s = pd.Series(0, index=prices.index)
    for pos, val in mapping.items():
        s.iloc[pos] = val
    return s


def test_long_and_short_close_execution_by_hand():
    prices = make_prices([100, 105, 110, 100, 90, 95])
    res = evaluate_signals(prices, signal_at(prices, {0: 1, 3: -1}), spec())
    t = res.trades
    assert len(t) == 2
    assert t["side"].tolist() == [1, -1]
    assert t["net_return"].tolist() == pytest.approx([0.10, 0.05])   # 110/100-1 oraz 1-95/100
    assert t["hit"].tolist() == [True, True]
    assert str(t["entry_date"].iloc[0].date()) == str(prices.index[0].date())
    assert str(t["exit_date"].iloc[0].date()) == str(prices.index[2].date())


def test_open_next_session_execution():
    prices = make_prices([100] * 6, open_=[100, 101, 102, 103, 104, 105])
    res = evaluate_signals(prices, signal_at(prices, {0: 1}), spec(execution="open_t+1"))
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(101)      # otwarcie sesji t+1
    assert t["exit_price"] == pytest.approx(103)       # otwarcie sesji t+1+N
    assert t["net_return"] == pytest.approx(103 / 101 - 1)


def test_costs_commission_and_borrow():
    prices = make_prices([100, 90, 81])
    sig = signal_at(prices, {0: -1})
    costs = CostModel(enabled=True, commission_bps=10, borrow_bps_annual=252)
    res = evaluate_signals(prices, sig, spec(costs=costs))
    # short: 1 - 0.81 = 0.19; koszt wejścia i wyjścia 2*10bp = 0.002; pożyczka 252bp/252*2 sesje = 0.0002
    assert res.trades["net_return"].iloc[0] == pytest.approx(0.19 - 0.002 - 0.0002)


def test_costs_flag_off_ignores_values():
    prices = make_prices([100, 110, 121])
    sig = signal_at(prices, {0: 1})
    res = evaluate_signals(prices, sig, spec(costs=CostModel(enabled=False, commission_bps=500)))
    assert res.trades["net_return"].iloc[0] == pytest.approx(0.21)


def test_target_threshold():
    prices = make_prices([100, 102, 104])           # zwrot 4%
    sig = signal_at(prices, {0: 1})
    hit = evaluate_signals(prices, sig, spec(target="threshold", target_threshold_pct=3.0)).trades["hit"].iloc[0]
    miss = evaluate_signals(prices, sig, spec(target="threshold", target_threshold_pct=5.0)).trades["hit"].iloc[0]
    assert bool(hit) is True and bool(miss) is False


def test_no_overlap_vs_overlap():
    prices = make_prices(np.linspace(100, 120, 10))
    sig = signal_at(prices, {0: 1, 1: 1, 2: 1, 3: 1})
    assert len(evaluate_signals(prices, sig, spec(no_overlap=True)).trades) == 2      # sesje 0 i 2
    assert len(evaluate_signals(prices, sig, spec(no_overlap=False)).trades) == 4


def test_incomplete_forward_window_is_dropped():
    prices = make_prices([100, 101, 102, 103, 104, 105])
    res = evaluate_signals(prices, signal_at(prices, {4: 1}), spec(horizon=3))
    assert res.summaries["full"]["n_trades"] == 0


def test_zero_volume_window_excluded_or_kept():
    prices = make_prices([100, 101, 102, 103, 104], volume=[1e6, 1e6, 0, 1e6, 1e6])
    sig = signal_at(prices, {0: 1})                  # wyjście w sesji 2 ma zerowy wolumen
    assert evaluate_signals(prices, sig, spec(exclude_zero_volume=True)).summaries["full"]["n_trades"] == 0
    assert evaluate_signals(prices, sig, spec(exclude_zero_volume=False)).summaries["full"]["n_trades"] == 1


def test_extreme_day_window_excluded():
    prices = make_prices([100, 100, 120, 120, 120, 120])   # +20% w sesji 2
    sig = signal_at(prices, {0: 1, 3: 1})
    res = evaluate_signals(prices, sig, spec(exclude_extreme_abs_return=0.10))
    # okno sygnału 0 obejmuje sesje 1..2 (z dniem +20%) i odpada; okno sygnału 3 jest czyste
    assert res.trades["signal_date"].tolist() == [prices.index[3]]


def test_baseline_on_rising_series_gives_zero_edge():
    prices = make_prices(100 * 1.01 ** np.arange(60))
    sig = signal_at(prices, {5: 1, 15: 1, 25: 1, 35: 1})
    full = evaluate_signals(prices, sig, spec(horizon=3)).summaries["full"]
    assert full["hit_rate"] == 1.0 and full["baseline_hit_rate"] == 1.0
    assert full["hit_rate_diff"] == 0.0 and full["p_value_binomial"] == 1.0


def test_holdout_split_is_purged():
    prices = random_walk_prices(3, n=1000)
    sig = pd.Series(1, index=prices.index)
    res = evaluate_signals(prices, sig, spec(horizon=5, holdout_frac=0.3, no_overlap=False))
    split = pd.Timestamp(res.split_date)
    ins, hold = res.trades_by_segment["in_sample"], res.trades_by_segment["holdout"]
    assert len(ins) > 0 and len(hold) > 0
    assert (ins["exit_date"] < split).all()          # próba nie sięga w hold-out
    assert (hold["signal_date"] >= split).all()


def test_sides_filter():
    prices = make_prices(np.linspace(100, 110, 8))
    sig = signal_at(prices, {0: 1, 3: -1})
    assert evaluate_signals(prices, sig, spec(sides="long")).trades["side"].tolist() == [1]
    assert evaluate_signals(prices, sig, spec(sides="short")).trades["side"].tolist() == [-1]
    assert evaluate_signals(prices, sig, spec(sides="both")).trades["side"].tolist() == [1, -1]


def test_sensitivity_flags_dependence_on_extreme_days():
    # jedna gigantyczna sesja generuje cały "sukces" hipotezy
    close = [100.0] * 40
    close[20:] = [150.0] * 20
    prices = make_prices(close)
    sig = signal_at(prices, {19: 1, 5: 1, 8: 1, 11: 1, 14: 1, 30: 1, 33: 1})
    res = evaluate_signals(prices, sig, spec(horizon=2, sensitivity_top_days=1, no_overlap=True))
    assert res.sensitivity["excluded_days"] == [str(prices.index[20].date())]
    assert res.sensitivity["n_trades"] < res.summaries["full"]["n_trades"]
