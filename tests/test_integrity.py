"""Testy poprawności metodologicznej: brak wycieku przyszłości i brak 'wykrywania' przewagi w szumie."""
import numpy as np
import pandas as pd
import pytest

from hypothesis_lab import HypothesisSpec, run_hypothesis
from hypothesis_lab.signals import PARAM_SPECS, compute_signals
from hypothesis_lab.synthetic import random_walk_prices

PARAMS = {
    "mean_reversion": {"W": 20, "k": 1.5},
    "momentum": {"L": 10, "threshold": 0.01},
    "shock_rebound": {"x_pct": 1.0},
}


CUTS = [250, 400, 550, 700, 850, 1000]


def leaks(signal_fn, prices) -> bool:
    """True, jeśli sygnał z sesji t zależy od danych po t (obcięcie lub zmiana przyszłych cen)."""
    full = signal_fn(prices)
    rng = np.random.default_rng(0)
    for t0 in CUTS:
        if not full.iloc[: t0 + 1].equals(signal_fn(prices.iloc[: t0 + 1])):
            return True
        tampered = prices.copy()
        for col in ("adj_close", "close", "open", "adj_open", "high", "low"):
            tampered.iloc[t0 + 1:, tampered.columns.get_loc(col)] *= rng.uniform(0.5, 2.0, len(prices) - t0 - 1)
        if not full.iloc[: t0 + 1].equals(signal_fn(tampered).iloc[: t0 + 1]):
            return True
    return False


@pytest.mark.parametrize("htype", sorted(PARAM_SPECS))
def test_signals_do_not_use_future_data(htype):
    prices = random_walk_prices(11, n=1200)
    assert not leaks(lambda p: compute_signals(p, htype, PARAMS[htype]), prices)


def test_leak_detector_catches_known_leaks():
    """Meta-test: sam detektor musi wykrywać sygnały z wyciekiem, inaczej nic nie dowodzi."""
    prices = random_walk_prices(11, n=1200)

    def centered_window(p, W=20, k=1.5):
        c = p["adj_close"]
        z = (c - c.rolling(W, center=True).mean()) / c.rolling(W, center=True).std()
        return pd.Series(np.select([z < -k, z > k], [1, -1], 0).astype("int8"), index=c.index)

    def next_day_return(p):
        c = p["adj_close"]
        return pd.Series(np.where(c.shift(-1) > c, 1, -1).astype("int8"), index=c.index)

    assert leaks(centered_window, prices)
    assert leaks(next_day_return, prices)


def test_signals_exist_in_noise_so_the_test_is_not_vacuous():
    prices = random_walk_prices(11, n=1200)
    for htype in PARAM_SPECS:
        assert (compute_signals(prices, htype, PARAMS[htype]) != 0).sum() > 20


def test_random_walk_has_no_systematic_edge():
    """Na błądzeniu losowym odsetek 'istotnych' wyników nie powinien przekraczać poziomu istotności."""
    spec = HypothesisSpec(symbol="SYN", hypothesis_type="mean_reversion", params={"W": 20, "k": 1.5},
                          horizon=5, holdout_frac=0.0, n_boot=50, sensitivity_top_days=0)
    p_values, diffs = [], []
    for seed in range(150):
        full = run_hypothesis(random_walk_prices(seed), spec).summaries["full"]
        if full["n_trades"] >= 20:
            p_values.append(full["p_value_binomial"])
            diffs.append(full["hit_rate_diff"])
    assert len(p_values) >= 120
    false_positive_rate = np.mean(np.array(p_values) < 0.05)
    assert false_positive_rate <= 0.10, f"zbyt wiele fałszywych wyników: {false_positive_rate:.2%}"
    assert abs(np.mean(diffs)) < 0.01


def test_momentum_on_noise_also_has_no_edge():
    spec = HypothesisSpec(symbol="SYN", hypothesis_type="momentum", params={"L": 10, "threshold": 0.01},
                          horizon=5, holdout_frac=0.0, n_boot=50, sensitivity_top_days=0)
    diffs = []
    for seed in range(100):
        full = run_hypothesis(random_walk_prices(seed + 1000), spec).summaries["full"]
        if full["n_trades"] >= 20:
            diffs.append(full["hit_rate_diff"])
    assert abs(np.mean(diffs)) < 0.01
