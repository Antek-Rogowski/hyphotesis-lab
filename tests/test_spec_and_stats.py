import json

import numpy as np
import pytest

from hypothesis_lab import CostModel, HypothesisSpec, run_hypothesis
from hypothesis_lab.stats import binom_two_sided_p, block_bootstrap_ci
from hypothesis_lab.synthetic import random_walk_prices


def test_spec_roundtrip_and_hash():
    s = HypothesisSpec(symbol="gld", hypothesis_type="mean_reversion", params={"W": 30, "k": 1.5},
                       horizon=7, costs=CostModel(enabled=True, commission_bps=2))
    s2 = HypothesisSpec.from_json(s.to_json())
    assert s2 == s and s2.spec_hash() == s.spec_hash()
    assert s.symbol == "GLD"
    other = HypothesisSpec(symbol="gld", hypothesis_type="mean_reversion", params={"W": 31, "k": 1.5}, horizon=7,
                           costs=CostModel(enabled=True, commission_bps=2))
    assert other.spec_hash() != s.spec_hash()


def test_defaults_are_filled_in_params():
    s = HypothesisSpec(symbol="GLD", hypothesis_type="momentum")
    assert s.params == {"L": 20, "threshold": 0.0}


@pytest.mark.parametrize("kwargs", [
    dict(horizon=0),
    dict(execution="open_t+2"),
    dict(sides="flat"),
    dict(target="threshold"),                           # brak progu
    dict(holdout_frac=0.95),
    dict(params={"nie_ma_takiego": 1}),
    dict(params={"W": 2.5}),                            # nie-całkowite W
    dict(params={"k": 0.0}),                            # k musi być > 0
    dict(hypothesis_type="nieznana"),
])
def test_invalid_specs_raise(kwargs):
    base = dict(symbol="GLD", hypothesis_type="mean_reversion")
    base.update(kwargs)
    with pytest.raises(ValueError):
        HypothesisSpec(**base)


def test_binomial_known_values():
    assert binom_two_sided_p(60, 100, 0.5) == pytest.approx(0.0569, abs=5e-4)   # binom.test(60, 100)
    assert binom_two_sided_p(50, 100, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert binom_two_sided_p(10, 10, 1.0) == 1.0
    assert binom_two_sided_p(3, 10, 0.0) == 0.0


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.random.default_rng(1).normal(1.0, 1.0, 200)
    lo, hi = block_bootstrap_ci(x, n_boot=500, seed=5)
    assert lo < x.mean() < hi
    assert (lo, hi) == block_bootstrap_ci(x, n_boot=500, seed=5)
    assert all(np.isnan(block_bootstrap_ci([1.0])))


def test_result_is_deterministic_and_json_safe():
    prices = random_walk_prices(7)
    spec = HypothesisSpec(symbol="SYN", hypothesis_type="shock_rebound", params={"x_pct": 1.5},
                          n_boot=200, seed=3)
    a, b = run_hypothesis(prices, spec).to_dict(), run_hypothesis(prices, spec).to_dict()
    assert a == b
    json.dumps(a)                                        # nie może zawierać NaN ani typów numpy
