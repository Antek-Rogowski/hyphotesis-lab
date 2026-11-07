"""Hypothesis Lab - silnik testowania hipotez giełdowych (Faza 1)."""
from .data import load_prices, prepare_prices
from .engine import HypothesisResult, evaluate_signals, run_hypothesis
from .spec import CostModel, HypothesisSpec

__all__ = [
    "CostModel",
    "HypothesisResult",
    "HypothesisSpec",
    "evaluate_signals",
    "load_prices",
    "prepare_prices",
    "run_hypothesis",
]
__version__ = "0.1.0"
