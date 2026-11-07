"""Statystyka: dokładny test dwumianowy i bootstrap blokowy (bez scipy)."""
from __future__ import annotations

from math import lgamma, log

import numpy as np


def binom_two_sided_p(k: int, n: int, p0: float) -> float:
    """Dokładny dwustronny test dwumianowy: P(wynik co najmniej tak mało prawdopodobny jak k)."""
    if n <= 0:
        return float("nan")
    if p0 <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p0 >= 1.0:
        return 1.0 if k == n else 0.0
    ks = np.arange(n + 1)
    log_comb = lgamma(n + 1) - np.array([lgamma(i + 1) + lgamma(n - i + 1) for i in ks])
    log_pmf = log_comb + ks * log(p0) + (n - ks) * log(1.0 - p0)
    pmf = np.exp(log_pmf)
    p_obs = pmf[k]
    return float(min(1.0, pmf[pmf <= p_obs * (1.0 + 1e-7)].sum()))


def block_bootstrap_ci(
    values,
    block_len: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Przedział ufności dla średniej metodą bootstrapu blokowego (uwzględnia autokorelację)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    b = block_len if block_len is not None else max(1, int(round(n ** (1.0 / 3.0))))
    b = int(min(max(b, 1), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, n_blocks * b)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
