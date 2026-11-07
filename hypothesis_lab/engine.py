"""Silnik: z sygnałów i cen liczy zdarzenia, trafność, bazę porównawczą i miary niepewności.

Konwencje:
  * sygnał w sesji t jest liczony z danych do zamknięcia sesji t,
  * execution="close_t":  wejście po adj_close[t],      wyjście po adj_close[t + N],
  * execution="open_t+1": wejście po adj_open[t + 1],   wyjście po adj_open[t + 1 + N],
  * long: zwrot = wyjście / wejście - 1, short: zwrot = 1 - wyjście / wejście,
  * koszty (jeśli włączone) są odejmowane od zwrotu; trafienie liczymy na zwrocie netto,
  * przy no_overlap=True nowy sygnał jest pomijany, dopóki trwa poprzednia pozycja
    (kolejny sygnał dopuszczamy w sesji t + N).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import data_fingerprint, slice_period
from .signals import compute_signals
from .spec import HypothesisSpec
from .stats import binom_two_sided_p, block_bootstrap_ci

MIN_TRADES_WARN = 30
MIN_HOLDOUT_TRADES_WARN = 15


@dataclass
class HypothesisResult:
    spec: dict
    spec_hash: str
    data_fingerprint: str
    n_sessions: int
    first_date: str
    last_date: str
    split_date: str | None
    summaries: dict
    sensitivity: dict
    trades_by_segment: dict[str, pd.DataFrame]
    warnings: list[str] = field(default_factory=list)

    @property
    def trades(self) -> pd.DataFrame:
        """Zdarzenia z całego okresu (bez podziału na próbę i hold-out)."""
        return self.trades_by_segment["full"]

    def to_dict(self) -> dict:
        return _clean(
            {
                "spec": self.spec,
                "spec_hash": self.spec_hash,
                "data_fingerprint": self.data_fingerprint,
                "period": {
                    "sessions": self.n_sessions,
                    "first_date": self.first_date,
                    "last_date": self.last_date,
                    "holdout_split_date": self.split_date,
                },
                "summaries": self.summaries,
                "sensitivity": self.sensitivity,
                "warnings": self.warnings,
            }
        )


def _clean(obj):
    """Zamienia obiekty numpy i NaN na wartości bezpieczne dla JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _select(signal_idx: np.ndarray, horizon: int, no_overlap: bool) -> np.ndarray:
    """Wybiera zdarzenia; przy no_overlap tylko takie, które nie nakładają się na poprzednią pozycję."""
    if not no_overlap or len(signal_idx) == 0:
        return signal_idx
    chosen: list[int] = []
    next_free = -1
    for i in signal_idx:
        if i >= next_free:
            chosen.append(int(i))
            next_free = int(i) + horizon
    return np.array(chosen, dtype=int)


def _round(x, digits=6):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), digits)


def _empty_summary(n_sessions: int) -> dict:
    return {"n_trades": 0, "sessions_in_segment": n_sessions, "note": "brak zdarzeń do oceny"}


def _summarize(trades: pd.DataFrame, elig: np.ndarray, hit: dict, net: dict,
               spec: HypothesisSpec, n_sessions: int) -> dict:
    n = len(trades)
    if n == 0:
        return _empty_summary(n_sessions)

    horizon = spec.horizon
    sides = trades["side"].to_numpy()
    weights = {1: float((sides == 1).mean()), -1: float((sides == -1).mean())}

    # baza: ta sama mieszanka pozycji, liczona na wszystkich kwalifikujących się sesjach segmentu
    base_hit = base_mean = 0.0
    for side, w in weights.items():
        if w == 0.0:
            continue
        base_hit += w * float(hit[side][elig].mean())
        base_mean += w * float(net[side][elig].mean())

    hits = int(trades["hit"].sum())
    hit_arr = trades["hit"].to_numpy(float)
    net_arr = trades["net_return"].to_numpy(float)
    hit_rate = hits / n

    block_len = None if spec.no_overlap else max(horizon, int(round(n ** (1.0 / 3.0))))
    ci_hit = block_bootstrap_ci(hit_arr, block_len, spec.n_boot, seed=spec.seed)
    ci_net = block_bootstrap_ci(net_arr, block_len, spec.n_boot, seed=spec.seed)

    wins, losses = net_arr[net_arr > 0], net_arr[net_arr <= 0]
    win_loss = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None
    equity_final = float(np.prod(1.0 + net_arr) - 1.0) if spec.no_overlap else None

    max_windows = n_sessions // horizon
    return {
        "n_trades": n,
        "side_mix": {"long": int((sides == 1).sum()), "short": int((sides == -1).sum())},
        "hits": hits,
        "hit_rate": _round(hit_rate),
        "hit_rate_ci95": [_round(ci_hit[0]), _round(ci_hit[1])],
        "baseline_hit_rate": _round(base_hit),
        "hit_rate_diff": _round(hit_rate - base_hit),
        "p_value_binomial": _round(binom_two_sided_p(hits, n, base_hit)),
        "mean_net_return": _round(net_arr.mean()),
        "mean_net_return_ci95": [_round(ci_net[0]), _round(ci_net[1])],
        "median_net_return": _round(np.median(net_arr)),
        "baseline_mean_net_return": _round(base_mean),
        "avg_win_loss_ratio": _round(win_loss),
        "equity_final": _round(equity_final),
        "sessions_in_segment": n_sessions,
        "max_independent_windows": max_windows,
        "n_effective": min(n, max_windows),
    }


def _evaluate(prices: pd.DataFrame, signal: pd.Series, spec: HypothesisSpec,
              excluded_days: np.ndarray | None = None):
    """Rdzeń obliczeń. Zwraca (summaries, trades_by_segment, meta)."""
    T = len(prices)
    N = spec.horizon
    open_exec = spec.execution == "open_t+1"
    shift = 1 if open_exec else 0
    px = (prices["adj_open"] if open_exec else prices["adj_close"]).to_numpy(float)

    i = np.arange(T)
    entry = i + shift
    exit_ = entry + N
    valid = exit_ <= T - 1                      # pełne okno do wyjścia musi mieścić się w danych
    e = np.where(valid, entry, 0)
    x = np.where(valid, exit_, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = px[x] / px[e]
    valid &= np.isfinite(ratio) & (px[e] > 0)

    if spec.exclude_zero_volume:
        zv = prices["zero_volume"].to_numpy(bool)
        valid &= ~(zv | zv[e] | zv[x])

    # okna z wykluczonymi dniami: dni zwrotów realizowanych w trakcie pozycji to sesje t+1 .. wyjście
    ext = np.zeros(T, dtype=bool)
    if spec.exclude_extreme_abs_return is not None:
        c = prices["adj_close"]
        day_ret = (c / c.shift(1) - 1.0).abs().to_numpy(float)
        ext |= np.nan_to_num(day_ret, nan=0.0) > spec.exclude_extreme_abs_return
    if excluded_days is not None:
        ext |= excluded_days
    if ext.any():
        cs = np.concatenate([[0], np.cumsum(ext)])
        first_day = np.minimum(i + 1, T)
        hits_ext = (cs[x + 1] - cs[first_day]) > 0
        valid &= ~hits_ext

    cost_rt = spec.costs.round_trip_fraction()
    borrow = spec.costs.borrow_fraction(N)
    gross = {1: ratio - 1.0, -1: 1.0 - ratio}
    net = {1: gross[1] - cost_rt, -1: gross[-1] - cost_rt - borrow}
    thr = spec.target_threshold_pct / 100.0
    if spec.target == "sign":
        hit = {s: net[s] > 0.0 for s in (1, -1)}
    else:
        hit = {s: net[s] >= thr for s in (1, -1)}

    sig = signal.reindex(prices.index).fillna(0).to_numpy(int)
    if spec.sides == "long":
        sig = np.where(sig == 1, 1, 0)
    elif spec.sides == "short":
        sig = np.where(sig == -1, -1, 0)
    sig_ok = (sig != 0) & valid

    split = int(T * (1.0 - spec.holdout_frac)) if spec.holdout_frac > 0 else None
    masks = {"full": np.ones(T, dtype=bool)}
    if split is not None and 0 < split < T:
        masks["in_sample"] = (i < split) & (exit_ < split)   # oczyszczenie: wyjście przed granicą
        masks["holdout"] = i >= split
    else:
        split = None

    dates = prices.index
    summaries: dict[str, dict] = {}
    trades_by_segment: dict[str, pd.DataFrame] = {}
    for name, mask in masks.items():
        chosen = _select(np.flatnonzero(sig_ok & mask), N, spec.no_overlap)
        if len(chosen):
            side = sig[chosen]
            net_ret = np.where(side == 1, net[1][chosen], net[-1][chosen])
            gross_ret = np.where(side == 1, gross[1][chosen], gross[-1][chosen])
            hit_flag = np.where(side == 1, hit[1][chosen], hit[-1][chosen])
            trades = pd.DataFrame(
                {
                    "signal_date": dates[chosen],
                    "side": side,
                    "entry_date": dates[entry[chosen]],
                    "exit_date": dates[exit_[chosen]],
                    "entry_price": px[entry[chosen]],
                    "exit_price": px[exit_[chosen]],
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "hit": hit_flag.astype(bool),
                }
            )
            trades["equity"] = (1.0 + trades["net_return"]).cumprod() if spec.no_overlap else np.nan
        else:
            trades = pd.DataFrame(columns=["signal_date", "side", "entry_date", "exit_date", "entry_price",
                                           "exit_price", "gross_return", "net_return", "hit", "equity"])
        trades_by_segment[name] = trades
        summaries[name] = _summarize(trades, valid & mask, hit, net, spec, int(mask.sum()))

    meta = {
        "n_signals_raw": int((sig != 0).sum()),
        "n_signals_evaluable": int(sig_ok.sum()),
        "split_index": split,
    }
    return summaries, trades_by_segment, meta


def evaluate_signals(prices: pd.DataFrame, signal: pd.Series, spec: HypothesisSpec) -> HypothesisResult:
    """Ocenia gotowy szereg sygnałów (np. z własnej reguły) zgodnie ze specyfikacją."""
    prices = slice_period(prices, spec.start, spec.end)
    min_rows = spec.horizon + 1 + (1 if spec.execution == "open_t+1" else 0)
    if len(prices) < min_rows:
        raise ValueError("za mało danych w wybranym okresie")
    signal = signal.reindex(prices.index).fillna(0)

    summaries, trades_by_segment, meta = _evaluate(prices, signal, spec)

    # test wrażliwości: ten sam test bez N największych dni (wg |zwrotu| w okresie)
    sensitivity: dict = {}
    k = spec.sensitivity_top_days
    if k > 0 and summaries["full"]["n_trades"] > 0:
        c = prices["adj_close"]
        abs_ret = (c / c.shift(1) - 1.0).abs()
        top_days = abs_ret.nlargest(k).index
        s2, _, _ = _evaluate(prices, signal, spec, excluded_days=prices.index.isin(top_days))
        full2 = s2["full"]
        sensitivity = {
            "excluded_days": [str(d.date()) for d in sorted(top_days)],
            "n_trades": full2.get("n_trades", 0),
            "hit_rate": full2.get("hit_rate"),
            "baseline_hit_rate": full2.get("baseline_hit_rate"),
            "hit_rate_diff": full2.get("hit_rate_diff"),
            "mean_net_return": full2.get("mean_net_return"),
        }

    warnings = _warnings(summaries, sensitivity, spec, len(prices))
    split_index = meta["split_index"]
    return HypothesisResult(
        spec=spec.to_dict(),
        spec_hash=spec.spec_hash(),
        data_fingerprint=data_fingerprint(prices),
        n_sessions=len(prices),
        first_date=str(prices.index[0].date()),
        last_date=str(prices.index[-1].date()),
        split_date=None if split_index is None else str(prices.index[split_index].date()),
        summaries=summaries,
        sensitivity=sensitivity,
        trades_by_segment=trades_by_segment,
        warnings=warnings,
    )


def _warnings(summaries: dict, sensitivity: dict, spec: HypothesisSpec, n_sessions: int) -> list[str]:
    out: list[str] = []
    full = summaries["full"]
    n = full.get("n_trades", 0)
    if n == 0:
        out.append("Brak zdarzeń: hipoteza nie wygenerowała żadnego oceniania sygnału w tym okresie.")
        return out
    if n < MIN_TRADES_WARN:
        out.append(f"Mała próba: tylko {n} zdarzeń. Wynik jest bardzo niepewny.")
    if "holdout" in summaries:
        hn = summaries["holdout"].get("n_trades", 0)
        if hn < MIN_HOLDOUT_TRADES_WARN:
            out.append(f"Zbiór hold-out ma tylko {hn} zdarzeń: wynik poza próbą jest mało wiarygodny.")
    if not spec.no_overlap:
        out.append("Zdarzenia nakładają się na siebie: p-wartość dwumianowa zawyża istotność. "
                   "Opieraj się na przedziałach ufności z bootstrapu.")
    if sensitivity:
        d_full, d_sens = full.get("hit_rate_diff"), sensitivity.get("hit_rate_diff")
        if d_full is not None and d_sens is not None and d_full != 0:
            if d_full * d_sens < 0 or abs(d_sens) < 0.5 * abs(d_full):
                out.append(
                    f"Wynik silnie zależy od {spec.sensitivity_top_days} największych dni "
                    f"(różnica wobec bazy: {d_full:+.3f} z nimi, {d_sens:+.3f} bez nich)."
                )
    if n_sessions < 252 * 12:
        out.append("Okres około 10 lat to jeden reżim rynkowy: wynik może nie przenosić się na inne okresy.")
    return out


def run_hypothesis(prices: pd.DataFrame, spec: HypothesisSpec) -> HypothesisResult:
    """Pełny przebieg: sygnały z gotowej hipotezy, a następnie ocena zdarzeń."""
    signal = compute_signals(prices, spec.hypothesis_type, spec.params)
    return evaluate_signals(prices, signal, spec)
