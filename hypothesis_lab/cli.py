"""Uruchamianie hipotezy z wiersza poleceń.

Przykłady:
    python -m hypothesis_lab.cli --symbol GLD --type mean_reversion --param W=20 --param k=2 --horizon 5
    python -m hypothesis_lab.cli --symbol SLV --type shock_rebound --param x_pct=3 --horizon 3 --costs --commission-bps 1 --spread-bps 3
    python -m hypothesis_lab.cli --synthetic --type momentum --json wynik.json
"""
from __future__ import annotations

import argparse
import json
import sys

from .data import load_prices
from .engine import HypothesisResult, run_hypothesis
from .signals import PARAM_SPECS
from .spec import CostModel, HypothesisSpec
from .synthetic import random_walk_prices


def _parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"parametr musi mieć postać nazwa=wartość (dostałem {item!r})")
        key, value = item.split("=", 1)
        params[key.strip()] = float(value)
    return params


def _pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def _block(name: str, s: dict) -> list[str]:
    if s.get("n_trades", 0) == 0:
        return [f"{name}: brak zdarzeń"]
    ci = s["hit_rate_ci95"]
    ci_net = s["mean_net_return_ci95"]
    mix = s["side_mix"]
    return [
        f"{name}: {s['n_trades']} zdarzeń (long {mix['long']}, short {mix['short']}), efektywnie ok. {s['n_effective']}",
        f"  trafność {_pct(s['hit_rate'])}  (95% CI {_pct(ci[0])} do {_pct(ci[1])})",
        f"  baza     {_pct(s['baseline_hit_rate'])}  różnica {s['hit_rate_diff']:+.1%}  p dwumianowe {s['p_value_binomial']:.3f}",
        f"  średni zwrot netto {_pct(s['mean_net_return'])} (95% CI {_pct(ci_net[0])} do {_pct(ci_net[1])}), baza {_pct(s['baseline_mean_net_return'])}",
        f"  mediana {_pct(s['median_net_return'])}, zysk/strata {s['avg_win_loss_ratio'] if s['avg_win_loss_ratio'] is not None else '-'}, "
        f"kapitał końcowy {_pct(s['equity_final'])}",
    ]


def format_report(res: HypothesisResult) -> str:
    sp = res.spec
    lines = [
        f"Hipoteza {sp['hypothesis_type']} {sp['params']} | {sp['symbol']} | horyzont {sp['horizon']} sesji | "
        f"pozycje: {sp['sides']} | wykonanie: {sp['execution']} | koszty: {'tak' if sp['costs']['enabled'] else 'nie'}",
        f"Okres {res.first_date} do {res.last_date} ({res.n_sessions} sesji), skrót specyfikacji {res.spec_hash}, "
        f"skrót danych {res.data_fingerprint}",
        "",
    ]
    lines += _block("CAŁY OKRES", res.summaries["full"])
    if "in_sample" in res.summaries:
        lines += ["", f"Podział próby: granica {res.split_date}"]
        lines += _block("PRÓBA (oczyszczona)", res.summaries["in_sample"])
        lines += _block("HOLD-OUT", res.summaries["holdout"])
    if res.sensitivity:
        se = res.sensitivity
        lines += ["", f"Wrażliwość: bez dni {', '.join(se['excluded_days'])}",
                  f"  zdarzeń {se['n_trades']}, trafność {_pct(se['hit_rate'])}, "
                  f"baza {_pct(se['baseline_hit_rate'])}, różnica {se['hit_rate_diff']:+.1%}"
                  if se.get("n_trades") else "  brak zdarzeń po wykluczeniu"]
    if res.warnings:
        lines += ["", "Ostrzeżenia:"] + [f"  - {w}" for w in res.warnings]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="GLD")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--synthetic", action="store_true", help="losowe błądzenie zamiast pliku Parquet")
    ap.add_argument("--type", dest="htype", required=True, choices=sorted(PARAM_SPECS))
    ap.add_argument("--param", action="append", default=[], help="nazwa=wartość, można powtarzać")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--sides", choices=["long", "short", "both"], default="both")
    ap.add_argument("--exec", dest="execution", choices=["close_t", "open_t+1"], default="close_t")
    ap.add_argument("--target", choices=["sign", "threshold"], default="sign")
    ap.add_argument("--target-threshold-pct", type=float, default=0.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--overlap", action="store_true", help="pozwól na nakładające się pozycje")
    ap.add_argument("--exclude-extreme", type=float, default=None, help="pomijaj okna z dniem |zwrot| > próg (ułamek)")
    ap.add_argument("--costs", action="store_true", help="włącz koszty transakcyjne")
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--spread-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--borrow-bps", type=float, default=0.0)
    ap.add_argument("--json", dest="json_path", help="zapisz pełny wynik do pliku JSON")
    args = ap.parse_args(argv)

    spec = HypothesisSpec(
        symbol=args.symbol, hypothesis_type=args.htype, params=_parse_params(args.param),
        horizon=args.horizon, sides=args.sides, target=args.target,
        target_threshold_pct=args.target_threshold_pct, execution=args.execution,
        costs=CostModel(enabled=args.costs, commission_bps=args.commission_bps, spread_bps=args.spread_bps,
                        slippage_bps=args.slippage_bps, borrow_bps_annual=args.borrow_bps),
        start=args.start, end=args.end, holdout_frac=args.holdout, no_overlap=not args.overlap,
        exclude_extreme_abs_return=args.exclude_extreme,
    )
    prices = random_walk_prices() if args.synthetic else load_prices(args.symbol, args.data_dir)
    result = run_hypothesis(prices, spec)
    print(format_report(result))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
