#!/usr/bin/env python3
"""Codzienna aktualizacja danych (uruchamiana z GitHub Actions lub ręcznie).

Przykłady:
    python scripts/update_data.py                      # domyślne symbole, aktualizacja przyrostowa
    python scripts/update_data.py --force-full         # pełne odświeżenie 10 lat
    python scripts/update_data.py --symbols GLD,SLV
Kod wyjścia: 0 = poprawnie (także z ostrzeżeniami), 1 = błąd (alarm w GitHub Actions).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis_lab.pipeline import (  # noqa: E402
    DEFAULT_SYMBOLS,
    MAX_YEARS,
    YFinanceProvider,
    render_markdown,
    update_all,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="symbole po przecinku")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-years", type=int, default=MAX_YEARS)
    ap.add_argument("--force-full", action="store_true", help="pełne odświeżenie zamiast przyrostowego")
    ap.add_argument("--pause", type=float, default=2.0, help="przerwa między symbolami (s)")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    status = update_all(symbols, YFinanceProvider(), args.data_dir, force_full=args.force_full,
                        max_years=args.max_years, pause=args.pause)

    markdown = render_markdown(status)
    print(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown)
    return 1 if status["overall_status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
