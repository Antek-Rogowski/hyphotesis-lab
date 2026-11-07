"""Specyfikacja hipotezy: serializowalny, wersjonowany opis jednego testu."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .signals import PARAM_SPECS, resolve_params

TRADING_DAYS = 252
SIDES = ("long", "short", "both")
EXECUTIONS = ("close_t", "open_t+1")
TARGETS = ("sign", "threshold")


@dataclass(frozen=True)
class CostModel:
    """Koszty transakcyjne. Wszystkie wartości w punktach bazowych (1 bp = 0,01%)."""

    enabled: bool = False
    commission_bps: float = 0.0       # prowizja na stronę transakcji
    spread_bps: float = 0.0           # pełny spread; na stronę płacimy połowę
    slippage_bps: float = 0.0         # poślizg na stronę
    borrow_bps_annual: float = 0.0    # koszt pożyczki, tylko short, w skali roku

    def __post_init__(self) -> None:
        for name in ("commission_bps", "spread_bps", "slippage_bps", "borrow_bps_annual"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} nie może być ujemne")

    def round_trip_fraction(self) -> float:
        """Koszt wejścia i wyjścia jako ułamek wartości pozycji."""
        if not self.enabled:
            return 0.0
        per_side = self.commission_bps + self.spread_bps / 2.0 + self.slippage_bps
        return 2.0 * per_side / 10_000.0

    def borrow_fraction(self, horizon: int) -> float:
        """Koszt pożyczki za cały okres trzymania pozycji short."""
        if not self.enabled:
            return 0.0
        return self.borrow_bps_annual / 10_000.0 * horizon / TRADING_DAYS


@dataclass(frozen=True)
class HypothesisSpec:
    symbol: str
    hypothesis_type: str
    params: dict = field(default_factory=dict)
    horizon: int = 5                          # sesji trzymania pozycji
    sides: str = "both"                       # long, short, both
    target: str = "sign"                      # sign: wynik netto > 0; threshold: wynik netto >= próg
    target_threshold_pct: float = 0.0
    execution: str = "close_t"                # close_t: cena zamknięcia sesji sygnału; open_t+1: otwarcie następnej
    costs: CostModel = field(default_factory=CostModel)
    start: str | None = None
    end: str | None = None
    holdout_frac: float = 0.3                 # ostatnia część okresu jako zbiór poza próbą (0 = brak)
    no_overlap: bool = True                   # jedna otwarta pozycja naraz
    exclude_zero_volume: bool = True          # pomijaj okna z sesją o zerowym wolumenie
    exclude_extreme_abs_return: float | None = None  # np. 0.10: pomijaj okna z dniem |zwrot| > 10%
    sensitivity_top_days: int = 5             # test wrażliwości: wynik bez N największych dni
    n_boot: int = 2000
    seed: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", resolve_params(self.hypothesis_type, self.params))
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if self.hypothesis_type not in PARAM_SPECS:
            raise ValueError(f"nieznany typ hipotezy: {self.hypothesis_type!r}")
        if self.horizon < 1 or int(self.horizon) != self.horizon:
            raise ValueError("horizon musi być całkowitą liczbą sesji >= 1")
        if self.sides not in SIDES:
            raise ValueError(f"sides musi być jednym z {SIDES}")
        if self.execution not in EXECUTIONS:
            raise ValueError(f"execution musi być jednym z {EXECUTIONS}")
        if self.target not in TARGETS:
            raise ValueError(f"target musi być jednym z {TARGETS}")
        if self.target == "threshold" and self.target_threshold_pct <= 0:
            raise ValueError("dla target='threshold' podaj target_threshold_pct > 0")
        if not 0.0 <= self.holdout_frac < 0.9:
            raise ValueError("holdout_frac musi być w przedziale [0, 0.9)")
        if self.exclude_extreme_abs_return is not None and self.exclude_extreme_abs_return <= 0:
            raise ValueError("exclude_extreme_abs_return musi być > 0")
        if self.sensitivity_top_days < 0 or self.n_boot < 1:
            raise ValueError("sensitivity_top_days >= 0 i n_boot >= 1")

    # --- serializacja ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HypothesisSpec":
        data = dict(data)
        data["costs"] = CostModel(**data.get("costs", {}))
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "HypothesisSpec":
        return cls.from_dict(json.loads(text))

    def spec_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]
