"""Opis instrumentów pokazywany w interfejsie (struktura funduszu wpływa na interpretację wyników)."""
from __future__ import annotations

INSTRUMENTS: dict[str, dict[str, str]] = {
    "GLD": {
        "name": "SPDR Gold Shares",
        "asset": "Złoto",
        "structure": "physical",
        "description": "Fundusz oparty na fizycznym złocie.",
    },
    "SLV": {
        "name": "iShares Silver Trust",
        "asset": "Srebro",
        "structure": "physical",
        "description": "Fundusz oparty na fizycznym srebrze.",
    },
    "CPER": {
        "name": "United States Copper Index Fund",
        "asset": "Miedź",
        "structure": "futures",
        "description": "Fundusz oparty na kontraktach terminowych na miedź. Koszt rolowania kontraktów "
                       "(contango i backwardation) jest wbudowany w cenę, więc wynik dotyczy funduszu, a nie miedzi.",
    },
    "USO": {
        "name": "United States Oil Fund",
        "asset": "Ropa WTI",
        "structure": "futures",
        "description": "Fundusz oparty na kontraktach terminowych na rop\u0119 WTI. Koszt rolowania kontraktów "
                       "(contango i backwardation) jest wbudowany w cenę, więc wynik dotyczy funduszu, a nie ropy.",
    },
}


def describe(symbol: str) -> dict[str, str]:
    return INSTRUMENTS.get(symbol.upper(), {"name": symbol, "asset": "-", "structure": "unknown",
                                            "description": "Brak opisu instrumentu."})
