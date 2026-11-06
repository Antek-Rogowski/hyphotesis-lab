"""Faza 0: minimalna aplikacja Streamlit.

Cel: sprawdzić, czy aplikacja wdrożona na Streamlit Community Cloud potrafi
odczytać migawkę danych (pliki Parquet) z repozytorium. Bez logiki hipotez.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))

st.set_page_config(page_title="Hypothesis Lab - Faza 0", layout="wide")
st.title("Hypothesis Lab - Faza 0")
st.caption("Test wdrożenia: odczyt migawki danych z repozytorium. Bez analizy hipotez.")


@st.cache_data(ttl=3600)
def load_all(data_dir: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(data_dir).glob("*.parquet")):
        frames[path.stem] = pd.read_parquet(path)
    return frames


frames = load_all(str(DATA_DIR))

if not frames:
    st.warning(
        f"Brak plików Parquet w katalogu: {DATA_DIR}. "
        "Uruchom najpierw workflow 'Phase 0 probe' w GitHub Actions."
    )
    st.stop()

summary = pd.DataFrame(
    [
        {
            "Symbol": symbol,
            "Od": df.index.min().date(),
            "Do": df.index.max().date(),
            "Wiersze": len(df),
            "Ostatnie zamknięcie": round(float(df["close"].iloc[-1]), 2),
            "Źródło": str(df["source"].iloc[-1]),
            "Pobrano (UTC)": df["ingested_at"].max(),
        }
        for symbol, df in frames.items()
    ]
)
st.subheader("Dostępne dane")
st.dataframe(summary, hide_index=True)

symbol = st.selectbox("Instrument", list(frames))
selected = frames[symbol]

st.subheader(f"{symbol}: cena skorygowana")
st.line_chart(selected["adj_close"])

st.subheader("Ostatnie sesje")
st.dataframe(selected.tail(10))

report_path = DATA_DIR / "probe_report.json"
if report_path.exists():
    with st.expander("Raport sondy"):
        st.json(json.loads(report_path.read_text(encoding="utf-8")))

st.caption(f"Katalog danych: {DATA_DIR}")
