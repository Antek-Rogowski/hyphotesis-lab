"""Hypothesis Lab: interfejs do testowania hipotez giełdowych na danych historycznych.

Interfejs jest cienką warstwą: cała logika (sygnały, zdarzenia, statystyka) jest w pakiecie hypothesis_lab.
Dane to pliki Parquet w katalogu data/ (zapisuje je codzienny job), konfigurowalne zmienną DATA_DIR.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from hypothesis_lab import CostModel, HypothesisSpec, load_prices, run_hypothesis  # noqa: E402
from hypothesis_lab.instruments import describe  # noqa: E402
from hypothesis_lab.pipeline import STATUS_FILE  # noqa: E402
from hypothesis_lab.signals import REGISTRY  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
SYMBOL_ORDER = ["GLD", "SLV", "CPER", "USO"]

# Hipotezy, ich opisy i parametry pochodzą z rejestru (hypothesis_lab/signals.py oraz classics.py).
HYPOTHESES = {key: (d.title, d.description) for key, d in REGISTRY.items()}
SIDES = {"both": "Long i short", "long": "Tylko long", "short": "Tylko short"}
TARGETS = {"sign": "Wynik netto powyżej zera", "threshold": "Wynik netto co najmniej X%"}
EXECUTIONS = {"close_t": "Cena zamknięcia sesji sygnału", "open_t+1": "Otwarcie następnej sesji"}
SEGMENTS = {"full": "Cały okres", "in_sample": "Próba (oczyszczona)", "holdout": "Hold-out"}


# --------------------------------------------------------------------------
# Dane
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_prices(symbol: str, data_dir: str, mtime: float) -> pd.DataFrame:
    return load_prices(symbol, data_dir)


def available_symbols() -> list[str]:
    found = sorted(p.stem for p in DATA_DIR.glob("*.parquet"))
    return [s for s in SYMBOL_ORDER if s in found] + [s for s in found if s not in SYMBOL_ORDER]


def prices_for(symbol: str) -> pd.DataFrame:
    return get_prices(symbol, str(DATA_DIR), (DATA_DIR / f"{symbol}.parquet").stat().st_mtime)


def read_status() -> dict | None:
    path = DATA_DIR / STATUS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Formatowanie
# --------------------------------------------------------------------------
def pct(x, digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def trials_label(n: int) -> str:
    """Odmiana po 'na': 1 próbę, 2-4 próby, 5+ prób."""
    if n == 1:
        return "próbę"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "próby"
    return "prób"


def settings_line(spec: dict) -> str:
    """Najważniejsze ustawienia wyniku w jednym wierszu, żeby było widać, czym wynik się różni."""
    costs = spec["costs"]
    if costs["enabled"]:
        parts = [f"prowizja {costs['commission_bps']:g} bp", f"spread {costs['spread_bps']:g} bp",
                 f"poślizg {costs['slippage_bps']:g} bp"]
        if spec["sides"] != "long":
            parts.append(f"pożyczka {costs['borrow_bps_annual']:g} bp rocznie")
        cost_text = f"**koszty: tak** ({', '.join(parts)})"
    else:
        cost_text = "koszty: nie"
    return (f"Pozycje: {SIDES[spec['sides']].lower()} | horyzont {spec['horizon']} sesji | "
            f"wykonanie: {EXECUTIONS[spec['execution']].lower()} | {cost_text} | "
            f"hold-out {round(spec['holdout_frac'] * 100)}%")


def build_spec(cfg: dict) -> HypothesisSpec:
    return HypothesisSpec(
        symbol=cfg["symbol"], hypothesis_type=cfg["htype"], params=cfg["params"], horizon=cfg["horizon"],
        sides=cfg["sides"], target=cfg["target"], target_threshold_pct=cfg["threshold_pct"],
        execution=cfg["execution"],
        costs=CostModel(enabled=cfg["costs_on"], commission_bps=cfg["commission"], spread_bps=cfg["spread"],
                        slippage_bps=cfg["slippage"], borrow_bps_annual=cfg["borrow"]),
        start=cfg["start"], end=cfg["end"], holdout_frac=cfg["holdout_pct"] / 100.0,
        no_overlap=cfg["no_overlap"], exclude_zero_volume=cfg["excl_zero"],
        exclude_extreme_abs_return=cfg["ext_pct"] / 100.0 if cfg["excl_ext"] else None,
        sensitivity_top_days=cfg["top_days"],
    )


# --------------------------------------------------------------------------
# Panel boczny: konfiguracja
# --------------------------------------------------------------------------
st.set_page_config(page_title="Hypothesis Lab", layout="wide")
st.title("Hypothesis Lab")
st.caption("Test hipotez giełdowych na historycznych danych dziennych. Narzędzie badawcze, nie rekomendacja inwestycyjna.")

symbols = available_symbols()
if not symbols:
    st.warning(f"Brak plików Parquet w katalogu {DATA_DIR}. Uruchom workflow aktualizacji danych w GitHub Actions.")
    st.stop()

with st.sidebar:
    st.subheader("1. Instrument")
    symbol = st.selectbox("Instrument", symbols, key="symbol",
                          format_func=lambda s: f"{s}: {describe(s)['asset']}")
    info = describe(symbol)
    st.caption(f"{info['name']}. {info['description']}")
    prices = prices_for(symbol)
    first_day, last_day = prices.index.min().date(), prices.index.max().date()
    period = st.date_input("Okres analizy", value=(first_day, last_day), min_value=first_day,
                           max_value=last_day, key=f"period_{symbol}")
    start_day, end_day = (period[0], period[1]) if isinstance(period, tuple) and len(period) == 2 \
        else (first_day, last_day)

    st.subheader("2. Hipoteza")
    htype = st.selectbox("Hipoteza", list(HYPOTHESES), key="htype", format_func=lambda k: HYPOTHESES[k][0])
    definition = REGISTRY[htype]
    st.caption(f"{definition.family}. {definition.description}"
               + (f" Źródło: {definition.reference}." if definition.reference else ""))
    params: dict = {}
    for name, pdef in definition.params.items():
        key = f"param_{htype}_{name}"
        if pdef.typ is bool:
            params[name] = st.checkbox(pdef.label, value=bool(pdef.default), key=key)
        elif pdef.typ is int:
            params[name] = st.number_input(pdef.label, min_value=int(pdef.lo), max_value=int(pdef.hi),
                                           value=int(pdef.default), step=1, key=key)
        else:
            step = pdef.step or 0.1
            params[name] = st.number_input(pdef.label,
                                           min_value=float(pdef.lo if pdef.lo_inclusive else pdef.lo + step),
                                           max_value=float(pdef.hi), value=float(pdef.default), step=step,
                                           format="%.3f" if step < 0.01 else "%.1f", key=key)

    st.subheader("3. Pozycje i cel")
    horizon = st.number_input("Horyzont (sesje trzymania pozycji)", min_value=1, max_value=60, value=5,
                              step=1, key="horizon")
    sides = st.selectbox("Kierunek pozycji", list(SIDES), key="sides", format_func=SIDES.get)
    target = st.radio("Kiedy hipoteza się sprawdziła?", list(TARGETS), key="target", format_func=TARGETS.get)
    threshold_pct = st.number_input("Próg X (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1,
                                    key="threshold_pct") if target == "threshold" else 0.0
    execution = st.radio("Moment wykonania transakcji", list(EXECUTIONS), key="execution",
                         format_func=EXECUTIONS.get)

    with st.expander("Koszty transakcyjne"):
        costs_on = st.checkbox("Uwzględnij koszty", value=False, key="costs_on")
        st.caption("Wartości przykładowe w punktach bazowych (1 bp = 0,01%). Ustaw własne.")
        commission = st.number_input("Prowizja na stronę (bp)", 0.0, 200.0, 1.0, 0.5, key="commission",
                                     disabled=not costs_on)
        spread = st.number_input("Pełny spread (bp)", 0.0, 200.0, 3.0, 0.5, key="spread", disabled=not costs_on)
        slippage = st.number_input("Poślizg na stronę (bp)", 0.0, 200.0, 1.0, 0.5, key="slippage",
                                   disabled=not costs_on)
        borrow = st.number_input("Koszt pożyczki, short (bp rocznie)", 0.0, 2000.0, 50.0, 10.0, key="borrow",
                                 disabled=not costs_on)

    with st.expander("Walidacja i wykluczenia"):
        holdout_pct = st.slider("Zbiór hold-out: ostatnia część okresu (%)", 0, 50, 30, key="holdout")
        no_overlap = st.checkbox("Jedna otwarta pozycja naraz", value=True, key="no_overlap")
        excl_zero = st.checkbox("Pomijaj sesje z zerowym wolumenem", value=True, key="excl_zero")
        excl_ext = st.checkbox("Pomijaj okna z dniem o dużej zmianie", value=False, key="excl_ext")
        ext_pct = st.number_input("Próg dużej zmiany (% w ciągu dnia)", 1.0, 50.0, 10.0, 0.5, key="ext_pct",
                                  disabled=not excl_ext)
        top_days = st.number_input("Test wrażliwości: wynik bez N największych dni", 0, 20, 5, 1, key="top_days")

    run_clicked = st.button("Uruchom analizę", type="primary", key="run")

config = {
    "symbol": symbol, "htype": htype, "params": params, "horizon": int(horizon), "sides": sides,
    "target": target, "threshold_pct": float(threshold_pct), "execution": execution,
    "costs_on": bool(costs_on), "commission": float(commission), "spread": float(spread),
    "slippage": float(slippage), "borrow": float(borrow),
    "start": str(start_day), "end": str(end_day), "holdout_pct": int(holdout_pct),
    "no_overlap": bool(no_overlap), "excl_zero": bool(excl_zero), "excl_ext": bool(excl_ext),
    "ext_pct": float(ext_pct), "top_days": int(top_days),
}
config_sig = json.dumps(config, sort_keys=True)

# --------------------------------------------------------------------------
# Uruchomienie
# --------------------------------------------------------------------------
history: list[dict] = st.session_state.setdefault("history", [])

if run_clicked:
    try:
        spec = build_spec(config)
        with st.spinner("Liczę..."):
            result = run_hypothesis(prices, spec)
    except ValueError as exc:
        st.session_state.pop("run_state", None)
        st.error(f"Nieprawidłowa konfiguracja: {exc}")
    else:
        st.session_state["run_state"] = {
            "result": result,
            "prices": prices.loc[config["start"]:config["end"]],
            "config_sig": config_sig,
        }
        run_key = (result.spec_hash, result.data_fingerprint)
        if run_key not in {h["key"] for h in history}:
            full = result.summaries["full"]
            history.append({
                "key": run_key, "czas": datetime.now().strftime("%H:%M:%S"), "symbol": result.spec["symbol"],
                "hipoteza": HYPOTHESES[result.spec["hypothesis_type"]][0], "parametry": str(result.spec["params"]),
                "horyzont": result.spec["horizon"], "wykonanie": result.spec["execution"],
                "koszty": "tak" if result.spec["costs"]["enabled"] else "nie", "skrót": result.spec_hash,
                "zdarzenia": full.get("n_trades", 0), "trafność": full.get("hit_rate"),
                "różnica vs baza": full.get("hit_rate_diff"), "p": full.get("p_value_binomial"),
            })


# --------------------------------------------------------------------------
# Widoki wyniku
# --------------------------------------------------------------------------
def show_summary_table(res) -> None:
    rows = []
    for key, s in res.summaries.items():
        if s.get("n_trades", 0) == 0:
            rows.append({"Segment": SEGMENTS[key], "Zdarzenia": 0})
            continue
        lo, hi = s["hit_rate_ci95"]
        ratio = s["avg_win_loss_ratio"]
        rows.append({
            "Segment": SEGMENTS[key], "Zdarzenia": s["n_trades"], "Trafność": pct(s["hit_rate"]),
            "95% CI": f"{pct(lo)} do {pct(hi)}", "Baza": pct(s["baseline_hit_rate"]),
            "Różnica": f"{s['hit_rate_diff'] * 100:+.1f} pp", "p dwumianowe": f"{s['p_value_binomial']:.3f}",
            "Śr. zwrot netto": pct(s["mean_net_return"], 2), "Mediana": pct(s["median_net_return"], 2),
            "Zysk/strata": "-" if ratio is None else f"{ratio:.2f}", "Kapitał końcowy": pct(s["equity_final"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if res.split_date:
        st.caption(f"Granica podziału próby i hold-out: {res.split_date}. Zdarzenia próby muszą się zakończyć przed tą datą.")


def show_charts(res, period_prices: pd.DataFrame) -> None:
    trades = res.trades
    price_df = period_prices[["adj_close"]].reset_index().rename(columns={"adj_close": "cena"})
    line = alt.Chart(price_df).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("cena:Q", title="Cena skorygowana", scale=alt.Scale(zero=False)))
    if len(trades):
        pts = trades.assign(
            wynik=np.where(trades["hit"], "trafione", "chybione"),
            kierunek=np.where(trades["side"] == 1, "long", "short"))
        marks = alt.Chart(pts).mark_point(filled=True, size=45).encode(
            x="entry_date:T", y="entry_price:Q",
            color=alt.Color("wynik:N", scale=alt.Scale(domain=["trafione", "chybione"],
                                                       range=["#2e7d32", "#c62828"])),
            shape=alt.Shape("kierunek:N"),
            tooltip=["entry_date:T", "exit_date:T", "kierunek:N", "net_return:Q", "wynik:N"])
        st.altair_chart(line + marks)
    else:
        st.altair_chart(line)

    if not len(trades):
        return
    if res.spec["no_overlap"]:
        st.caption("Kapitał: pozycje kolejno, jedna naraz, bez dźwigni.")
        eq = trades[["exit_date", "equity"]]
        st.altair_chart(alt.Chart(eq).mark_line(interpolate="step-after").encode(
            x=alt.X("exit_date:T", title=None), y=alt.Y("equity:Q", title="Kapitał", scale=alt.Scale(zero=False))))
    else:
        st.caption("Krzywa kapitału jest dostępna tylko przy jednej otwartej pozycji naraz.")
    st.caption("Rozkład zwrotów netto ze zdarzeń.")
    st.altair_chart(alt.Chart(trades[["net_return"]]).mark_bar().encode(
        x=alt.X("net_return:Q", bin=alt.Bin(maxbins=30), title="Zwrot netto"),
        y=alt.Y("count()", title="Liczba zdarzeń")))


def show_result(state: dict) -> None:
    res = state["result"]
    spec = res.spec
    full = res.summaries["full"]
    st.subheader(f"{spec['symbol']}: {HYPOTHESES[spec['hypothesis_type']][0]} {spec['params']}")
    st.caption(f"Okres {res.first_date} do {res.last_date} ({res.n_sessions} sesji) | skrót specyfikacji "
               f"{res.spec_hash} | skrót danych {res.data_fingerprint}")
    st.caption(settings_line(spec))
    if state["config_sig"] != config_sig:
        st.info("Ustawienia w panelu różnią się od wyświetlanego wyniku. Kliknij Uruchom analizę, aby je zastosować.")

    if full["n_trades"] == 0:
        st.info("Hipoteza nie wygenerowała żadnych zdarzeń w wybranym okresie i z wybranymi ustawieniami.")
        for w in res.warnings:
            st.warning(w)
        return

    trials = sum(1 for h in history if h["symbol"] == spec["symbol"])
    p_value = full["p_value_binomial"]
    cols = st.columns(4)
    cols[0].metric("Trafność", pct(full["hit_rate"]), f"{full['hit_rate_diff'] * 100:+.1f} pp vs baza",
                   delta_color="off")
    cols[1].metric("Baza porównawcza", pct(full["baseline_hit_rate"]))
    cols[2].metric("Zdarzenia", full["n_trades"])
    cols[2].caption(f"efektywnie ok. {full['n_effective']}")
    cols[3].metric("p (dwumianowe)", f"{p_value:.3f}")
    cols[3].caption(f"po korekcie na {trials} {trials_label(trials)}: {min(1.0, p_value * trials):.3f}")

    lo, hi = full["hit_rate_ci95"]
    if lo <= full["baseline_hit_rate"] <= hi and p_value >= 0.05:
        st.info("Brak dowodów na przewagę nad bazą: baza mieści się w 95% przedziale ufności trafności, a p wynosi co najmniej 0,05.")
    else:
        st.warning("Trafność odbiega od bazy. To jedna z kilku prób w tej sesji, więc przed wnioskami sprawdź hold-out, test wrażliwości i p po korekcie.")
    for w in res.warnings:
        st.warning(w)

    tab_result, tab_trades, tab_charts, tab_spec, tab_history = st.tabs(
        ["Wynik", "Zdarzenia", "Wykresy", "Specyfikacja", "Historia prób"])
    with tab_result:
        show_summary_table(res)
        if res.sensitivity and res.sensitivity.get("n_trades"):
            se = res.sensitivity
            st.markdown(
                f"**Test wrażliwości** (bez dni: {', '.join(se['excluded_days'])}): {se['n_trades']} zdarzeń, "
                f"trafność {pct(se['hit_rate'])}, baza {pct(se['baseline_hit_rate'])}, "
                f"różnica {se['hit_rate_diff'] * 100:+.1f} pp.")
    with tab_trades:
        shown = res.trades.assign(side=np.where(res.trades["side"] == 1, "long", "short")).rename(columns={
            "signal_date": "Data sygnału", "side": "Kierunek", "entry_date": "Wejście", "exit_date": "Wyjście",
            "entry_price": "Cena wejścia", "exit_price": "Cena wyjścia", "gross_return": "Zwrot brutto",
            "net_return": "Zwrot netto", "hit": "Trafione", "equity": "Kapitał"})
        st.dataframe(shown, hide_index=True)
        st.download_button("Pobierz zdarzenia (CSV)", res.trades.to_csv(index=False).encode("utf-8"),
                           file_name=f"zdarzenia_{res.spec_hash}.csv", mime="text/csv")
    with tab_charts:
        show_charts(res, state["prices"])
    with tab_spec:
        st.caption("Pełna specyfikacja pozwala odtworzyć wynik dokładnie (razem ze skrótem danych).")
        st.json(res.spec)
        st.download_button("Pobierz wynik (JSON)", json.dumps(res.to_dict(), ensure_ascii=False, indent=2),
                           file_name=f"wynik_{res.spec_hash}.json", mime="application/json")
    with tab_history:
        show_history()


def show_history() -> None:
    if not history:
        st.caption("Brak prób w tej sesji.")
        return
    table = pd.DataFrame([{k: v for k, v in h.items() if k != "key"} for h in history])
    table["trafność"] = table["trafność"].map(pct)
    table["różnica vs baza"] = table["różnica vs baza"].map(lambda x: "-" if x is None else f"{x * 100:+.1f} pp")
    table["p"] = table["p"].map(lambda x: "-" if x is None else f"{x:.3f}")
    st.dataframe(table, hide_index=True)
    st.caption("Każda próba zwiększa ryzyko, że jakiś wynik wyjdzie dobry przypadkiem. Korekta p to mnożenie przez liczbę "
               "prób dla danego instrumentu (Bonferroni). Historia dotyczy bieżącej sesji i znika po odświeżeniu strony.")


# --------------------------------------------------------------------------
# Strona główna
# --------------------------------------------------------------------------
status = read_status()
fresh_msg = f"Dane do sesji {last_day}"
if status:
    fresh_msg += f", ostatnia aktualizacja {status.get('generated_at', '?')} UTC"
if status and status.get("overall_status") == "fail":
    st.error(f"{fresh_msg}. Ostatnia aktualizacja zakończyła się błędem: {'; '.join(status.get('problems', []))}")
elif status and status.get("overall_status") == "warn":
    st.warning(f"{fresh_msg}. Ostatnia aktualizacja zakończyła się ostrzeżeniami.")
else:
    st.caption(fresh_msg + ".")

state = st.session_state.get("run_state")
if state:
    show_result(state)
else:
    st.markdown("Wybierz instrument, hipotezę i opcje w panelu po lewej, a potem kliknij **Uruchom analizę**. "
                "Wynik zawsze porównujemy z bazą: trafnością tej samej pozycji na wszystkich sesjach okresu.")
    overview = []
    for sym in symbols:
        p = prices_for(sym)
        meta = describe(sym)
        overview.append({"Symbol": sym, "Instrument": meta["name"], "Struktura":
                         "fizyczny metal" if meta["structure"] == "physical" else "kontrakty terminowe",
                         "Od": p.index.min().date(), "Do": p.index.max().date(), "Sesje": len(p),
                         "Ostatnie zamknięcie": round(float(p["close"].iloc[-1]), 2)})
    st.dataframe(pd.DataFrame(overview), hide_index=True)
    with st.expander("Historia prób w tej sesji"):
        show_history()
