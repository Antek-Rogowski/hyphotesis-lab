"""Testy interfejsu Streamlit uruchamianego bez przeglądarki (streamlit.testing)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from hypothesis_lab.synthetic import random_walk_prices

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def write_store(directory: Path, symbols) -> None:
    for i, symbol in enumerate(symbols):
        p = random_walk_prices(i, n=1500, start="2020-01-02")
        df = p[["open", "high", "low", "close", "adj_close", "volume"]].copy()
        df.insert(0, "symbol", symbol)
        df["source"] = "fake"
        df["ingested_at"] = pd.Timestamp("2026-11-06 22:50")
        df.to_parquet(directory / f"{symbol}.parquet")


@pytest.fixture
def app(tmp_path, monkeypatch):
    write_store(tmp_path, ["GLD", "SLV"])
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_loads_with_overview_and_no_errors(app):
    assert not app.exception
    assert app.title[0].value == "Hypothesis Lab"
    assert app.sidebar.selectbox(key="symbol").options == ["GLD: Złoto", "SLV: Srebro"]
    assert len(app.metric) == 0                                  # jeszcze nie uruchomiono analizy
    assert len(app.dataframe) == 1                               # tabela przeglądowa instrumentów


def test_run_shows_result_metrics(app):
    app.sidebar.button(key="run").click().run()
    assert not app.exception
    assert [m.label for m in app.metric] == ["Trafność", "Baza porównawcza", "Zdarzenia", "p (dwumianowe)"]
    result = app.session_state["run_state"]["result"]
    assert result.spec["symbol"] == "GLD" and result.spec["hypothesis_type"] == "mean_reversion"
    assert len(app.tabs) == 5


def test_other_hypothesis_costs_and_options_reach_the_spec(app):
    app.sidebar.selectbox(key="htype").select("shock_rebound").run()
    app.sidebar.number_input(key="param_shock_rebound_x_pct").set_value(1.5).run()
    app.sidebar.number_input(key="horizon").set_value(3).run()
    app.sidebar.selectbox(key="sides").select("long").run()
    app.sidebar.radio(key="execution").set_value("open_t+1").run()
    app.sidebar.checkbox(key="costs_on").check().run()
    app.sidebar.checkbox(key="excl_ext").check().run()
    app.sidebar.button(key="run").click().run()
    assert not app.exception
    spec = app.session_state["run_state"]["result"].spec
    assert spec["hypothesis_type"] == "shock_rebound" and spec["params"] == {"x_pct": 1.5}
    assert spec["horizon"] == 3 and spec["sides"] == "long" and spec["execution"] == "open_t+1"
    assert spec["costs"]["enabled"] is True and spec["exclude_extreme_abs_return"] == pytest.approx(0.10)


def test_threshold_target_is_passed_to_the_engine(app):
    app.sidebar.radio(key="target").set_value("threshold").run()
    app.sidebar.number_input(key="threshold_pct").set_value(0.5).run()
    app.sidebar.button(key="run").click().run()
    assert not app.exception
    spec = app.session_state["run_state"]["result"].spec
    assert spec["target"] == "threshold" and spec["target_threshold_pct"] == pytest.approx(0.5)


def test_history_counts_distinct_runs_only(app):
    app.sidebar.button(key="run").click().run()
    app.sidebar.button(key="run").click().run()                      # ta sama konfiguracja: bez nowej próby
    assert len(app.session_state["history"]) == 1
    app.sidebar.number_input(key="horizon").set_value(7).run()
    app.sidebar.button(key="run").click().run()
    assert len(app.session_state["history"]) == 2
    assert not app.exception


def test_stale_result_notice_after_changing_settings(app):
    app.sidebar.button(key="run").click().run()
    app.sidebar.number_input(key="horizon").set_value(9).run()
    assert any("różnią się" in i.value for i in app.info)


def test_switching_symbol_keeps_the_app_working(app):
    app.sidebar.selectbox(key="symbol").select("SLV").run()
    app.sidebar.button(key="run").click().run()
    assert not app.exception
    assert app.session_state["run_state"]["result"].spec["symbol"] == "SLV"


def test_empty_data_directory_shows_a_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Brak plików Parquet" in w.value for w in at.warning)


def test_update_status_banner(tmp_path, monkeypatch):
    write_store(tmp_path, ["GLD"])
    (tmp_path / "update_status.json").write_text(
        '{"overall_status": "fail", "problems": ["SLV: błąd sieci"], "generated_at": "2026-11-06 22:50:00"}',
        encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("SLV: błąd sieci" in e.value for e in at.error)
