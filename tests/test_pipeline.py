"""Testy codziennej aktualizacji danych na sztucznym dostawcy (bez internetu)."""
import json

import pandas as pd
import pytest

from hypothesis_lab.pipeline import (
    STATUS_FILE,
    STORED_COLUMNS,
    detect_revision,
    read_stored,
    render_markdown,
    update_all,
    update_symbol,
    write_atomic,
)
from hypothesis_lab.synthetic import random_walk_prices

END = "2026-11-06"                                              # piątek
AFTER_CLOSE = pd.Timestamp("2026-11-06 23:00", tz="UTC")        # 18:00 w Nowym Jorku, po sesji
DURING_SESSION = pd.Timestamp("2026-11-06 18:12", tz="UTC")     # 13:12 w Nowym Jorku, sesja trwa


def make_raw(seed=0, n=800, end=END, tz=None) -> pd.DataFrame:
    """Ramka w formacie yfinance (Open, High, Low, Close, Adj Close, Volume)."""
    idx = pd.bdate_range(end=end, periods=n)
    base = random_walk_prices(seed, n=n, start=str(idx[0].date()))
    raw = pd.DataFrame({"Open": base["open"], "High": base["high"], "Low": base["low"], "Close": base["close"],
                        "Adj Close": base["adj_close"], "Volume": base["volume"]})
    if tz:
        raw.index = raw.index.tz_localize(tz)
    return raw


class FakeProvider:
    name = "fake"

    def __init__(self, frames):
        self.frames = {k: v.copy() for k, v in frames.items()}
        self.calls = []
        self.fail = set()

    def fetch_daily(self, symbol, period):
        self.calls.append((symbol, period))
        if symbol in self.fail:
            raise RuntimeError("symulowany błąd sieci")
        df = self.frames[symbol]
        if period == "3mo":
            df = df[df.index >= df.index.max() - pd.Timedelta(days=92)]
        return df.copy()


def test_initial_load_writes_file(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1)})
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["status"] == "ok" and res["mode"] == "initial" and res["written"]
    stored = read_stored(tmp_path / "GLD.parquet")
    assert list(stored.columns) == STORED_COLUMNS
    assert (stored["source"] == "fake").all() and res["last_date"] == END
    assert prov.calls == [("GLD", "10y")]


def test_incremental_appends_only_the_new_session(tmp_path):
    full = make_raw(1, n=801)
    prov = FakeProvider({"GLD": full.iloc[:-1]})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    prov.frames["GLD"] = full
    prov.calls.clear()
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "incremental" and res["new_rows"] == 1 and res["last_date"] == END
    assert prov.calls == [("GLD", "3mo")]                       # bez pełnego pobierania
    stored = read_stored(tmp_path / "GLD.parquet")
    assert len(stored) == 801
    assert stored["close"].iloc[-1] == pytest.approx(full["Close"].iloc[-1])


def test_rerun_is_idempotent(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1)})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    before = (tmp_path / "GLD.parquet").read_bytes()
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "unchanged" and res["written"] is False and res["new_rows"] == 0
    assert (tmp_path / "GLD.parquet").read_bytes() == before


def test_partial_session_is_dropped(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1)})                   # dane obejmują piątek
    res = update_symbol("GLD", prov, tmp_path, now_utc=DURING_SESSION)
    assert res["partial_dropped"] is True and res["last_date"] == "2026-11-05"
    assert any("niepełną" in w for w in res["warnings"])


def test_recent_rows_are_overwritten_to_heal_bad_data(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1)})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    stored = read_stored(tmp_path / "GLD.parquet")
    good_close = stored["close"].iloc[-1]
    stored.loc[stored.index[-1], "close"] = good_close * 1.05    # uszkodzona świeczka w pliku
    write_atomic(stored, tmp_path / "GLD.parquet")
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "incremental" and res["written"] is True
    assert read_stored(tmp_path / "GLD.parquet")["close"].iloc[-1] == pytest.approx(good_close)


def test_revision_of_history_triggers_full_refresh(tmp_path):
    raw = make_raw(1)
    prov = FakeProvider({"GLD": raw})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    restated = raw.copy()
    restated[["Open", "High", "Low", "Close", "Adj Close"]] *= 0.5     # np. split zmienia całą historię
    prov.frames["GLD"] = restated
    prov.calls.clear()
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "full_refresh" and "split" in res["refresh_reason"]
    assert ("GLD", "10y") in prov.calls
    assert read_stored(tmp_path / "GLD.parquet")["close"].iloc[0] == pytest.approx(restated["Close"].iloc[0])


def test_dividend_style_adjustment_change_triggers_full_refresh(tmp_path):
    raw = make_raw(1)
    prov = FakeProvider({"GLD": raw})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    adjusted = raw.copy()
    adjusted["Adj Close"] *= 0.99                                       # zmienił się tylko współczynnik korekty
    prov.frames["GLD"] = adjusted
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "full_refresh" and "korekty" in res["refresh_reason"]


def test_gap_in_stored_data_triggers_full_refresh(tmp_path):
    full = make_raw(1, n=800)
    prov = FakeProvider({"GLD": full.iloc[:-200]})                      # zapis kończy się ok. 9 miesięcy wcześniej
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    prov.frames["GLD"] = full
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["mode"] == "full_refresh" and "luka" in res["refresh_reason"]
    assert res["rows_after"] == 800 and res["last_date"] == END


def test_invalid_bar_fails_and_leaves_file_untouched(tmp_path):
    full = make_raw(1, n=801)
    prov = FakeProvider({"GLD": full.iloc[:-1]})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    before = (tmp_path / "GLD.parquet").read_bytes()
    broken = full.copy()
    broken.iloc[-1, broken.columns.get_loc("High")] = broken["Low"].iloc[-1] - 1.0   # high < low
    prov.frames["GLD"] = broken
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    assert res["status"] == "fail" and "high < low" in res["error"]
    assert (tmp_path / "GLD.parquet").read_bytes() == before


def test_history_is_trimmed_to_max_years(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1, n=3000)})                   # ok. 11,5 roku
    res = update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE, max_years=10)
    stored = read_stored(tmp_path / "GLD.parquet")
    assert len(stored) < 3000 and res["rows_after"] == len(stored)
    assert stored.index.min() >= stored.index.max() - pd.DateOffset(years=10)


def test_timezone_aware_index_is_normalised(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1, tz="America/New_York")})
    update_symbol("GLD", prov, tmp_path, now_utc=AFTER_CLOSE)
    stored = read_stored(tmp_path / "GLD.parquet")
    assert stored.index.tz is None and str(stored.index.max().date()) == END


def test_detect_revision_unit():
    from hypothesis_lab.pipeline import standardize
    a = standardize(make_raw(1), "GLD", "fake", AFTER_CLOSE)
    assert detect_revision(a, a.copy()) == ""
    b = a.copy()
    b[["open", "high", "low", "close", "adj_close"]] *= 0.5
    assert detect_revision(a, b) != ""


def test_update_all_ok_writes_status_file(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1), "SLV": make_raw(2)})
    status = update_all(["GLD", "SLV"], prov, tmp_path, now_utc=AFTER_CLOSE, pause=0)
    assert status["overall_status"] == "ok" and status["latest_session"] == END and status["problems"] == []
    on_disk = json.loads((tmp_path / STATUS_FILE).read_text(encoding="utf-8"))
    assert set(on_disk["symbols"]) == {"GLD", "SLV"} and on_disk["provider"] == "fake"
    md = render_markdown(status)
    assert "| GLD |" in md and "| SLV |" in md and "OK" in md


def test_update_all_flags_out_of_sync_symbols(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1), "SLV": make_raw(2, end="2026-11-05")})
    status = update_all(["GLD", "SLV"], prov, tmp_path, now_utc=AFTER_CLOSE, pause=0)
    assert status["overall_status"] == "fail"
    assert any("rozjazd" in p for p in status["problems"])


def test_update_all_flags_stale_data(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1), "SLV": make_raw(2)})
    status = update_all(["GLD", "SLV"], prov, tmp_path, now_utc=AFTER_CLOSE + pd.Timedelta(days=20), pause=0)
    assert status["overall_status"] == "fail" and any("przestarzałe" in p for p in status["problems"])


def test_one_failing_symbol_does_not_stop_the_others(tmp_path):
    prov = FakeProvider({"GLD": make_raw(1), "SLV": make_raw(2)})
    prov.fail.add("SLV")
    status = update_all(["GLD", "SLV"], prov, tmp_path, now_utc=AFTER_CLOSE, pause=0)
    assert status["overall_status"] == "fail" and any("SLV" in p for p in status["problems"])
    assert (tmp_path / "GLD.parquet").exists() and not (tmp_path / "SLV.parquet").exists()
    assert "FAIL" in render_markdown(status)
