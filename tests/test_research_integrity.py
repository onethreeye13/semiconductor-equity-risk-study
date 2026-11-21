"""Small hand-checkable cases for economic timing, costs and leakage."""

import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest
from src.backtest import run_backtest
from src.metrics import summarize
from src.indicators import add_features, trading_signals, rsi_wilder
from src.data import normalize_and_validate, validate_window, write_sqlite


def bars(opens, closes=None):
    closes = opens if closes is None else closes
    idx = pd.bdate_range("2025-01-01", periods=len(opens))
    return pd.DataFrame({"adj_open": opens, "adj_close": closes}, index=idx, dtype=float)


def run(x, signals, cost=0, liquidate=True):
    return run_backtest(x, pd.Series(signals, index=x.index), str(x.index[1].date()),
                        str((x.index[-1] + pd.Timedelta(days=1)).date()), cost, liquidate)


def test_entry_cannot_capture_signal_day_or_entry_overnight_jump():
    x = bars([100, 200, 300], [100, 200, 330])
    result = run(x, [0, 1, 1])
    assert result.daily.equity.iloc[-1] == pytest.approx(1.1)
    assert result.daily.position_during_session.tolist() == [0, 1]


def test_exit_retains_overnight_exposure_but_not_exit_session_return():
    x = bars([100, 100, 150], [100, 110, 300])
    result = run(x, [1, 0, 0])
    assert result.daily.equity.iloc[-1] == pytest.approx(1.5)
    assert result.trades.net_return.iloc[0] == pytest.approx(.5)


def test_flat_price_round_trip_cost_is_exact_and_counts_two_sides():
    result = run(bars([100, 100, 100]), [1, 1, 1], cost=10)
    expected = (1 - .001) / (1 + .001)
    assert result.daily.equity.iloc[-1] == pytest.approx(expected)
    assert result.daily.turnover.sum() == 2
    assert result.trades.net_return.iloc[0] == pytest.approx(expected - 1)
    assert result.trades.exit_reason.iloc[0] == "terminal_liquidation"


def test_first_day_loss_is_in_drawdown():
    result = run(bars([100, 100, 90], [100, 90, 90]), [1, 1, 1])
    assert summarize(result)["maximum_drawdown"] == pytest.approx(-.1)


def test_initial_window_starts_cash_even_if_signal_was_long():
    result = run(bars([10, 100, 100]), [1, 1, 1])
    assert result.daily.equity.iloc[-1] == pytest.approx(1.0)


def test_no_trade_has_nan_trade_win_rate_and_sharpe():
    result = run(bars([100, 100, 100]), [0, 0, 0])
    stats = summarize(result)
    assert stats["closed_trades"] == 0
    assert np.isnan(stats["win_rate_closed_trades"])
    assert np.isnan(stats["sharpe_rf0"])


def test_features_and_signals_do_not_change_when_future_rows_are_appended():
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    p = pd.Series(100 + np.sin(np.arange(100) / 5) * 8 + np.arange(100) / 3,
                  index=pd.bdate_range("2025-01-01", periods=100))
    frame = pd.DataFrame({"adj_close": p, "volume": 1000 + np.arange(100)})
    full, prefix = add_features(frame, config), add_features(frame.iloc[:75], config)
    pd.testing.assert_frame_equal(full.iloc[:75], prefix)
    for name, sig in trading_signals(prefix).items():
        pd.testing.assert_series_equal(trading_signals(full)[name].iloc[:75], sig)


@pytest.mark.parametrize("values,expected", [([1,2,3,4,5],100),([5,4,3,2,1],0),([2,2,2,2,2],50)])
def test_rsi_zero_gain_loss_cases(values, expected):
    assert rsi_wilder(pd.Series(values,dtype=float), 3).iloc[-1] == expected


def test_missing_warmup_raises_instead_of_filling_with_future_information():
    with pytest.raises(ValueError, match="warmup"):
        run(bars([100, 100, 100]), [np.nan, 1, 1])


def test_missing_prices_and_duplicate_dates_are_rejected():
    raw = pd.DataFrame({"Open":[100.,101.],"High":[101.,102.],"Low":[99.,100.],
                        "Close":[100.,101.],"Adj Close":[100.,101.],"Volume":[1000.,1100.]},
                       index=pd.bdate_range("2025-01-01",periods=2))
    bad = raw.copy(); bad.iloc[1, 4] = np.nan
    with pytest.raises(ValueError,match="missing/non-finite"):
        normalize_and_validate(bad, "MU")
    with pytest.raises(ValueError,match="duplicate"):
        normalize_and_validate(pd.concat([raw,raw]), "MU")


def test_calendar_gap_cannot_be_silently_intersected():
    x = bars([100,101,102,103])
    with pytest.raises(ValueError, match="unequal"):
        validate_window({"SOXX":x,"MU":x.drop(x.index[2])}, ["MU","SOXX"], "SOXX", "2025-01-01", "2025-02-01")


def test_sqlite_round_trip_and_primary_key(tmp_path):
    raw = pd.DataFrame({"Open":[100.,101.],"High":[101.,102.],"Low":[99.,100.],
                        "Close":[100.,101.],"Adj Close":[99.,100.],"Volume":[1000.,1100.]},
                       index=pd.bdate_range("2025-01-01",periods=2))
    normalized = normalize_and_validate(raw,"MU")
    path = tmp_path / "test.sqlite"
    write_sqlite({"MU":normalized},path)
    with sqlite3.connect(path) as conn:
        result = conn.execute("SELECT adj_close FROM daily_bars ORDER BY date").fetchall()
        assert result == [(99.,),(100.,)]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO daily_bars SELECT * FROM daily_bars LIMIT 1")
