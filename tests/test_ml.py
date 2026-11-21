"""Hand-checkable ML timing, fit isolation and account invariants."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.ml import (FEATURES, make_features, make_dataset, date_block,
                    fit_predict, open_interval_backtest)


def bars(opens):
    dates = pd.bdate_range("2020-01-01", periods=len(opens))
    return pd.DataFrame({"adj_open": opens, "adj_close": opens,
                         "volume": 1000+np.arange(len(opens))}, index=dates, dtype=float)


def intervals(b):
    return make_dataset(b).iloc[:-2]


def test_features_unchanged_when_future_prices_and_volumes_change():
    b = bars(100+np.sin(np.arange(100)) + np.arange(100)/10)
    prefix = make_features(b.iloc[:70])
    b.loc[b.index[70]:, ["adj_close", "adj_open", "volume"]] *= 10
    pd.testing.assert_frame_equal(prefix, make_features(b).iloc[:70])
    assert list(prefix.columns) == FEATURES


def test_labels_use_next_two_opens_and_unknown_tail_stays_missing():
    b = bars([100, 200, 220, 198])
    d = make_dataset(b)
    assert d.forward_return.iloc[0] == pytest.approx(.1)
    assert d.forward_return.iloc[1] == pytest.approx(-.1)
    assert d.target.iloc[:2].tolist() == [1, 0]
    assert d.target.iloc[-2:].isna().all()
    assert d.entry_date.iloc[0] == b.index[1]
    assert d.label_end_date.iloc[0] == b.index[2]
    with pytest.raises(ValueError, match="only one-session"):
        make_dataset(b, horizon=5)


def test_purge_uses_label_end_dates_including_boundary_equality():
    b = bars(100+np.sin(np.arange(100)))
    d = make_dataset(b)
    boundary = b.index[80]
    kept, removed = date_block(d, str(b.index[50].date()), str(boundary.date()))
    assert kept.label_end_date.max() < boundary
    assert list(removed.index) == list(b.index[78:80])
    assert removed.reason.eq("label_reaches_next_block").all()
    broken = d.copy()
    broken.loc[b.index[60], "volume_change_1"] = np.nan
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        date_block(broken, str(b.index[50].date()), str(boundary.date()))


def test_scaler_fits_train_only_and_predictions_ignore_evaluation_labels():
    config = json.loads((Path(__file__).parents[1]/"config_ml.json").read_text())
    d = make_dataset(bars(100+5*np.sin(np.arange(160)/2)))
    train = d.iloc[50:100].copy()
    evaluation = d.iloc[102:140].copy()
    p, fitted = fit_predict(train, evaluation, config)
    np.testing.assert_allclose(fitted["LogisticRegression"]["scale"].mean_, train[FEATURES].mean())
    assert p.Prior.eq(train.target.mean()).all()
    altered = evaluation.copy()
    altered["target"] = 1-altered.target
    altered["forward_return"] = 9999
    p2, _ = fit_predict(train, altered, config)
    pd.testing.assert_frame_equal(p, p2)
    overlapping = d.iloc[99:140]
    with pytest.raises(ValueError, match="purge first"):
        fit_predict(train, overlapping, config)


def test_open_account_cannot_earn_pre_entry_gap_or_post_exit_rebound():
    b = bars([100, 200, 180, 900])
    d = intervals(b)
    result = open_interval_backtest(b, d, pd.Series([1, 0], index=d.index), cost_bps=0)
    assert result.daily.equity.iloc[-1] == pytest.approx(.9)
    assert result.trades.net_return.iloc[0] == pytest.approx(-.1)
    assert result.trades.exit_date.iloc[0] == b.index[2]
    assert result.daily.drawdown.min() == pytest.approx(-.1)


def test_consecutive_longs_merge_and_costs_match_one_round_trip():
    b = bars([100, 100, 100, 100, 100])
    d = intervals(b)
    result = open_interval_backtest(b, d, pd.Series(1, index=d.index), cost_bps=10)
    assert len(result.trades) == 1
    assert result.trades.holding_sessions.iloc[0] == 3
    assert result.daily.turnover.sum() == 2
    expected = .999/1.001
    assert result.daily.equity.iloc[-1] == pytest.approx(expected)
    assert result.trades.net_return.iloc[0] == pytest.approx(expected-1)


def test_disjoint_trade_ledger_compounds_to_account_equity():
    b = bars([100, 200, 220, 110, 121, 150, 160])
    d = intervals(b)
    result = open_interval_backtest(b, d, pd.Series([1, 0, 1, 1, 0], index=d.index), 25)
    assert result.daily.equity.iloc[-1] == pytest.approx((1+result.trades.net_return).prod())
    assert result.daily.equity.iloc[-1] == pytest.approx((1+result.daily.net_return).prod())
    assert result.daily.turnover.sum() == 4


def test_interval_gaps_or_wrong_next_open_are_rejected():
    b = bars([100]*7)
    d = intervals(b).drop(b.index[2])
    with pytest.raises(ValueError, match="no overlap or gap"):
        open_interval_backtest(b, d, pd.Series(1, index=d.index))
    d = intervals(b).iloc[:1].copy()
    d["entry_date"] = b.index[2]
    d["label_end_date"] = b.index[3]
    with pytest.raises(ValueError, match="next two actual"):
        open_interval_backtest(b, d, pd.Series(1, index=d.index))
