"""Economic invariants for the inverse-ETF extension; prices are test fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.sndk_inverse import (bootstrap_range_difference, fixed_horizon_backtest,
                              range_observations, research_features,
                              summarize_horizon, training_leaders)


def example_bars(opens, closes=None):
    closes = opens if closes is None else closes
    return pd.DataFrame({"adj_open": opens, "adj_close": closes},
                        index=pd.bdate_range("2026-01-01", periods=len(opens)), dtype=float)


def backtest(bars, signal, h=1, cost=0, stop=None):
    return fixed_horizon_backtest(bars, pd.Series(signal, index=bars.index),
                                  str(bars.index[1].date()),
                                  stop or str((bars.index[-1] + pd.Timedelta(days=1)).date()), h, cost)


def test_close_signal_earns_only_next_open_to_close_for_one_day_hold():
    bars = example_bars([10, 100, 200, 300], [10, 110, 220, 330])
    result = backtest(bars, [True, False, False, False])
    assert result.trades.iloc[0].signal_date == bars.index[0]
    assert result.trades.iloc[0].entry_date == bars.index[1]
    assert result.trades.iloc[0].exit_date == bars.index[1]
    assert result.daily.equity.tolist() == pytest.approx([1.1, 1.1, 1.1])


def test_fixed_horizon_holds_overnight_and_persistent_signal_never_overlaps():
    bars = example_bars([100]*7, [100, 110, 90, 130, 110, 110, 110])
    result = backtest(bars, [True]*7, h=3)
    assert result.trades.entry_date.tolist() == [bars.index[1], bars.index[4]]
    assert result.trades.exit_date.tolist() == [bars.index[3], bars.index[6]]
    assert result.trades.net_return.tolist() == pytest.approx([.3, .1])
    assert result.daily.equity.iloc[-1] == pytest.approx(1.43)
    assert result.daily.equity.iloc[1] == pytest.approx(.9)


def test_cross_boundary_trade_is_purged_and_test_starts_cash():
    bars = example_bars([100]*7, [100, 110, 90, 130, 110, 110, 110])
    signal = pd.Series(True, index=bars.index)
    split = str(bars.index[3].date())
    train = backtest(bars, signal, h=3, stop=split)
    assert train.trades.empty  # The day-1 entry would exit on the excluded boundary.
    test = fixed_horizon_backtest(bars, signal, split,
                                  str((bars.index[-1] + pd.Timedelta(days=1)).date()), 3, 0)
    assert test.trades.entry_date.iloc[0] == bars.index[3]
    assert test.trades.signal_date.iloc[0] == bars.index[2]
    assert test.daily.equity.iloc[-1] == pytest.approx(1.1)


def test_costs_compound_per_round_trip_and_flat_equity_has_no_free_profit():
    result = backtest(example_bars([100]*5), [True]*5, cost=10)
    one = (1-.001)/(1+.001)
    assert result.trades.net_return.tolist() == pytest.approx([one-1]*4)
    assert result.daily.equity.iloc[-1] == pytest.approx(one**4)
    assert result.daily.drawdown.min() == pytest.approx(one**4-1)


def test_empty_or_all_winning_accounts_do_not_have_infinite_payoff():
    bars = example_bars([100]*3, [100, 110, 110])
    empty = summarize_horizon(backtest(bars, [False]*3))
    assert empty['trades'] == 0 and empty['cumulative_return'] == 0
    assert np.isnan(empty['win_rate']) and np.isnan(empty['payoff_ratio'])
    wins = summarize_horizon(backtest(bars, [True]*3))
    assert wins['win_rate'] == 1 and np.isnan(wins['payoff_ratio'])
    assert wins['win_ci_low'] < 1


def test_future_append_cannot_change_past_feature_values():
    prices = pd.DataFrame({'adj_close': 100 + np.arange(80) + np.sin(np.arange(80))},
                          index=pd.bdate_range('2025-01-01', periods=80))
    pd.testing.assert_frame_equal(research_features(prices.iloc[:55]),
                                  research_features(prices).iloc[:55])
    flat = prices.copy(); flat['adj_close'] = 100
    assert research_features(flat).zscore_20.isna().all()


def test_training_selection_rejects_validation_and_filters_tiny_perfect_sample():
    frame = pd.DataFrame([
        {'segment':'train','ticker':'SNDQ','rule_id':'tiny','holding_sessions':1,
         'trades':2,'wins':2,'losses':0,'mean_net_return':.50,'win_rate':1.,'payoff_ratio':np.nan},
        {'segment':'train','ticker':'SNDQ','rule_id':'eligible','holding_sessions':1,
         'trades':20,'wins':11,'losses':9,'mean_net_return':.01,'win_rate':.55,'payoff_ratio':1.5}])
    assert training_leaders(frame,20)['selected']['rule_id'] == 'eligible'
    changed = frame.copy(); changed.loc[1,'mean_net_return'] = -.01
    assert training_leaders(changed,20)['selected'] is None
    changed = frame.copy(); changed.loc[1,'segment'] = 'validation'
    with pytest.raises(ValueError,match='training rows only'):
        training_leaders(changed,20)


def test_calendar_gaps_fail_instead_of_turning_next_session_into_two_days_later():
    bars = example_bars([100]*5)
    signals = pd.Series(True,index=bars.index)
    with pytest.raises(ValueError, match='calendars differ'):
        fixed_horizon_backtest(bars.drop(bars.index[2]),signals,
                              str(bars.index[1].date()),'2026-02-01',1)


def test_range_is_relative_and_close_day_label_can_disagree_with_intraday():
    idx = pd.bdate_range('2026-01-01',periods=3)
    bars = pd.DataFrame({'open':[100,80,190], 'close':[100,90,180],
                         'high':[105,95,200], 'low':[95,75,170],
                         'adj_close':[50,45,90]},index=idx,dtype=float)
    obs = range_observations(bars)
    assert obs.iloc[0]['range'] == pytest.approx(.2)
    assert obs.iloc[0].close_to_close == 'down'
    assert obs.iloc[0].open_to_close == 'up'
    scaled = bars * 10
    pd.testing.assert_frame_equal(obs,range_observations(scaled))


def test_block_bootstrap_reproducible_with_hand_checkable_difference():
    obs = pd.DataFrame({'range':[.2,.1,.2,.1,.2,.1],
                        'close_to_close':['down','up']*3})
    a = bootstrap_range_difference(obs,'close_to_close',200,2,42)
    b = bootstrap_range_difference(obs,'close_to_close',200,2,42)
    assert a == b
    assert a['down_minus_up_mean_range'] == pytest.approx(.1)
    assert a['ci_low'] == pytest.approx(.1)
    assert a['ci_high'] == pytest.approx(.1)


def test_snapshot_reuse_checks_hashes_and_refresh_preserves_old_version(tmp_path, monkeypatch):
    import json
    from src import sndk_inverse_data as module

    idx = pd.bdate_range('2026-01-02', periods=4)
    frame = pd.DataFrame({'open':100., 'close':100., 'adj_close':100.,
                          'high':101., 'low':99., 'volume':1000.}, index=idx)
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'data/processed').mkdir(parents=True)
    for ticker in ['SNDK','SOXX']:
        frame.to_csv(tmp_path/f'data/processed/{ticker}.csv')
    config = {'underlying':'SNDK', 'etfs':['SNDQ','SNXX'],
              'download_end_exclusive':'2026-01-08',
              'listing_floors':{'SNDQ':'2026-01-02','SNXX':'2026-01-02'}}
    monkeypatch.setattr(module,'download_data',lambda *args: {'SNDK':frame,'SOXX':frame})
    calls = []

    def fetch(ticker, *args):
        calls.append(ticker)
        return ticker.encode()

    monkeypatch.setattr(module,'fetch_chart_response',fetch)
    monkeypatch.setattr(module,'parse_chart_response',lambda *args: frame)
    _, first = module.load_research_data(tmp_path,config)
    _, reused = module.load_research_data(tmp_path,config)
    assert calls == ['SNDQ','SNXX'] and first == reused
    old_path = tmp_path/'data/raw/sndk_inverse'/first['snapshot_id']/'SNDQ.csv'
    original = old_path.read_bytes()
    _, refreshed = module.load_research_data(tmp_path,config,refresh=True)
    assert refreshed['snapshot_id'] != first['snapshot_id']
    assert old_path.read_bytes() == original
    new_path = tmp_path/'data/raw/sndk_inverse'/refreshed['snapshot_id']/'SNDQ.csv'
    new_path.write_bytes(new_path.read_bytes() + b'\n')
    with pytest.raises(ValueError,match='integrity mismatch'):
        module.load_research_data(tmp_path,config)
