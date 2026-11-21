"""Daily long/cash backtest with next-open execution and explicit costs.

Prices are in adjusted total-return units, not a literal share/dividend ledger.
Signals calculated at close t cannot change exposure before open t+1.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame


def run_backtest(bars: pd.DataFrame, signal: pd.Series, start: str,
                 end_exclusive: str, cost_bps: float = 10,
                 liquidate: bool = True) -> BacktestResult:
    """Start each evaluation window with cash=1, including buy-and-hold.

    Fees and slippage are represented by one proportional cost per trade.
    All-in entry invests cash/(1+cost); exit receives proceeds*(1-cost).
    Overnight exposure belongs to yesterday's position, including on exits.
    Any terminal liquidation happens at the final close and pays exit cost.
    """
    if not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be in [0, 10000)")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Bars must have unique, chronological dates")
    if not signal.index.equals(bars.index):
        raise ValueError("Signal and bars must have exactly the same index")
    if not signal.dropna().isin([0, 1]).all():
        raise ValueError("Only long/cash holdings (1/0) are supported")
    price = bars[["adj_open", "adj_close"]]
    if not np.isfinite(price.to_numpy()).all() or (price <= 0).any().any():
        raise ValueError("Adjusted prices must be finite and positive")
    # Shift on the FULL price history, before slicing the evaluation window.
    desired = signal.shift(1)
    mask = (bars.index >= pd.Timestamp(start)) & (bars.index < pd.Timestamp(end_exclusive))
    selected = bars.loc[mask]
    if selected.empty:
        raise ValueError("No observations in evaluation window")
    if desired.loc[mask].isna().any():
        raise ValueError("Insufficient warmup or missing signal in evaluation window")
    previous_close = bars["adj_close"].shift(1)
    cost = cost_bps / 10_000
    equity, held = 1.0, 0
    entry = None
    rows, trades = [], []

    for i, (date, bar) in enumerate(selected.iterrows()):
        previous_equity = equity
        target = int(desired.loc[date])
        # The first session starts in cash; no pre-window return is earned.
        if held:
            equity *= bar["adj_open"] / previous_close.loc[date]
        turnover = abs(target - held)
        fee = 0.0
        if target != held:
            if target == 1:
                entry = {"entry_date": date, "entry_adj_price": bar["adj_open"],
                         "capital_before_entry": equity, "entry_i": i}
                fee = equity * cost / (1 + cost)
                equity -= fee
            else:
                fee = equity * cost
                equity -= fee
                trades.append({"entry_date": entry["entry_date"], "exit_date": date,
                               "entry_adj_price": entry["entry_adj_price"], "exit_adj_price": bar["adj_open"],
                               "holding_sessions": i - entry["entry_i"],
                               "net_return": equity / entry["capital_before_entry"] - 1,
                               "exit_reason": "signal"})
                entry = None
        held = target
        if held:
            equity *= bar["adj_close"] / bar["adj_open"]
        terminal_exit = bool(liquidate and i == len(selected) - 1 and held)
        if terminal_exit:
            exit_fee = equity * cost
            equity -= exit_fee
            fee += exit_fee
            turnover += 1
            trades.append({"entry_date": entry["entry_date"], "exit_date": date,
                           "entry_adj_price": entry["entry_adj_price"], "exit_adj_price": bar["adj_close"],
                           "holding_sessions": i - entry["entry_i"] + 1,
                           "net_return": equity / entry["capital_before_entry"] - 1,
                           "exit_reason": "terminal_liquidation"})
            held = 0
        rows.append({"date": date, "signal_previous_close": desired.loc[date],
                     "position_during_session": target, "position_end": held,
                     "turnover": turnover, "cost_in_initial_capital_units": fee,
                     "net_return": equity / previous_equity - 1, "equity": equity,
                     "terminal_exit": terminal_exit})
    daily = pd.DataFrame(rows).set_index("date")
    # Include initial capital in the high-water mark: a first-day loss counts.
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax().clip(lower=1.0) - 1
    columns = ["entry_date", "exit_date", "entry_adj_price", "exit_adj_price",
               "holding_sessions", "net_return", "exit_reason"]
    return BacktestResult(daily, pd.DataFrame(trades, columns=columns))
