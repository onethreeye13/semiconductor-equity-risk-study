"""Causal SNDK features, fixed-horizon ETF accounts and descriptive range tests."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.indicators import rsi_wilder


def research_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["adj_close"]
    sma = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=1).replace(0, np.nan)
    return pd.DataFrame({
        "momentum_5": close.pct_change(5, fill_method=None),
        "ma_distance_20": close / sma - 1,
        "rsi_14": rsi_wilder(close, 14),
        "zscore_20": (close - sma) / std,
    })


def candidate_rules(features: pd.DataFrame, config: dict):
    """Yield the entire declared grid, including never-triggered rules."""
    for feature, thresholds in config["features"].items():
        for threshold in thresholds:
            rule_id = f"{feature}_ge_{threshold:g}"
            # Unknown warmup features cannot trigger an entry.
            signal = features[feature].ge(threshold) & features[feature].notna()
            yield rule_id, feature, float(threshold), signal


@dataclass
class HorizonResult:
    daily: pd.DataFrame
    trades: pd.DataFrame


def fixed_horizon_backtest(bars: pd.DataFrame, signal: pd.Series, start: str,
                           end_exclusive: str, holding_sessions: int,
                           cost_bps: float = 10) -> HorizonResult:
    """Close signal -> next open; precommitted h-th close exit, no overlap.

    Shift on the full underlying calendar before slicing. End-of-segment trades
    without a complete horizon are excluded, rather than shortened. Each segment
    starts cash. Return marks use adjusted prices, as a total-return approximation.
    """
    if not isinstance(holding_sessions, int) or holding_sessions < 1:
        raise ValueError("holding_sessions must be a positive integer")
    if not 0 <= cost_bps < 10000:
        raise ValueError("Invalid one-way cost")
    for idx in [bars.index, signal.index]:
        if not idx.is_unique or not idx.is_monotonic_increasing:
            raise ValueError("Dates must be unique and increasing")
    x = bars.loc[(bars.index >= pd.Timestamp(start)) &
                 (bars.index < pd.Timestamp(end_exclusive))]
    if x.empty:
        raise ValueError("Empty backtest segment")
    expected = signal.loc[(signal.index >= pd.Timestamp(start)) &
                          (signal.index < pd.Timestamp(end_exclusive))].index
    if not x.index.equals(expected):
        raise ValueError("ETF and underlying calendars differ; no silent intersection")
    prices = x[["adj_open", "adj_close"]].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid ETF prices")
    known = signal.shift(1).reindex(x.index)
    signal_dates = pd.Series(signal.index, index=signal.index).shift(1).reindex(x.index)
    if known.isna().any() or not known.isin([True, False, 0, 1]).all():
        raise ValueError("Missing prior-close signal or invalid signal")
    c = cost_bps / 10000
    equity = np.ones(len(x))
    exposed = np.zeros(len(x), dtype=int)
    rows, cash, i = [], 1.0, 0
    while i < len(x):
        exit_i = i + holding_sessions - 1
        if bool(known.iloc[i]) and exit_i < len(x):
            entry_price, exit_price = prices[i, 0], prices[exit_i, 1]
            units = cash / (entry_price * (1 + c))
            equity[i:exit_i + 1] = units * prices[i:exit_i + 1, 1]
            equity[exit_i] *= 1 - c
            exposed[i:exit_i + 1] = 1
            net_return = exit_price / entry_price * (1 - c) / (1 + c) - 1
            rows.append({"signal_date": signal_dates.iloc[i], "entry_date": x.index[i],
                         "exit_date": x.index[exit_i], "holding_sessions": holding_sessions,
                         "entry_adj_open": entry_price, "exit_adj_close": exit_price,
                         "gross_return": exit_price / entry_price - 1,
                         "net_return": net_return})
            cash = equity[exit_i]
            i = exit_i + 1
        else:
            equity[i] = cash
            i += 1
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    daily = pd.DataFrame({"equity": equity, "drawdown": equity / peak - 1,
                          "net_return": equity / np.r_[1., equity[:-1]] - 1,
                          "exposed": exposed}, index=x.index)
    columns = ["signal_date", "entry_date", "exit_date", "holding_sessions",
               "entry_adj_open", "exit_adj_close", "gross_return", "net_return"]
    return HorizonResult(daily, pd.DataFrame(rows, columns=columns))


def summarize_horizon(result: HorizonResult) -> dict:
    returns = result.trades["net_return"].astype(float)
    wins, losses = returns[returns > 0], returns[returns < 0]
    n, nw, nl = len(returns), len(wins), len(losses)
    mean_win = wins.mean() if nw else np.nan
    mean_loss = -losses.mean() if nl else np.nan
    rate = nw / n if n else np.nan
    lo = hi = np.nan
    if n:
        z = 1.959963984540054
        denom = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denom
        half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
        lo, hi = center - half, center + half
    return {"trades": n, "wins": nw, "losses": nl, "flat_trades": n - nw - nl,
            "win_rate": rate, "win_ci_low": lo, "win_ci_high": hi,
            "mean_win": mean_win, "mean_loss_magnitude": mean_loss,
            "payoff_ratio": mean_win / mean_loss if nw and nl else np.nan,
            "profit_factor": wins.sum() / -losses.sum() if nw and nl else np.nan,
            "mean_net_return": returns.mean() if n else np.nan,
            "median_net_return": returns.median() if n else np.nan,
            "cumulative_return": result.daily.equity.iloc[-1] - 1,
            "maximum_drawdown": result.daily.drawdown.min(),
            "exposure": result.daily.exposed.mean(), "sessions": len(result.daily)}


def training_leaders(training: pd.DataFrame, minimum_trades: int) -> dict:
    """Accept training rows only; prohibit accidental validation-based selection."""
    if set(training["segment"]) != {"train"}:
        raise ValueError("Selection accepts training rows only")
    eligible = training.loc[(training.ticker == "SNDQ") &
                            (training.rule_id != "unconditional") &
                            (training.trades >= minimum_trades) &
                            (training.wins > 0) & (training.losses > 0)]
    leaders = {}
    for metric in ["mean_net_return", "win_rate", "payoff_ratio"]:
        ranked = eligible.sort_values([metric, "trades", "rule_id", "holding_sessions"],
                                      ascending=[False, False, True, True])
        leaders[metric] = None if ranked.empty else {
            "rule_id": str(ranked.iloc[0].rule_id),
            "holding_sessions": int(ranked.iloc[0].holding_sessions),
            "training_trades": int(ranked.iloc[0].trades),
            "training_value": float(ranked.iloc[0][metric])}
    best = leaders["mean_net_return"]
    leaders["selected"] = best if best and best["training_value"] > 0 else None
    leaders["eligible_candidates"] = len(eligible)
    return leaders


def range_observations(bars: pd.DataFrame) -> pd.DataFrame:
    """Realized explanatory labels, not observable-at-open trading signals."""
    factor = bars.adj_close / bars.close
    previous = bars.adj_close.shift(1)
    adj_open = bars.open * factor
    close_return = bars.adj_close / previous - 1
    intraday_return = bars.close / bars.open - 1
    result = pd.DataFrame({
        "range": (bars.high - bars.low) * factor / previous,
        "close_return": close_return,
        "abs_close_return": close_return.abs(),
        "abs_intraday_return": intraday_return.abs(),
        "abs_overnight_gap": (adj_open / previous - 1).abs(),
    })
    for name, values in [("close_to_close", close_return), ("open_to_close", intraday_return)]:
        result[name] = np.select([values > 0, values < 0], ["up", "down"], default="flat")
        result.loc[values.isna(), name] = "unavailable"
    return result.dropna(subset=["range", "close_return"])


def range_summary(observations: pd.DataFrame, window: str) -> pd.DataFrame:
    rows = []
    for classification in ["close_to_close", "open_to_close"]:
        for direction in ["down", "up", "flat"]:
            x = observations.loc[observations[classification] == direction]
            rows.append({"window": window, "classification": classification,
                         "direction": direction, "days": len(x),
                         "mean_range": x["range"].mean(), "median_range": x["range"].median(),
                         "range_p90": x["range"].quantile(.9),
                         "mean_abs_close_return": x.abs_close_return.mean(),
                         "mean_abs_intraday_return": x.abs_intraday_return.mean(),
                         "mean_abs_overnight_gap": x.abs_overnight_gap.mean()})
    return pd.DataFrame(rows)


def bootstrap_range_difference(observations: pd.DataFrame, classification: str,
                               repetitions: int, block: int, seed: int) -> dict:
    """Circular moving-block percentile interval; no stationarity guarantee."""
    if repetitions < 1 or not 1 <= block <= len(observations):
        raise ValueError("Invalid bootstrap settings")
    labels = observations[classification].to_numpy()
    ranges = observations["range"].to_numpy()

    def difference(indices):
        selected, values = labels[indices], ranges[indices]
        down, up = values[selected == "down"], values[selected == "up"]
        return down.mean() - up.mean() if len(down) and len(up) else np.nan

    rng = np.random.default_rng(seed)
    n = len(observations)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        draws.append(difference(indices))
    finite = np.asarray(draws)[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
    return {"down_minus_up_mean_range": difference(np.arange(n)),
            "ci_low": lo, "ci_high": hi, "valid_repetitions": len(finite),
            "block_sessions": block, "seed": seed}
