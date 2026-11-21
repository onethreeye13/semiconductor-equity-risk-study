"""Past-and-present-only features. No backfilling and no centered windows."""

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI: seed with the first period mean, then smooth recursively.

    A flat price gets RSI=50; all gains=100; all losses=0. Warmup stays NaN.
    """
    if period < 1:
        raise ValueError("RSI period must be positive")
    if close.isna().any():
        raise ValueError("Missing prices must be resolved explicitly before RSI")
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    out = pd.Series(np.nan, index=close.index, name=f"rsi_{period}")
    if len(close) <= period:
        return out
    avg_gain = gain.iloc[1:period + 1].mean()
    avg_loss = loss.iloc[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            avg_gain = ((period - 1) * avg_gain + gain.iloc[i]) / period
            avg_loss = ((period - 1) * avg_loss + loss.iloc[i]) / period
        if avg_loss == 0:
            out.iloc[i] = 50.0 if avg_gain == 0 else 100.0
        else:
            out.iloc[i] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def add_features(bars: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute each ticker separately, retaining the full warmup history."""
    x = bars.copy()
    p = x["adj_close"]
    fast, slow = config["sma_fast"], config["sma_slow"]
    horizon = config["momentum_days"]
    x["daily_return"] = p.pct_change(fill_method=None)
    x["cumulative_return"] = p / p.iloc[0] - 1
    x["sma_fast"] = p.rolling(fast, min_periods=fast).mean()
    x["sma_slow"] = p.rolling(slow, min_periods=slow).mean()
    x["ma_distance"] = p / x["sma_slow"] - 1
    x["momentum"] = p.pct_change(horizon, fill_method=None)
    x["rsi"] = rsi_wilder(p, config["rsi_days"])
    window = config["volatility_days"]
    x["rolling_volatility"] = x["daily_return"].rolling(window, min_periods=window).std(ddof=1) * np.sqrt(config["annualization_sessions"])
    x["volume_change"] = x["volume"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    x["price_drawdown"] = p / p.cummax() - 1
    x.attrs["signal_parameters"] = {"sma_fast": fast, "sma_slow": slow, "momentum_days": horizon}
    return x


def trading_signals(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Desired holdings known AFTER today's close; execution is tomorrow's open.

    SMA: long when fast > slow, otherwise cash (including equality).
    Momentum: long when trailing 20-session return > 0, otherwise cash.
    Warmup is unavailable, represented by NaN, not a trading opinion.
    """
    valid_sma = features[["sma_fast", "sma_slow"]].notna().all(axis=1)
    sma = (features["sma_fast"] > features["sma_slow"]).astype(float).where(valid_sma)
    mom = (features["momentum"] > 0).astype(float).where(features["momentum"].notna())
    params = features.attrs.get("signal_parameters", {"sma_fast": 20, "sma_slow": 50, "momentum_days": 20})
    return {f"SMA_{params['sma_fast']}_{params['sma_slow']}": sma.rename("signal"),
            f"Momentum_{params['momentum_days']}": mom.rename("signal")}
