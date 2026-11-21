"""Performance conventions are explicit and shared by every baseline."""

import numpy as np
from .backtest import BacktestResult


def summarize(result: BacktestResult, sessions_per_year: int = 252) -> dict:
    d, trades = result.daily, result.trades
    r = d["net_return"]
    vol = r.std(ddof=1)
    n = len(d)
    return {
        "start": d.index[0].date().isoformat(),
        "end": d.index[-1].date().isoformat(),
        "sessions": n,
        "cumulative_return": d["equity"].iloc[-1] - 1,
        "annualized_return": d["equity"].iloc[-1] ** (sessions_per_year / n) - 1,
        "annualized_volatility": vol * np.sqrt(sessions_per_year),
        "maximum_drawdown": d["drawdown"].min(),
        "sharpe_rf0": r.mean() / vol * np.sqrt(sessions_per_year) if vol > 1e-12 else np.nan,
        "win_rate_closed_trades": (trades["net_return"] > 0).mean() if len(trades) else np.nan,
        "closed_trades": len(trades),
        "one_way_transactions": int(d["turnover"].sum()),
        "transactions_per_year": d["turnover"].sum() * sessions_per_year / n,
        "session_exposure": d["position_during_session"].mean(),
        "cost_in_initial_capital_units": d["cost_in_initial_capital_units"].sum(),
    }
