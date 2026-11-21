# Data and metric definitions

## Data columns

| Column | Meaning |
|---|---|
| date | New York exchange session date, timezone removed after conversion |
| ticker | Fixed instrument identifier within this study; SNDK restricted by listing floor |
| open / high / low / close | Vendor-reported daily OHLC values, USD |
| adj_close | Vendor-adjusted closing series |
| volume | Vendor-reported share volume; historical adjustment conventions are vendor-dependent |
| dividends / stock_splits | Corporate-action values returned by the provider, retained for audit |
| adjustment_factor | adj_close / close |
| adj_open | open × adjustment_factor; adjusted execution proxy |

The SQLite primary key is `(ticker, date)`. SQL queries compute returns using `LAG`, calculate trailing averages with window functions, and join stocks to SOXX by date.

## Features

Daily return = P[t]/P[t−1]−1; cumulative asset return = P[t]/P[first]−1; momentum uses P[t]/P[t−20]−1. Rolling annualized volatility is the sample standard deviation of 20 daily returns × sqrt(252). SMA20 and SMA50 require complete windows. RSI14 uses Wilder's mean seed and smoothing coefficient 1/14; all-up, all-down and flat paths are explicitly handled. Volume change is V[t]/V[t−1]−1, with division by zero left unavailable. These features are descriptive, not proof of predictability.

## Backtest metrics

Let E[0]=1 denote initial capital, E[n] ending wealth, and r[t] the net close-valued daily account return. The first session starts at the open in cash and contains no pre-window overnight return.

| Metric | Definition / interpretation |
|---|---|
| Cumulative return | E[n] / E[0] − 1, after all modeled costs |
| Annualized return | (E[n] / E[0])^(252/n) − 1; session-based annualization, not a calendar-year CAGR |
| Annualized volatility | std(r, ddof=1) × sqrt(252) |
| Maximum drawdown | min(E[t] / max(E[0],…,E[t]) − 1); negative sign retained |
| Sharpe (rf=0) | mean(r) / std(r, ddof=1) × sqrt(252); undefined for zero variance |
| Win rate | Fraction of completed round-trip trades with net return > 0; terminally closed trades included |
| One-way transactions | Number of entries + exits, including initial and forced terminal trades |
| Exposure | Fraction of sessions held from open to close; not an hour-weighted market-exposure estimate |
| Total cost | Nominal sum of fees expressed in units of initial wealth; not the entire compounded performance drag |

Trade win rate is not the fraction of winning days and is not classifier accuracy. No trades gives an undefined win rate. A buy-and-hold baseline has one closed trade, so its 0% or 100% win rate contains little statistical information.

Adjusted-price factors approximate dividend reinvestment. They do not reproduce precise dividend entitlement, ex-dividend timing or cash reinvestment at the actual open. For a higher-fidelity extension, use a split-adjusted share ledger, explicit cash dividends and a separately tested execution model.
