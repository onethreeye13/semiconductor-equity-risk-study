# V1 exploratory results

This file is generated from the saved market-data snapshot.
Rules were specified on 2026-10-01 after the historical period had occurred. The historical test is a retrospective chronological comparison, not a prospective or untouched validation sample.

All results below include 10 bps each way, initial entry, and next-open execution. Terminal liquidation: True. Cash and Sharpe reference rates are zero. Annualization uses 252 sessions.

## Common-period comparison

| Asset | Strategy | Period | Return | Annualized return | Volatility | Max drawdown | Sharpe (rf=0) | Closed trades | Trade win rate | Exposure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | SMA_20_50 | 2025-06-02 to 2026-09-30 | 458.7% | 264.8% | 67.5% | -38.9% | 2.25 | 4 | 100.0% | 81.2% |
| MU | Momentum_20 | 2025-06-02 to 2026-09-30 | 803.0% | 423.5% | 63.9% | -21.2% | 2.91 | 18 | 77.8% | 76.7% |
| MU | Buy_Hold | 2025-06-02 to 2026-09-30 | 1026.8% | 518.3% | 73.6% | -39.1% | 2.85 | 1 | 100.0% | 100.0% |
| SNDK | SMA_20_50 | 2025-06-02 to 2026-09-30 | 3236.7% | 1299.3% | 97.7% | -45.7% | 3.19 | 3 | 66.7% | 86.0% |
| SNDK | Momentum_20 | 2025-06-02 to 2026-09-30 | 3010.1% | 1227.2% | 92.8% | -37.9% | 3.25 | 11 | 63.6% | 79.1% |
| SNDK | Buy_Hold | 2025-06-02 to 2026-09-30 | 4525.5% | 1688.9% | 105.5% | -56.5% | 3.26 | 1 | 100.0% | 100.0% |
| SOXX | Buy_Hold | 2025-06-02 to 2026-09-30 | 179.7% | 116.8% | 41.5% | -29.0% | 2.08 | 1 | 100.0% | 100.0% |

## Long-history comparison

| Asset | Strategy | Period | Return | Annualized return | Volatility | Max drawdown | Sharpe (rf=0) | Closed trades | Trade win rate | Exposure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | SMA_20_50 | 2020-01-02 to 2026-09-30 | 1106.3% | 44.8% | 41.5% | -42.3% | 1.10 | 17 | 58.8% | 60.4% |
| MU | Momentum_20 | 2020-01-02 to 2026-09-30 | 597.5% | 33.5% | 42.1% | -58.8% | 0.90 | 93 | 40.9% | 59.2% |
| MU | Buy_Hold | 2020-01-02 to 2026-09-30 | 1887.4% | 56.0% | 55.3% | -57.6% | 1.08 | 1 | 100.0% | 100.0% |
| SOXX | Buy_Hold | 2020-01-02 to 2026-09-30 | 607.3% | 33.8% | 39.0% | -45.8% | 0.94 | 1 | 100.0% | 100.0% |

## Retrospective historical test

| Asset | Strategy | Period | Return | Annualized return | Volatility | Max drawdown | Sharpe (rf=0) | Closed trades | Trade win rate | Exposure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | SMA_20_50 | 2024-01-02 to 2026-09-30 | 691.7% | 113.1% | 53.0% | -38.9% | 1.69 | 8 | 87.5% | 67.8% |
| MU | Momentum_20 | 2024-01-02 to 2026-09-30 | 531.8% | 96.2% | 52.3% | -58.8% | 1.55 | 39 | 51.3% | 64.4% |
| MU | Buy_Hold | 2024-01-02 to 2026-09-30 | 1175.8% | 153.8% | 66.5% | -57.6% | 1.73 | 1 | 100.0% | 100.0% |
| SOXX | Buy_Hold | 2024-01-02 to 2026-09-30 | 205.2% | 50.4% | 40.8% | -41.4% | 1.20 | 1 | 100.0% | 100.0% |

## Interpretation boundaries

- Compare strategies only within the same window. A higher return alone is not risk-adjusted outperformance.
- Few round trips make win rate highly uncertain; buy-and-hold has just one terminally closed trade.
- Lower volatility or drawdown may reflect less time invested. Check exposure alongside returns.
- The SNDK sample is short and starts after its 2025 separation. It cannot support strong cross-cycle claims.
- These are selected current securities, not a point-in-time constituent universe; selection and survivorship bias remain.
- Data are a vendor snapshot with adjusted-price execution approximations, not exchange-quality execution records.
- Inspect cost_sensitivity.csv, the trade ledger, and research_design.md before drawing conclusions.
- No ML model, statistical significance test, or prospective performance claim is included in V1.
