# ML001: retrospective MU machine-learning results

Fixed logistic regression and a shallow random forest; no parameter or threshold search.
The historical prices had already been reviewed in V1. These are chronologically separated
model-fit evaluations, **not** prospective or pristine unseen-market evidence.

Target: positive adjusted-open return from t+1 to t+2, using features known at close t.
Both models are retained regardless of results. Training-majority is fitted on training labels only.

## Validation

Feature dates: 2023-01-03 to 2023-12-27; execution: 2023-01-04 open to 2023-12-29 open; 248 intervals.

| Model/rule | Accuracy | Balanced accuracy | ROC-AUC | Brier ↓ | Log loss ↓ | Predicted up |
|---|---:|---:|---:|---:|---:|---:|
| Prior | 52.42% | 50.00% | 0.500 | 0.249 | 0.692 | 100.0% |
| LogisticRegression | 56.85% | 55.25% | 0.493 | 0.250 | 0.694 | 83.5% |
| RandomForest | 50.00% | 48.87% | 0.472 | 0.255 | 0.703 | 73.4% |
| SMA_20_50 | 43.55% | 42.32% | n/a | n/a | n/a | 75.0% |
| Momentum_20 | 45.97% | 45.22% | n/a | n/a | n/a | 65.3% |

Observed positive fraction: 52.42%. Rule outputs are hard labels, not probability estimates.

All accounts include 10 bps per side and final-open liquidation.

| Asset | Account | Net return | Annualized return | Volatility | Open-marked MDD | Sharpe rf=0 | Exposure | Trades | Trade win rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | Prior | 62.6% | 63.9% | 37.9% | -18.1% | 1.49 | 100.0% | 1 | 100.0% |
| MU | LogisticRegression | 87.2% | 89.1% | 33.6% | -14.0% | 2.06 | 83.5% | 33 | 69.7% |
| MU | RandomForest | 46.5% | 47.4% | 32.5% | -15.0% | 1.36 | 73.4% | 24 | 70.8% |
| MU | SMA_20_50 | -6.1% | -6.2% | 31.5% | -24.6% | -0.05 | 75.0% | 4 | 25.0% |
| MU | Momentum_20 | 19.5% | 19.9% | 31.9% | -19.4% | 0.73 | 65.3% | 10 | 60.0% |
| MU | Buy_Hold | 62.6% | 63.9% | 37.9% | -18.1% | 1.49 | 100.0% | 1 | 100.0% |
| SOXX | Buy_Hold | 66.5% | 67.9% | 28.6% | -17.7% | 1.96 | 100.0% | 1 | 100.0% |

## Test

Feature dates: 2024-01-02 to 2026-09-28; execution: 2024-01-03 open to 2026-09-30 open; 687 intervals.

| Model/rule | Accuracy | Balanced accuracy | ROC-AUC | Brier ↓ | Log loss ↓ | Predicted up |
|---|---:|---:|---:|---:|---:|---:|
| Prior | 53.86% | 50.00% | 0.500 | 0.249 | 0.690 | 100.0% |
| LogisticRegression | 50.22% | 50.21% | 0.520 | 0.252 | 0.697 | 50.1% |
| RandomForest | 51.67% | 50.93% | 0.513 | 0.250 | 0.693 | 59.7% |
| SMA_20_50 | 52.69% | 51.34% | n/a | n/a | n/a | 67.7% |
| Momentum_20 | 51.09% | 49.99% | n/a | n/a | n/a | 64.3% |

Observed positive fraction: 53.86%. Rule outputs are hard labels, not probability estimates.

All accounts include 10 bps per side and final-open liquidation.

| Asset | Account | Net return | Annualized return | Volatility | Open-marked MDD | Sharpe rf=0 | Exposure | Trades | Trade win rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | Prior | 1234.2% | 158.7% | 67.7% | -59.1% | 1.74 | 100.0% | 1 | 100.0% |
| MU | LogisticRegression | 190.6% | 47.9% | 46.9% | -48.6% | 1.07 | 50.1% | 95 | 62.1% |
| MU | RandomForest | 369.4% | 76.3% | 52.8% | -36.4% | 1.34 | 59.7% | 89 | 65.2% |
| MU | SMA_20_50 | 728.0% | 117.1% | 53.2% | -41.0% | 1.72 | 67.7% | 8 | 87.5% |
| MU | Momentum_20 | 560.7% | 99.9% | 53.2% | -59.3% | 1.57 | 64.3% | 39 | 53.8% |
| MU | Buy_Hold | 1234.2% | 158.7% | 67.7% | -59.1% | 1.74 | 100.0% | 1 | 100.0% |
| SOXX | Buy_Hold | 216.9% | 52.7% | 39.9% | -42.8% | 1.26 | 100.0% | 1 | 100.0% |

## What this experiment shows

- LogisticRegression: test accuracy is -3.64 percentage points relative to Prior; net return is 190.6%, versus 1234.2% for MU buy-and-hold. Open-marked maximum drawdown is -48.6% and exposure is 50.1%.
- RandomForest: test accuracy is -2.18 percentage points relative to Prior; net return is 369.4%, versus 1234.2% for MU buy-and-hold. Open-marked maximum drawdown is -36.4% and exposure is 59.7%.

## Test cost sensitivity

| Account | Cost per side | Net return | One-way transactions |
|---|---:|---:|---:|
| LogisticRegression | 0 bps | 251.4% | 190 |
| LogisticRegression | 10 bps | 190.6% | 190 |
| LogisticRegression | 25 bps | 118.5% | 190 |
| RandomForest | 0 bps | 460.9% | 178 |
| RandomForest | 10 bps | 369.4% | 178 |
| RandomForest | 25 bps | 259.4% | 178 |

## Limits and reading guide

- Accuracy and trade win rate measure different things. Daily long predictions can merge into one trade.
- A high majority accuracy can reflect an imbalanced rising sample. Balanced accuracy, AUC and probability scores provide additional checks.
- Lower drawdown may come from lower exposure. Neither a single accuracy gain nor a return point estimate establishes durable predictive/economic value.
- Open-sampled drawdown excludes intraday extremes. V1 close-marked metrics are not directly interchangeable with this table.
- Inspect yearly_classification.csv and yearly_account_returns.csv for regime variation. They are descriptive slices of one fixed model, not independent replications; 2026 is partial.
- No significance test or independence assumption is used to claim an edge. This is one selected asset and few market regimes.
- Both final estimators were fitted only on eligible pre-2024 labels and then frozen; staleness can hurt performance.
- Vendor-adjusted data are a current snapshot, not point-in-time corporate-action records. Volume changes can reflect corporate actions.
- This result does not transfer automatically to SNDK, SNDQ/SNXX, future periods, or personal account P&L.
- Do not tune on this test and call it independent again. See docs/ml_protocol.md, split/fit audits, predictions and trade ledger.
