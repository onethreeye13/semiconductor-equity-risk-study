# ML001: fixed first machine-learning experiment

Protocol written on 2026-10-08 UTC, before fitting these models or inspecting their
outputs. This is a retrospective extension: the underlying 2024–2026 history was
already examined in V1. A protocol file and its hash document this experiment;
they are not external preregistration or evidence of a pristine future holdout.

## Scope and question

Can a small logistic regression or random forest extract useful next-session
directional information from past MU prices and volume, beyond a training-majority
baseline and the existing SMA/momentum rules? Does any classification improvement
survive the same costs in a long/cash account?

MU is the only training asset. Its longer history permits chronological blocks;
SNDK's short post-separation sample is deferred. SOXX is a sector buy-and-hold
reference, not a feature or a claim of factor-adjusted alpha. The original V1 and
SNDQ/SNXX studies remain separate and unchanged.

## One target, one execution rule

For a feature date t, use information available by that day's close. Define:

`forward_return = adj_open[t+2] / adj_open[t+1] - 1`

The label is 1 only when this gross return is strictly positive; zero belongs to
class 0. Retain `feature_date`, `entry_date` (t+1), and `label_end_date` (t+2).
The last two feature dates have no complete target and are excluded, not labeled 0.
Labels are outcomes, never model inputs.

A probability strictly above 0.5 means long at the next open; otherwise cash.
Consecutive long predictions keep one position. Labels span adjacent one-session
open-to-open intervals, so they do not overlap in time. Sharing a boundary price
does not make observations statistically independent. N>1 is deliberately unsupported.
No short positions, leverage, stop losses, probability-based sizing, or threshold search.

Every account starts at 1 in cash, enters no earlier than the first entry open,
and liquidates at the final label-end open. Entry divides capital by (1+c), exit
multiplies it by (1-c), with c=10 bps per side. Sensitivity uses 0/10/25 bps.
Unchanged holdings incur no new fee. Cash return and Sharpe reference rate are 0.
Mark-to-market returns and drawdowns are sampled at opens; intraday extremes are
not measured. This differs from V1's close-marked series and final-close exit:
re-run all comparators on the ML dates and convention rather than copy V1 metrics.

Adjusted opens use the frozen vendor adjusted-close factor, as in V1. These are
total-return approximations, not a share/cash corporate-action ledger or historically
point-in-time vendor data. Costs simplify execution; taxes and capacity are omitted.

## Fixed features and estimators

Nine features: 1-, 2-, and 5-session adjusted close returns; 20-session momentum;
Wilder RSI14; 20-session sample return volatility annualized by sqrt(252);
adjusted-close distance to SMA20 and SMA50; one-session volume percentage change.
No absolute price, date, target, future return, or future open enters X. Rolling
features use past/present observations including pre-2020 warmup. Do not backfill.
Unexpected missing/nonfinite evaluation features stop the run; do not silently
drop internal dates or impute using later data. Vendor volume is not split-adjusted
by this feature code, so corporate actions can distort volume changes.

- **Prior:** constant training positive fraction as probability, majority label
  at the same 0.5 threshold. No test-set majority fitting.
- **LogisticRegression:** StandardScaler followed by L2 logistic regression;
  C=1, lbfgs, max_iter=2000, random_state=42. Fit scaling inside each training fit.
- **RandomForest:** 300 trees, max_depth=3, min_samples_leaf=25,
  max_features=sqrt, bootstrap=True, random_state=42, n_jobs=1. No scaling needed.

No feature selection, oversampling, class weighting, calibration fitting, model
selection or hyperparameter tuning. All model parameters and thresholds stay fixed.
Keep both models' results even if either loses to the baseline.

## Chronological blocks and label purging

Dates below refer to feature dates. Actual retained bounds and removed rows are
reported in `reports/ml/split_audit.csv` and `purged_rows.csv`.

| Role | Feature-date range | Label requirement |
|---|---|---|
| Initial training | 2020-01-02 to before 2023-01-03 | label end < 2023-01-03 |
| Validation diagnostic | 2023-01-03 to before 2024-01-02 | label end < 2024-01-02 |
| Final training refit | 2020-01-02 to before 2024-01-02 | label end < 2024-01-02 |
| Retrospective test | 2024-01-02 to before 2026-10-01 | complete label end < 2026-10-01 |

The validation report is a diagnostic, not a search. Refit the identical estimators
on all eligible pre-test labels, including formerly training/validation-boundary
rows that are now known. There is no fitting or updating during the final test.
Assert each fit's last label-end date precedes its evaluation's first feature date.
Purge by actual dates rather than blindly assuming a generic gap. The multi-year
fixed test model may become stale; retraining schedules are a future experiment.

## Evaluation and interpretation

Classification: counts, class balance, predicted-positive fraction, accuracy,
balanced accuracy, confusion counts; ROC-AUC, Brier score and log loss for genuine
probabilities. Rules produce hard labels, not calibrated probabilities. A constant
majority can exceed 50% accuracy without learning conditional predictability.
Brier/log loss assess probabilities without fitting a calibration layer.

Trading: cumulative/annualized return, annualized volatility, open-sampled maximum
drawdown, Sharpe (rf=0), exposure, turnover, closed-trade win rate, costs, full ledger.
Comparators are training-majority, SMA20/50, momentum20, MU buy-and-hold, and SOXX
buy-and-hold on identical execution intervals. Classification accuracy is not trade
win rate; merged holdings mean one trade may span many daily predictions.

Report validation and test separately, plus descriptive test-year slices. Annual
classification and account returns use the same fixed model; year returns are slices
of the continuous account, not reset/retrained strategies. Partial years are labeled.
No claim of statistical significance or stability comes from point estimates. Serial
dependence, a selected asset, few market regimes, prior historical exposure, provider
revisions, model staleness and friction uncertainty limit conclusions.

Record ML001, its protocol/config/source/data hashes, package versions, output hashes
and actual fit dates. Re-running the same code/data is replication, not a new held-out
experiment. Any result-driven model/threshold/feature change becomes a new documented
development experiment; the present test cannot validate that change independently.

## Implementation references

- [scikit-learn: common pitfalls and leakage](https://scikit-learn.org/1.8/common_pitfalls.html)
- [LogisticRegression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [RandomForestClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
