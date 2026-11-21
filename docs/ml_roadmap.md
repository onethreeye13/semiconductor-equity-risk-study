# Machine-learning extension

ML001 is implemented in [Notebook 06](../notebooks/06_machine_learning.ipynb).
The fixed [protocol](ml_protocol.md), [configuration](../config_ml.json), and
[results](../reports/ml/findings.md) are the authoritative experiment records.

The first experiment uses MU's longer history, nine causal price/volume features,
training-only scaling, logistic regression and a shallow random forest. It predicts
one next-open-to-following-open interval, purges labels at chronological boundaries,
and compares against a training-majority baseline, simple rules and buy-and-hold.
Both models are retained; no threshold or hyperparameter search is performed.

The core V1 questions have been discussed; independent code reproduction and owner
interpretation remain in progress. The next step is to reproduce and explain this
fixed experiment, not to keep tuning until test results improve.

Possible later work, requiring a separate protocol/experiment ID:

- A declared rolling/expanding retraining schedule with date-purged training fits.
- Dependence-aware uncertainty estimates and exposure-matched comparisons.
- A clearly defined multi-session holding rule before introducing N>1 labels.
- Additional assets or genuinely new observations after locking the full research process.

The historical 2024–2026 data were already reviewed in V1. The ML split limits
model-fitting leakage but does not erase researcher exposure. Reusing this test
for result-driven changes converts it into development data for those changes.
