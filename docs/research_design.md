# Research design, V1.0

Protocol recorded 2026-10-01 (America/Toronto), before this project's first strategy-output review. Historical prices and market narratives were already observable, so this is not a claim of independent preregistration or a genuinely untouched test.

## Questions and fixed decisions

Compare two simple long/cash rules to same-asset buy-and-hold and SOXX. Evaluate return, drawdown, volatility, exposure and transaction intensity together. No optimization, parameter grid or return-maximizing asset selection is implemented.

Universe: MU and SNDK, with SOXX as the investable sector reference. SOXX is an ETF rather than a pure index series; its fund costs and tracking differences are embedded in its prices. The stock-specific baseline isolates timing effects, while SOXX supplies sector context. Neither comparison is a formal factor-adjusted alpha estimate.

SNDK is limited to its current listing from 2025-02-24. Data download ends exclusively at 2026-10-01, keeping all of September but excluding a changing partial next month. Long history begins evaluation on 2020-01-02 with 2019 warmup; common evaluation begins 2025-06-02 after SNDK's indicator warmup. Windows never silently shrink to an intersection.

## Data

Use one Yahoo Finance public chart response per symbol via Python's standard HTTPS library. Retain the original JSON, parsed vendor OHLCV / adjusted close / corporate actions, normalized CSV, download timestamp, dependency versions and SHA-256 hashes. This is an unofficial, best-effort interface; stop on missing fields, non-finite prices, identity/currency mismatch, duplicate dates, OHLC inconsistencies or invalid values.

An initial yfinance download encountered a rate limit in its authentication workflow. The public chart endpoint was separately observed to return valid complete data. V1 therefore uses the explicit chart client, not a silent fallback inside yfinance. It does not retry rate-limited requests, use proxies, or replace failures with generated prices.

Calendar coverage is checked against observed SOXX dates and exact evaluation-window equality is required. This detects symbol-specific omissions; it does not independently certify the full exchange calendar if the provider omits a session from every symbol. Price jumps above 40% and zero-volume rows are flagged for investigation rather than automatically deleted. No forward-fill/backfill creates artificial price returns.

OHLC values are the vendor's reported price basis, not a claim of original unadjusted exchange ticks. Set adjusted open = open × adjusted close / close. Signals, returns and comparisons use a consistent adjustment basis. Split/dividend adjustment revisions and lack of a point-in-time corporate-action archive remain limitations.

## Signals and execution

At close t, SMA-long = 1 when SMA20 > SMA50, otherwise 0. Momentum-long = 1 when adjusted close[t] / adjusted close[t−20] − 1 > 0, otherwise 0. Equality produces cash. Insufficient lookback is NaN and an evaluation window containing unavailable lagged signals is rejected.

Signals use no future rows; rolling windows are trailing and never centered. A shift of one session maps close-time information to next-open decisions. Each evaluation begins in cash with wealth 1, and both strategies and baselines pay initial entry and terminal exit costs.

For each session: (1) old holdings earn the previous-close-to-current-open move; (2) execute the previously known target at the open, paying cost; (3) new holdings earn open-to-close; (4) optionally liquidate at the final close, paying exit cost. This avoids falsely earning an entry-day overnight gap or missing an exit-day overnight loss.

Positions are binary, so entry and exit cost factors are exact for this simplified model: entry factor 1/(1+c), exit factor (1−c), where c = bps / 10000. This treats cost as a proportional notional charge, including a stylized spread/slippage allowance. Ten bps per side is a scenario assumption, not an empirically calibrated execution estimate. Test 0 and 25 bps with identical signals.

## Evaluation

- Long and common windows are separate comparisons, each with common dates and capital conventions.
- A development block (2020–2023) and chronological historical-test block (2024–2026-09) are also reported for MU and SOXX. Signals carry prior lookback information into each block, but each block starts with cash and a fresh cost ledger.
- The historical test is retrospective; asset/parameter choices were informed by an already observed world. Genuine forward testing requires locking a version and evaluating data arriving after that lock, without retuning on it.
- SNDK's common sample overlaps the long-window test and is not an additional independent test sample.
- Annualized numbers use 252 sessions and may exaggerate the apparent stability of short samples. No hypothesis test, confidence interval or statistical-significance claim is made in V1.

## Planned interpretation

Report negative results. If a rule reduces drawdown but sacrifices return, quantify both and check market exposure. If results diverge between MU and SNDK, consider sample length, volatility, market regime and chance; do not infer causality from two stocks. Trading costs should reduce returns for a fixed signal path, and high turnover should receive explicit attention.

## V1 completion versus later work

V1 produces real-data snapshots, notebooks 01–04, SQL, checked backtest calculations, metrics, trade ledgers and six figures. No predictive model is part of this core V1 study. The separate [ML001 protocol](ml_protocol.md) now governs logistic regression and random forest in Notebook 06; its outputs live in `reports/ml/` and do not replace V1 results.
