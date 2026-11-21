# SNDK extensions: range asymmetry and conditional inverse-ETF trades

Protocol recorded on 2026-10-02, before running this extension's outcome tables.
The historical period is already observable. This is a retrospective exploratory
study with a chronological validation segment, **not** an untouched or prospective
out-of-sample claim. The V1 equity study and its snapshot remain separate.

## Questions

1. After a large recent SNDK rise or a high price-relative-to-history reading,
   do next-open SNDQ trades have favorable net outcomes?
2. Is SNDK's daily high–low range larger on down days than on up days?
3. How do SNDQ and SNXX behave for the **same** signal dates, holding periods,
   initial capital and costs? This does not explain an individual's actual P&L.

## Product identity and data

- [Tradr SNDQ](https://www.tradretfs.com/sndq): targets **−200% of SNDK's daily
  performance**, before fees and expenses; inception 2026-04-22.
- [Tradr SNXX](https://www.tradretfs.com/snxx): targets **+200% daily**;
  inception 2026-01-26. Inception and first available trading bar can differ.
- [SEC corporate-action supplement](https://www.sec.gov/Archives/edgar/data/1587982/000121390026074424/ea0296765-05_497.htm):
  SNDQ 1-for-10 reverse split; split-adjusted trading starts 2026-07-21.
- [SEC SNXX supplement](https://www.sec.gov/Archives/edgar/data/1587982/000121390026057930/ea0291169-05_497.htm):
  SNXX 8-for-1 forward split; split-adjusted trading starts 2026-06-03.

Sources accessed 2026-10-02. Use actual Yahoo ETF OHLC/adjusted-close observations
through 2026-09-30. Do not synthesize pre-inception ETF history or multiply a
multi-day SNDK return by ±2. Daily resetting, compounding and tracking differences
make that invalid. Apply the same adjusted-open approximation used in V1;
do not apply corporate-action splits a second time to vendor-adjusted prices.
Fund operating expenses are already reflected in observed ETF prices and are
not deducted again. Costs below represent additional trading friction.

The loader verifies the existing SNDK/SOXX snapshot and ETF session coverage.
The pipeline checks these known split events against the vendor and reports the
largest close-to-close gap from each ETF's daily leverage target as a diagnostic,
not a guarantee of exact tracking. It does not modify prices to force tracking.
Each explicit ETF refresh creates a new version directory with original JSON,
normalized CSV and SHA-256 hashes. A hash identifies bytes, not data correctness;
a later download is not guaranteed to recreate this version. Public reports
record the snapshot ID and hashes, while raw vendor prices remain uncommitted.

## Fixed candidate grid

Each rule uses one feature, not a fitted combination. All rolling windows are
trailing and include the signal day's close.

| Feature | Definition | Entry threshold (greater than or equal) |
|---|---|---|
| Momentum 5 | adjusted close / close 5 sessions earlier − 1 | 5%, 10%, 20% |
| MA distance 20 | adjusted close / trailing SMA(20) − 1 | 5%, 10%, 20% |
| RSI 14 | Wilder RSI, as in V1 | 70, 80, 90 |
| Z-score 20 | (adjusted close − SMA20) / trailing price standard deviation, ddof=1 | 1.5, 2.0, 2.5 |

Z-score here standardizes **price levels**, not returns. It is not a normal-tail
probability or a promise of mean reversion; a trending price can remain extreme.
A zero rolling standard deviation leaves Z-score unavailable.

Twelve rules × holding periods of **1, 3 or 5 sessions** = **36 candidates**.
An always-eligible entry schedule at each horizon is a baseline, not a candidate.
For each rule/account, a qualifying close can initiate a trade when flat; an
ongoing position prevents overlapping trades. A condition that persists may
initiate another trade after the preceding position exits. These are not assumed
to be statistically independent events.

## Execution and chronological selection

- Observe SNDK at close on day t; enter the ETF at the next session's open.
- Hold h sessions including entry day; exit at close of the h-th session.
  Thus h=1 is next-day open-to-close. Entry-day overnight movement is not earned.
- Use identical entry/exit dates for both ETFs. Full capital, fractional shares,
  no borrowing at the account level, no additional leverage, zero cash yield.
- Start both segments in cash. Training entries/exits precede **2026-08-03**;
  validation entries begin on that date. The preceding close may supply its
  signal. Drop a trade if its full horizon would cross a segment's end. Do not
  shorten it using future knowledge. This is a boundary purge.
- Base friction: **10 bps per side**. Show 0 / 10 / 25 / 50 bps sensitivity.
  Net round-trip return = exit adjusted close / entry adjusted open ×
  (1 − c) / (1 + c) − 1. Daily marks include fees and overnight holding exposure.
- Rank eligible SNDQ training candidates by mean net trade return, then by
  trade count, then rule ID and horizon. Require **20 training trades** and at
  least one win and loss. Freeze this choice before scoring validation outcomes.
  A non-positive best training mean yields **no selected candidate**.
- Also report the training leaders for win rate and payoff ratio separately;
  these need not agree. They use the same eligibility filter and are descriptive.
- Report **all 36** training and validation rows, including empty candidates;
  never reselect a winner based on validation. Fewer than **10 validation trades**
  is explicitly marked insufficient. These cutoffs are reporting guardrails,
  not proof of significance. Multiple testing still creates selection bias.

Metrics: completed trades, wins/losses/flat trades, win rate, descriptive Wilson
95% interval, mean win, mean loss magnitude, payoff ratio (mean win / mean loss),
profit factor (sum wins / absolute sum losses), mean net trade return, account
cumulative return, daily-close maximum drawdown and exposure. Undefined ratios
remain blank when there are no wins or no losses; no infinite "best" ratio.
The Wilson interval assumes independent Bernoulli trials, which these trades may
violate; it is an uncertainty illustration, not a corrected inference result.
Daily-close drawdown misses intraday extremes. No annualized short-history alpha
claim is made. 10 bps can understate real ETF spreads and market impact.

## Up/down-day range study

Primary daily range = (adjusted high − adjusted low) / previous adjusted close.
Adjusted high/low use the same-day adj_close / close factor. Up/down classification
uses adjusted close-to-close return, with flat days kept separate. A second
classification uses close versus open because a gap-down day can rally intraday.
Also show median range, absolute close-to-close return, absolute open-to-close
return and absolute overnight gap. Range has no direction and is not achievable
profit; daily bars do not tell us whether the high or low occurred first.

Analyze both the full available post-separation SNDK history (the first day has no
previous close) and the exact common ETF period. A 5-session circular moving-block
bootstrap, 2,000 repetitions with a fixed seed, estimates a descriptive 95%
interval for the down-minus-up **mean range** difference. This preserves short
local dependence approximately, but not regime changes; the samples are short,
selected and retrospective. No multiple-comparison significance claim is made.
The realized day label is explanatory **after the close**, never an entry signal.

## Interpretation boundaries

A larger down-day range does not establish a profitable short entry or explain
personal P&L. The latter requires fills, share counts, holding times, commissions,
spreads and any currency conversion. Here we compare standardized hypothetical
trades only. A future intraday study needs timestamped intraday data and its own
execution protocol; OHLC bars cannot validate an intraday stop or profit target.
Keep failed rules and small samples visible. A prospective paper-trading period
after freezing a rule is a later research step, not a result of this notebook.
