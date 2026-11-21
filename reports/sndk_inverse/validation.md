# Extension validation — 2026-10-02

- `python -m pytest -q`: **25 passed** (14 existing V1 cases and 11 extension cases).
- Notebook 05 validates against the nbformat schema; all **7 code cells** ran in
  order in a fresh in-process Python namespace, with cached real data. This is
  not a claim of testing the Jupyter browser interface or kernel startup.
- The full extension pipeline completed: **156** base-cost accounts and **624**
  account/cost rows (36 candidates + 3 unconditional horizons, two ETFs, two
  segments; four cost levels for sensitivity). All rules, including no-trade
  rules, are retained.
- Every base-cost account reconciled to the compounded net trade ledger. Signal
  dates precede entries; trades do not overlap; full holding periods stay within
  their segment. SNDQ/SNXX entry and exit dates match for every paired account.
- All original V1 input file hashes still match the original private manifest.
  ETF snapshot reuse was verified. Automated fixtures verify that modified bytes
  are rejected and an explicit refresh preserves the preceding ETF version.
- Actual known split events match primary SEC records: SNDQ 1-for-10 on
  2026-07-21; SNXX 8-for-1 on 2026-06-03. Prices are already adjusted by the
  vendor; no second split adjustment is applied. Daily target tracking gaps are
  diagnostic fields in `data_audit.csv`, not corrected away.
- The three figures were visually inspected for labels, clipping and consistency
  with reported tables. Committed notebook outputs are cleared; its markdown
  records this run's key findings and points to the generated tables.

This verifies implementation and internal consistency, not vendor accuracy,
statistical significance, achievable fills or future performance. Price history
is not a point-in-time revision database, and the historical validation period
was already observable when the research question was posed.
