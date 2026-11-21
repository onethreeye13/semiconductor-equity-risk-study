# ML001 verification record

Verified on 2026-10-08 UTC using the existing 2026-10-02 V1 snapshot. No new market
download, model/threshold search, or change to original V1 / inverse-ETF results.

- `python -m pytest -q`: **33 passed**, including 8 new ML integrity tests.
  These check feature prefix causality, next-two-open labels, missing future labels,
  purging by label-end dates, training-only scaler fitting, evaluation-label isolation,
  pre-entry/post-exit exclusion, merged positions, exact fees, ledger reconciliation,
  and rejection of gapped or misaligned execution intervals.
- `python -m src.ml_pipeline`: completed on real cached market data. All source
  snapshot hashes were checked by the existing V1 loader.
- `nbformat.validate` and Python parsing passed for Notebook 06. All **7 code cells**
  were executed in order in one Python process, starting from `notebooks/` to check
  root discovery, imports, model fitting, display calls and report generation.
- A separate Jupyter kernel could not start: this environment rejected socket
  binding for both TCP and IPC (`Operation not permitted`). Therefore no successful
  `nbclient` kernel execution or interactive Jupyter UI check is claimed.
- The notebook's repeat pipeline run reproduced **15 output hashes exactly**:
  11 CSV tables, findings, fit diagnostics, and 2 SVG figures. The provenance run
  timestamp changes intentionally. Source/data/output hashes are in `provenance.json`.
- Both SVG figures were rendered to PNG and visually inspected for clipping,
  labels, legend, scales and dates. Review PNGs are temporary; SVGs are the exports.

The retained initial training / validation / refit / test sizes are respectively
**754 / 248 / 1004 / 687**. Each boundary excludes two candidate feature rows;
the final two test candidates lack complete future labels. Both fitting label ends
strictly precede the next evaluation feature date. The final refit deliberately
reuses eligible pre-test observations, not test labels.

Training-majority always predicts up in this snapshot. Its long/cash account
therefore equals MU buy-and-hold under the shared execution/cost convention.
Test trades and daily account returns reconcile multiplicatively. Cost sensitivity
uses the same predictions and trade dates at 0/10/25 bps per side.

These checks establish implementation consistency for this experiment; they do
not establish statistical significance, economic causality, prospective predictive
skill, provider correctness, or that the owner has independently reproduced it yet.
