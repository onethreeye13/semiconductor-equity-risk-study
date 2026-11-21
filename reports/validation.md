# V1 validation record

Validated on 2026-10-01 (America/Toronto).

- **14 automated checks passed:** next-open entry timing, overnight exit exposure, exact two-sided cost factors, first-session drawdown, cash initialization, undefined no-trade statistics, causal feature prefixes, RSI edge cases, warmup rejection, invalid data rejection, calendar mismatch rejection and SQLite uniqueness.
- **Real market-data pipeline completed** using the retained Yahoo chart responses: MU 1,947 rows, SNDK 403 rows, SOXX 1,947 rows. Last session: 2026-09-30. No missing or extra symbol-specific dates versus observed SOXX sessions in each overlap, no zero-volume observations, and no absolute daily adjusted return above 40%.
- **Independent arithmetic checks passed:** every buy-and-hold endpoint matches its direct adjusted-open/adjusted-close ratio after entry and exit costs; increasing costs cannot increase terminal wealth for a fixed signal sequence.
- **Four SQL queries executed** against the generated SQLite database.
- **Four notebooks, all 21 code cells, executed in order** using in-process Python and IPython rich output. No code-cell error remained. Notebook structure validated with nbformat.
- **Six research figures inspected visually.** Risk-return label overlap was corrected.

The hosted environment disallowed socket binding, so Jupyter's TCP and IPC kernels could not start. Notebook code and outputs were verified in process; interactive JupyterLab startup was not verified here. See `environment.json` for the exact tested Python/package versions.

These checks validate specific implementation properties. They do not establish market-data perfection, predictive value, statistically significant outperformance or real-world execution feasibility.
