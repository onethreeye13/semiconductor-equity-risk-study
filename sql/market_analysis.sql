-- Run against data/market.sqlite. Dates use ISO format so string sorting is chronological.

-- 1. Data coverage. SNDK must not include the old, pre-acquisition listing.
SELECT ticker, MIN(date) AS first_date, MAX(date) AS last_date,
       COUNT(*) AS observations, AVG(volume) AS average_volume
FROM daily_bars
GROUP BY ticker
ORDER BY ticker;

-- 2. Compute lagged returns BEFORE filtering to the research window.
WITH lagged AS (
    SELECT ticker, date, adj_close,
           LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS previous_close
    FROM daily_bars
)
SELECT ticker, date, adj_close / previous_close - 1.0 AS daily_return
FROM lagged
WHERE date >= '2025-06-02' AND date < '2026-10-01'
ORDER BY ticker, date;

-- 3. A true 20-observation SMA: do not emit a partial warmup value.
WITH rolling AS (
    SELECT ticker, date, adj_close,
           COUNT(*) OVER w AS window_n,
           AVG(adj_close) OVER w AS sma20
    FROM daily_bars
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT ticker, date, CASE WHEN window_n = 20 THEN sma20 END AS sma20
FROM rolling
ORDER BY ticker, date;

-- 4. Match each asset's daily return to SOXX on the same session.
WITH lagged AS (
    SELECT ticker, date, adj_close,
           LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS previous_close
    FROM daily_bars
), returns AS (
    SELECT ticker, date, adj_close / previous_close - 1.0 AS r FROM lagged
)
SELECT a.ticker, a.date, a.r AS asset_return, b.r AS soxx_return,
       a.r - b.r AS daily_return_difference
FROM returns a
JOIN returns b ON a.date = b.date AND b.ticker = 'SOXX'
WHERE a.ticker IN ('MU','SNDK') AND a.date >= '2025-06-02' AND a.date < '2026-10-01'
ORDER BY a.ticker, a.date;
