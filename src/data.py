"""Download, validate, snapshot and query daily market observations."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import platform
import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(root: Path) -> dict:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def normalize_and_validate(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    x = raw.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if len(x.columns.get_level_values(-1).unique()) != 1:
            raise ValueError("Expected one ticker per download")
        x.columns = x.columns.get_level_values(0)
    x.columns = [str(c).lower().replace(" ", "_") for c in x.columns]
    required = PRICE_COLUMNS + ["volume"]
    missing = set(required) - set(x.columns)
    if missing or x.empty:
        raise ValueError(f"{ticker}: empty response or missing columns {sorted(missing)}")
    idx = pd.DatetimeIndex(x.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    x.index = idx.normalize()
    x.index.name = "date"
    if not x.index.is_unique or not x.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: duplicate or unordered dates")
    x[required] = x[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x[required].to_numpy()).all():
        raise ValueError(f"{ticker}: missing/non-finite observations; no automatic filling")
    if (x[PRICE_COLUMNS] <= 0).any().any() or (x["volume"] < 0).any():
        raise ValueError(f"{ticker}: invalid prices or volume")
    eps = 1e-6 * x["high"]
    if ((x["high"] + eps < x[["open", "close", "low"]].max(axis=1)) |
        (x["low"] - eps > x[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{ticker}: inconsistent OHLC range")
    for column in ["dividends", "stock_splits"]:
        if column not in x:
            x[column] = 0.0
    x["adjustment_factor"] = x["adj_close"] / x["close"]
    x["adj_open"] = x["open"] * x["adjustment_factor"]
    x["ticker"] = ticker
    columns = ["ticker"] + required + ["dividends", "stock_splits", "adjustment_factor", "adj_open"]
    return x[columns]


def data_request(config: dict) -> dict:
    return {k: config[k] for k in ["tickers", "download_start", "download_end_exclusive", "listing_floors"]}


def parse_chart_response(payload: bytes, ticker: str) -> pd.DataFrame:
    """Parse the public Yahoo chart response without discarding missing bars."""
    envelope = json.loads(payload)["chart"]
    if envelope.get("error") or not envelope.get("result"):
        raise ValueError(f"{ticker}: provider error: {envelope.get('error')}")
    result = envelope["result"][0]
    meta = result["meta"]
    if meta.get("symbol") != ticker or meta.get("currency") != "USD":
        raise ValueError(f"{ticker}: unexpected identity or currency")
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize()
    quote = result["indicators"]["quote"][0]
    raw = pd.DataFrame({name.title(): quote[name] for name in ["open", "high", "low", "close", "volume"]}, index=idx)
    adjusted = result["indicators"].get("adjclose")
    if not adjusted:
        raise ValueError(f"{ticker}: vendor did not supply adjusted close")
    raw["Adj Close"] = adjusted[0]["adjclose"]
    raw["Dividends"], raw["Stock Splits"] = 0.0, 0.0
    for kind, column in [("dividends", "Dividends"), ("splits", "Stock Splits")]:
        for event in result.get("events", {}).get(kind, {}).values():
            date = pd.Timestamp(event["date"], unit="s", tz="UTC").tz_convert("America/New_York").normalize()
            if date in raw.index:
                raw.loc[date, column] = event["amount"] if kind == "dividends" else event["numerator"] / event["denominator"]
    return raw


def fetch_chart_response(ticker: str, start: str, end_exclusive: str) -> bytes:
    """One public HTTPS request per ticker, no credentials or automatic retry.

    This endpoint is not a contracted/stable market-data service. Fail visibly
    if it becomes unavailable. Preserve the original JSON for provenance.
    """
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
              "period2": int(pd.Timestamp(end_exclusive, tz="UTC").timestamp()),
              "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "semiconductor-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("Market-data service rate limited the request. Stop and try later; no retry or synthetic fallback was used.") from exc
        raise RuntimeError(f"Market-data HTTP error {exc.code} for {ticker}") from exc


def download_data(root: Path, config: dict, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Use an integrity-checked local snapshot unless refresh is explicit.

    Never replace a failed download with synthetic prices or a different ticker.
    """
    for folder in ["data/raw", "data/processed", "reports", "figures"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = root / "reports/manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["request"] != data_request(config):
            raise ValueError("Data request changed. Use --refresh for a deliberate new snapshot.")
        for name, digest in manifest["sha256"].items():
            path = root / name
            if not path.exists() or sha256(path) != digest:
                raise ValueError(f"Snapshot integrity mismatch: {name}. Restore it or use --refresh.")
        return {ticker: pd.read_csv(root / f"data/processed/{ticker}.csv", parse_dates=["date"], index_col="date")
                for ticker in config["tickers"]}

    downloaded, hashes = {}, {}
    for ticker in config["tickers"]:
        start = max(config["download_start"], config["listing_floors"].get(ticker, config["download_start"]))
        payload = fetch_chart_response(ticker, start, config["download_end_exclusive"])
        raw = parse_chart_response(payload, ticker)
        x = normalize_and_validate(raw, ticker)
        if (x.index < pd.Timestamp(start)).any() or (x.index >= pd.Timestamp(config["download_end_exclusive"])).any():
            raise ValueError(f"{ticker}: response outside requested date bounds")
        # Stage in memory: validate every symbol before writing the snapshot.
        downloaded[ticker] = (raw, x, payload)
    data = {ticker: pair[1] for ticker, pair in downloaded.items()}
    quality = audit_coverage(data, config["benchmark"])
    for ticker, (raw, x, payload) in downloaded.items():
        raw_path, processed_path = root / f"data/raw/{ticker}.csv", root / f"data/processed/{ticker}.csv"
        json_path = root / f"data/raw/{ticker}.json"
        json_path.write_bytes(payload)
        raw.to_csv(raw_path, index_label="date")
        x.to_csv(processed_path, float_format="%.17g")
        for path in [raw_path, processed_path, json_path]:
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    quality.to_csv(root / "reports/data_quality.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "matplotlib"]}
    versions["python"] = platform.python_version()
    manifest = {"provider": "Yahoo Finance public chart endpoint via Python urllib (unofficial interface)",
                "endpoint": "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "request": data_request(config), "auto_adjust": False, "repair": False,
                "price_basis": "Vendor OHLC and Adj Close; adj_open = open * adj_close / close",
                "packages": versions, "sha256": hashes,
                "config_at_download": config, "config_sha256": sha256(root / "config.json")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Return the same serialized values as every later cached run.
    return {ticker: pd.read_csv(root / f"data/processed/{ticker}.csv", parse_dates=["date"], index_col="date")
            for ticker in config["tickers"]}


def audit_coverage(data: dict[str, pd.DataFrame], benchmark: str) -> pd.DataFrame:
    """Compare coverage to SOXX's observed sessions; report, never impute gaps.

    This catches symbol-specific gaps, not a simultaneous omission by the vendor.
    """
    reference = data[benchmark].index
    rows = []
    for ticker, bars in data.items():
        expected = reference[(reference >= bars.index.min()) & (reference <= bars.index.max())]
        missing = expected.difference(bars.index)
        extra = bars.index.difference(reference)
        rows.append({"ticker": ticker, "first_date": str(bars.index.min().date()),
                     "last_date": str(bars.index.max().date()), "rows": len(bars),
                     "missing_vs_soxx": len(missing), "extra_vs_soxx": len(extra),
                     "zero_volume_rows": int((bars["volume"] == 0).sum()),
                     "absolute_return_over_40pct": int((bars["adj_close"].pct_change(fill_method=None).abs() > .4).sum())})
        if len(missing) or len(extra):
            raise ValueError(f"{ticker}: calendar mismatch vs SOXX; missing={list(missing)}, extra={list(extra)}")
    return pd.DataFrame(rows)


def validate_window(data: dict[str, pd.DataFrame], tickers: list[str], benchmark: str,
                    start: str, end_exclusive: str) -> None:
    ref = data[benchmark].loc[start:]
    ref = ref.loc[ref.index < pd.Timestamp(end_exclusive)]
    if len(ref) < 2:
        raise ValueError("Evaluation window is too short")
    for ticker in tickers:
        actual = data[ticker].loc[start:]
        actual = actual.loc[actual.index < pd.Timestamp(end_exclusive)]
        if not actual.index.equals(ref.index):
            raise ValueError(f"{ticker}: unequal evaluation calendar; do not compare different periods")


def write_sqlite(data: dict[str, pd.DataFrame], path: Path) -> None:
    """One row per ticker/date; repeated runs replace this project's table."""
    combined = pd.concat([x.reset_index() for x in data.values()], ignore_index=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    columns = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "adj_open"]
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS daily_bars (
            ticker TEXT NOT NULL, date TEXT NOT NULL,
            open REAL NOT NULL CHECK(open>0), high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL CHECK(close>0),
            adj_close REAL NOT NULL CHECK(adj_close>0), volume REAL NOT NULL CHECK(volume>=0),
            dividends REAL NOT NULL, stock_splits REAL NOT NULL,
            adj_open REAL NOT NULL CHECK(adj_open>0), PRIMARY KEY(ticker,date))""")
        conn.execute("DELETE FROM daily_bars")
        placeholders = ",".join("?" for _ in columns)
        conn.executemany(f"INSERT INTO daily_bars ({','.join(columns)}) VALUES ({placeholders})",
                         combined[columns].itertuples(index=False, name=None))
