"""Versioned ETF snapshots; reuse and verify the original V1 underlying data."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import (audit_coverage, download_data, fetch_chart_response,
                      normalize_and_validate, parse_chart_response, read_config,
                      sha256)


def load_research_data(root: Path, config: dict, refresh: bool = False):
    """Return bars and provenance. Refresh appends a version, never rewrites one.

    The V1 snapshot is verified/reused even when ETF refresh=True. Refreshing V1
    remains a separate, explicit action through src.pipeline.
    """
    baseline = download_data(root, read_config(root))
    end = pd.Timestamp(config["download_end_exclusive"])
    reference = baseline["SOXX"].loc[lambda x: x.index < end]
    if reference.empty or reference.index[-1] < end - pd.Timedelta(days=4):
        raise ValueError("Baseline snapshot does not cover the ETF study endpoint")
    underlying = baseline[config["underlying"]].loc[lambda x: x.index < end]
    sources = {f"data/processed/{t}.csv": sha256(root / f"data/processed/{t}.csv")
               for t in [config["underlying"], "SOXX"]}
    request = {k: config[k] for k in ["etfs", "listing_floors", "download_end_exclusive"]}
    cache = root / "data/raw/sndk_inverse"
    pointer = cache / "current.json"
    if pointer.exists() and not refresh:
        current = json.loads(pointer.read_text())
        version = cache / current["snapshot_id"]
        manifest_path = version / "manifest.json"
        if sha256(manifest_path) != current["manifest_sha256"]:
            raise ValueError("ETF manifest integrity mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest["request"] != request or manifest["baseline_sha256"] != sources:
            raise ValueError("ETF request/baseline changed; use an explicit ETF --refresh")
        for name, digest in manifest["sha256"].items():
            if sha256(version / name) != digest:
                raise ValueError(f"ETF snapshot integrity mismatch: {name}")
    else:
        staged = {}
        for ticker in config["etfs"]:
            start = config["listing_floors"][ticker]
            payload = fetch_chart_response(ticker, start, str(end.date()))
            bars = normalize_and_validate(parse_chart_response(payload, ticker), ticker)
            if (bars.index < pd.Timestamp(start)).any() or (bars.index >= end).any():
                raise ValueError(f"{ticker}: provider returned dates outside request")
            if bars.index[-1] != reference.index[-1]:
                raise ValueError(f"{ticker}: missing final study session")
            staged[ticker] = (payload, bars)
        audit_coverage({"SOXX": reference, config["underlying"]: underlying,
                        **{t: pair[1] for t, pair in staged.items()}}, "SOXX")
        content_id = hashlib.sha256(b"".join(pair[0] for pair in staged.values())).hexdigest()[:12]
        now = datetime.now(timezone.utc)
        snapshot_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + content_id
        version = cache / snapshot_id
        version.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for ticker, (payload, bars) in staged.items():
            (version / f"{ticker}.json").write_bytes(payload)
            bars.to_csv(version / f"{ticker}.csv", float_format="%.17g")
            for suffix in ["json", "csv"]:
                name = f"{ticker}.{suffix}"
                hashes[name] = sha256(version / name)
        manifest = {"snapshot_id": snapshot_id, "downloaded_at_utc": now.isoformat(),
                    "provider": "Yahoo Finance public chart endpoint (unofficial)",
                    "request": request, "baseline_sha256": sources, "sha256": hashes}
        (version / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        current = {"snapshot_id": snapshot_id,
                   "manifest_sha256": sha256(version / "manifest.json")}
        temporary_pointer = cache / "current.tmp"
        temporary_pointer.write_text(json.dumps(current, indent=2) + "\n")
        temporary_pointer.replace(pointer)
    data = {config["underlying"]: underlying}
    for ticker in config["etfs"]:
        data[ticker] = pd.read_csv(version / f"{ticker}.csv", index_col="date", parse_dates=["date"])
    audit_coverage({"SOXX": reference, **data}, "SOXX")
    return data, {**manifest, "manifest_sha256": current["manifest_sha256"]}
