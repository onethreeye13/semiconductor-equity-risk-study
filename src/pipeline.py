"""Run with: python -m src.pipeline (from the repository root)."""

import argparse
import json
from pathlib import Path
import pandas as pd
from .data import read_config, download_data, audit_coverage, validate_window, write_sqlite, sha256
from .indicators import add_features, trading_signals
from .backtest import run_backtest
from .metrics import summarize


def run_study(data: dict, config: dict):
    if config["cash_return_annual"] != 0 or config["sharpe_reference_rate_annual"] != 0:
        raise ValueError("V1 implements zero cash and Sharpe reference rates only; extend the model before changing them.")
    features = {ticker: add_features(bars, config) for ticker, bars in data.items()}
    windows = {"long": config["long_window"], "common": config["common_window"]}
    windows["development"] = {**config["long_window"], "end_exclusive": config["historical_test_start"]}
    windows["historical_test"] = {**config["long_window"], "start": config["historical_test_start"]}
    rows, sensitivity, results = [], [], {}
    for window_name, window in windows.items():
        validate_window(data, window["tickers"], config["benchmark"], window["start"], window["end_exclusive"])
        for ticker in window["tickers"]:
            signals = {} if ticker == config["benchmark"] else trading_signals(features[ticker])
            signals["Buy_Hold"] = pd.Series(1.0, index=data[ticker].index)
            for strategy, signal in signals.items():
                for cost in sorted(set(config["cost_sensitivity_bps"] + [config["one_way_cost_bps"]])):
                    result = run_backtest(data[ticker], signal, window["start"], window["end_exclusive"],
                                          cost_bps=cost, liquidate=config["liquidate_at_end"])
                    stats = {"window": window_name, "ticker": ticker, "strategy": strategy,
                             "one_way_cost_bps": cost, **summarize(result, config["annualization_sessions"])}
                    sensitivity.append(stats)
                    if cost == config["one_way_cost_bps"]:
                        rows.append(stats)
                        results[window_name, ticker, strategy] = result
    return pd.DataFrame(rows), pd.DataFrame(sensitivity), results, features


def report_text(summary: pd.DataFrame, config: dict) -> str:
    text = ["# V1 exploratory results", "", "This file is generated from the saved market-data snapshot.",
            "Rules were specified on 2026-10-01 after the historical period had occurred. The historical test is a retrospective chronological comparison, not a prospective or untouched validation sample.", "",
            f"All results below include {config['one_way_cost_bps']} bps each way, initial entry, and next-open execution. Terminal liquidation: {config['liquidate_at_end']}. Cash and Sharpe reference rates are zero. Annualization uses {config['annualization_sessions']} sessions.", ""]
    for window, title in [("common", "Common-period comparison"), ("long", "Long-history comparison"), ("historical_test", "Retrospective historical test")]:
        text += [f"## {title}", "", "| Asset | Strategy | Period | Return | Annualized return | Volatility | Max drawdown | Sharpe (rf=0) | Closed trades | Trade win rate | Exposure |",
                 "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in summary[summary.window == window].iterrows():
            win = "n/a" if pd.isna(r.win_rate_closed_trades) else f"{r.win_rate_closed_trades:.1%}"
            text.append(f"| {r.ticker} | {r.strategy} | {r.start} to {r.end} | {r.cumulative_return:.1%} | {r.annualized_return:.1%} | {r.annualized_volatility:.1%} | {r.maximum_drawdown:.1%} | {r.sharpe_rf0:.2f} | {int(r.closed_trades)} | {win} | {r.session_exposure:.1%} |")
        text.append("")
    text += ["## Interpretation boundaries", "",
             "- Compare strategies only within the same window. A higher return alone is not risk-adjusted outperformance.",
             "- Few round trips make win rate highly uncertain; buy-and-hold has just one terminally closed trade.",
             "- Lower volatility or drawdown may reflect less time invested. Check exposure alongside returns.",
             "- The SNDK sample is short and starts after its 2025 separation. It cannot support strong cross-cycle claims.",
             "- These are selected current securities, not a point-in-time constituent universe; selection and survivorship bias remain.",
             "- Data are a vendor snapshot with adjusted-price execution approximations, not exchange-quality execution records.",
             "- Inspect cost_sensitivity.csv, the trade ledger, and research_design.md before drawing conclusions.",
             "- No ML model, statistical significance test, or prospective performance claim is included in V1.", ""]
    return "\n".join(text)


def run_pipeline(root: Path, refresh: bool = False):
    config = read_config(root)
    data = download_data(root, config, refresh=refresh)
    audit_coverage(data, config["benchmark"]).to_csv(root / "reports/data_quality.csv", index=False)
    write_sqlite(data, root / "data/market.sqlite")
    summary, sensitivity, results, features = run_study(data, config)
    summary.to_csv(root / "reports/metrics.csv", index=False)
    sensitivity.to_csv(root / "reports/cost_sensitivity.csv", index=False)
    ledger = []
    daily = []
    for (window, ticker, strategy), result in results.items():
        ledger.append(result.trades.assign(window=window, ticker=ticker, strategy=strategy))
        daily.append(result.daily.reset_index().assign(window=window, ticker=ticker, strategy=strategy))
    pd.concat(ledger, ignore_index=True).to_csv(root / "reports/trades.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(root / "reports/daily_results.csv", index=False)
    (root / "reports/findings.md").write_text(report_text(summary, config), encoding="utf-8")
    run_record = {"config": config, "config_sha256": sha256(root / "config.json"),
                  "data_manifest_sha256": sha256(root / "reports/manifest.json"),
                  "status": "retrospective_exploratory", "ml_implemented": False}
    (root / "reports/run_config.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    from .plots import make_figures
    make_figures(root, data, features, results, summary, config)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Deliberately replace the local vendor snapshot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = run_pipeline(root, refresh=args.refresh)
    print(summary.loc[summary.window == "common", ["ticker", "strategy", "cumulative_return", "maximum_drawdown", "sharpe_rf0"]].to_string(index=False))
    print("Saved reports/, figures/, and data/market.sqlite. No ML model has been fitted.")


if __name__ == "__main__":
    main()
