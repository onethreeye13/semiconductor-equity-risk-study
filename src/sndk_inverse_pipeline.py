"""Run the declared SNDK extension: python -m src.sndk_inverse_pipeline."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.data import sha256, validate_window
from src.sndk_inverse import (bootstrap_range_difference, candidate_rules,
                              fixed_horizon_backtest, range_observations,
                              range_summary, research_features, summarize_horizon,
                              training_leaders)
from src.sndk_inverse_data import load_research_data


def make_figures(root, ranges, intervals, metrics, selection, results, base_cost):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "sndk-inverse-v1.1"})
    colors = {"SNDQ": "#b74735", "SNXX": "#247b91"}
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        svg = figures / f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        # PNG previews are private, ignored intermediates for visual validation.
        previews = root / "data/processed/sndk_inverse_previews"
        previews.mkdir(exist_ok=True)
        fig.savefig(previews / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for ax, window in zip(axes, ["full_sndk", "etf_common"]):
        rows = ranges.loc[(ranges.window == window) &
                          (ranges.classification == "close_to_close") &
                          ranges.direction.isin(["down", "up"])].set_index("direction")
        ax.bar(["Down days", "Up days"], rows.loc[["down", "up"], "mean_range"],
               color=[colors["SNDQ"], colors["SNXX"]], width=.55)
        for i, direction in enumerate(["down", "up"]):
            row = rows.loc[direction]
            ax.text(i, row.mean_range, f"{row.mean_range:.2%}\nn={int(row.days)}",
                    ha="center", va="bottom")
        interval = intervals.loc[(intervals.window == window) &
                                 (intervals.classification == "close_to_close")].iloc[0]
        ax.set_title("Full SNDK history" if window == "full_sndk" else "Common ETF period")
        ax.set_ylim(0, rows.mean_range.max() * 1.32)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(f"Down − up: {100*interval.down_minus_up_mean_range:+.2f} pp\n"
                      f"Block-bootstrap 95% CI [{100*interval.ci_low:+.2f}, {100*interval.ci_high:+.2f}] pp")
    axes[0].set_ylabel("Mean (adjusted high − low) / previous close")
    fig.suptitle("SNDK daily range: classify by close-to-close return", fontsize=13)
    fig.tight_layout()
    save(fig, "sndk_range_asymmetry")

    base = metrics.loc[(metrics.ticker == "SNDQ") & metrics.rule_id.ne("unconditional")]
    comparison = base.pivot(index=["rule_id", "holding_sessions"], columns="segment",
                            values="mean_net_return").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    for h, color in zip([1, 3, 5], ["#247b91", "#b74735", "#7b61a8"]):
        rows = comparison.loc[comparison.index.get_level_values("holding_sessions") == h]
        ax.scatter(rows.train, rows.validation, label=f"Hold {h} session(s)", color=color, alpha=.8)
    chosen = selection["selected"]
    if chosen:
        key = (chosen["rule_id"], chosen["holding_sessions"])
        if key in comparison.index:
            point = comparison.loc[key]
            ax.scatter([point.train], [point.validation], s=210, facecolors="none",
                       edgecolors="black", linewidths=1.5, label="Training-selected rule")
    ax.axhline(0, color="gray", linewidth=.8)
    ax.axvline(0, color="gray", linewidth=.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Training mean net trade return", ylabel="Validation mean net trade return",
           title="SNDQ: all candidates with trades in both segments")
    ax.legend(fontsize=8)
    fig.text(.5, -.015, "Retrospective grid; sparse/related trades. Empty candidates remain in CSV tables.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "sndq_train_validation")

    rule = chosen["rule_id"] if chosen else "unconditional"
    h = chosen["holding_sessions"] if chosen else 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ticker in ["SNDQ", "SNXX"]:
        result = results[("validation", ticker, rule, h, base_cost)]
        axes[0].plot(result.daily.index, result.daily.equity, label=ticker, color=colors[ticker])
        axes[1].plot(result.daily.index, result.daily.drawdown, label=ticker, color=colors[ticker])
    axes[0].axhline(1, color="gray", linewidth=.7)
    axes[0].set_ylabel("Equity (start = 1)")
    axes[1].set_ylabel("Daily-close drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    label = "Training-selected" if chosen else "Unconditional baseline; no selected rule"
    axes[0].set_title(f"{label}: {rule}, hold {h} session(s)\n"
                      f"Chronological validation; {base_cost:g} bps per side")
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "sndk_etf_validation")


def write_findings(path, config, ranges, intervals, selection, metrics, all_metrics,
                   common_start, data):
    def pct(value):
        return "—" if pd.isna(value) else f"{100*value:.2f}%"

    def ratio(value):
        return "—" if pd.isna(value) else f"{value:.2f}"

    lines = ["# SNDK range and inverse-ETF research", "",
             "Retrospective exploratory extension; rules and split fixed on 2026-10-02. "
             "Not an untouched out-of-sample or personal-P&L attribution result.", "",
             f"SNDK: {len(data['SNDK'])} bars; SNDQ: {len(data['SNDQ'])}; SNXX: {len(data['SNXX'])}. "
             f"ETF comparison: {common_start} through {data['SNDQ'].index[-1].date()}. "
             f"Validation starts {config['test_start']}. Base cost: {config['one_way_cost_bps']} bps per side.",
             "", "## Daily range", "",
             "Up/down uses adjusted close-to-close returns. Range is (adjusted high − low) / previous adjusted close.",
             "", "| Window | Down days | Up days | Down mean range | Up mean range | Difference (pp) | 95% block interval (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for window in ["full_sndk", "etf_common"]:
        sub = ranges.loc[(ranges.window == window) &
                         (ranges.classification == "close_to_close")].set_index("direction")
        ci = intervals.loc[(intervals.window == window) &
                           (intervals.classification == "close_to_close")].iloc[0]
        lines.append(f"| {window} | {int(sub.loc['down','days'])} | {int(sub.loc['up','days'])} | "
                     f"{pct(sub.loc['down','mean_range'])} | {pct(sub.loc['up','mean_range'])} | "
                     f"{100*ci.down_minus_up_mean_range:+.2f} | [{100*ci.ci_low:+.2f}, {100*ci.ci_high:+.2f}] |")
    lines += ["", "A negative difference means down days have a smaller mean range. "
              "Intervals are descriptive and depend on the 5-session block assumption. "
              "Median range, intraday body, gap and open-to-close classification are in range_summary.csv.",
              "", "![Range comparison](../../figures/sndk_range_asymmetry.svg)",
              "", "## Training selection", "",
              f"{selection['eligible_candidates']} of 36 candidates meet the training eligibility rule "
              f"(at least {config['minimum_train_trades']} trades and both wins and losses). "
              "Selection maximizes training mean net trade return; a non-positive maximum selects nothing.",
              "", "| Training criterion | Rule | Hold sessions | Trades | Training value |",
              "|---|---|---:|---:|---:|"]
    for metric in ["mean_net_return", "win_rate", "payoff_ratio"]:
        leader = selection[metric]
        if leader:
            value = ratio(leader["training_value"]) if metric == "payoff_ratio" else pct(leader["training_value"])
            lines.append(f"| {metric} | {leader['rule_id']} | {leader['holding_sessions']} | "
                         f"{leader['training_trades']} | {value} |")
        else:
            lines.append(f"| {metric} | No eligible candidate | — | — | — |")
    chosen = selection["selected"]
    if chosen:
        lines += ["", f"Frozen candidate: **{chosen['rule_id']}, hold {chosen['holding_sessions']} session(s)**."]
        shown = metrics.loc[(metrics.rule_id == chosen["rule_id"]) &
                            (metrics.holding_sessions == chosen["holding_sessions"])]
    else:
        lines += ["", "**No candidate selected.** The following comparison is the unconditional "
                  "one-session baseline, not a replacement winner."]
        shown = metrics.loc[(metrics.rule_id == "unconditional") & (metrics.holding_sessions == 1)]
    lines += ["", "| Segment | ETF | Trades | Win rate | Payoff | Mean net/trade | Account return | Max drawdown |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in shown.itertuples():
        lines.append(f"| {row.segment} | {row.ticker} | {row.trades} | {pct(row.win_rate)} | "
                     f"{ratio(row.payoff_ratio)} | {pct(row.mean_net_return)} | {pct(row.cumulative_return)} | "
                     f"{pct(row.maximum_drawdown)} |")
    if chosen:
        test = shown.loc[(shown.segment == "validation") & (shown.ticker == "SNDQ")].iloc[0]
        lines += ["", f"SNDQ validation win-rate Wilson interval: [{pct(test.win_ci_low)}, {pct(test.win_ci_high)}]. "
                  f"Validation sample flag: **{test.sample_status}**. "
                  "Even a sufficient-count flag does not establish statistical significance or future edge."]
        lines += ["", "A payoff below 1 means the average winning trade is smaller than the average "
                  "losing trade. A higher win rate alone is not enough. The account reinvests capital, "
                  "so arithmetic mean trade return is not compounded account return.",
                  "", "### Unconditional SNDQ comparison at the same holding horizon", "",
                  "This baseline enters whenever flat without a SNDK feature threshold. "
                  "Its exposure and trading costs can differ from the conditional account.", "",
                  "| Segment | Trades | Win rate | Mean net/trade | Account return | Max drawdown | Exposure |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        baseline = metrics.loc[(metrics.ticker == "SNDQ") & (metrics.rule_id == "unconditional") &
                               (metrics.holding_sessions == chosen["holding_sessions"])]
        for row in baseline.itertuples():
            lines.append(f"| {row.segment} | {row.trades} | {pct(row.win_rate)} | "
                         f"{pct(row.mean_net_return)} | {pct(row.cumulative_return)} | "
                         f"{pct(row.maximum_drawdown)} | {pct(row.exposure)} |")
        lines += ["", "### Frozen SNDQ candidate: cost sensitivity", "",
                  "10 bps = 0.10% per side; 50 bps = 0.50% per side. No rule is reselected at other costs.",
                  "", "| Segment | Cost/side (bps) | Win rate | Mean net/trade | Account return | Max drawdown |",
                  "|---|---:|---:|---:|---:|---:|"]
        costs = all_metrics.loc[(all_metrics.ticker == "SNDQ") &
                                (all_metrics.rule_id == chosen["rule_id"]) &
                                (all_metrics.holding_sessions == chosen["holding_sessions"])]
        for row in costs.itertuples():
            lines.append(f"| {row.segment} | {row.cost_bps} | {pct(row.win_rate)} | "
                         f"{pct(row.mean_net_return)} | {pct(row.cumulative_return)} | "
                         f"{pct(row.maximum_drawdown)} |")
    lines += ["", "![Candidate stability](../../figures/sndq_train_validation.svg)",
              "", "![ETF comparison](../../figures/sndk_etf_validation.svg)",
              "", "## What this can and cannot explain", "",
              "- These are identical-rule hypothetical accounts, not the researcher's real trades. "
              "Personal P&L needs execution timestamps, position sizes, costs and FX records.",
              "- An up/down-day label is only known after the day ends. A large range is neither "
              "a known-at-open signal nor a return that could all be captured.",
              "- SNDQ and SNXX reset leverage daily. Observed ETF prices include path dependence "
              "and fund expenses; no pre-inception synthetic history or double expense charge is used.",
              "- Training maxima are vulnerable to multiple testing. All candidates, including "
              "empty or losing ones, are retained. Validation is not used to reselect a rule.",
              "- A few months of ETF data cannot establish a stable threshold across market regimes. "
              "No stop/target fill claims can be made from daily bars.",
              "", "## Outputs and reproduction", "",
              "Run `python -m src.sndk_inverse_pipeline` from the repository root. "
              "Use `--refresh` only for a deliberate new ETF snapshot; it preserves older ETF versions. "
              "The V1 underlying snapshot is reused and verified.", "",
              "- `candidate_metrics.csv`: every declared candidate and unconditional baseline, "
              "both ETFs and both chronological segments, at base cost.",
              "- `cost_sensitivity.csv`: the same frozen grid at 0/10/25/50 bps per side; no reselection.",
              "- `trades.csv`: base-cost trade ledger, including signal/entry/exit dates.",
              "- `range_summary.csv`, `range_intervals.csv`: both windows and day classifications.",
              "- `data_audit.csv`: date coverage, observed split events and large-return checks.",
              "- `selection.json`, `run_config.json`, `provenance.json`: rule choice, protocol and input hashes.",
              "", "See [protocol and primary product sources](../../docs/sndk_inverse_protocol.md) for exact definitions."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(root: Path, refresh: bool = False):
    config = json.loads((root / "config_sndk_inverse.json").read_text())
    if config["one_way_cost_bps"] not in config["cost_sensitivity_bps"]:
        raise ValueError("Base cost must be included in cost sensitivity")
    if config["selection_metric"] != "mean_net_return":
        raise ValueError("This protocol selects by training mean_net_return only")
    data, provenance = load_research_data(root, config, refresh=refresh)
    common_start = str(max(frame.index[0] for frame in data.values()).date())
    end, split = config["download_end_exclusive"], config["test_start"]
    if not pd.Timestamp(common_start) < pd.Timestamp(split) < pd.Timestamp(end):
        raise ValueError("Invalid chronological split")
    validate_window(data, list(data), "SNDK", common_start, end)
    features = research_features(data["SNDK"])
    rules = list(candidate_rules(features, config))
    rules.append(("unconditional", "none", np.nan, pd.Series(True, index=features.index)))
    results, rows, trade_tables, selection = {}, [], [], None
    base_cost = config["one_way_cost_bps"]
    for segment, start, stop in [("train", common_start, split), ("validation", split, end)]:
        for rule_id, feature, threshold, signal in rules:
            for h in config["holding_sessions"]:
                for ticker in config["etfs"]:
                    for cost in config["cost_sensitivity_bps"]:
                        result = fixed_horizon_backtest(data[ticker], signal, start, stop, h, cost)
                        results[(segment, ticker, rule_id, h, cost)] = result
                        meta = {"segment": segment, "ticker": ticker, "rule_id": rule_id,
                                "feature": feature, "threshold": threshold,
                                "holding_sessions": h, "cost_bps": cost}
                        stats = summarize_horizon(result)
                        minimum = config["minimum_train_trades"] if segment == "train" else config["minimum_test_trades"]
                        stats["sample_status"] = "insufficient_trades" if stats["trades"] < minimum else "count_threshold_met"
                        rows.append({**meta, **stats})
                        if cost == base_cost and not result.trades.empty:
                            trades = result.trades.copy()
                            for key, value in meta.items():
                                trades[key] = value
                            trades["signal_feature_value"] = (
                                features[feature].reindex(pd.DatetimeIndex(trades.signal_date)).to_numpy()
                                if feature != "none" else np.nan)
                            trade_tables.append(trades)
        if segment == "train":
            training = pd.DataFrame(rows).loc[lambda x: x.cost_bps == base_cost]
            selection = training_leaders(training, config["minimum_train_trades"])
            # Freeze now, before any validation account is evaluated.
    all_metrics = pd.DataFrame(rows)
    metrics = all_metrics.loc[all_metrics.cost_bps == base_cost].reset_index(drop=True)
    observations = range_observations(data["SNDK"])
    ranges, intervals = [], []
    for window, obs in [("full_sndk", observations),
                        ("etf_common", observations.loc[common_start:])]:
        ranges.append(range_summary(obs, window))
        for classification in ["close_to_close", "open_to_close"]:
            intervals.append({"window": window, "classification": classification,
                              **bootstrap_range_difference(obs, classification,
                                  config["bootstrap_repetitions"], config["bootstrap_block_sessions"],
                                  config["random_seed"])})
    ranges, intervals = pd.concat(ranges, ignore_index=True), pd.DataFrame(intervals)
    audit_rows = []
    expected_actions = {"SNDQ": ("2026-07-21", .1), "SNXX": ("2026-06-03", 8.)}
    underlying_returns = data["SNDK"].adj_close.pct_change(fill_method=None)
    for ticker, bars in data.items():
        splits = {str(date.date()): float(row.stock_splits)
                  for date, row in bars.loc[bars.stock_splits.ne(0)].iterrows()}
        returns = bars.adj_close.pct_change(fill_method=None)
        tracking_gap = pd.Series(dtype=float)
        if ticker in expected_actions:
            action_date, split_ratio = expected_actions[ticker]
            if bars.index[0] <= pd.Timestamp(action_date) <= bars.index[-1]:
                if not np.isclose(splits.get(action_date, np.nan), split_ratio):
                    raise ValueError(f"{ticker}: known corporate action missing/mismatched; review data")
            multiplier = -2 if ticker == "SNDQ" else 2
            tracking_gap = (returns - multiplier * underlying_returns.reindex(bars.index)).dropna()
        audit_rows.append({"ticker": ticker, "first_date": str(bars.index[0].date()),
                           "last_date": str(bars.index[-1].date()), "bars": len(bars),
                           "zero_volume_days": int(bars.volume.eq(0).sum()),
                           "max_abs_daily_return": returns.abs().max(),
                           "max_abs_return_date": str(returns.abs().idxmax().date()),
                           "max_abs_daily_target_gap": tracking_gap.abs().max(),
                           "split_events_json": json.dumps(splits)})
    audit = pd.DataFrame(audit_rows)
    out = root / "reports/sndk_inverse"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("candidate_metrics", metrics), ("cost_sensitivity", all_metrics),
                        ("trades", pd.concat(trade_tables, ignore_index=True)),
                        ("range_summary", ranges), ("range_intervals", intervals), ("data_audit", audit)]:
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
    provenance.update({"run_at_utc": datetime.now(timezone.utc).isoformat(),
                       "protocol_sha256": sha256(root / "config_sndk_inverse.json"),
                       "packages": {"python": platform.python_version(), "pandas": pd.__version__,
                                    "numpy": np.__version__,
                                    "matplotlib": importlib.metadata.version("matplotlib")},
                       "code_sha256": {f"src/{name}.py": sha256(root / f"src/{name}.py")
                                        for name in ["sndk_inverse", "sndk_inverse_data",
                                                     "sndk_inverse_pipeline", "data", "indicators"]}})
    for name, value in [("selection", selection), ("run_config", config), ("provenance", provenance)]:
        (out / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    make_figures(root, ranges, intervals, metrics, selection, results, base_cost)
    write_findings(out / "findings.md", config, ranges, intervals, selection, metrics,
                   all_metrics, common_start, data)
    return {"metrics": metrics, "ranges": ranges, "intervals": intervals,
            "selection": selection, "data_audit": audit, "output": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Create a new ETF snapshot version")
    args = parser.parse_args()
    result = run_study(Path(__file__).resolve().parents[1], refresh=args.refresh)
    print(f"Wrote {result['output']}")
    print(json.dumps(result["selection"], indent=2))
