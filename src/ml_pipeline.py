"""Reproduce fixed ML001 from the verified V1 snapshot: python -m src.ml_pipeline."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib
from .data import read_config, download_data, audit_coverage, validate_window, sha256
from .indicators import add_features, trading_signals
from .metrics import summarize
from .ml import (FEATURES, make_dataset, make_blocks, fit_predict,
                 classification_metrics, open_interval_backtest)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_protocol(root: Path):
    lock = json.loads((root / "reports/ml/protocol_lock.json").read_text())
    for rel, expected in lock["sha256"].items():
        if sha256(root / rel) != expected:
            raise ValueError(f"Frozen ML001 protocol changed: {rel}; document a new experiment")
    return lock


def run_experiment(data: dict, base_config: dict, config: dict):
    if config["cash_return_annual"] != 0 or config["sharpe_reference_rate_annual"] != 0:
        raise ValueError("Only zero cash/reference rates are implemented")
    if config["probability_threshold"] != .5 or config["threshold_rule"] != "strictly_greater_than":
        raise ValueError("ML001 uses a fixed probability > 0.5 rule")
    ticker, benchmark = config["ticker"], config["benchmark"]
    validate_window(data, [ticker, benchmark], benchmark,
                    config["training_start"], config["end_exclusive"])
    dataset = make_dataset(data[ticker], config["horizon_sessions"])
    blocks, split_audit, purged = make_blocks(dataset, config)
    rule_signals = trading_signals(add_features(data[ticker], base_config))
    classification, predictions, costs, results, fits, yearly, yearly_accounts = [], [], [], {}, {}, [], []
    fit_audit = []
    for block_name, train_name in [("validation", "train"), ("test", "refit")]:
        train, block = blocks[train_name], blocks[block_name]
        prob, estimators = fit_predict(train, block, config)
        fits[block_name] = estimators
        fit_audit.append({"evaluation": block_name, "fit_block": train_name,
                          "fit_rows": len(train), "fit_feature_start": train.index[0],
                          "fit_feature_end": train.index[-1], "fit_label_end": train.label_end_date.max(),
                          "evaluation_first_feature": block.index[0],
                          "training_positive_rate": train.target.mean()})
        signals = {name: (prob[name] > config["probability_threshold"]).astype(int) for name in prob}
        signals.update({name: signal.loc[block.index].astype(int) for name, signal in rule_signals.items()})
        for name, prediction in signals.items():
            probability = prob[name] if name in prob else None
            classification.append({"block": block_name, "model": name,
                                   **classification_metrics(block.target, prediction, probability)})
            record = block[["entry_date", "label_end_date", "target", "forward_return"]].copy()
            record["prediction"] = prediction
            record["probability_up"] = probability if probability is not None else np.nan
            predictions.append(record.reset_index().assign(block=block_name, model=name))
            if block_name == "test":
                for year in sorted(block.index.year.unique()):
                    mask = block.index.year == year
                    yearly.append({"feature_year": int(year), "model": name,
                                   **classification_metrics(block.target.loc[mask], prediction.loc[mask],
                                                            probability.loc[mask] if probability is not None else None)})
        accounts = {(ticker, name): sig for name, sig in signals.items()}
        accounts.update({(asset, "Buy_Hold"): pd.Series(1, index=block.index) for asset in (ticker, benchmark)})
        for (asset, strategy), signal in accounts.items():
            for cost in sorted(set(config["cost_sensitivity_bps"] + [config["one_way_cost_bps"]])):
                result = open_interval_backtest(data[asset], block, signal, cost)
                costs.append({"block": block_name, "ticker": asset, "strategy": strategy,
                              "one_way_cost_bps": cost,
                              "entry_start": block.entry_date.iloc[0].date().isoformat(),
                              **summarize(result, config["annualization_sessions"])})
                if cost == config["one_way_cost_bps"]:
                    results[block_name, asset, strategy] = result
                    if block_name == "test":
                        for year, d in result.daily.groupby(result.daily.index.year):
                            yearly_accounts.append({"mark_year": int(year), "ticker": asset, "strategy": strategy,
                                                    "first_end_open": d.index[0], "last_end_open": d.index[-1],
                                                    "intervals": len(d), "return": (1+d.net_return).prod()-1,
                                                    "exposure": d.position_during_session.mean(),
                                                    "transactions": int(d.turnover.sum())})
    sensitivity = pd.DataFrame(costs)
    return {"dataset": dataset, "blocks": blocks, "split_audit": split_audit, "purged_rows": purged,
            "fit_audit": pd.DataFrame(fit_audit), "classification_metrics": pd.DataFrame(classification),
            "predictions": pd.concat(predictions, ignore_index=True), "cost_sensitivity": sensitivity,
            "trading_metrics": sensitivity.loc[sensitivity.one_way_cost_bps == config["one_way_cost_bps"]].copy(),
            "yearly_classification": pd.DataFrame(yearly), "yearly_account_returns": pd.DataFrame(yearly_accounts),
            "results": results, "fits": fits}


def findings(study: dict, config: dict) -> str:
    lines = ["# ML001: retrospective MU machine-learning results", "",
             "Fixed logistic regression and a shallow random forest; no parameter or threshold search.",
             "The historical prices had already been reviewed in V1. These are chronologically separated",
             "model-fit evaluations, **not** prospective or pristine unseen-market evidence.", "",
             "Target: positive adjusted-open return from t+1 to t+2, using features known at close t.",
             "Both models are retained regardless of results. Training-majority is fitted on training labels only.", ""]
    for block_name in ("validation", "test"):
        audit = study["split_audit"].set_index("block").loc[block_name]
        lines += [f"## {block_name.title()}", "",
                  f"Feature dates: {audit.feature_start.date()} to {audit.feature_end.date()}; "
                  f"execution: {audit.entry_start.date()} open to {audit.label_end.date()} open; {audit.n} intervals.", "",
                  "| Model/rule | Accuracy | Balanced accuracy | ROC-AUC | Brier ↓ | Log loss ↓ | Predicted up |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        cls = study["classification_metrics"].query("block == @block_name")
        for r in cls.itertuples():
            scores = ["n/a" if pd.isna(v) else f"{v:.3f}" for v in (r.roc_auc, r.brier_score, r.log_loss)]
            lines.append(f"| {r.model} | {r.accuracy:.2%} | {r.balanced_accuracy:.2%} | "
                         + " | ".join(scores) + f" | {r.predicted_positive_rate:.1%} |")
        lines += ["", f"Observed positive fraction: {cls.positive_rate.iloc[0]:.2%}. "
                  "Rule outputs are hard labels, not probability estimates.", "",
                  f"All accounts include {config['one_way_cost_bps']} bps per side and final-open liquidation.", "",
                  "| Asset | Account | Net return | Annualized return | Volatility | Open-marked MDD | Sharpe rf=0 | Exposure | Trades | Trade win rate |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in study["trading_metrics"].query("block == @block_name").itertuples():
            win = f"{r.win_rate_closed_trades:.1%}" if pd.notna(r.win_rate_closed_trades) else "n/a"
            lines.append(f"| {r.ticker} | {r.strategy} | {r.cumulative_return:.1%} | {r.annualized_return:.1%} | "
                         f"{r.annualized_volatility:.1%} | {r.maximum_drawdown:.1%} | {r.sharpe_rf0:.2f} | "
                         f"{r.session_exposure:.1%} | {r.closed_trades} | {win} |")
        lines.append("")
    cls = study["classification_metrics"].query("block == 'test'").set_index("model")
    trade = study["trading_metrics"]
    trade = trade.loc[(trade.block == "test") & (trade.ticker == config["ticker"])].set_index("strategy")
    lines += ["## What this experiment shows", ""]
    for name in ("LogisticRegression", "RandomForest"):
        delta = cls.loc[name, "accuracy"] - cls.loc["Prior", "accuracy"]
        lines.append(f"- {name}: test accuracy is {delta*100:+.2f} percentage points relative to Prior; "
                     f"net return is {trade.loc[name, 'cumulative_return']:.1%}, versus "
                     f"{trade.loc['Buy_Hold', 'cumulative_return']:.1%} for MU buy-and-hold. "
                     f"Open-marked maximum drawdown is {trade.loc[name, 'maximum_drawdown']:.1%} and "
                     f"exposure is {trade.loc[name, 'session_exposure']:.1%}.")
    lines += ["", "## Test cost sensitivity", "",
              "| Account | Cost per side | Net return | One-way transactions |", "|---|---:|---:|---:|"]
    for r in study["cost_sensitivity"].query("block == 'test' and strategy in ['LogisticRegression', 'RandomForest']").itertuples():
        lines.append(f"| {r.strategy} | {r.one_way_cost_bps} bps | {r.cumulative_return:.1%} | {r.one_way_transactions} |")
    lines += ["", "## Limits and reading guide", "",
              "- Accuracy and trade win rate measure different things. Daily long predictions can merge into one trade.",
              "- A high majority accuracy can reflect an imbalanced rising sample. Balanced accuracy, AUC and probability scores provide additional checks.",
              "- Lower drawdown may come from lower exposure. Neither a single accuracy gain nor a return point estimate establishes durable predictive/economic value.",
              "- Open-sampled drawdown excludes intraday extremes. V1 close-marked metrics are not directly interchangeable with this table.",
              "- Inspect yearly_classification.csv and yearly_account_returns.csv for regime variation. They are descriptive slices of one fixed model, not independent replications; 2026 is partial.",
              "- No significance test or independence assumption is used to claim an edge. This is one selected asset and few market regimes.",
              "- Both final estimators were fitted only on eligible pre-2024 labels and then frozen; staleness can hurt performance.",
              "- Vendor-adjusted data are a current snapshot, not point-in-time corporate-action records. Volume changes can reflect corporate actions.",
              "- This result does not transfer automatically to SNDK, SNDQ/SNXX, future periods, or personal account P&L.",
              "- Do not tune on this test and call it independent again. See docs/ml_protocol.md, split/fit audits, predictions and trade ledger.", ""]
    return "\n".join(lines)


def run_ml_pipeline(root: Path):
    lock = verify_protocol(root)
    config = json.loads((root / "config_ml.json").read_text())
    base_config = read_config(root)
    if not (root / "reports/manifest.json").exists():
        raise FileNotFoundError("First run src.pipeline or restore the private V1 snapshot; ML never silently downloads new prices")
    data = download_data(root, base_config, refresh=False)
    audit_coverage(data, config["benchmark"])
    study = run_experiment(data, base_config, config)
    out = root / "reports/ml"
    out.mkdir(exist_ok=True)
    table_names = ["split_audit", "purged_rows", "fit_audit", "classification_metrics", "predictions",
                   "trading_metrics", "cost_sensitivity", "yearly_classification", "yearly_account_returns"]
    for name in table_names:
        study[name].to_csv(out / f"{name}.csv", index=False)
    daily, trades = [], []
    for (block, ticker, strategy), result in study["results"].items():
        daily.append(result.daily.reset_index().assign(block=block, ticker=ticker, strategy=strategy))
        trades.append(result.trades.assign(block=block, ticker=ticker, strategy=strategy))
    pd.concat(daily, ignore_index=True).to_csv(out / "daily_results.csv", index=False)
    pd.concat(trades, ignore_index=True).to_csv(out / "trades.csv", index=False)
    (out / "findings.md").write_text(findings(study, config), encoding="utf-8")
    from .ml_plots import make_ml_figures
    make_ml_figures(root, study)
    fit_details = {}
    for name, estimators in study["fits"].items():
        lr, forest = estimators["LogisticRegression"], estimators["RandomForest"]
        fit_details[name] = {"features": FEATURES, "scaler_mean": lr["scale"].mean_.tolist(),
                             "scaler_scale": lr["scale"].scale_.tolist(),
                             "logistic_coefficients": lr["model"].coef_[0].tolist(),
                             "logistic_intercept": float(lr["model"].intercept_[0]),
                             "forest_training_impurity_importance": forest.feature_importances_.tolist(),
                             "note": "Training diagnostics, not causal effects or test permutation importance."}
    write_json(out / "fit_details.json", fit_details)
    source_files = ["config.json", "config_ml.json", "docs/ml_protocol.md", "src/ml.py", "src/ml_pipeline.py",
                    "src/ml_plots.py", "src/data.py", "src/indicators.py", "src/backtest.py", "src/metrics.py", "requirements.txt"]
    outputs = sorted([p for p in out.iterdir() if p.suffix == ".csv"] +
                     [out / "findings.md", out / "fit_details.json"] + list((root / "figures").glob("ml_*.svg")))
    manifest = json.loads((root / "reports/manifest.json").read_text())
    record = {"experiment_id": config["experiment_id"], "run_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": config["status"], "config": config, "protocol_lock": lock,
              "source_sha256": {p: sha256(root / p) for p in source_files},
              "data_manifest_sha256": sha256(root / "reports/manifest.json"),
              "input_sha256": manifest["sha256"], "data_downloaded_at_utc": manifest["downloaded_at_utc"],
              "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                              "scipy": scipy.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                              "matplotlib": matplotlib.__version__},
              "output_sha256": {str(p.relative_to(root)): sha256(p) for p in outputs},
              "experiment_log": [{"id": "ML001", "parameter_variants": 1, "model_families": 2,
                                  "test_used_for_selection": False, "history_previously_viewed": True,
                                  "note": "Fixed design; validation diagnostic, deterministic pre-test refit. Both models reported."}]}
    write_json(out / "provenance.json", record)
    return study


def main():
    study = run_ml_pipeline(Path(__file__).resolve().parents[1])
    print(study["classification_metrics"].to_string(index=False))
    print("Saved reports/ml/ and figures/ml_*.svg; original V1 outputs unchanged.")


if __name__ == "__main__":
    main()
