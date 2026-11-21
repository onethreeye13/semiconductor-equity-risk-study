"""ML001: causal features, date-purged fits and aligned open-to-open accounts."""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .backtest import BacktestResult
from .indicators import rsi_wilder


FEATURES = ["return_1", "return_2", "return_5", "momentum_20", "rsi_14",
            "volatility_20", "ma_distance_20", "ma_distance_50", "volume_change_1"]


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """X_t is known after close t. No targets, dates or future prices in X."""
    close = bars.adj_close
    x = pd.DataFrame(index=bars.index)
    for n in (1, 2, 5):
        x[f"return_{n}"] = close.pct_change(n, fill_method=None)
    x["momentum_20"] = close.pct_change(20, fill_method=None)
    x["rsi_14"] = rsi_wilder(close, 14)
    x["volatility_20"] = x.return_1.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    for n in (20, 50):
        x[f"ma_distance_{n}"] = close / close.rolling(n, min_periods=n).mean() - 1
    x["volume_change_1"] = bars.volume.pct_change(fill_method=None)
    return x.replace([np.inf, -np.inf], np.nan)[FEATURES]


def make_dataset(bars: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if horizon != 1:
        raise ValueError("ML001 supports only one-session non-overlapping labels")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Bars need unique chronological dates")
    d = make_features(bars)
    dates = pd.Series(bars.index, index=bars.index)
    d.index = d.index.rename("feature_date")
    d["entry_date"] = dates.shift(-1)
    d["label_end_date"] = dates.shift(-2)
    d["forward_return"] = bars.adj_open.shift(-2) / bars.adj_open.shift(-1) - 1
    d["target"] = (d.forward_return > 0).astype("Int64").where(d.forward_return.notna())
    return d


def date_block(dataset: pd.DataFrame, start: str, end: str):
    """Keep only labels ending strictly before the block boundary; audit removals."""
    candidate = dataset.loc[(dataset.index >= pd.Timestamp(start)) &
                            (dataset.index < pd.Timestamp(end))].copy()
    eligible = candidate.label_end_date.notna() & (candidate.label_end_date < pd.Timestamp(end))
    removed = candidate.loc[~eligible, ["entry_date", "label_end_date"]].copy()
    removed["reason"] = np.where(removed.label_end_date.isna(), "incomplete_future_label",
                                 "label_reaches_next_block")
    kept = candidate.loc[eligible]
    if kept.empty:
        raise ValueError("Empty chronological block")
    if not np.isfinite(kept[FEATURES].to_numpy()).all():
        raise ValueError("Missing/nonfinite features inside block; no silent dropping or filling")
    if not np.isfinite(kept.forward_return).all() or kept.target.isna().any():
        raise ValueError("Missing target inside block")
    if not (kept.index < kept.entry_date).all() or not (kept.entry_date < kept.label_end_date).all():
        raise ValueError("Invalid feature/entry/label time order")
    return kept, removed


def make_blocks(dataset: pd.DataFrame, config: dict):
    if config["features"] != FEATURES:
        raise ValueError("ML001 feature list changed: define a new experiment explicitly")
    starts_ends = {
        "train": (config["training_start"], config["validation_start"]),
        "validation": (config["validation_start"], config["test_start"]),
        "refit": (config["training_start"], config["test_start"]),
        "test": (config["test_start"], config["end_exclusive"]),
    }
    blocks, audits, removals = {}, [], []
    for name, (start, end) in starts_ends.items():
        b, removed = date_block(dataset, start, end)
        blocks[name] = b
        audits.append({"block": name, "n": len(b), "feature_start": b.index[0],
                       "feature_end": b.index[-1], "entry_start": b.entry_date.iloc[0],
                       "label_end": b.label_end_date.iloc[-1], "positive_rate": b.target.mean(),
                       "removed_rows": len(removed), "boundary_exclusive": end})
        removals.append(removed.reset_index().assign(block=name))
    return blocks, pd.DataFrame(audits), pd.concat(removals, ignore_index=True)


def fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame, config: dict):
    """All fit operations see training only; evaluation labels are not read here."""
    if train.label_end_date.max() >= evaluation.index.min():
        raise ValueError("Training labels reach evaluation feature dates: purge first")
    x, y = train[FEATURES], train.target.astype(int)
    if y.nunique() != 2:
        raise ValueError("Both target classes are required in training")
    estimators = {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**config["logistic_regression"]))]),
        "RandomForest": RandomForestClassifier(**config["random_forest"]),
    }
    probabilities = pd.DataFrame(index=evaluation.index)
    probabilities["Prior"] = float(y.mean())
    for name, estimator in estimators.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x, y)
        class_one = list(estimator.classes_).index(1)
        probabilities[name] = estimator.predict_proba(evaluation[FEATURES])[:, class_one]
    return probabilities, estimators


def classification_metrics(y, prediction, probability=None) -> dict:
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    result = {"observations": len(y), "positive_rate": y.mean(),
              "predicted_positive_rate": prediction.mean(),
              "accuracy": accuracy_score(y, prediction),
              "balanced_accuracy": balanced_accuracy_score(y, prediction) if both else np.nan,
              "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
              "roc_auc": np.nan, "brier_score": np.nan, "log_loss": np.nan}
    if probability is not None:
        p = np.asarray(probability, dtype=float)
        result.update(roc_auc=roc_auc_score(y, p) if both else np.nan,
                      brier_score=brier_score_loss(y, p),
                      log_loss=log_loss(y, np.column_stack([1-p, p]), labels=[0, 1]))
    return result


def open_interval_backtest(bars: pd.DataFrame, intervals: pd.DataFrame,
                           desired: pd.Series, cost_bps: float = 10) -> BacktestResult:
    """One contiguous account over disjoint next-open to following-open intervals.

    End-open marks precede the next rebalance; that rebalance fee belongs to the
    next interval. The last mark includes terminal liquidation at that open.
    """
    if not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be in [0, 10000)")
    if intervals.empty or not desired.index.equals(intervals.index):
        raise ValueError("Desired holdings must match a nonempty interval index")
    if not intervals.index.is_unique or not intervals.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and chronological")
    if not desired.isin([0, 1]).all():
        raise ValueError("Only complete long/cash signals are supported")
    if not (intervals.index < intervals.entry_date).all() or not (intervals.entry_date < intervals.label_end_date).all():
        raise ValueError("Invalid signal and execution date order")
    if not np.array_equal(intervals.label_end_date.iloc[:-1].to_numpy(),
                          intervals.entry_date.iloc[1:].to_numpy()):
        raise ValueError("Intervals must be contiguous, with no overlap or gap")
    # Validate actual bar positions, not just ordered timestamps.
    fi = bars.index.get_indexer(intervals.index)
    ei = bars.index.get_indexer(intervals.entry_date)
    li = bars.index.get_indexer(intervals.label_end_date)
    if (fi < 0).any() or not ((ei == fi+1) & (li == fi+2)).all():
        raise ValueError("Intervals must use the next two actual trading opens")
    prices = bars.adj_open
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Adjusted opens must be finite and positive")
    equity, held, cost, entry = 1.0, 0, cost_bps / 10_000, None
    daily, trades = [], []
    for i, (feature_date, row) in enumerate(intervals.iterrows()):
        target = int(desired.loc[feature_date])
        before, fee, turnover = equity, 0.0, 0
        if target != held:
            turnover += 1
            if target:
                entry = {"date": row.entry_date, "price": prices.loc[row.entry_date],
                         "capital": equity, "i": i}
                fee = equity * cost / (1 + cost)
                equity -= fee
            else:
                fee = equity * cost
                equity -= fee
                trades.append({"entry_date": entry["date"], "exit_date": row.entry_date,
                               "entry_adj_price": entry["price"], "exit_adj_price": prices.loc[row.entry_date],
                               "holding_sessions": i-entry["i"],
                               "net_return": equity/entry["capital"]-1, "exit_reason": "signal"})
                entry = None
        held = target
        if held:
            equity *= prices.loc[row.label_end_date] / prices.loc[row.entry_date]
        terminal = i == len(intervals)-1 and bool(held)
        if terminal:
            exit_fee = equity * cost
            equity -= exit_fee
            fee += exit_fee
            turnover += 1
            trades.append({"entry_date": entry["date"], "exit_date": row.label_end_date,
                           "entry_adj_price": entry["price"], "exit_adj_price": prices.loc[row.label_end_date],
                           "holding_sessions": i-entry["i"]+1,
                           "net_return": equity/entry["capital"]-1, "exit_reason": "terminal_open"})
            held = 0
        daily.append({"date": row.label_end_date, "feature_date": feature_date,
                      "entry_date": row.entry_date, "position_during_session": target,
                      "position_end": held, "turnover": turnover,
                      "cost_in_initial_capital_units": fee,
                      "net_return": equity/before-1, "equity": equity,
                      "terminal_exit": terminal})
    d = pd.DataFrame(daily).set_index("date")
    d["drawdown"] = d.equity/d.equity.cummax().clip(lower=1.0)-1
    columns = ["entry_date", "exit_date", "entry_adj_price", "exit_adj_price",
               "holding_sessions", "net_return", "exit_reason"]
    return BacktestResult(d, pd.DataFrame(trades, columns=columns))
