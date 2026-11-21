"""Deterministic, exportable ML research figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"LogisticRegression": "#2864b7", "RandomForest": "#b25226", "Momentum_20": "#23856d",
          "Buy_Hold": "#242d3d", "SOXX": "#8b8798", "Prior": "#747b84"}


def make_ml_figures(root, study):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "ML001"})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
    for asset, name in [("MU", "LogisticRegression"), ("MU", "RandomForest"), ("MU", "Momentum_20"),
                        ("MU", "Buy_Hold"), ("SOXX", "Buy_Hold")]:
        d = study["results"]["test", asset, name].daily
        idx = pd.DatetimeIndex([d.entry_date.iloc[0]]).append(d.index)
        color = COLORS["SOXX" if asset == "SOXX" else name]
        label = f"{asset} {name.replace('_', ' ')}"
        axes[0].plot(idx, np.r_[1., d.equity], label=label, color=color, linewidth=1.6)
        axes[1].plot(idx, np.r_[0., d.drawdown*100], color=color, linewidth=1.25)
    axes[0].set(yscale="log", ylabel="Account value (log scale)", title="MU: fixed-model historical test, net of 10 bps per side")
    axes[0].legend(ncol=2, fontsize=8, loc="upper left")
    axes[1].set(ylabel="Open-marked drawdown (%)", xlabel="Open at interval end")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.text(.08, .015, "Features 2024-01-02–2026-09-28; trade 2024-01-03 open–2026-09-30 open. Retrospective; intraday drawdowns excluded.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(root / "figures/ml_equity_drawdown.svg", metadata={"Date": None})
    plt.close(fig)

    metrics = study["classification_metrics"]
    names = ["Prior", "LogisticRegression", "RandomForest"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, block in zip(axes, ["validation", "test"]):
        d = metrics.loc[metrics.block == block].set_index("model").loc[names]
        x = np.arange(3)
        ax.bar(x-.18, 100*d.accuracy, .36, color="#2864b7", label="Accuracy")
        ax.bar(x+.18, 100*d.balanced_accuracy, .36, color="#23856d", label="Balanced accuracy")
        ax.axhline(50, color="#727272", linestyle="--", linewidth=.8)
        ax.set(xticks=x, xticklabels=["Training prior", "Logistic", "Random forest"], ylim=(0, 100),
               title=block.title(), ylabel="Score (%)" if block == "validation" else "")
        ax.grid(axis="y", alpha=.15)
        for i, r in enumerate(d.itertuples()):
            ax.text(i-.18, r.accuracy*100+1.4, f"{r.accuracy:.1%}", ha="center", fontsize=8)
            ax.text(i+.18, r.balanced_accuracy*100+1.4, f"{r.balanced_accuracy:.1%}", ha="center", fontsize=8)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("A majority baseline can exceed 50% without conditional prediction skill")
    fig.text(.07, .02, "Fixed threshold > 0.5. Chronological, purged fits; previously reviewed history. Accuracy is not trading profit.", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.savefig(root / "figures/ml_classification.svg", metadata={"Date": None})
    plt.close(fig)
