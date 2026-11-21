"""Reproducible static research figures for the README."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

COLORS = {"SMA_20_50": "#247c9d", "Momentum_20": "#8b5fbf", "Buy_Hold": "#dd7854", "SOXX": "#27364d"}


def make_figures(root: Path, data: dict, features: dict, results: dict, summary: pd.DataFrame, config: dict):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .16,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    assets = [t for t in config["common_window"]["tickers"] if t != config["benchmark"]]
    sma_name = f"SMA_{config['sma_fast']}_{config['sma_slow']}"
    momentum_name = f"Momentum_{config['momentum_days']}"
    colors = {sma_name: COLORS["SMA_20_50"], momentum_name: COLORS["Momentum_20"], "Buy_Hold": COLORS["Buy_Hold"], "SOXX": COLORS["SOXX"]}
    benchmark = results["common", config["benchmark"], "Buy_Hold"].daily
    start, end = config["common_window"]["start"], config["common_window"]["end_exclusive"]
    caption = f"Next-open execution · {config['one_way_cost_bps']} bps each way · cash return 0% · terminal liquidation"

    def save(fig, name):
        fig.savefig(directory / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    for ticker in assets:
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for name in ["Buy_Hold", sma_name, momentum_name]:
            d = results["common", ticker, name].daily
            # Explicit initial capital point makes entry costs and first-day loss visible.
            points = pd.concat([pd.Series([1.0], index=[d.index[0] - pd.Timedelta(hours=12)]), d.equity])
            ax.plot(points.index, points, color=colors[name], label=name.replace("_", " "), lw=1.8)
        ax.plot(benchmark.index, benchmark.equity, color=COLORS["SOXX"], ls="--", label="SOXX Buy & Hold", lw=1.5)
        ax.set(title=f"{ticker} | Growth of $1", ylabel="Wealth / initial capital", xlabel=caption)
        ax.legend(frameon=False, loc="upper left")
        save(fig, f"equity_{ticker}.png")

    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 7), sharex=True, layout="constrained", squeeze=False)
    for ticker, ax in zip(assets, axes[:, 0]):
        for name in ["Buy_Hold", sma_name, momentum_name]:
            d = results["common", ticker, name].daily
            ax.plot(d.index, d.drawdown, color=colors[name], label=name.replace("_", " "), lw=1.4)
        ax.plot(benchmark.index, benchmark.drawdown, color=COLORS["SOXX"], ls="--", label="SOXX Buy & Hold")
        ax.set(title=f"{ticker} | Drawdown from prior wealth peak", ylabel="Drawdown")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")
    axes[-1, 0].set_xlabel(caption)
    save(fig, "drawdowns.png")

    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 7), sharex=True, layout="constrained", squeeze=False)
    for ticker, ax in zip(assets, axes[:, 0]):
        p = data[ticker].loc[start:]
        p = p.loc[p.index < pd.Timestamp(end)]
        trades = results["common", ticker, sma_name].trades
        ax.plot(p.index, p.adj_close, lw=1.4, color=COLORS["SOXX"], label="Adjusted close")
        ax.scatter(trades.entry_date, trades.entry_adj_price, marker="^", color="#07827a", s=48, zorder=5, label="Entry at next open")
        ax.scatter(trades.exit_date, trades.exit_adj_price, marker="v", color="#d14e54", s=48, zorder=5, label="Exit (last may be final close)")
        ax.set(title=f"{ticker} | SMA {config['sma_fast']} / {config['sma_slow']} executions", ylabel="Adjusted price (USD units)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel("Markers show executions, not the earlier close that generated the signal")
    save(fig, "sma_executions.png")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for ticker, color in zip(config["tickers"], [COLORS["SMA_20_50"], COLORS["Momentum_20"], COLORS["SOXX"]]):
        x = features[ticker].loc[start:]
        x = x.loc[x.index < pd.Timestamp(end)]
        ax.plot(x.index, x.rolling_volatility, label=ticker, color=color, lw=1.6)
    ax.set(title="20-session rolling volatility | Common window", ylabel="Annualized volatility", xlabel="Daily adjusted-close returns · sample standard deviation × √252")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    save(fig, "rolling_volatility.png")

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    rows = summary[summary.window == "common"]
    markers = {"MU": "o", "SNDK": "s", "SOXX": "D"}
    offsets = [(-70, -15), (-75, 12), (8, 4), (8, 9), (8, -17), (8, 5), (8, -12)]
    for (_, r), offset in zip(rows.iterrows(), offsets):
        color = COLORS["SOXX"] if r.ticker == "SOXX" else colors[r.strategy]
        ax.scatter(r.annualized_volatility, r.annualized_return, color=color, marker=markers[r.ticker], s=80)
        ax.annotate(f"{r.ticker} · {r.strategy.replace('_', ' ')}", (r.annualized_volatility, r.annualized_return), xytext=offset, textcoords="offset points", fontsize=8)
    ax.margins(x=.35, y=.20)
    ax.set(title="Risk and return | Same common window", xlabel="Annualized volatility", ylabel="Annualized compounded return")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "risk_return.png")
