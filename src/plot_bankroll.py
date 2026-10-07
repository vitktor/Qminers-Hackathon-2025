"""Plot the bankroll of every variant through the evaluation loop on all training seasons.

Writes docs/bankroll.png and docs/bankroll_dark.png (light and dark theme for the README).

Usage: python src/plot_bankroll.py
"""

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from environment import Environment
from model import DEFAULT_VARIANT, VARIANTS, Model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "games.csv"
INIT_BANKROLL = 1000

# The variants compared in the README. Default first so it gets the first colour and is drawn on top.
ORDER = [DEFAULT_VARIANT, "xgb_all", "logreg", "logreg_all"]
assert set(ORDER) <= set(VARIANTS)

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def bankroll_history(variant: str) -> pd.Series:
    games = pd.read_csv(DATA_PATH, index_col=0, parse_dates=["Date", "Open"])
    env = Environment(games, Model(variant), init_bankroll=INIT_BANKROLL, min_bet=5, max_bet=100)
    with contextlib.redirect_stdout(io.StringIO()):
        env.run()
    history = env.get_history()
    # Keep only the states saved after bets were settled, not right after new stakes went out.
    return history.loc[history.Cash_Invested == 0, "Bankroll"]


def spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Push end labels apart vertically so they do not overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def plot(histories: dict[str, pd.Series], theme: str, path: Path) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    plt.rcParams["font.family"] = "sans-serif"

    start = min(h.index[0] for h in histories.values())
    end = max(h.index[-1] for h in histories.values())
    ax.hlines(INIT_BANKROLL, start, end, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    lines = {}
    for i, variant in reversed(list(enumerate(ORDER))):
        h = histories[variant]
        lines[variant], = ax.plot(h.index, h.values, color=t["series"][i], lw=2.2 if i == 0 else 1.6,
                                  label=variant, zorder=3 + (i == 0), solid_joinstyle="round")

    finals = [histories[v].iloc[-1] for v in ORDER]
    ymin, ymax = ax.get_ylim()
    for variant, final, y in zip(ORDER, finals, spread_labels(finals, (ymax - ymin) * 0.06)):
        ax.annotate(f"{variant}  {final:,.0f}", xy=(end, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=t["ink"],
                    fontweight="bold" if variant == DEFAULT_VARIANT else "normal",
                    annotation_clip=False)

    ax.set_title("Bankroll through the evaluation loop, all training seasons (start = 1000)",
                 loc="left", fontsize=12, color=t["ink"], pad=30)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    legend = ax.legend([lines[v] for v in ORDER], ORDER, loc="lower left", bbox_to_anchor=(0, 1.0),
                       frameon=False, fontsize=9, ncol=4, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(t["ink2"])

    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    histories = {}
    for variant in ORDER:
        histories[variant] = bankroll_history(variant)
        print(f"{variant}: final bankroll {histories[variant].iloc[-1]:.2f}")
    plot(histories, "light", ROOT / "docs" / "bankroll.png")
    plot(histories, "dark", ROOT / "docs" / "bankroll_dark.png")


if __name__ == "__main__":
    main()
