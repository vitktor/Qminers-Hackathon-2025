"""Season-by-season walk-forward check of the outcome model variants.

For each test season every variant is fit only on earlier seasons, then compared with the
bookmaker on log-loss and on the flat-stake ROI of bets whose estimated edge exceeds a
threshold. A model is only worth betting with if it beats the market log-loss.

Usage: python src/validate.py [variant ...]   (default: all variants)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import MARKET_COL, VARIANTS, FeatureBuilder, OutcomeModel

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "games.csv"
EDGES = [0.0, 0.05, 0.1]
RECENT = 2002  # seasons without draw odds, closest to the hidden validation seasons


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def flat_roi(p: np.ndarray, rows: pd.DataFrame, min_edge: float) -> tuple[int, float]:
    """Number of bets and ROI of 1-unit bets on the side with the larger edge (draws lose)."""
    edge_h = p * (1 - rows.p_draw) * rows.odds_h - 1
    edge_a = (1 - p) * (1 - rows.p_draw) * rows.odds_a - 1
    bet_h = (edge_h >= edge_a) & (edge_h > min_edge)
    bet_a = (edge_a > edge_h) & (edge_a > min_edge)
    payout = (np.where(bet_h & (rows.result == "H"), rows.odds_h, 0)
              + np.where(bet_a & (rows.result == "A"), rows.odds_a, 0))
    n = int(bet_h.sum() + bet_a.sum())
    return n, (payout.sum() / n - 1) if n else 0.0


def walk_forward(variant: str, builder: FeatureBuilder, rows: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    report = []
    for season in sorted(rows.season.unique())[5:]:
        model = OutcomeModel.from_variant(variant)
        model.maybe_refit(builder.rows[: int((rows.season < season).sum())])
        test = rows[rows.season == season]
        p = model.predict(X[test.index])
        # Log-loss is measured on decided games, matching the model's target P(H | no draw).
        decided = (test.result != "D").values
        p_market = 1 / (1 + np.exp(-X[test.index, MARKET_COL]))
        y = (test.result == "H").values
        line = {"season": season, "n": len(test),
                "ll_market": log_loss(p_market[decided], y[decided]),
                "ll_model": log_loss(p[decided], y[decided])}
        for e in EDGES:
            line[f"bets@{e}"], line[f"roi@{e}"] = flat_roi(p, test, e)
        report.append(line)
    return pd.DataFrame(report).set_index("season")


def summary(report: pd.DataFrame) -> dict:
    gain = report.ll_market - report.ll_model
    out = {"ll_gain": gain.mean(), f"ll_gain_{RECENT}+": gain[report.index >= RECENT].mean()}
    for e in EDGES:
        n = report[f"bets@{e}"]
        out[f"bets@{e}"] = int(n.sum())
        out[f"roi@{e}"] = (report[f"roi@{e}"] * n).sum() / max(n.sum(), 1)
    return out


def main() -> None:
    variants = sys.argv[1:] or list(VARIANTS)
    games = pd.read_csv(DATA_PATH, index_col=0, parse_dates=["Date", "Open"])
    builder = FeatureBuilder()
    builder.ingest(games)
    rows = pd.DataFrame(builder.rows)
    X = np.stack(rows.x)

    results = {}
    for variant in variants:
        report = walk_forward(variant, builder, rows, X)
        if len(variants) == 1:
            print(report.round(4).to_string(), "\n")
        results[variant] = summary(report)

    print("Log-loss gain over market (positive = better) and pooled flat-stake ROI:")
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
