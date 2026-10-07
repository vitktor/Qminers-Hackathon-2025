"""Walk-forward experiments on the outcome model, scored against the bookmaker.

Each test season, a model is fit only on earlier seasons. Settings are chosen on the
development seasons; the holdout seasons are only looked at afterwards, as a final check.

  1. era      - train only on the no-draw era (2002+), or down-weight old seasons
  2. shrink   - pull the model's log-odds toward the market's
  3. calib    - predicted edge vs realised return of the bets it would place
  4. features - does each feature add anything beyond the market price?

Usage: python src/experiments.py [era|shrink|calib|features ...]   (default: all)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import ALL_FEATURES, MARKET_COL, VARIANTS, FeatureBuilder, OutcomeModel
from validate import flat_roi, log_loss

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "games.csv"
DEV = [2005, 2006]
HOLDOUT = [2007, 2008, 2009, 2010]
EDGE = 0.05


def load() -> tuple[FeatureBuilder, pd.DataFrame, np.ndarray]:
    games = pd.read_csv(DATA_PATH, index_col=0, parse_dates=["Date", "Open"])
    builder = FeatureBuilder()
    builder.ingest(games)
    rows = pd.DataFrame(builder.rows)
    return builder, rows, np.stack(rows.x)


def predictions(config: dict, builder: FeatureBuilder, rows: pd.DataFrame, X: np.ndarray) -> pd.Series:
    """Walk-forward P(home | no draw) for every DEV and HOLDOUT game."""
    out = []
    for season in DEV + HOLDOUT:
        model = OutcomeModel(**config)
        model.maybe_refit(builder.rows[: int((rows.season < season).sum())])
        test = rows[rows.season == season]
        out.append(pd.Series(model.predict(X[test.index]), index=test.index))
    return pd.concat(out)


def score(p: pd.Series, rows: pd.DataFrame, X: np.ndarray) -> dict:
    line = {}
    for name, seasons in (("dev", DEV), ("hold", HOLDOUT)):
        test = rows.loc[p.index][rows.loc[p.index].season.isin(seasons)]
        y = (test.result == "H").values
        p_market = 1 / (1 + np.exp(-X[test.index, MARKET_COL]))
        line[f"ll_gain_{name}"] = 1000 * (log_loss(p_market, y) - log_loss(p[test.index].values, y))
        line[f"bets_{name}"], line[f"roi_{name}"] = flat_roi(p[test.index].values, test, EDGE)
    return line


def compare(configs: dict, data) -> pd.DataFrame:
    return pd.DataFrame({name: score(predictions(c, *data), data[1], data[2])
                         for name, c in configs.items()}).T


def era(data) -> None:
    options = {"last 8 seasons": {}, "no-draw era only (2002+)": {"first_season": 2002},
               "season decay 0.8": {"season_decay": 0.8}, "season decay 0.6": {"season_decay": 0.6}}
    configs = {f"{variant}: {name}": {**VARIANTS[variant], **option}
               for variant in ("xgb_market", "logreg") for name, option in options.items()}
    print_table("1. Training era", compare(configs, data))


def shrink(data) -> None:
    base = VARIANTS["xgb_market"]
    configs = {f"shrink {s}": {**base, "shrink": s} for s in (1.0, 0.75, 0.5, 0.25)}
    print_table("2. Shrink toward the market", compare(configs, data))


def calib(data) -> None:
    builder, rows, X = data
    p = predictions(VARIANTS["xgb_market"], *data)
    test = rows.loc[p.index]
    p_h = p.values * (1 - test.p_draw)
    p_a = (1 - p.values) * (1 - test.p_draw)
    edge_h, edge_a = p_h * test.odds_h - 1, p_a * test.odds_a - 1
    home = edge_h >= edge_a
    bets = pd.DataFrame({
        "edge": np.where(home, edge_h, edge_a),
        "ret": np.where(home, (test.result == "H") * test.odds_h, (test.result == "A") * test.odds_a) - 1,
    })
    bins = [-1, -0.15, -0.1, -0.05, 0, 0.05, 1]
    table = bets.groupby(pd.cut(bets.edge, bins), observed=True).agg(
        n=("ret", "size"), predicted=("edge", "mean"), realised=("ret", "mean"))
    print_table("3. Calibration: best side of each game, 2005-2010 (xgb_market)", table)


def features(data) -> None:
    configs = {"market only": dict(features=["market_logit"], classifier="logreg")}
    for f in ALL_FEATURES[1:]:
        configs[f"market + {f}"] = dict(features=["market_logit", f], classifier="logreg")
    table = compare(configs, data)[["ll_gain_dev", "ll_gain_hold"]]
    table = table - table.loc["market only"]
    print_table("4. Log-loss gain (x1000) of each feature added to the market, vs market alone",
                table.sort_values("ll_gain_dev", ascending=False))

    # Compact models from the features that helped most on the dev seasons only.
    top = list(table.drop("market only").ll_gain_dev.nlargest(5).index.str.removeprefix("market + "))
    configs = {f"logreg: market + top {k}": dict(features=["market_logit"] + top[:k], classifier="logreg")
               for k in (2, 3, 5)}
    configs["xgb_market: top 5"] = {**VARIANTS["xgb_market"], "features": top}
    print_table(f"4b. Compact models (dev top 5: {', '.join(top)})", compare(configs, data))


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n{table.round(3).to_string()}")


if __name__ == "__main__":
    experiments = {"era": era, "shrink": shrink, "calib": calib, "features": features}
    data = load()
    print(f"dev seasons {DEV}, holdout seasons {HOLDOUT}; ll_gain is x1000, "
          f"positive = better than market; bets/roi are flat stakes at edge >= {EDGE}")
    for name in sys.argv[1:] or experiments:
        experiments[name](data)
