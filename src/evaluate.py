"""Run a model variant through the evaluation loop on the training data.

Usage: python src/evaluate.py [variant] [--from-season YEAR]

  variant         one of model.VARIANTS (default: model.DEFAULT_VARIANT)
  --from-season   hand all earlier games to the model as the first increment and only
                  bet from that season on, like the submission system does with the
                  hidden seasons (e.g. --from-season 2007 scores the last 4 seasons)
"""

import argparse
from pathlib import Path

import pandas as pd

from model import DEFAULT_VARIANT, VARIANTS, Model
from environment import Environment

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "games.csv"

parser = argparse.ArgumentParser()
parser.add_argument("variant", nargs="?", default=DEFAULT_VARIANT, choices=list(VARIANTS))
parser.add_argument("--from-season", type=int)
args = parser.parse_args()

games = pd.read_csv(DATA_PATH, index_col=0, parse_dates=["Date", "Open"])
start = games.loc[games["Season"] == args.from_season, "Open"].min() if args.from_season else None

env = Environment(games, Model(args.variant), start_date=start, init_bankroll=1000, min_bet=5, max_bet=100)

evaluation = env.run()

print()
print(f"Final bankroll ({args.variant}): {env.bankroll:.2f}")

history = env.get_history()
