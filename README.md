# Ice Hockey Betting Model — Qminers Quant Hackathon 2025

A betting agent for NHL ice hockey, built for the
[Qminers Quant Hackathon 2025](http://hyperion.felk.cvut.cz/). Each day it sees the latest
results and the bookmaker's odds, estimates win probabilities and decides how much to stake.
The hard part is finding games where the bookmaker is wrong by more than its ~17 % margin.

Gradient boosting that learns a correction on top of the bookmaker's probabilities beats the
market's log-loss in walk-forward validation and grows the bankroll by 16 % over 20+ seasons.

**Stack:** Python 3.12, pandas, NumPy, scikit-learn, XGBoost

## Approach

Everything is in [`src/model.py`](src/model.py) (the submission system accepts one file);
the maths is in [`docs/model_math.pdf`](docs/model_math.pdf).

1. **Features, built online.** Games are processed in date order and each game's features
   are saved before it updates any team state, so there is no look-ahead leakage.
   Features: the bookmaker's margin-free probability, Elo, Poisson attack/defence ratings,
   exponentially weighted team form (goals, shots, special teams, save %) and rest days.
2. **Model.** XGBoost starts from the market's log-odds as its base margin, so it only learns
   where the bookmaker is wrong. Refit every 150 games on the last 8 seasons.
3. **Staking.** 25 % fractional Kelly on bets with ≥5 % expected return, capped at 5 % of the
   bankroll per bet and 20 % per day.

## Results

Walk-forward validation ([`src/validate.py`](src/validate.py)): each season is predicted by
a model trained only on earlier seasons and compared with the bookmaker on log-loss.

| Variant | Log-loss gain vs market (all / 2002+) | Bets at ≥5 % edge | Flat-stake ROI | Final bankroll, all seasons | Final bankroll, 2007–2011 |
|---|---|---:|---:|---:|---:|
| `logreg` (market + Elo + goals) | −0.0008 / −0.0005 | 24 | −17.9 % | 1070.74 | 1000.00 |
| `logreg_all` (all features) | −0.0015 / −0.0044 | 333 | +0.8 % | 556.99 | 691.23 |
| `xgb_all` | +0.0012 / −0.0016 | 209 | +17.6 % | 1017.95 | 941.59 |
| **`xgb_market`** (default) | **+0.0023 / +0.0005** | 33 | +26.5 % | **1160.55** | 987.83 |

Bankrolls start at 1000 and come from the evaluation loop ([`src/evaluate.py`](src/evaluate.py)).

![Bankroll of the four variants over all training seasons. xgb_market ends at 1161, logreg at 1071, xgb_all at 1018 after peaking near 1650, and logreg_all at 557.](docs/bankroll.png)

**Takeaways**
- The bookmaker is a very strong baseline: only models built on top of its prediction beat it.
- More features made logistic regression overfit and lose ~45 % of its bankroll.
- `xgb_market` roughly broke even on 2007–2011, betting rarely. With so few bets the ROI is
  not statistically significant, so the edge is thin.

Design choices were tuned in [`src/experiments.py`](src/experiments.py) on 2005–06, with
2007–10 kept as a holdout.

## Running it

```bash
conda env create -f qqh-2025-env.yml && conda activate qqh-2025   # or: uv sync
cd src
python validate.py                              # walk-forward comparison of all variants
python evaluate.py xgb_all --from-season 2007   # one variant through the evaluation loop
python experiments.py                           # design experiments
python plot_bankroll.py                         # regenerate the chart
```

## Repository structure

| Path | Description | Author |
|---|---|---|
| [`src/model.py`](src/model.py) | Features, models and staking | me |
| [`src/validate.py`](src/validate.py), [`src/experiments.py`](src/experiments.py), [`src/plot_bankroll.py`](src/plot_bankroll.py) | Validation, experiments, chart | me |
| [`docs/model_math.pdf`](docs/model_math.pdf) | Maths behind the model | me |
| [`src/evaluate.py`](src/evaluate.py) | Evaluation runner | organisers, extended by me |
| [`src/environment.py`](src/environment.py), [`data/games.csv`](data/games.csv), [`problem_info_en.md`](problem_info_en.md) | Evaluation loop, data (1989–2011), problem statement (my translation) | organisers |

## License

[MIT](LICENSE), covering my own code. The evaluation environment, data and problem
statement belong to the hackathon organisers.
