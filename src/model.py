"""Hockey betting model for the Qminers Quant Hackathon 2025.

Pipeline (runs inside every `Model.place_bets` call):

1. `FeatureBuilder` ingests newly finished games strictly in date order. Before a game
   updates the team state, its pre-game feature vector is recorded, so every training
   row only contains information that was available when the game was bet on.
2. `OutcomeModel` maps features -> P(home win | no draw). The bookmaker's own probability
   is one of the features, so the model only has to learn where the market is wrong. It
   is refit whenever enough new games have been recorded.
3. `KellyStaker` bets a fraction of the bankroll on outcomes whose expected value
   clears a margin that is large compared with the bookmaker's overround (~17 %).

Several model variants are available, see `VARIANTS`; `Model()` uses `DEFAULT_VARIANT`.

The whole file is self-contained because the submission system accepts a single module.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------

def has_odds(odds_h: float, odds_a: float) -> bool:
    return odds_h > 1.0 and odds_a > 1.0


def market_probs(odds_h: float, odds_a: float, odds_d: float) -> tuple[float, float, float]:
    """De-vigged (normalised) bookmaker probabilities for home / away / draw."""
    inv = np.array([1.0 / odds_h, 1.0 / odds_a, 1.0 / odds_d if odds_d > 1.0 else 0.0])
    p_h, p_a, p_d = inv / inv.sum()
    return p_h, p_a, p_d


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


# ---------------------------------------------------------------------------
# Team strength ratings (all updated online, one finished game at a time)
# ---------------------------------------------------------------------------

@dataclass
class Elo:
    """Elo with home advantage, margin-of-victory scaling and between-season regression."""

    k: float = 8.0
    home_adv: float = 60.0
    season_carry: float = 0.9
    base: float = 1500.0
    ratings: dict = field(init=False)

    def __post_init__(self) -> None:
        self.ratings = defaultdict(lambda: self.base)

    def expected_home(self, hid: int, aid: int) -> float:
        diff = self.ratings[hid] + self.home_adv - self.ratings[aid]
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, hid: int, aid: int, home_score: float, goal_diff: int) -> None:
        mov = np.log1p(abs(goal_diff)) if goal_diff else 1.0
        delta = self.k * mov * (home_score - self.expected_home(hid, aid))
        self.ratings[hid] += delta
        self.ratings[aid] -= delta

    def new_season(self) -> None:
        for team, rating in self.ratings.items():
            self.ratings[team] = self.base + self.season_carry * (rating - self.base)


@dataclass
class GoalRatings:
    """Online Poisson attack/defence ratings: E[goals] = exp(mu + home + att - def)."""

    lr: float = 0.02
    mu: float = float(np.log(3.0))
    home: float = 0.1
    attack: dict = field(default_factory=lambda: defaultdict(float))
    defence: dict = field(default_factory=lambda: defaultdict(float))

    def expected_goals(self, hid: int, aid: int) -> tuple[float, float]:
        lam_h = np.exp(self.mu + self.home + self.attack[hid] - self.defence[aid])
        lam_a = np.exp(self.mu + self.attack[aid] - self.defence[hid])
        return lam_h, lam_a

    def update(self, hid: int, aid: int, goals_h: int, goals_a: int) -> None:
        lam_h, lam_a = self.expected_goals(hid, aid)
        err_h, err_a = goals_h - lam_h, goals_a - lam_a
        self.attack[hid] += self.lr * err_h
        self.defence[aid] -= self.lr * err_h
        self.attack[aid] += self.lr * err_a
        self.defence[hid] -= self.lr * err_a
        self.mu += 0.1 * self.lr * (err_h + err_a) / 2
        self.home += 0.1 * self.lr * (err_h - err_a) / 2


# Box-score stats tracked per team: goals for/against, shots on goal for/against, wins,
# power-play goals for/against, minor penalties and goalie save percentage.
FORM_STATS = ["gf", "ga", "sf", "sa", "win", "ppg", "ppga", "pen", "svp"]


@dataclass
class TeamForm:
    """Exponentially weighted per-team stats at a fast and a slow speed, plus schedule info."""

    fast: float = 0.15  # ~ last 6-7 games, forgotten between seasons
    slow: float = 0.04  # ~ last 25 games, carried over
    avg: dict = field(default_factory=lambda: defaultdict(dict))  # team -> {"stat_speed": value}
    last_date: dict = field(default_factory=dict)
    record: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # season W, L

    def get(self, team: int, stat: str, speed: str) -> float:
        return self.avg[team].get(f"{stat}_{speed}", np.nan)

    def update(self, team: int, date: pd.Timestamp, stats: dict) -> None:
        for speed, alpha in (("f", self.fast), ("s", self.slow)):
            for stat, value in stats.items():
                if np.isnan(value):
                    continue
                key = f"{stat}_{speed}"
                old = self.avg[team].get(key)
                self.avg[team][key] = value if old is None else (1 - alpha) * old + alpha * value
        self.last_date[team] = date
        win = stats["win"]
        if win != 0.5:
            self.record[team][0 if win == 1 else 1] += 1

    def new_season(self) -> None:
        self.record.clear()
        for stats in self.avg.values():
            for key in [k for k in stats if k.endswith("_f")]:
                del stats[key]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

BASE_FEATURES = ["market_logit", "elo_logit", "goal_diff"]
ALL_FEATURES = (BASE_FEATURES
                + ["overround", "rest_h", "b2b_h", "winpct_h", "rest_a", "b2b_a", "winpct_a"]
                + [f"d_{stat}_{speed}" for stat in FORM_STATS for speed in ("f", "s")])


class FeatureBuilder:
    """Keeps team state and produces leakage-free feature vectors (ordered as ALL_FEATURES)."""

    def __init__(self) -> None:
        self.elo = Elo()
        self.goals = GoalRatings()
        self.form = TeamForm()
        self.season: Optional[int] = None
        self.rows: list[dict] = []  # pre-game features + outcome of every finished game

    def features(self, hid: int, aid: int, date: pd.Timestamp,
                 odds_h: float, odds_a: float, odds_d: float) -> np.ndarray:
        p_h, p_a, _ = market_probs(odds_h, odds_a, odds_d)
        lam_h, lam_a = self.goals.expected_goals(hid, aid)
        f = [logit(p_h / (p_h + p_a)),
             logit(self.elo.expected_home(hid, aid)),
             lam_h - lam_a,
             1.0 / odds_h + 1.0 / odds_a + (1.0 / odds_d if odds_d > 1.0 else 0.0)]
        for team in (hid, aid):
            last = self.form.last_date.get(team)
            rest = (date - last).days if last is not None else 10
            wins, losses = self.form.record[team]
            f += [min(rest, 10), float(rest <= 1), (wins + 2) / (wins + losses + 4)]
        for stat in FORM_STATS:
            for speed in ("f", "s"):
                f.append(self.form.get(hid, stat, speed) - self.form.get(aid, stat, speed))
        return np.nan_to_num(np.array(f))  # teams without history: no difference

    def ingest(self, games: pd.DataFrame) -> None:
        """Record pre-game features, then update team state, day by day."""
        games = games.dropna(subset=["HID", "AID", "HS", "AS"])
        for _, day in games.sort_values("Date").groupby("Date", sort=False):
            for g in day.itertuples():
                if self.season is not None and g.Season != self.season:
                    self.elo.new_season()
                    self.form.new_season()
                self.season = g.Season
                if has_odds(g.OddsH, g.OddsA):
                    x = self.features(g.HID, g.AID, g.Date, g.OddsH, g.OddsA, g.OddsD)
                    result = "H" if g.H else "A" if g.A else "D"
                    self.rows.append({"x": x, "result": result, "season": g.Season,
                                      "odds_h": g.OddsH, "odds_a": g.OddsA,
                                      "p_draw": market_probs(g.OddsH, g.OddsA, g.OddsD)[2]})
            # Update only after the whole day is recorded: same-day games are unknown at bet time.
            for g in day.itertuples():
                self._update(g)

    def _update(self, g) -> None:
        home_score = 1.0 if g.H else 0.0 if g.A else 0.5
        self.elo.update(g.HID, g.AID, home_score, int(g.HS - g.AS))
        self.goals.update(g.HID, g.AID, int(g.HS), int(g.AS))
        # Shots on goal are only recorded since 2009; rebuild them as opponent saves + own
        # goals (shoot-out goals excluded), which is available for every season.
        so_h = 0 if pd.isna(g.H_SO) else g.H_SO
        so_a = 0 if pd.isna(g.A_SO) else g.A_SO
        shots_h, shots_a = g.A_SV + g.HS - so_h, g.H_SV + g.AS - so_a
        for team, date, gf, ga, sf, sa, ppg, ppga, pen, saves in (
                (g.HID, g.Date, g.HS, g.AS, shots_h, shots_a, g.H_PPG, g.A_PPG, g.H_PEN, g.H_SV),
                (g.AID, g.Date, g.AS, g.HS, shots_a, shots_h, g.A_PPG, g.H_PPG, g.A_PEN, g.A_SV)):
            win = 1.0 if gf > ga else 0.0 if gf < ga else 0.5
            svp = saves / (saves + ga) if saves + ga > 0 else np.nan
            self.form.update(team, date, dict(gf=gf, ga=ga, sf=sf, sa=sa, win=win,
                                              ppg=ppg, ppga=ppga, pen=pen, svp=svp))


# ---------------------------------------------------------------------------
# Probability model
# ---------------------------------------------------------------------------

XGB_PARAMS = dict(n_estimators=300, max_depth=1, learning_rate=0.02, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  n_jobs=1, verbosity=0, random_state=0)

# Model variants, with final bankroll from 1000 in src/evaluate.py (all seasons /
# --from-season 2007). Not betting at all scores 1000 / 1000; "xgb_market" is the default
# because it scored best over all seasons and bets rarely in the recent ones.
VARIANTS = {
    # Logistic regression on market, Elo and goal ratings.
    "logreg": dict(features=BASE_FEATURES, classifier="logreg", C=1.0),
    # Logistic regression on all features; strongly regularised, still overfits.
    "logreg_all": dict(features=ALL_FEATURES, classifier="logreg", C=0.05),
    # Depth-1 gradient boosting on all features.
    "xgb_all": dict(features=ALL_FEATURES, classifier="xgb", n_estimators=300),
    # Same, but boosting starts from the market log-odds and only learns a correction to it.
    "xgb_market": dict(features=ALL_FEATURES, classifier="xgb_market", n_estimators=200),
    # Market plus the five features that each improved on the market most on the 2005-06
    # seasons (src/experiments.py): shots against, season win rate, away back-to-back and
    # save percentage. Beats the market log-loss on later seasons too, but bets more and
    # loses: 924 / 895.
    "logreg_compact": dict(features=["market_logit", "d_sa_f", "d_sa_s", "d_win_s", "b2b_a", "d_svp_s"],
                           classifier="logreg"),
}
DEFAULT_VARIANT = "xgb_market"
MARKET_COL = ALL_FEATURES.index("market_logit")


class OutcomeModel:
    """Classifier for P(home win | no draw), refit on recent recorded games."""

    def __init__(self, features: list[str] = BASE_FEATURES, classifier: str = "logreg",
                 train_seasons: int = 8, min_train_games: int = 1000,
                 refit_every: int = 150, first_season: Optional[int] = None,
                 season_decay: float = 1.0, shrink: float = 1.0, **params) -> None:
        self.cols = [ALL_FEATURES.index(f) for f in features]
        self.classifier = classifier
        self.params = params
        self.first_season = first_season  # ignore older seasons, e.g. the era with draws
        self.season_decay = season_decay  # sample weight per season of age
        self.shrink = shrink  # 1 = raw model, 0 = market; applied in log-odds
        self.train_seasons = train_seasons
        self.min_train_games = min_train_games  # ~3 seasons; fewer makes the fit overconfident
        self.refit_every = refit_every
        self.clf = None
        self._rows_at_fit = 0

    @classmethod
    def from_variant(cls, name: str) -> "OutcomeModel":
        return cls(**VARIANTS[name])

    def maybe_refit(self, rows: list[dict]) -> None:
        if self.clf is not None and len(rows) - self._rows_at_fit < self.refit_every:
            return
        last_season = rows[-1]["season"] if rows else 0
        train = [r for r in rows
                 if r["season"] > last_season - self.train_seasons and r["result"] != "D"
                 and (self.first_season is None or r["season"] >= self.first_season)]
        if len(train) < self.min_train_games:
            return
        X = np.stack([r["x"] for r in train])
        y = np.array([r["result"] == "H" for r in train])
        w = self.season_decay ** np.array([last_season - r["season"] for r in train], dtype=float)
        if self.classifier == "logreg":
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **self.params))
            self.clf.fit(X[:, self.cols], y, logisticregression__sample_weight=w)
        else:
            self.clf = XGBClassifier(**{**XGB_PARAMS, **self.params})
            margin = X[:, MARKET_COL] if self.classifier == "xgb_market" else None
            self.clf.fit(X[:, self.cols], y, base_margin=margin, sample_weight=w)
        self._rows_at_fit = len(rows)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(home | no draw) for full feature rows (ordered as ALL_FEATURES)."""
        market = X[:, MARKET_COL]
        if self.classifier == "xgb_market":
            z = self.clf.predict(X[:, self.cols], output_margin=True, base_margin=market)
        else:
            p = np.clip(self.clf.predict_proba(X[:, self.cols])[:, 1], 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
        z = market + self.shrink * (z - market)
        return 1.0 / (1.0 + np.exp(-z))

    def prob_home(self, x: np.ndarray) -> Optional[float]:
        if self.clf is None:
            return None
        return float(self.predict(x.reshape(1, -1))[0])


# ---------------------------------------------------------------------------
# Staking
# ---------------------------------------------------------------------------

@dataclass
class KellyStaker:
    """Fractional Kelly on the single best outcome of a game, if its edge is big enough."""

    min_edge: float = 0.05        # required expected return per unit staked
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.05     # of bankroll, per game
    max_daily_fraction: float = 0.2    # of bankroll, per day

    def stake(self, prob: float, odds: float, bankroll: float) -> float:
        edge = prob * odds - 1.0
        if edge < self.min_edge:
            return 0.0
        kelly = edge / (odds - 1.0)
        return bankroll * min(self.kelly_fraction * kelly, self.max_bet_fraction)


# ---------------------------------------------------------------------------
# Submission entry point
# ---------------------------------------------------------------------------

BET_COLS = ["BetH", "BetA", "BetD"]


class Model:
    def __init__(self, variant: str = DEFAULT_VARIANT, staker: Optional[KellyStaker] = None,
                 outcome_model: Optional[OutcomeModel] = None) -> None:
        self.features = FeatureBuilder()
        self.outcome_model = outcome_model or OutcomeModel.from_variant(variant)
        self.staker = staker or KellyStaker()

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
        s = summary.iloc[0]
        bankroll, today = float(s["Bankroll"]), pd.Timestamp(s["Date"])
        min_bet, max_bet = float(s["Min_bet"]), float(s["Max_bet"])

        if not inc.empty:
            self.features.ingest(inc)
            self.outcome_model.maybe_refit(self.features.rows)

        bets = pd.DataFrame(0.0, index=opps.index, columns=BET_COLS)
        # Bet on the match day only: the ratings then include every earlier result.
        todays = opps[opps["Date"] == today]
        budget = bankroll * self.staker.max_daily_fraction

        for g in todays.itertuples():
            if not has_odds(g.OddsH, g.OddsA):
                continue
            x = self.features.features(g.HID, g.AID, g.Date, g.OddsH, g.OddsA, g.OddsD)
            p_home = self.outcome_model.prob_home(x)
            if p_home is None:
                continue
            p_draw = market_probs(g.OddsH, g.OddsA, g.OddsD)[2]
            candidates = {"BetH": (p_home * (1 - p_draw), g.OddsH),
                          "BetA": ((1 - p_home) * (1 - p_draw), g.OddsA)}
            col, (prob, odds) = max(candidates.items(), key=lambda kv: kv[1][0] * kv[1][1])
            stake = min(self.staker.stake(prob, odds, bankroll), max_bet, budget)
            if stake > 0 and stake >= min_bet:
                bets.at[g.Index, col] = round(stake, 2)
                budget -= stake

        return bets
