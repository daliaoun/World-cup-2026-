"""
Gradient-boosted goal model (XGBoost, Poisson objective).

Design notes, because the shape of this model is the whole point of including it:

LONG FORMAT. Each match becomes two rows, one per team's point of view:
(own features, opponent features, is_home) -> goals scored by that team. This doubles
the training data and, more importantly, forces the model to be symmetric: it learns
one function "how many goals does a team like this score against an opponent like
that", rather than two unrelated home- and away-goal functions that could disagree
with each other. Dixon-Coles gets this symmetry for free from its structure; XGBoost
has to be handed it.

POISSON OBJECTIVE. We predict a goal RATE, not a win probability. This matters: it
means the GBM's output is the same kind of object as Dixon-Coles' output, so the two
can be fed into the identical score-matrix -> extra-time -> shootout simulator, and
can be blended. A softmax 1X2 classifier would have been easier and useless downstream.

RHO. Independent Poissons famously misprice 0-0, 1-0, 0-1 and 1-1. We re-use the
Dixon-Coles tau correction, fitting the single parameter rho by maximum likelihood on
the GBM's own rates, so the GBM is not handicapped by a flaw Dixon-Coles has already
solved.

REFIT CADENCE. Dixon-Coles must be refitted at every match date, because its parameters
ARE the team ratings - they go stale immediately. XGBoost's parameters are a mapping
from features to goals, which is stable; the freshness lives in the features (Elo, form,
rest, current-tournament goals), and those update after every single match. So refitting
the GBM once a year is not a shortcut that flatters it - the information it sees at
prediction time is exactly as current as Dixon-Coles'. It is refit on an expanding
window, strictly on the past.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC
from src.dixon_coles import MAX_GOALS

# features from the perspective of "own" team vs "opp" team
LONG_FEATURES = [
    "own_elo", "opp_elo", "elo_adv",
    "own_gf", "own_ga", "own_pts",
    "opp_gf", "opp_ga", "opp_pts",
    "own_rest", "opp_rest", "rest_adv",
    "own_last_et", "opp_last_et",
    "own_t_gfpg", "own_t_gapg", "opp_t_gfpg", "opp_t_gapg",
    "own_t_n", "opp_t_n",
    "h2h_gd", "h2h_n",
    "is_home", "weight_imp",
]

PARAMS = dict(
    objective="count:poisson",
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=2.0,
    n_estimators=450,
    max_delta_step=0.7,     # xgboost's own recommendation for Poisson stability
    tree_method="hist",
    n_jobs=1,
    random_state=7,
)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows."""
    home = pd.DataFrame({
        "date": df.date, "match_ix": df.index, "is_home": df.home_adv,
        "own_elo": df.elo_home, "opp_elo": df.elo_away, "elo_adv": df.elo_diff,
        "own_gf": df.h_gf, "own_ga": df.h_ga, "own_pts": df.h_pts,
        "opp_gf": df.a_gf, "opp_ga": df.a_ga, "opp_pts": df.a_pts,
        "own_rest": df.h_rest, "opp_rest": df.a_rest, "rest_adv": df.rest_diff,
        "own_last_et": df.h_last_et, "opp_last_et": df.a_last_et,
        "own_t_gfpg": df.h_t_gfpg, "own_t_gapg": df.h_t_gapg,
        "opp_t_gfpg": df.a_t_gfpg, "opp_t_gapg": df.a_t_gapg,
        "own_t_n": df.h_t_n, "opp_t_n": df.a_t_n,
        "h2h_gd": df.h2h_gd, "h2h_n": df.h2h_n,
        "weight_imp": df.weight_imp, "y": df.hs, "side": "H",
    })
    away = pd.DataFrame({
        "date": df.date, "match_ix": df.index, "is_home": 0.0,
        "own_elo": df.elo_away, "opp_elo": df.elo_home, "elo_adv": -df.elo_diff,
        "own_gf": df.a_gf, "own_ga": df.a_ga, "own_pts": df.a_pts,
        "opp_gf": df.h_gf, "opp_ga": df.h_ga, "opp_pts": df.h_pts,
        "own_rest": df.a_rest, "opp_rest": df.h_rest, "rest_adv": -df.rest_diff,
        "own_last_et": df.a_last_et, "opp_last_et": df.h_last_et,
        "own_t_gfpg": df.a_t_gfpg, "own_t_gapg": df.a_t_gapg,
        "opp_t_gfpg": df.h_t_gfpg, "opp_t_gapg": df.h_t_gapg,
        "own_t_n": df.a_t_n, "opp_t_n": df.h_t_n,
        "h2h_gd": -df.h2h_gd, "h2h_n": df.h2h_n,
        "weight_imp": df.weight_imp, "y": df.as_, "side": "A",
    })
    # away rows carry is_home=0; home rows carry the match's actual home_adv
    # (0 at a neutral venue, 1 at a true home ground) so the model learns the
    # size of home advantage rather than having it assumed.
    return pd.concat([home, away], ignore_index=True)


def _tau(h, a, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((h == 0) & (a == 0), 1 - lam * mu * rho, t)
    t = np.where((h == 0) & (a == 1), 1 + lam * rho, t)
    t = np.where((h == 1) & (a == 0), 1 + mu * rho, t)
    t = np.where((h == 1) & (a == 1), 1 - rho, t)
    return t


def fit_rho(hs, as_, lam, mu) -> float:
    """MLE for the low-score correlation parameter, given fixed GBM rates."""
    def nll(rho):
        t = _tau(hs, as_, lam, mu, rho)
        if np.any(t <= 1e-9):
            return 1e9
        return -np.sum(np.log(t))
    r = minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded")
    return float(r.x)


class GBMGoals:
    """Poisson GBM over the long format, plus a Dixon-Coles rho correction."""

    def __init__(self, xi: float = 0.0015, **kw):
        self.xi = xi
        self.params = {**PARAMS, **kw}
        self.model_: xgb.XGBRegressor | None = None
        self.rho_: float = 0.0

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp) -> "GBMGoals":
        tr = df[(df.date < as_of) & df.hs.notna() & df.elo_home.notna()]
        L = to_long(tr).dropna(subset=["y"])
        days = (as_of - L.date).dt.days.values.astype(float)
        w = np.exp(-self.xi * days) * L.weight_imp.values

        self.model_ = xgb.XGBRegressor(**self.params)
        self.model_.fit(L[LONG_FEATURES], L.y.values, sample_weight=w, verbose=False)

        # fit rho on the most recent slice of training data, using the model's own rates
        recent = tr[tr.date >= as_of - pd.DateOffset(years=8)]
        if len(recent) > 200:
            lam, mu = self.rates(recent)
            self.rho_ = fit_rho(recent.hs.values, recent.as_.values, lam, mu)
        return self

    def rates(self, df: pd.DataFrame):
        L = to_long(df)
        p = self.model_.predict(L[LONG_FEATURES])
        n = len(df)
        return np.clip(p[:n], 0.05, 6.0), np.clip(p[n:], 0.05, 6.0)

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        M[0, 0] *= 1 - lam * mu * self.rho_
        M[0, 1] *= 1 + lam * self.rho_
        M[1, 0] *= 1 + mu * self.rho_
        M[1, 1] *= 1 - self.rho_
        return M / M.sum()

    def proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        lam, mu = self.rates(df)
        out = np.zeros((len(df), 3))
        for i in range(len(df)):
            M = self.score_matrix(lam[i], mu[i])
            out[i] = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
        return out
