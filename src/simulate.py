"""
Monte Carlo match simulator: 90 minutes -> extra time -> penalties.

A 1X2 probability is not what a knockout match needs. "Draw" is not an outcome in a
semi-final; somebody walks off having won. So we simulate the whole cascade and report
the thing that actually exists: P(reaches the final).

Three stages, each calibrated on data rather than assumed:

  90 MINUTES   Scoreline drawn from the Dixon-Coles score matrix (bivariate Poisson with
               the low-score tau correction). Neutral venue, so no home advantage.

  EXTRA TIME   Only if level. Goal rates scaled by (30/90) * 1.087, where the damping
               factor is measured from the true population of extra-time matches - which
               crucially includes the ones that went to penalties without scoring. Left
               out, those matches would have made extra time look like a goal-fest.
               Independent Poisson here, not the tau correction: tau is estimated on
               90-minute scorelines and there is no warrant for reusing it on a 30-minute
               increment.

  SHOOTOUT     Only if still level. Not a coin flip: on 681 historical shootouts the
               stronger side on Elo wins 53.7%, and the logistic coefficient on the Elo
               gap is significant (z = 2.89). Small, real, and we model it. Who shoots
               first is a coin toss, so any first-mover advantage integrates out of a
               forecast and is correctly ignored.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, N_SIMS, SEED

ET_FRACTION = 30.0 / 90.0


def fit_shootout_model(df: pd.DataFrame, shootouts: pd.DataFrame):
    """Logistic: P(home wins shootout) = sigmoid(b0 + b1 * elo_diff/100)."""
    from scipy.optimize import minimize
    K = ["date", "home_team", "away_team"]
    so = shootouts.merge(df[K + ["elo_diff"]], on=K, how="inner")
    so = so[so.elo_diff.notna()]
    X = so.elo_diff.to_numpy() / 100.0
    Y = (so.winner == so.home_team).to_numpy().astype(float)

    def nll(t):
        p = np.clip(1 / (1 + np.exp(-(t[0] + t[1] * X))), 1e-9, 1 - 1e-9)
        return -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    r = minimize(nll, [0.0, 0.0], method="BFGS")
    return float(r.x[0]), float(r.x[1]), len(so)


def simulate(lam: float, kap: float, score_matrix: np.ndarray, elo_diff: float,
             so_b0: float, so_b1: float, et_damp: float,
             n_sims: int = N_SIMS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    G = score_matrix.shape[0]

    # --- 90 minutes: sample scorelines from the flattened DC matrix ---
    flat = score_matrix.ravel() / score_matrix.sum()
    idx = rng.choice(flat.size, size=n_sims, p=flat)
    h90, a90 = np.divmod(idx, G)

    level = h90 == a90
    home_win = h90 > a90
    away_win = h90 < a90

    # --- extra time, for the level ones only ---
    n_et = int(level.sum())
    lam_et = lam * ET_FRACTION * et_damp
    kap_et = kap * ET_FRACTION * et_damp
    h_et = rng.poisson(lam_et, n_et)
    a_et = rng.poisson(kap_et, n_et)

    adv = home_win.copy()                      # True = home advances
    et_ix = np.flatnonzero(level)
    et_home = h_et > a_et
    et_away = h_et < a_et
    still_level = h_et == a_et
    adv[et_ix[et_home]] = True
    adv[et_ix[et_away]] = False

    # --- shootout, for whoever is still level after 120 ---
    n_so = int(still_level.sum())
    p_home_so = 1.0 / (1.0 + np.exp(-(so_b0 + so_b1 * elo_diff / 100.0)))
    so_home = rng.random(n_so) < p_home_so
    adv[et_ix[np.flatnonzero(still_level)]] = so_home

    return {
        "p_home_90": float(home_win.mean()),
        "p_draw_90": float(level.mean()),
        "p_away_90": float(away_win.mean()),
        "p_home_adv": float(adv.mean()),
        "p_away_adv": float(1 - adv.mean()),
        "p_reach_et": float(level.mean()),
        "p_reach_pens": float(level.mean() * (still_level.mean() if n_et else 0.0)),
        "p_home_shootout": float(p_home_so),
        "exp_goals_home": float(h90.mean()),
        "exp_goals_away": float(a90.mean()),
        "score_matrix": score_matrix,
        "lam": lam, "kap": kap,
    }


def top_scorelines(M: np.ndarray, k: int = 6):
    flat = [(M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
    flat.sort(reverse=True)
    return [(f"{i}-{j}", p) for p, i, j in flat[:k]]
