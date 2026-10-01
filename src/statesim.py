"""
Exact scoreline distribution under the state-dependent in-play model.

Monte Carlo would work here, but it is both slower and noisier than necessary. The match
state (home goals, away goals) lives on a small grid, so we can propagate the full
probability distribution forward one minute at a time and get the answer exactly.

    P_t(h, a)  ->  P_{t+1}(h, a)

At each minute the two scoring rates depend on where the match currently stands, which is
precisely what makes this process non-Poisson and what the static model cannot represent.
A team a goal up scores at 0.88x its baseline; the same team level scores at 1.02x. Since
the rate depends on the state, and the state depends on the goals already scored, the
final scoreline is no longer a bivariate Poisson draw.

There is no tau correction here. Dixon-Coles introduced tau as a patch for the fact that
independent Poissons misprice 0-0, 1-0, 0-1 and 1-1. The hypothesis this module tests is
that score-state dependence is the mechanism tau was approximating - if so, modelling the
mechanism directly should do at least as well without the patch.

Extra time reuses the same machinery, continuing the recursion from minute 91 to 120 at
the intensity of the closing stage of a match, with the score state carried over.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inplay import N_BUCKETS, STATES, SIDX, _bucket

MAXG = 9


def _state_idx_grid(G: int) -> tuple[np.ndarray, np.ndarray]:
    """For every (h,a) cell, the index into state_mult for home and for away."""
    h = np.arange(G)[:, None]
    a = np.arange(G)[None, :]
    dh = np.clip(h - a, -2, 2)
    da = np.clip(a - h, -2, 2)
    ih = np.vectorize(SIDX.get)(dh)
    ia = np.vectorize(SIDX.get)(da)
    return ih, ia


def score_distribution(lam: float, kap: float, time_mult: np.ndarray,
                       state_mult: np.ndarray, minutes: int = 90,
                       et_mult: float | None = None,
                       P0: np.ndarray | None = None) -> np.ndarray:
    """
    Full joint distribution over (home goals, away goals) after `minutes`.

    lam, kap are 90-minute baseline rates. Set et_mult to run minutes 91-120 at a fixed
    intensity, starting from P0.
    """
    G = MAXG + 1
    ih, ia = _state_idx_grid(G)
    P = np.zeros((G, G)) if P0 is None else P0.copy()
    if P0 is None:
        P[0, 0] = 1.0

    lh_base = lam / 90.0
    la_base = kap / 90.0

    for t in range(1, minutes + 1):
        tm = et_mult if et_mult is not None else time_mult[_bucket(t)]
        rh = lh_base * tm * state_mult[ih]
        ra = la_base * tm * state_mult[ia]
        ph = 1.0 - np.exp(-rh)          # P(home scores this minute)
        pa = 1.0 - np.exp(-ra)
        stay = P * (1 - ph) * (1 - pa)
        gh = P * ph * (1 - pa)
        ga = P * (1 - ph) * pa
        gb = P * ph * pa                # both score in the same minute (rare)
        Q = stay
        Q[1:, :] += gh[:-1, :]
        Q[:, 1:] += ga[:, :-1]
        Q[1:, 1:] += gb[:-1, :-1]
        # absorb probability that would leave the grid
        Q[-1, :] += gh[-1, :]
        Q[:, -1] += ga[:, -1]
        P = Q
    return P / P.sum()


def proba_1x2(P: np.ndarray) -> np.ndarray:
    return np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])


def knockout(lam: float, kap: float, time_mult: np.ndarray, state_mult: np.ndarray,
             elo_diff: float, so_b0: float, so_b1: float) -> dict:
    """Full cascade: 90 minutes -> extra time -> shootout, all exact."""
    from src.inplay import et_multiplier
    P90 = score_distribution(lam, kap, time_mult, state_mult, 90)
    p_home_90 = float(np.tril(P90, -1).sum())
    p_draw_90 = float(np.trace(P90))
    p_away_90 = float(np.triu(P90, 1).sum())

    # carry the level states into extra time, renormalised
    Plevel = np.zeros_like(P90)
    np.fill_diagonal(Plevel, np.diag(P90))
    if Plevel.sum() > 0:
        Plevel = Plevel / Plevel.sum()
    etm = et_multiplier(time_mult)
    P120 = score_distribution(lam, kap, time_mult, state_mult, 30,
                              et_mult=etm, P0=Plevel)
    p_h_et = float(np.tril(P120, -1).sum())
    p_a_et = float(np.triu(P120, 1).sum())
    p_lvl_et = float(np.trace(P120))

    p_home_so = 1.0 / (1.0 + np.exp(-(so_b0 + so_b1 * elo_diff / 100.0)))
    adv = p_home_90 + p_draw_90 * (p_h_et + p_lvl_et * p_home_so)
    return {
        "p_home_90": p_home_90, "p_draw_90": p_draw_90, "p_away_90": p_away_90,
        "p_home_adv": float(adv), "p_away_adv": float(1 - adv),
        "p_reach_et": p_draw_90, "p_reach_pens": p_draw_90 * p_lvl_et,
        "p_home_shootout": float(p_home_so),
        "P90": P90, "lam": lam, "kap": kap,
    }
