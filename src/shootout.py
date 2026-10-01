"""
A shootout model calibrated on shootouts.

The previous version predicted penalties from Elo alone, which is really a statement that
penalty ability does not exist as a separate thing. That is a strong claim to make by
omission, so here it is tested properly instead.

THE MODEL. Every team gets its own penalty rating, and shootouts are fitted directly:

    logit P(team A beats team B) = beta * (elo gap / 100)
                                 + theta_A - theta_B
                                 + gamma * (A shoots first)

theta is a team's penalty ability over and above its general quality. If Argentina really
are better at this than their overall strength implies, theta_Argentina is where it shows.

THE HONEST PART: SHRINKAGE. With 682 shootouts and ~120 nations, unconstrained team
ratings would fit noise perfectly - Argentina's 4-0 run would become "Argentina are
unbeatable on penalties" and the model would be worse for it. So the team ratings carry a
penalty term controlled by lambda, and lambda is chosen by cross-validation on held-out
shootouts. The data decides how much team identity matters:

    lambda small  -> team ratings survive, penalty ability is real and worth modelling
    lambda large  -> ratings collapse to zero, and the honest answer is that it is not

This is the difference between assuming there is no penalty skill and measuring how much
there is. The conclusion may be the same; the claim is not.

FIRST SHOOTER. Included because it is genuinely penalty-specific and measurable. It does
not affect a forecast - who shoots first is decided by a coin toss after the match ends -
but leaving it out would push its effect into the other coefficients.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, RAW


def build_design(so: pd.DataFrame, teams: list[str]):
    tix = {t: i for i, t in enumerate(teams)}
    n, k = len(so), len(teams)
    X_team = np.zeros((n, k))
    for r, row in enumerate(so.itertuples(index=False)):
        if row.home_team in tix:
            X_team[r, tix[row.home_team]] += 1.0
        if row.away_team in tix:
            X_team[r, tix[row.away_team]] -= 1.0
    elo = (so.elo_diff.to_numpy() / 100.0).reshape(-1, 1)
    first = np.zeros((n, 2))
    if "first_shooter" in so.columns:
        fs = so.first_shooter.fillna("")
        first[:, 0] = np.where(fs == so.home_team, 1.0,
                               np.where(fs == so.away_team, -1.0, 0.0))
    # home advantage gets its own column. Without it the intercept absorbs it and
    # the model quietly hands a crowd bonus to a team playing at a neutral venue.
    first[:, 1] = so.home_adv.to_numpy().astype(float)
    y = (so.winner == so.home_team).to_numpy().astype(float)
    return elo, first, X_team, y


def fit(elo, first, X_team, y, lam: float):
    """Penalised logistic regression. Only the team ratings are penalised."""
    k = X_team.shape[1]
    X = np.hstack([np.ones((len(y), 1)), elo, first, X_team])

    def nll(t):
        z = X @ t
        p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
        ll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return ll + 0.5 * lam * np.sum(t[4:] ** 2)

    def grad(t):
        z = X @ t
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y)
        g[4:] += lam * t[4:]
        return g

    r = minimize(nll, np.zeros(4 + k), jac=grad, method="L-BFGS-B",
                 options={"maxiter": 800})
    return r.x


def cv_lambda(elo, first, X_team, y, lams, folds=5, seed=3):
    """Choose the shrinkage by held-out log loss."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    out = []
    for lam in lams:
        ll = 0.0
        for f in range(folds):
            tr, te = fold != f, fold == f
            t = fit(elo[tr], first[tr], X_team[tr], y[tr], lam)
            X = np.hstack([np.ones((te.sum(), 1)), elo[te], first[te], X_team[te]])
            p = np.clip(1 / (1 + np.exp(-(X @ t))), 1e-9, 1 - 1e-9)
            ll += -np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        out.append(ll / len(y))
    return np.array(out)


def main(verbose=True):
    so = pd.read_csv(RAW / "shootouts.csv", parse_dates=["date"])
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    K = ["date", "home_team", "away_team"]
    so = so.merge(df[K + ["elo_diff", "home_adv"]], on=K, how="inner")
    so = so[so.elo_diff.notna()].sort_values("date").reset_index(drop=True)

    counts = pd.concat([so.home_team, so.away_team]).value_counts()
    teams = sorted(counts[counts >= 2].index.tolist())
    elo, first, X_team, y = build_design(so, teams)

    lams = np.array([0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000])
    cv = cv_lambda(elo, first, X_team, y, lams)
    best = float(lams[int(np.argmin(cv))])

    if verbose:
        print(f"  {len(so)} shootouts, {len(teams)} teams with 2 or more\n")
        print("  Cross-validated shrinkage (held-out log loss, lower is better):")
        for l, c in zip(lams, cv):
            mark = "  <- best" if l == best else ""
            bar = "#" * int((c - cv.min()) * 900)
            print(f"     lambda {l:>7.1f}   {c:.5f} {bar}{mark}")

    theta = fit(elo, first, X_team, y, best)
    # pure-Elo comparison: infinite shrinkage on team effects
    cv_elo = cv_lambda(elo, first, np.zeros_like(X_team), y, [0.0])[0]
    if verbose:
        print(f"\n  best model (lambda={best:g})       held-out log loss {cv.min():.5f}")
        print(f"  Elo only, no team ratings      held-out log loss {cv_elo:.5f}")
        gain = cv_elo - cv.min()
        print(f"  gain from team penalty ratings: {gain:+.5f}")
        print(f"     -> {'WORTH KEEPING' if gain > 0.002 else 'NOT WORTH KEEPING'}")
        print(f"\n  fitted coefficients:")
        print(f"     intercept (NEUTRAL venue)  {theta[0]:+.4f}")
        print(f"     Elo gap / 100              {theta[1]:+.4f}")
        print(f"     shoots first               {theta[2]:+.4f}")
        print(f"     playing at true home       {theta[3]:+.4f}")
        tt = pd.Series(theta[4:], index=teams).sort_values(ascending=False)
        print(f"\n  team penalty ratings after shrinkage "
              f"(range {tt.min():+.4f} to {tt.max():+.4f}):")
        for t in ["Argentina", "Spain"]:
            if t in tt.index:
                rank = int((tt > tt[t]).sum()) + 1
                print(f"     {t:<12}{tt[t]:+.5f}   (rank {rank} of {len(tt)})")
        print(f"\n  best and worst:")
        for t in list(tt.index[:3]) + list(tt.index[-3:]):
            print(f"     {t:<24}{tt[t]:+.5f}")
    import json
    tt = pd.Series(theta[4:], index=teams).sort_values(ascending=False)
    rec = {}
    for t in ["Argentina", "Spain"]:
        m = so[(so.home_team == t) | (so.away_team == t)]
        w = int((m.winner == t).sum())
        m18 = m[m.date >= pd.Timestamp("2018-01-01")]
        w18 = int((m18.winner == t).sum())
        rec[t] = dict(w=w, l=int(len(m) - w), w18=w18, l18=int(len(m18) - w18),
                      rating=float(tt[t]) if t in tt.index else 0.0,
                      rank=int((tt > tt[t]).sum()) + 1 if t in tt.index else 0)
    with open(PROC / "shootout_summary.json", "w") as fh:
        json.dump(dict(n_shootouts=int(len(so)), n_teams=int(len(tt)),
                       cv_with=float(cv.min()), cv_without=float(cv_elo),
                       best_lambda=float(best), b0=float(theta[0]),
                       b_elo=float(theta[1]), b_first=float(theta[2]),
                       b_home=float(theta[3]), best_teams=list(tt.index[:3]),
                       worst_teams=list(tt.index[-3:]), records=rec), fh, indent=1)
    return theta, teams, best, so


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  A SHOOTOUT MODEL BUILT FROM SHOOTOUTS")
    print("=" * 70 + "\n")
    theta, teams, best, so = main()
    tix = {t: i for i, t in enumerate(teams)}
    ed = 83.3
    b0, b_elo = theta[0], theta[1]
    th_s = theta[4 + tix["Spain"]] if "Spain" in tix else 0.0
    th_a = theta[4 + tix["Argentina"]] if "Argentina" in tix else 0.0
    z = b0 + b_elo * ed / 100 + th_s - th_a
    p = 1 / (1 + np.exp(-z))
    print(f"\n  SPAIN vs ARGENTINA SHOOTOUT")
    print(f"     Elo contribution      {b_elo*ed/100:+.4f}")
    print(f"     Spain penalty rating  {th_s:+.5f}")
    print(f"     Argentina penalty     {th_a:+.5f}")
    print(f"     neutral intercept     {b0:+.4f}")
    print(f"     ------")
    print(f"     P(Spain wins shootout) = {p:.1%}")
    np.save(PROC / "so_hier.npy", np.array([b0, b_elo, th_s, th_a]))
    print()
