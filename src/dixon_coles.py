"""
Time-weighted Dixon-Coles model (Dixon & Coles, 1997, Applied Statistics 46(2)).

    lambda (home goals) = exp( mu + attack[h] - defence[a] + gamma * is_home )
    kappa  (away goals) = exp( mu + attack[a] - defence[h] )

    P(X=x, Y=y) = tau(x,y | lambda,kappa,rho) * Pois(x|lambda) * Pois(y|kappa)

`tau` inflates the four low-scoring cells (0-0, 0-1, 1-0, 1-1). Independent Poisson
systematically under-predicts draws; this is the fix, and it is why the model still
beats far heavier machinery on football.

Two additions on top of the 1997 paper, both of which matter here:

1. WEIGHTED LIKELIHOOD.  Each match carries weight
       w = exp(-xi * days_ago) * importance(tournament)
   The exponential decay is Dixon & Coles' own suggestion (a team's strength drifts).
   The importance term downweights friendlies, where teams rotate heavily and the
   scoreline says little about strength. xi is tuned by walk-forward validation.

2. RIDGE SHRINKAGE.  A penalty  0.5 * lam2 * (sum a^2 + sum d^2)  is added.
   This does double duty: it pins down the two flat directions in the likelihood
   (attack and defence are only identified up to a shared constant), and it shrinks
   teams with few matches toward the global average — essential in international
   football, where the sample per team is thin and wildly unbalanced.

Fitted by L-BFGS-B with analytic gradients, because the walk-forward backtest
refits the model a few hundred times.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_GOALS = 12


# --------------------------------------------------------------------- tau
def _tau_and_grads(x, y, lam, kap, rho):
    """tau, dtau/dlam, dtau/dkap, dtau/drho -- vectorised over matches."""
    t = np.ones_like(lam)
    dl = np.zeros_like(lam)
    dk = np.zeros_like(lam)
    dr = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    t[m00] = 1.0 - lam[m00] * kap[m00] * rho
    dl[m00] = -kap[m00] * rho
    dk[m00] = -lam[m00] * rho
    dr[m00] = -lam[m00] * kap[m00]

    t[m01] = 1.0 + lam[m01] * rho
    dl[m01] = rho
    dr[m01] = lam[m01]

    t[m10] = 1.0 + kap[m10] * rho
    dk[m10] = rho
    dr[m10] = kap[m10]

    t[m11] = 1.0 - rho
    dr[m11] = -1.0

    return t, dl, dk, dr


class DixonColes:
    def __init__(self, xi: float = 0.0015, lam2: float = 5.0, max_iter: int = 2000):
        self.xi = xi          # time-decay rate, per day
        self.lam2 = lam2      # ridge strength
        self.max_iter = max_iter
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defence_: dict[str, float] = {}
        self.mu_ = 0.0
        self.gamma_ = 0.0
        self.rho_ = 0.0

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp,
            warm: "DixonColes | None" = None) -> "DixonColes":
        """
        df must contain: date, home_team, away_team, hs, as_, home_adv, weight_imp.
        Only matches strictly BEFORE `as_of` are used.

        `warm`: a previously-fitted model to initialise from. In the walk-forward
        backtest consecutive refits differ by a handful of matches, so warm-starting
        cuts L-BFGS iterations by roughly an order of magnitude. It changes only the
        starting point of the optimiser, never the objective, so the fitted model is
        the same one we'd reach from a cold start.
        """
        d = df[(df.date < as_of) & df.hs.notna()].copy()
        teams = sorted(set(d.home_team) | set(d.away_team))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.teams_, self._idx = teams, idx

        hi = d.home_team.map(idx).to_numpy()
        ai = d.away_team.map(idx).to_numpy()
        x = d.hs.to_numpy(float)
        y = d.as_.to_numpy(float)
        ha = d.home_adv.to_numpy(float)

        days = (as_of - d.date).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days) * d.weight_imp.to_numpy(float)
        w = w / w.mean()                      # scale-free: keeps ridge comparable

        self._n = n

        def nll(theta):
            a = theta[:n]
            dfc = theta[n:2 * n]
            mu, gam, rho = theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]

            eta_h = mu + a[hi] - dfc[ai] + gam * ha
            eta_a = mu + a[ai] - dfc[hi]
            lam = np.exp(np.clip(eta_h, -8, 4))
            kap = np.exp(np.clip(eta_a, -8, 4))

            t, dtl, dtk, dtr = _tau_and_grads(x, y, lam, kap, rho)
            t = np.clip(t, 1e-9, None)

            ll = w * (np.log(t) + x * np.log(lam) - lam + y * np.log(kap) - kap)
            obj = -ll.sum() + 0.5 * self.lam2 * (np.sum(a ** 2) + np.sum(dfc ** 2))

            # gradients
            dll_dlam = w * (dtl / t + x / lam - 1.0)
            dll_dkap = w * (dtk / t + y / kap - 1.0)
            gl = dll_dlam * lam          # chain through exp()
            gk = dll_dkap * kap

            ga = np.zeros(n)
            gd = np.zeros(n)
            np.add.at(ga, hi, gl)        # d lambda / d attack[home]
            np.add.at(ga, ai, gk)        # d kappa  / d attack[away]
            np.add.at(gd, ai, -gl)       # d lambda / d defence[away]
            np.add.at(gd, hi, -gk)       # d kappa  / d defence[home]

            g_mu = gl.sum() + gk.sum()
            g_gam = (gl * ha).sum()
            g_rho = (w * (dtr / t)).sum()

            grad = np.concatenate([
                -ga + self.lam2 * a,
                -gd + self.lam2 * dfc,
                [-g_mu, -g_gam, -g_rho],
            ])
            return obj, grad

        if warm is not None and warm.attack_:
            th0 = np.concatenate([
                np.array([warm.attack_.get(t, 0.0) for t in teams]),
                np.array([warm.defence_.get(t, 0.0) for t in teams]),
                [warm.mu_, warm.gamma_, warm.rho_],
            ])
        else:
            th0 = np.concatenate([np.zeros(2 * n), [0.0, 0.25, -0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-2, 2), (-0.5, 1.0), (-0.35, 0.35)]
        res = minimize(nll, th0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "maxfun": 10 * self.max_iter})

        th = res.x
        self.attack_ = dict(zip(teams, th[:n]))
        self.defence_ = dict(zip(teams, th[n:2 * n]))
        self.mu_, self.gamma_, self.rho_ = th[2 * n], th[2 * n + 1], th[2 * n + 2]
        self.converged_ = bool(res.success)
        self.nll_ = float(res.fun)
        self.n_train_ = int(len(d))
        return self

    # -------------------------------------------------------------- predict
    def rates(self, home: str, away: str, home_adv: float = 0.0) -> tuple[float, float]:
        """Expected 90-minute goals (lambda, kappa). Unknown teams fall back to average."""
        ah = self.attack_.get(home, 0.0)
        aa = self.attack_.get(away, 0.0)
        dh = self.defence_.get(home, 0.0)
        da = self.defence_.get(away, 0.0)
        lam = np.exp(self.mu_ + ah - da + self.gamma_ * home_adv)
        kap = np.exp(self.mu_ + aa - dh)
        return float(lam), float(kap)

    def score_matrix(self, home: str, away: str, home_adv: float = 0.0,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        """Full joint P(home=x, away=y) with the Dixon-Coles low-score correction."""
        lam, kap = self.rates(home, away, home_adv)
        gx = poisson.pmf(np.arange(max_goals + 1), lam)
        gy = poisson.pmf(np.arange(max_goals + 1), kap)
        M = np.outer(gx, gy)
        r = self.rho_
        M[0, 0] *= 1.0 - lam * kap * r
        M[0, 1] *= 1.0 + lam * r
        M[1, 0] *= 1.0 + kap * r
        M[1, 1] *= 1.0 - r
        M = np.clip(M, 0.0, None)
        return M / M.sum()

    def proba_1x2(self, home: str, away: str, home_adv: float = 0.0) -> np.ndarray:
        M = self.score_matrix(home, away, home_adv)
        h = np.tril(M, -1).sum()   # home goals > away goals
        d = np.trace(M)
        a = np.triu(M, 1).sum()
        return np.array([h, d, a])

    # ----------------------------------------------------------------- misc
    def ratings_table(self) -> pd.DataFrame:
        t = pd.DataFrame({
            "team": self.teams_,
            "attack": [self.attack_[t] for t in self.teams_],
            "defence": [self.defence_[t] for t in self.teams_],
        })
        # net strength = expected goal difference vs an average opponent, neutral venue
        t["exp_gf"] = np.exp(self.mu_ + t.attack)
        t["exp_ga"] = np.exp(self.mu_ - t.defence)
        t["net"] = t.exp_gf - t.exp_ga
        return t.sort_values("net", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from config import PROC
    df = pd.read_csv(PROC / "matches_elo.csv", parse_dates=["date"])
    train = df[df.date >= pd.Timestamp("2026-07-14") - pd.DateOffset(years=8)]

    dc = DixonColes(xi=0.0015, lam2=0.02).fit(train, as_of=pd.Timestamp("2026-07-14"))
    print(f"fitted on {dc.n_train_:,} matches | {len(dc.teams_)} teams | "
          f"converged={dc.converged_}")
    print(f"mu={dc.mu_:+.3f}  home_adv(gamma)={dc.gamma_:+.3f}  rho={dc.rho_:+.3f}")
    print(f"  -> home advantage multiplies expected goals by {np.exp(dc.gamma_):.3f}x")
    print(f"  -> rho<0 means draws are MORE likely than independent Poisson implies\n")

    tab = dc.ratings_table()
    print("Dixon-Coles strength, top 12 (expected goals for/against an average team, neutral):\n")
    print(f"  {'team':<16}{'attack':>8}{'defence':>9}{'xGF':>7}{'xGA':>7}{'net':>7}")
    for _, r in tab.head(12).iterrows():
        star = "  <--" if r.team in ("France", "Spain", "England", "Argentina") else ""
        print(f"  {r.team:<16}{r.attack:>8.2f}{r.defence:>9.2f}"
              f"{r.exp_gf:>7.2f}{r.exp_ga:>7.2f}{r.net:>7.2f}{star}")

    print("\nRaw model rates for the two semi-finals (neutral venue, 90 minutes):")
    for h, a in [("France", "Spain"), ("England", "Argentina")]:
        lam, kap = dc.rates(h, a, home_adv=0.0)
        p = dc.proba_1x2(h, a, 0.0)
        print(f"  {h} {lam:.2f} - {kap:.2f} {a}   "
              f"| 1X2 = {p[0]:.1%} / {p[1]:.1%} / {p[2]:.1%}")
