"""
In-play hazard model: when goals actually arrive, and how the scoreline changes that.

The static Dixon-Coles model treats a match as a single draw from a bivariate Poisson.
That is a fine approximation, but it assumes two things that are both false:

  1. Goals arrive uniformly across the 90 minutes. They do not. The last 15 minutes
     produce roughly 1.9x the goals of the first 15.

  2. The scoreline does not affect play. It does. Teams that fall behind push, teams
     that lead sit deeper.

Both effects are estimated here as MULTIPLIERS on the team's baseline Dixon-Coles rate,
which keeps everything interpretable: 1.00 means "exactly the rate this team's strength
predicts", 1.15 means "15% faster than that".

THE CONFOUND, AND WHY IT MATTERS. Measured naively, teams that are two goals ahead look
like they score MORE. Putting that in a model would be catastrophic - scoring would
become self-reinforcing and the simulator would produce absurd blowouts. But the raw
number is an artefact of selection: the teams who are two goals up are mostly good teams
beating bad ones. Once each minute of exposure is offset by the two teams' own
Dixon-Coles rates, the effect reverses and the familiar football story appears.

SYMMETRY. A team leading by one and its opponent trailing by one are the same situation
seen from two sides, so the multiplier for state s must be estimated from both. Rates
are therefore accumulated per TEAM-minute rather than per match, and the resulting table
is automatically self-consistent: M[+1] describes the leader whether they are home or
away.

NORMALISATION. Multipliers are scaled so the exposure-weighted mean is exactly 1.0. The
overall goal level still comes from Dixon-Coles; this module only redistributes it across
time and match states.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, RAW

N_BUCKETS = 6                 # 15-minute blocks
STATES = np.array([-2, -1, 0, 1, 2])
SIDX = {s: i for i, s in enumerate(STATES)}


def _bucket(m: int) -> int:
    return min(int((m - 1) // 15), N_BUCKETS - 1)


def estimate(df: pd.DataFrame, goalscorers: pd.DataFrame,
             window_start="2014-01-01", verbose=True):
    """Returns (time_mult[6], state_mult[5]) as multipliers on the baseline rate."""
    from src.dixon_coles import DixonColes
    K = ["date", "home_team", "away_team"]
    ws = pd.Timestamp(window_start)

    gs = goalscorers.copy()
    gs["m"] = pd.to_numeric(gs.minute, errors="coerce")

    mt = df[(df.date >= ws) & df.played & df.hs.notna()].copy()
    mt = mt.merge(gs[K].drop_duplicates(), on=K, how="inner")

    # team-strength baseline. In-sample by design: we want the best available estimate
    # of how good these teams are, not a forecast of it.
    dc = DixonColes(xi=0.0, lam2=0.5).fit(df[df.date >= ws], as_of=df.date.max() + pd.Timedelta(days=1))
    rates = [dc.rates(r.home_team, r.away_team, home_adv=r.home_adv)
             for r in mt.itertuples(index=False)]
    mt["lam"] = [r[0] for r in rates]
    mt["kap"] = [r[1] for r in rates]

    g = gs.merge(mt[K + ["lam", "kap"]], on=K, how="inner")
    g = g[g.m.notna() & (g.m >= 1) & (g.m <= 90)]
    g["is_home"] = g.team == g.home_team

    # accumulate per TEAM-minute: observed goals and expected goals, by bucket and by
    # the scoring team's OWN goal difference
    obs_t = np.zeros(N_BUCKETS); exp_t = np.zeros(N_BUCKETS)
    obs_s = np.zeros(len(STATES)); exp_s = np.zeros(len(STATES))

    for _, grp in g.groupby(K, sort=False):
        lam = grp.lam.iloc[0] / 90.0
        kap = grp.kap.iloc[0] / 90.0
        ev = grp.sort_values("m")[["m", "is_home"]].values
        hs = as_ = 0
        prev = 0
        for m_, ih in ev:
            m_ = int(m_)
            for t in range(prev + 1, m_ + 1):
                b = _bucket(t)
                d = int(np.clip(hs - as_, -2, 2))
                exp_t[b] += lam + kap
                exp_s[SIDX[d]] += lam            # home team, own diff = d
                exp_s[SIDX[-d]] += kap           # away team, own diff = -d
            b = _bucket(m_)
            d = int(np.clip(hs - as_, -2, 2))
            obs_t[b] += 1
            obs_s[SIDX[d if ih else -d]] += 1
            if ih:
                hs += 1
            else:
                as_ += 1
            prev = m_
        for t in range(prev + 1, 91):
            b = _bucket(t)
            d = int(np.clip(hs - as_, -2, 2))
            exp_t[b] += lam + kap
            exp_s[SIDX[d]] += lam
            exp_s[SIDX[-d]] += kap

    time_mult = obs_t / exp_t
    state_mult = obs_s / exp_s
    # normalise to exposure-weighted mean 1 so the overall goal level is untouched
    time_mult /= np.average(time_mult, weights=exp_t)
    state_mult /= np.average(state_mult, weights=exp_s)

    if verbose:
        print(f"estimated on {len(mt):,} matches since {ws.date()}\n")
        print("  TIME multiplier (on the team's own baseline rate)")
        for b in range(N_BUCKETS):
            bar = "#" * int(time_mult[b] * 22)
            print(f"    {b*15+1:>2}-{b*15+15:>3}'   {time_mult[b]:.3f}  {bar}")
        print("\n  SCORE-STATE multiplier (from the scoring team's own point of view)")
        for s in STATES:
            lab = {-2: "2+ behind", -1: "1 behind", 0: "level", 1: "1 ahead", 2: "2+ ahead"}[s]
            print(f"    {lab:<12} {state_mult[SIDX[s]]:.3f}")
    return time_mult, state_mult


def et_multiplier(time_mult: np.ndarray) -> float:
    """
    Extra-time rate relative to the average regulation minute.

    Rather than a free parameter, we extrapolate: extra time is played at the intensity
    of the closing stage of a match. The final bucket is inflated by stoppage-time goals
    being coded at minute 90, so we use the mean of the last two buckets.
    """
    return float(time_mult[-2:].mean())


if __name__ == "__main__":
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    gs = pd.read_csv(RAW / "goalscorers.csv", parse_dates=["date"])
    tm, sm = estimate(df, gs)
    np.save(PROC / "time_mult.npy", tm)
    np.save(PROC / "state_mult.npy", sm)
    print(f"\n  implied extra-time intensity: {et_multiplier(tm):.3f} x an average regulation minute")
