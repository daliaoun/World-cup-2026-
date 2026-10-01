"""
"Grinta": can we measure a team's will to win, and does it predict anything?

The narrative version of this idea is untestable. Every finalist has a motivation story
available after the fact, and a quantity that can explain any outcome forecasts none. So
we replace the narrative with something a computer can be wrong about.

DEFINITION. At minute 75 of every match, the in-play model already knows what the game is
worth to each side: given the two teams' strengths and the current scoreline, it can
compute expected points over the closing quarter-hour. Grinta is what a team actually
earned, minus that.

    grinta = mean over matches of ( actual points - expected points at 75' )

This is deliberately strength- AND situation-controlled. A team is not credited for being
good, nor for being ahead. It is credited only for finishing matches better than a model
of its own ability says it should, from the exact positions it found itself in. If "will
to win" is a real and persistent property, this is where it would show up.

THE TEST THAT MATTERS. A number can be computed for any team; the question is whether it
means anything. The standard test in sports analytics is persistence: split each team's
matches into two halves at random and compute grinta separately on each. A real, stable
trait correlates across the split. Noise does not. Clutch-performance metrics in most
sports fail this test badly, and the honest prior here is that this one will too.

If it fails, the metric does not go in the model. That is the whole point of measuring it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, RAW
from src.statesim import score_distribution, MAXG

CUT_MINUTE = 75
WINDOW_START = "2014-01-01"


def _ep_from_state(lam, kap, tm, sm, h, a, minutes_left, cache):
    """Expected points for the home side from score (h,a) with `minutes_left` to play."""
    key = (round(lam, 2), round(kap, 2), int(np.clip(h - a, -3, 3)), minutes_left)
    if key in cache:
        return cache[key]
    P0 = np.zeros((MAXG + 1, MAXG + 1))
    hh, aa = (h, a) if h - a >= 0 else (0, a - h)
    hh = min(hh, MAXG); aa = min(aa, MAXG)
    P0[hh, aa] = 1.0
    P = score_distribution(lam, kap, tm, sm, minutes_left, P0=P0)
    pw = float(np.tril(P, -1).sum()); pd_ = float(np.trace(P))
    ep = 3 * pw + pd_
    cache[key] = ep
    return ep


def build(df: pd.DataFrame, gs: pd.DataFrame, tm, sm, verbose=True) -> pd.DataFrame:
    from src.dixon_coles import DixonColes
    K = ["date", "home_team", "away_team"]
    ws = pd.Timestamp(WINDOW_START)
    gs = gs.copy(); gs["m"] = pd.to_numeric(gs.minute, errors="coerce")

    mt = df[(df.date >= ws) & df.played & df.hs.notna()].copy()
    mt = mt.merge(gs[K].drop_duplicates(), on=K, how="inner")
    dc = DixonColes(xi=0.0, lam2=0.5).fit(df[df.date >= ws],
                                          as_of=df.date.max() + pd.Timedelta(days=1))
    rates = [dc.rates(r.home_team, r.away_team, home_adv=r.home_adv)
             for r in mt.itertuples(index=False)]
    mt["lam"] = [r[0] for r in rates]; mt["kap"] = [r[1] for r in rates]

    g = gs.merge(mt[K], on=K, how="inner")
    g = g[g.m.notna() & (g.m >= 1) & (g.m <= 90)]
    g["is_home"] = g.team == g.home_team
    # score at CUT_MINUTE for every match
    st = {}
    for key, grp in g.groupby(K, sort=False):
        h = int(((grp.m <= CUT_MINUTE) & grp.is_home).sum())
        a = int(((grp.m <= CUT_MINUTE) & ~grp.is_home).sum())
        st[key] = (h, a)

    cache = {}
    rows = []
    for r in mt.itertuples(index=False):
        key = (r.date, r.home_team, r.away_team)
        if key not in st:
            continue
        h75, a75 = st[key]
        ep_h = _ep_from_state(r.lam, r.kap, tm, sm, h75, a75, 90 - CUT_MINUTE, cache)
        ep_a = _ep_from_state(r.kap, r.lam, tm, sm, a75, h75, 90 - CUT_MINUTE, cache)
        hs, as_ = float(r.hs), float(r.as_)
        pts_h = 3.0 if hs > as_ else (1.0 if hs == as_ else 0.0)
        pts_a = 3.0 - pts_h if hs != as_ else 1.0
        rows.append(dict(date=r.date, team=r.home_team, opp=r.away_team,
                         trailing=h75 < a75, delta=pts_h - ep_h))
        rows.append(dict(date=r.date, team=r.away_team, opp=r.home_team,
                         trailing=a75 < h75, delta=pts_a - ep_a))
    G = pd.DataFrame(rows)
    if verbose:
        print(f"grinta computed on {len(mt):,} matches ({len(G):,} team-matches) since {ws.date()}")
    return G


def table(G: pd.DataFrame, min_matches=25) -> pd.DataFrame:
    t = G.groupby("team").agg(n=("delta", "size"), grinta=("delta", "mean"),
                              sd=("delta", "std")).query("n >= @min_matches")
    t["se"] = t.sd / np.sqrt(t.n)
    t["z"] = t.grinta / t.se
    return t.sort_values("grinta", ascending=False)


def persistence(G: pd.DataFrame, min_matches=30, n_rep=200, seed=5):
    """Split each team's matches at random into halves; correlate the two grinta values."""
    rng = np.random.default_rng(seed)
    teams = G.team.value_counts()
    teams = teams[teams >= min_matches].index
    sub = G[G.team.isin(teams)]
    rs = []
    for _ in range(n_rep):
        a, b = [], []
        for tm_, grp in sub.groupby("team"):
            d = grp.delta.to_numpy()
            ix = rng.permutation(len(d)); half = len(d) // 2
            a.append(d[ix[:half]].mean()); b.append(d[ix[half:2 * half]].mean())
        rs.append(np.corrcoef(a, b)[0, 1])
    return np.array(rs), len(teams)


if __name__ == "__main__":
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    gs = pd.read_csv(RAW / "goalscorers.csv", parse_dates=["date"])
    tm = np.load(PROC / "time_mult.npy"); sm = np.load(PROC / "state_mult.npy")

    G = build(df, gs, tm, sm)
    G.to_csv(PROC / "grinta.csv", index=False)
    T = table(G)

    print("\n=== GRINTA: points earned above in-play expectation, per match ===\n")
    print("  TOP 8")
    print(T.head(8)[["n", "grinta", "se", "z"]].round(3).to_string())
    print("\n  BOTTOM 5")
    print(T.tail(5)[["n", "grinta", "se", "z"]].round(3).to_string())

    print("\n=== THE FINALISTS ===")
    for t_ in ["Spain", "Argentina"]:
        if t_ in T.index:
            r = T.loc[t_]
            rank = int((T.grinta > r.grinta).sum()) + 1
            print(f"  {t_:<10} {r.grinta:+.3f} pts/match (rank {rank}/{len(T)}, "
                  f"z={r.z:+.2f}, n={int(r.n)})")

    print("\n=== IS IT A REAL TRAIT? split-half persistence ===")
    rs, nteams = persistence(G)
    print(f"  {nteams} teams, 200 random splits")
    print(f"  correlation between a team's grinta on one half of its matches")
    print(f"  and the other half:  r = {rs.mean():+.3f}  "
          f"(90% of splits in [{np.percentile(rs,5):+.3f}, {np.percentile(rs,95):+.3f}])")
    print(f"\n  P(r > 0) across splits = {(rs>0).mean():.1%}")
