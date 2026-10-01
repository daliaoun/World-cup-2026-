"""
The final: Spain vs Argentina.

Two things here that the semi-final forecast did not have.

UNCERTAINTY. A forecast of "57%" invites the question nobody usually answers: how sure
is that number? The team ratings are estimates from finite data, so the forecast built on
them is itself an estimate with a sampling distribution. We recover it by bootstrap:
resample the training matches with replacement, refit Dixon-Coles, recompute the
forecast, repeat. The spread of the resulting forecasts is the honest error bar. It is
usually wider than people expect, and reporting it is the difference between a
prediction and a guess with a decimal point.

LIVE WIN PROBABILITY. Because the in-play model tracks the match state minute by minute,
it can answer questions the static model structurally cannot: if Argentina lead 1-0 at
the hour, what then? The same forward recursion, started from a given score at a given
minute, gives the answer exactly.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, DC_TRAIN_YEARS
from src.dixon_coles import DixonColes
from src.statesim import score_distribution, knockout, MAXG
from src.simulate import simulate

XI, LAM2 = 0.0015, 0.5
FINAL_DATE = pd.Timestamp("2026-07-19")
HOME, AWAY = "Spain", "Argentina"


def bootstrap_forecast(df, tm, sm, elo_diff, b0, b1, B=200, seed=11):
    """Resample training matches, refit, re-forecast. Returns B advance-probabilities."""
    rng = np.random.default_rng(seed)
    tr = df[(df.date < FINAL_DATE) &
            (df.date >= FINAL_DATE - pd.DateOffset(years=DC_TRAIN_YEARS)) &
            df.hs.notna()].reset_index(drop=True)
    n = len(tr)
    out = []
    for b in range(B):
        samp = tr.iloc[rng.integers(0, n, n)]
        try:
            dc = DixonColes(xi=XI, lam2=LAM2).fit(samp, as_of=FINAL_DATE)
            lam, kap = dc.rates(HOME, AWAY, home_adv=0.0)
            r = knockout(lam, kap, tm, sm, elo_diff, b0, b1)
            out.append(r["p_home_adv"])
        except Exception:
            continue
    return np.array(out)


def win_prob_surface(lam, kap, tm, sm, elo_diff, b0, b1):
    """P(Spain wins the trophy) given the score at selected minutes."""
    from src.inplay import et_multiplier
    rows = []
    for minute in [0, 30, 60, 75, 90]:
        for (h, a) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
            P0 = np.zeros((MAXG + 1, MAXG + 1))
            P0[h, a] = 1.0
            rem = 90 - minute
            P = score_distribution(lam, kap, tm, sm, rem, P0=P0) if rem > 0 else P0
            p_h = float(np.tril(P, -1).sum())
            p_d = float(np.trace(P))
            Pl = np.zeros_like(P); np.fill_diagonal(Pl, np.diag(P))
            if Pl.sum() > 0:
                Pl /= Pl.sum()
                P120 = score_distribution(lam, kap, tm, sm, 30,
                                          et_mult=et_multiplier(tm), P0=Pl)
                p_h_et = float(np.tril(P120, -1).sum())
                p_lvl = float(np.trace(P120))
            else:
                p_h_et = p_lvl = 0.0
            pso = 1 / (1 + np.exp(-(b0 + b1 * elo_diff / 100)))
            rows.append(dict(minute=minute, score=f"{h}-{a}",
                             p=p_h + p_d * (p_h_et + p_lvl * pso)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    tm = np.load(PROC / "time_mult.npy")
    sm = np.load(PROC / "state_mult.npy")
    b0, b1 = np.load(PROC / "so_coef.npy")
    damp = float(np.load(PROC / "et_damp.npy")[0])

    row = df[(df.date == FINAL_DATE) & (df.home_team == HOME) & (df.away_team == AWAY)]
    if row.empty:
        cand = df[(df.date >= FINAL_DATE) & df.home_team.isin([HOME, AWAY])]
        raise SystemExit(f"final not found. nearby rows:\n{cand.head()}")
    row = row.iloc[[0]]
    elo_diff = float(row.elo_diff.iloc[0])

    tr = df[(df.date < FINAL_DATE) & (df.date >= FINAL_DATE - pd.DateOffset(years=DC_TRAIN_YEARS))]
    dc = DixonColes(xi=XI, lam2=LAM2).fit(tr, as_of=FINAL_DATE)
    lam, kap = dc.rates(HOME, AWAY, home_adv=0.0)

    ip = knockout(lam, kap, tm, sm, elo_diff, float(b0), float(b1))
    st = simulate(lam, kap, dc.score_matrix(HOME, AWAY, home_adv=0.0),
                  elo_diff, float(b0), float(b1), damp, n_sims=50000)

    print("=" * 76)
    print(f"  WORLD CUP FINAL   {HOME} vs {AWAY}   19 July 2026, East Rutherford")
    print("=" * 76)
    print(f"\n  Elo: {HOME} {row.elo_home.iloc[0]:.0f}  |  {AWAY} {row.elo_away.iloc[0]:.0f}"
          f"   (gap {elo_diff:+.0f})")
    print(f"  Expected goals (90'): {HOME} {lam:.2f} - {kap:.2f} {AWAY}\n")
    print(f"  {'':<26}{HOME:>12}{'draw':>10}{AWAY:>12}")
    print(f"  {'90 minutes':<26}{ip['p_home_90']:>11.1%}{ip['p_draw_90']:>10.1%}{ip['p_away_90']:>12.1%}")
    print(f"\n  {'TO LIFT THE TROPHY':<26}{ip['p_home_adv']:>11.1%}{'':>10}{ip['p_away_adv']:>12.1%}")
    print(f"  {'  (static model)':<26}{st['p_home_adv']:>11.1%}{'':>10}{st['p_away_adv']:>12.1%}")
    print(f"\n  reaches extra time {ip['p_reach_et']:.1%}   reaches penalties {ip['p_reach_pens']:.1%}")

    P = ip["P90"]
    flat = sorted(((P[i, j], i, j) for i in range(6) for j in range(6)), reverse=True)[:6]
    print("\n  most likely scorelines at 90':")
    print("    " + "   ".join(f"{i}-{j} {p:.1%}" for p, i, j in flat))

    print("\n  bootstrapping the forecast (200 refits on resampled history)...")
    print("  see src/ceiling.py for the parametric-bootstrap cross-check")
    bs = bootstrap_forecast(df, tm, sm, elo_diff, float(b0), float(b1), B=200)
    lo, hi = np.percentile(bs, [5, 95])
    print(f"    {HOME} to win the trophy: {ip['p_home_adv']:.1%}   "
          f"90% interval [{lo:.1%}, {hi:.1%}]   (n={len(bs)} refits)")
    print(f"    P(model still favours {HOME} on a resampled history) = {(bs>0.5).mean():.1%}")
    np.save(PROC / "final_bootstrap.npy", bs)

    print("\n  LIVE WIN PROBABILITY for Spain, by score and minute:")
    S = win_prob_surface(lam, kap, tm, sm, elo_diff, float(b0), float(b1))
    piv = S.pivot(index="score", columns="minute", values="p")
    print("\n" + (piv * 100).round(1).to_string())
    piv.to_csv(PROC / "win_prob_surface.csv")
    return ip, st, bs, piv


if __name__ == "__main__":
    main()
