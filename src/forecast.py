"""
Final forecast for the two 2026 World Cup semi-finals.

Uses the EXACT recipe the walk-forward backtest validated - Dixon-Coles, xi=0.0015,
lam2=0.5, 8-year training window - refit on every match played before each kickoff.
Deviating from the validated recipe here would make the backtest a decoration rather
than evidence, so the hyperparameters are read from one place and not touched.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, RAW, FIXTURES, MARKET_ODDS, DC_TRAIN_YEARS, N_SIMS
from src.dixon_coles import DixonColes
from src.gbm import GBMGoals
from src.simulate import simulate, top_scorelines
from src.evaluate import devig

XI, LAM2 = 0.0015, 0.5          # <- the backtest winner. Do not hand-tune.


def main():
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    damp = float(np.load(PROC / "et_damp.npy")[0])
    b0, b1 = np.load(PROC / "so_coef.npy")

    out = []
    for fx in FIXTURES:
        d = pd.Timestamp(fx["date"])
        home, away = fx["home"], fx["away"]

        row = df[(df.date == d) & (df.home_team == home) & (df.away_team == away)]
        if row.empty:
            raise SystemExit(f"fixture not found in data: {home} v {away} {d.date()}")
        row = row.iloc[[0]]
        elo_diff = float(row.elo_diff.iloc[0])

        # --- refit on everything before kickoff, same window as the backtest ---
        train = df[(df.date < d) & (df.date >= d - pd.DateOffset(years=DC_TRAIN_YEARS))]
        dc = DixonColes(xi=XI, lam2=LAM2).fit(train, as_of=d)
        # neutral venue -> home_adv = 0, so neither side gets a crowd bonus
        lam, kap = dc.rates(home, away, home_adv=0.0)
        M = dc.score_matrix(home, away, home_adv=0.0)

        r = simulate(lam, kap, M, elo_diff, float(b0), float(b1), damp, n_sims=N_SIMS)

        # --- robustness: does the ML model, built from different information, agree? ---
        g = GBMGoals(xi=XI).fit(df, as_of=d)
        gl, gk = g.rates(row)
        gM = g.score_matrix(float(gl[0]), float(gk[0]))
        rg = simulate(float(gl[0]), float(gk[0]), gM, elo_diff, float(b0), float(b1), damp,
                      n_sims=N_SIMS)

        mk = MARKET_ODDS.get((home, away)) or MARKET_ODDS.get((away, home))
        out.append(dict(fixture=fx, elo_diff=elo_diff, lam=lam, kap=kap,
                        dc=r, gbm=rg, market=mk,
                        top=top_scorelines(M, 6),
                        elo_home=float(row.elo_home.iloc[0]),
                        elo_away=float(row.elo_away.iloc[0])))
    return out


if __name__ == "__main__":
    res = main()
    for r in res:
        fx = r["fixture"]; d = r["dc"]; g = r["gbm"]
        h, a = fx["home"], fx["away"]
        print("=" * 74)
        print(f"{h}  vs  {a}   |  {fx['date']}  |  {fx.get('venue','neutral')}")
        print("=" * 74)
        print(f"  Elo: {h} {r['elo_home']:.0f}  |  {a} {r['elo_away']:.0f}   (gap {r['elo_diff']:+.0f})")
        print(f"  Expected goals (90'): {h} {d['lam']:.2f} - {d['kap']:.2f} {a}\n")
        print(f"  90 MINUTES     {h} win {d['p_home_90']:.1%}   draw {d['p_draw_90']:.1%}   {a} win {d['p_away_90']:.1%}")
        print(f"  TO THE FINAL   {h} {d['p_home_adv']:.1%}   {a} {d['p_away_adv']:.1%}")
        print(f"    reaches extra time {d['p_reach_et']:.1%} | reaches penalties {d['p_reach_pens']:.1%}")
        print(f"\n  most likely scorelines (90'):")
        print("    " + "   ".join(f"{s} {p:.1%}" for s, p in r["top"]))
        print(f"\n  ROBUSTNESS  XGBoost independently says: {h} {g['p_home_adv']:.1%} / {a} {g['p_away_adv']:.1%}")
        if r["market"]:
            print(f"  MARKET      {r['market']}")
        print()
