"""
Walk-forward backtest.

The only backtest worth running is one that cannot cheat. For every match in the
evaluation set we refit the model on matches strictly BEFORE that match's date, then
predict. Ratings and features are recomputed as they stood at kickoff. Nothing from
the match itself, or from any later match, is visible to the model.

Evaluation set = matches at the elite international tournaments (World Cup, Euro,
Copa America) since 2010. These are the matches that resemble the thing we actually
want to forecast: two strong national sides, high stakes, neutral-ish venue.

Refits happen once per match DATE (not once per match) — within a single day, every
match sees the same, correctly-lagged model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, DC_TRAIN_YEARS
from src.dixon_coles import DixonColes
from src.evaluate import summary, rps

ELITE = ["FIFA World Cup", "UEFA Euro", "Copa América"]


def eval_set(df: pd.DataFrame, start: str = "2010-01-01",
             tournaments: list[str] | None = None) -> pd.DataFrame:
    t = tournaments or ELITE
    e = df[(df.tournament.isin(t)) & (df.date >= start) & df.played & df.hs.notna()]
    return e.sort_values("date").reset_index(drop=True)


def y_true(e: pd.DataFrame) -> np.ndarray:
    return np.where(e.hs > e.as_, 0, np.where(e.hs == e.as_, 1, 2))


def backtest_dc(df: pd.DataFrame, e: pd.DataFrame, xi: float, lam2: float,
                train_years: int = DC_TRAIN_YEARS,
                verbose: bool = False) -> tuple[np.ndarray, list]:
    """Returns (n,3) probability matrix aligned with `e`, plus the fitted models."""
    P = np.zeros((len(e), 3))
    models = []
    prev = None
    for d, grp in e.groupby("date", sort=True):
        train = df[(df.date < d) & (df.date >= d - pd.DateOffset(years=train_years))]
        dc = DixonColes(xi=xi, lam2=lam2).fit(train, as_of=d, warm=prev)
        prev = dc
        models.append((d, dc))
        for i, row in zip(grp.index, grp.itertuples(index=False)):
            P[i] = dc.proba_1x2(row.home_team, row.away_team, row.home_adv)
        if verbose:
            print(f"    {d.date()}  n={len(grp)}", flush=True)
    return P, models


def backtest_elo(e: pd.DataFrame) -> np.ndarray:
    """
    Elo baseline. Elo gives P(win) for a two-way contest; we need three outcomes.
    Standard fix: map the Elo win expectancy to 1X2 with a draw model whose width is
    calibrated on the training data. We use a fixed, well-established mapping:
        P(draw) peaks when teams are level and decays with |rating gap|.
    """
    we = e.elo_p_home.to_numpy()
    gap = np.abs(e.elo_diff.to_numpy())
    pd_ = 0.30 * np.exp(-gap / 300.0)          # draw share shrinks as the gap widens
    ph = we * (1 - pd_)
    pa = (1 - we) * (1 - pd_)
    P = np.column_stack([ph, pd_, pa])
    return P / P.sum(1, keepdims=True)


if __name__ == "__main__":
    df = pd.read_csv(PROC / "matches_elo.csv", parse_dates=["date"])
    e = eval_set(df)
    y = y_true(e)
    print(f"Evaluation set: {len(e)} elite-tournament matches, "
          f"{e.date.min().date()} -> {e.date.max().date()} "
          f"({e.date.nunique()} distinct dates)")
    print(f"Outcome base rates: home {np.mean(y==0):.1%} / draw {np.mean(y==1):.1%} "
          f"/ away {np.mean(y==2):.1%}\n")

    # ---- baselines -------------------------------------------------------
    res = {}
    Pu = np.tile([1/3, 1/3, 1/3], (len(e), 1))
    res["Uniform (1/3 each)"] = summary(Pu, y)
    base = np.array([np.mean(y == 0), np.mean(y == 1), np.mean(y == 2)])
    res["Base rates"] = summary(np.tile(base, (len(e), 1)), y)
    res["Elo"] = summary(backtest_elo(e), y)

    print("Baselines:")
    for k, v in res.items():
        print(f"  {k:<22} RPS={v['RPS']:.4f}  logloss={v['logloss']:.4f}  acc={v['accuracy']:.1%}")

    # ---- Dixon-Coles hyperparameter sweep --------------------------------
    print("\nDixon-Coles walk-forward sweep (refit at every date, past data only):")
    print(f"  {'xi':>7}{'lam2':>7}{'RPS':>9}{'logloss':>10}{'brier':>8}{'acc':>8}")
    best = (None, 9e9)
    grid = [(xi, l2) for xi in [0.0000, 0.0005, 0.0010, 0.0020, 0.0035]
                     for l2 in [2.0, 8.0, 20.0]]
    rows = []
    for xi, l2 in grid:
        P, _ = backtest_dc(df, e, xi=xi, lam2=l2)
        s = summary(P, y)
        rows.append(dict(xi=xi, lam2=l2, **s))
        flag = ""
        if s["RPS"] < best[1]:
            best = ((xi, l2), s["RPS"]); flag = "  *"
        print(f"  {xi:>7.4f}{l2:>7.1f}{s['RPS']:>9.4f}{s['logloss']:>10.4f}"
              f"{s['brier']:>8.4f}{s['accuracy']:>7.1%}{flag}", flush=True)

    pd.DataFrame(rows).to_csv(PROC / "dc_sweep.csv", index=False)
    print(f"\nBest: xi={best[0][0]:.4f}, lam2={best[0][1]:.1f}  ->  RPS={best[1]:.4f}")
    print(f"half-life of a match's weight: "
          f"{np.log(2)/best[0][0]/365.25:.2f} years" if best[0][0] > 0 else
          "no time decay selected")


def backtest_gbm(df, e, xi=0.0015, **kw):
    """
    Walk-forward GBM. Refit on 1 January of each year in the evaluation window, on an
    expanding window of everything strictly before that date. Features are still fresh
    to the match (Elo/form/rest update after every game); only the learned mapping is
    held for a year. See src/gbm.py for why that's legitimate.
    """
    from src.gbm import GBMGoals
    P = np.zeros((len(e), 3))
    e = e.copy()
    e["_yr"] = e.date.dt.year
    for yr, grp in e.groupby("_yr", sort=True):
        cut = pd.Timestamp(f"{yr}-01-01")
        m = GBMGoals(xi=xi, **kw).fit(df, as_of=cut)
        P[e._yr.values == yr] = m.proba_1x2(grp)
    return P
