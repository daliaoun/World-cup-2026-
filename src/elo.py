"""
Elo ratings for national teams (World Football Elo Ratings convention).

    R' = R + K * G * (W - We)
    We = 1 / (1 + 10^(-dr/400)),  dr = R_home - R_away + home_advantage
    G  = 1           if |goal diff| <= 1
         1.5         if |goal diff| == 2
         (11+|gd|)/8 if |goal diff| >= 3
    K  = 60 (World Cup) ... 20 (friendly)

Note: ratings are updated on the REGULATION (90') result, not the after-extra-time
result, so that Elo measures the same thing the goals model does. A team that needed
120 minutes to beat Cape Verde did not beat Cape Verde in 90.

Ratings are produced as a *time series*: for every match we store each team's rating
as it stood immediately BEFORE kickoff. That is what makes them safe to use as
features — no information from the match itself, or any later match, leaks in.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ELO_K, ELO_K_DEFAULT, ELO_START, ELO_HOME_ADV


def _g_factor(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def run_elo(matches: pd.DataFrame,
            k_map: dict | None = None,
            home_adv: float = ELO_HOME_ADV,
            start: float = ELO_START) -> pd.DataFrame:
    """
    Walk the match history forward in time, updating ratings.

    Returns `matches` with four extra columns:
        elo_home, elo_away   ratings BEFORE the match (leak-free)
        elo_diff             elo_home - elo_away + home advantage
        elo_p_home           Elo's own win expectancy for the home side
    """
    k_map = k_map or ELO_K
    R: dict[str, float] = {}
    eh = np.full(len(matches), np.nan)
    ea = np.full(len(matches), np.nan)

    m = matches.sort_values("date").reset_index(drop=True)
    for i, row in enumerate(m.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        rh = R.setdefault(h, start)
        ra = R.setdefault(a, start)
        eh[i], ea[i] = rh, ra

        if not row.played or pd.isna(row.hs):
            continue

        adv = home_adv * row.home_adv
        we = 1.0 / (1.0 + 10 ** (-(rh - ra + adv) / 400.0))
        hs, as_ = float(row.hs), float(row.as_)
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = k_map.get(row.tournament, ELO_K_DEFAULT)
        delta = k * _g_factor(hs - as_) * (w - we)
        R[h] = rh + delta
        R[a] = ra - delta

    m["elo_home"] = eh
    m["elo_away"] = ea
    m["elo_diff"] = m.elo_home - m.elo_away + home_adv * m.home_adv
    m["elo_p_home"] = 1.0 / (1.0 + 10 ** (-m.elo_diff / 400.0))
    return m


def final_ratings(matches_with_elo: pd.DataFrame) -> pd.Series:
    """Current rating for every team, i.e. after the last match each one played."""
    m = matches_with_elo[matches_with_elo.played].sort_values("date")
    last: dict[str, float] = {}
    R: dict[str, float] = {}
    # replay to get post-match ratings
    for row in m.itertuples(index=False):
        R.setdefault(row.home_team, ELO_START)
        R.setdefault(row.away_team, ELO_START)
    # simplest: re-run and capture the dict at the end
    R = {}
    for row in m.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = R.setdefault(h, ELO_START)
        ra = R.setdefault(a, ELO_START)
        adv = ELO_HOME_ADV * row.home_adv
        we = 1.0 / (1.0 + 10 ** (-(rh - ra + adv) / 400.0))
        hs, as_ = float(row.hs), float(row.as_)
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = ELO_K.get(row.tournament, ELO_K_DEFAULT)
        d = k * _g_factor(hs - as_) * (w - we)
        R[h] = rh + d
        R[a] = ra - d
    return pd.Series(R).sort_values(ascending=False)


if __name__ == "__main__":
    from config import PROC
    df = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    df = run_elo(df)
    df.to_csv(PROC / "matches_elo.csv", index=False)

    r = final_ratings(df)
    print("Elo top 15 as of 2026-07-13 (regulation-time results):\n")
    for i, (t, v) in enumerate(r.head(15).items(), 1):
        star = "  <--" if t in ("France", "Spain", "England", "Argentina") else ""
        print(f"  {i:>2}. {t:<16} {v:7.1f}{star}")
