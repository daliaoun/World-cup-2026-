"""
Feature engineering for the gradient-boosted model.

Every feature is computed from matches STRICTLY BEFORE kickoff, in a single forward
pass over the history. There is no groupby-then-shift, no rolling window that peeks:
the state machine below simply cannot see the future, which is the only way to be sure.

Features fall into four families:

  RATING     Elo before kickoff, and the gap.
  FORM       Exponentially-weighted goals for/against and points, per team. Weighted by
             match importance too, so a 5-0 friendly win doesn't look like a 5-0 in a
             World Cup.
  FATIGUE    Days of rest, and whether the previous match went to extra time. This is
             the one thing a pure Dixon-Coles model is blind to, and at this World Cup
             it is not a footnote: England and Argentina have each played 120 minutes
             within the last few days, France and Spain have not.
  CONTEXT    Neutral venue, match importance, head-to-head history, current-tournament
             goals scored and conceded.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC

FORM_HL = 8.0          # half-life in matches for exponentially-weighted form
H2H_MAX = 6            # head-to-head lookback


class _TeamState:
    __slots__ = ("gf", "ga", "pts", "wsum", "last_date", "last_et",
                 "tourn", "t_gf", "t_ga", "t_n")

    def __init__(self):
        self.gf = 0.0; self.ga = 0.0; self.pts = 0.0; self.wsum = 0.0
        self.last_date = None; self.last_et = 0
        self.tourn = None; self.t_gf = 0.0; self.t_ga = 0.0; self.t_n = 0

    def form(self):
        if self.wsum < 1e-9:
            return np.nan, np.nan, np.nan
        return self.gf / self.wsum, self.ga / self.wsum, self.pts / self.wsum

    def update(self, gf, ga, imp, date, et, tournament):
        decay = 0.5 ** (1.0 / FORM_HL)
        w = imp
        self.gf = self.gf * decay + w * gf
        self.ga = self.ga * decay + w * ga
        p = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
        self.pts = self.pts * decay + w * p
        self.wsum = self.wsum * decay + w
        self.last_date = date
        self.last_et = int(et)
        # current-tournament running totals (reset when the tournament changes)
        if tournament != self.tourn:
            self.tourn = tournament; self.t_gf = 0.0; self.t_ga = 0.0; self.t_n = 0
        self.t_gf += gf; self.t_ga += ga; self.t_n += 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """`df` must already carry Elo columns. Returns df + feature columns."""
    df = df.sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)
    S: dict[str, _TeamState] = defaultdict(_TeamState)
    H2H: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_MAX))

    cols = {c: np.full(len(df), np.nan) for c in [
        "h_gf", "h_ga", "h_pts", "a_gf", "a_ga", "a_pts",
        "h_rest", "a_rest", "h_last_et", "a_last_et",
        "h_t_gf", "h_t_ga", "h_t_n", "a_t_gf", "a_t_ga", "a_t_n",
        "h2h_n", "h2h_gd",
    ]}

    for i, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        sh, sa = S[h], S[a]

        cols["h_gf"][i], cols["h_ga"][i], cols["h_pts"][i] = sh.form()
        cols["a_gf"][i], cols["a_ga"][i], cols["a_pts"][i] = sa.form()
        cols["h_rest"][i] = (r.date - sh.last_date).days if sh.last_date else np.nan
        cols["a_rest"][i] = (r.date - sa.last_date).days if sa.last_date else np.nan
        cols["h_last_et"][i] = sh.last_et
        cols["a_last_et"][i] = sa.last_et

        # current-tournament form (0 if this is their first match of it)
        cols["h_t_gf"][i] = sh.t_gf if sh.tourn == r.tournament else 0.0
        cols["h_t_ga"][i] = sh.t_ga if sh.tourn == r.tournament else 0.0
        cols["h_t_n"][i] = sh.t_n if sh.tourn == r.tournament else 0
        cols["a_t_gf"][i] = sa.t_gf if sa.tourn == r.tournament else 0.0
        cols["a_t_ga"][i] = sa.t_ga if sa.tourn == r.tournament else 0.0
        cols["a_t_n"][i] = sa.t_n if sa.tourn == r.tournament else 0

        key = tuple(sorted([h, a]))
        hist = H2H[key]
        if hist:
            gd = [g if key[0] == h else -g for g in hist]
            cols["h2h_n"][i] = len(hist)
            cols["h2h_gd"][i] = float(np.mean(gd))
        else:
            cols["h2h_n"][i] = 0
            cols["h2h_gd"][i] = 0.0

        # ---- update state AFTER the features are recorded ----
        if r.played and not pd.isna(r.hs):
            hs, as_ = float(r.hs), float(r.as_)
            sh.update(hs, as_, r.weight_imp, r.date, r.went_et, r.tournament)
            sa.update(as_, hs, r.weight_imp, r.date, r.went_et, r.tournament)
            hist.append(hs - as_ if key[0] == h else as_ - hs)

    for c, v in cols.items():
        df[c] = v

    # derived
    df["rest_diff"] = df.h_rest - df.a_rest
    df["form_gf_diff"] = df.h_gf - df.a_gf
    df["form_ga_diff"] = df.h_ga - df.a_ga
    df["form_pts_diff"] = df.h_pts - df.a_pts
    df["h_t_gfpg"] = np.where(df.h_t_n > 0, df.h_t_gf / df.h_t_n.clip(lower=1), np.nan)
    df["h_t_gapg"] = np.where(df.h_t_n > 0, df.h_t_ga / df.h_t_n.clip(lower=1), np.nan)
    df["a_t_gfpg"] = np.where(df.a_t_n > 0, df.a_t_gf / df.a_t_n.clip(lower=1), np.nan)
    df["a_t_gapg"] = np.where(df.a_t_n > 0, df.a_t_ga / df.a_t_n.clip(lower=1), np.nan)
    return df


FEATURES = [
    "elo_home", "elo_away", "elo_diff", "elo_p_home",
    "h_gf", "h_ga", "h_pts", "a_gf", "a_ga", "a_pts",
    "form_gf_diff", "form_ga_diff", "form_pts_diff",
    "h_rest", "a_rest", "rest_diff", "h_last_et", "a_last_et",
    "h_t_gfpg", "h_t_gapg", "a_t_gfpg", "a_t_gapg", "h_t_n", "a_t_n",
    "h2h_n", "h2h_gd",
    "home_adv", "weight_imp",
]


if __name__ == "__main__":
    df = pd.read_csv(PROC / "matches_elo.csv", parse_dates=["date"])
    df = build_features(df)
    df.to_csv(PROC / "matches_feat.csv", index=False)
    print(f"built {len(FEATURES)} features for {len(df):,} matches")

    fx = df[df.date.isin([pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")])]
    show = ["date", "home_team", "away_team", "elo_diff", "h_rest", "a_rest",
            "h_last_et", "a_last_et", "h_t_gfpg", "h_t_gapg", "a_t_gfpg", "a_t_gapg"]
    print("\nFeature snapshot for the two semi-finals:\n")
    print(fx[show].to_string(index=False))
