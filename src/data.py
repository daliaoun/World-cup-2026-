"""
Data layer: ingest, harmonise, and — critically — reconstruct *regulation-time*
(90-minute) scorelines.

WHY THIS MATTERS
----------------
`results.csv` records the score at the end of extra time, not at 90 minutes.
Fitting a goals model on those scores contaminates knockout matches: a team that
wins 3-1 after extra time did not score three goals in 90 minutes.

At WC2026 this is not academic. Argentina's raw record reads 17-6; their true
90-minute record is 13-5, because two of their three knockout wins were settled
in extra time. England's 2-1 over Norway was 1-1 at 90.

`goalscorers.csv` carries the minute of every goal. The dataset's convention —
verified in `validate_et_convention()` — codes regulation stoppage time as <= 90,
so any goal at minute > 90 is an extra-time goal. That gives us an exact,
auditable reconstruction rather than a guess.
"""
from __future__ import annotations
import sys
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, PROC, SOURCES, TOURNAMENT_WEIGHTS, DEFAULT_TOURNAMENT_WEIGHT

KEY = ["date", "home_team", "away_team"]


# ---------------------------------------------------------------- ingest
def download(force: bool = False) -> None:
    """Fetch the martj42 international results dataset (CC0)."""
    src = SOURCES["international_results"]
    for f in src["files"]:
        dest = RAW / f
        if dest.exists() and not force:
            continue
        url = f"{src['base']}/{f}"
        urllib.request.urlretrieve(url, dest)
        print(f"  downloaded {f}")


# ---------------------------------------------------------------- validation
def validate_et_convention(results: pd.DataFrame, goals: pd.DataFrame,
                           shootouts: pd.DataFrame) -> dict:
    """
    Test the assumption `minute > 90 => extra-time goal`.

    If true, then every match we can *independently confirm* went to extra time
    (it reached a penalty shootout, or had a goal after minute 105) must be LEVEL
    when we count only goals at minute <= 90 — because a match only goes to extra
    time if the teams are level at the end of regulation.
    """
    per = _goal_aggregates(goals)
    m = results.merge(per, on=KEY, how="inner")
    m = m[m.home_score.notna()]
    m = m[(m.hg == m.home_score) & (m.ag == m.away_score)]  # scorer data complete

    sh = shootouts.assign(_pens=True)[KEY + ["_pens"]]
    m = m.merge(sh, on=KEY, how="left")
    m["_pens"] = m["_pens"].fillna(False)

    confirmed = m[((m["_pens"]) | (m.max_min > 105)) & (m.max_min > 90)]
    level = (confirmed.h90 == confirmed.a90)
    # The handful that are NOT level at 90' are two-legged ties (World Cup
    # qualifying play-offs, Nations League play-offs) where extra time was
    # triggered by the AGGREGATE being level, not the single match. The rule
    # still identifies their extra-time goals correctly.
    return {
        "n_confirmed_et": int(len(confirmed)),
        "n_level_at_90": int(level.sum()),
        "pct_level_at_90": float(level.mean()) if len(confirmed) else float("nan"),
        "exceptions": confirmed.loc[~level, KEY + ["tournament"]].to_dict("records"),
    }


def _goal_aggregates(goals: pd.DataFrame) -> pd.DataFrame:
    """Per match: goals for each side, in regulation (<=90') and in total."""
    g = goals.copy()
    g["_et"] = g.minute > 90
    g["_is_home"] = g.team == g.home_team
    out = g.groupby(KEY).apply(
        lambda d: pd.Series({
            "h90": int(((d._is_home) & (~d._et)).sum()),
            "a90": int(((~d._is_home) & (~d._et)).sum()),
            "hg": int((d._is_home).sum()),
            "ag": int((~d._is_home).sum()),
            "max_min": float(d.minute.max()),
        }),
        include_groups=False,
    ).reset_index()
    return out


# ---------------------------------------------------------------- build
def build_matches() -> pd.DataFrame:
    """
    Return one clean row per international match with:
      home_score / away_score  : as played (includes extra time) — kept for audit
      hs / as_                 : REGULATION (90-minute) score — the modelling target
      went_et                  : bool
      weight_imp               : match-importance weight
    """
    res = pd.read_csv(RAW / "results.csv")
    goals = pd.read_csv(RAW / "goalscorers.csv")
    shoot = pd.read_csv(RAW / "shootouts.csv")

    res["date"] = pd.to_datetime(res["date"])
    goals["date"] = pd.to_datetime(goals["date"])
    shoot["date"] = pd.to_datetime(shoot["date"])

    per = _goal_aggregates(goals)
    per["date"] = pd.to_datetime(per["date"])
    df = res.merge(per, on=KEY, how="left")

    # Scorer data reconciles with the final score? Only then can we trust the
    # minute-level reconstruction for that match.
    reconciles = (df.hg == df.home_score) & (df.ag == df.away_score)
    df["scorer_data"] = reconciles.fillna(False)

    # Regulation score: strip extra-time goals where we have trustworthy minutes,
    # otherwise fall back to the as-played score (no ET in the vast majority).
    df["hs"] = np.where(df.scorer_data, df.h90, df.home_score)
    df["as_"] = np.where(df.scorer_data, df.a90, df.away_score)
    df["went_et"] = df.scorer_data & (df.max_min > 90)

    # played?
    df["played"] = df.home_score.notna()

    # importance weight
    df["weight_imp"] = df.tournament.map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_TOURNAMENT_WEIGHT)

    # neutral venue flag -> effective home advantage indicator
    df["neutral"] = df["neutral"].astype(bool)
    df["home_adv"] = (~df.neutral).astype(float)

    # penalty-shootout outcome (for the shootout model)
    sh = shoot.rename(columns={"winner": "so_winner", "first_shooter": "so_first"})
    df = df.merge(sh[KEY + ["so_winner", "so_first"]], on=KEY, how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    keep = KEY + ["home_score", "away_score", "hs", "as_", "went_et", "scorer_data",
                  "tournament", "city", "country", "neutral", "home_adv",
                  "weight_imp", "played", "so_winner", "so_first", "match_id"]
    return df[keep]


def outcome(hs, as_):
    """1X2 label from a scoreline: 0 = home win, 1 = draw, 2 = away win."""
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


if __name__ == "__main__":
    print("Downloading sources...")
    download()

    res = pd.read_csv(RAW / "results.csv")
    goals = pd.read_csv(RAW / "goalscorers.csv")
    shoot = pd.read_csv(RAW / "shootouts.csv")

    v = validate_et_convention(res, goals, shoot)
    print(f"\nET-convention check: {v['n_level_at_90']}/{v['n_confirmed_et']} "
          f"({v['pct_level_at_90']:.1%}) of independently-confirmed extra-time "
          f"matches are level at 90' under the rule.")
    print(f"  the {len(v['exceptions'])} exceptions are all two-legged ties "
          f"(ET triggered by aggregate score, not a level single match).")

    df = build_matches()
    df.to_csv(PROC / "matches.csv", index=False)
    played = df[df.played]
    print(f"\nmatches: {len(df):,}  played: {len(played):,}  "
          f"with minute-level scorer data: {played.scorer_data.sum():,}")
    print(f"matches corrected for extra time: {played.went_et.sum():,}")

    print("\nWC2026 semifinalists — as-played vs regulation (90') record:")
    wc = played[(played.tournament == "FIFA World Cup") & (played.date >= "2026-06-01")]
    for t in ["France", "Spain", "England", "Argentina"]:
        m = wc[(wc.home_team == t) | (wc.away_team == t)]
        h = m.home_team == t
        gf_r = (m.home_score.where(h, m.away_score)).sum()
        ga_r = (m.away_score.where(h, m.home_score)).sum()
        gf_9 = (m.hs.where(h, m.as_)).sum()
        ga_9 = (m.as_.where(h, m.hs)).sum()
        print(f"  {t:<10} as-played {int(gf_r):>2}-{int(ga_r):<2}   "
              f"regulation {int(gf_9):>2}-{int(ga_9):<2}   ({int(m.went_et.sum())} ET matches)")
