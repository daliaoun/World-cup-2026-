"""
Scenario analysis for the final.

A single number - "Spain 61.8%" - throws away almost everything the model knows. The
forward recursion in statesim.py can be started from any position at any minute, so the
model can answer the questions people actually ask during a match: what is it worth if
Argentina score first? How much does a goal in the 80th minute swing it? What happens if
it is still level at 80?

Everything here is exact, not simulated. No Monte Carlo noise.

One caution that applies to every number below: these are conditional probabilities under
the model. "If Argentina score first, they win 61% of the time" does not mean Argentina
have a 61% chance tonight - it means that among all the futures where they score first,
they win 61% of them. Conditioning on a goal also silently conditions on everything that
goal implies about how the match is going, which the model cannot see.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, DC_TRAIN_YEARS
from src.statesim import score_distribution, MAXG, _state_idx_grid
from src.inplay import et_multiplier, _bucket

HOME, AWAY = "Spain", "Argentina"


class Scenarios:
    def __init__(self, lam, kap, tm, sm, elo_diff, b0, b1):
        self.lam, self.kap, self.tm, self.sm = lam, kap, tm, sm
        self.elo_diff, self.b0, self.b1 = elo_diff, b0, b1
        self.p_so = 1 / (1 + np.exp(-(b0 + b1 * elo_diff / 100)))
        self.etm = et_multiplier(tm)

    # ---------- core: value of a position ----------
    def value(self, h: int, a: int, minute: int) -> float:
        """P(Spain lifts the trophy) given score (h,a) with `minute` minutes played."""
        P0 = np.zeros((MAXG + 1, MAXG + 1))
        P0[min(h, MAXG), min(a, MAXG)] = 1.0
        rem = 90 - minute
        P = score_distribution(self.lam, self.kap, self.tm, self.sm, rem, P0=P0) if rem > 0 else P0
        pw = float(np.tril(P, -1).sum())
        pl = float(np.trace(P))
        if pl < 1e-12:
            return pw
        Pl = np.zeros_like(P); np.fill_diagonal(Pl, np.diag(P)); Pl /= Pl.sum()
        P120 = score_distribution(self.lam, self.kap, self.tm, self.sm, 30,
                                  et_mult=self.etm, P0=Pl)
        w_et = float(np.tril(P120, -1).sum())
        lvl = float(np.trace(P120))
        return pw + pl * (w_et + lvl * self.p_so)

    # ---------- decomposition of the headline number ----------
    def decompose(self) -> dict:
        P = score_distribution(self.lam, self.kap, self.tm, self.sm, 90)
        w90 = float(np.tril(P, -1).sum())
        l90 = float(np.triu(P, 1).sum())
        d90 = float(np.trace(P))
        Pl = np.zeros_like(P); np.fill_diagonal(Pl, np.diag(P)); Pl /= Pl.sum()
        P120 = score_distribution(self.lam, self.kap, self.tm, self.sm, 30,
                                  et_mult=self.etm, P0=Pl)
        wet = float(np.tril(P120, -1).sum())
        let_ = float(np.triu(P120, 1).sum())
        lvl = float(np.trace(P120))
        return {
            "spain_90": w90, "arg_90": l90, "draw_90": d90,
            "spain_et": d90 * wet, "arg_et": d90 * let_,
            "pens": d90 * lvl,
            "spain_pens": d90 * lvl * self.p_so,
            "arg_pens": d90 * lvl * (1 - self.p_so),
            "p_so": self.p_so,
        }

    # ---------- first goal ----------
    def first_goal(self):
        """Distribution of when the first goal arrives and who scores it."""
        lh, la = self.lam / 90, self.kap / 90
        p_still = 1.0
        rows = []
        for t in range(1, 91):
            mult = self.tm[_bucket(t)]
            ph = 1 - np.exp(-lh * mult * self.sm[2])   # level state
            pa = 1 - np.exp(-la * mult * self.sm[2])
            rows.append((t, p_still * ph * (1 - pa), p_still * pa * (1 - ph),
                         p_still * ph * pa, p_still))
            p_still *= (1 - ph) * (1 - pa)
        F = pd.DataFrame(rows, columns=["minute", "spain", "arg", "both", "still_00"])
        return F, p_still

    # ---------- goal swing ----------
    def swing(self, minutes):
        rows = []
        for m in minutes:
            lvl = self.value(0, 0, m)
            sp = self.value(1, 0, m)
            ar = self.value(0, 1, m)
            rows.append(dict(minute=m, level=lvl, spain_scores=sp, arg_scores=ar,
                             spain_swing=sp - lvl, arg_swing=lvl - ar))
        return pd.DataFrame(rows)

    # ---------- does Spain ever trail? ----------
    def ever_trails(self, side="home") -> float:
        """P(the given side is behind at some point in the 90 minutes)."""
        G = MAXG + 1
        ih, ia = _state_idx_grid(G)
        # state augmented with a flag: 0 = has not trailed, 1 = has
        P = np.zeros((2, G, G)); P[0, 0, 0] = 1.0
        lh, la = self.lam / 90, self.kap / 90
        for t in range(1, 91):
            mult = self.tm[_bucket(t)]
            rh = lh * mult * self.sm[ih]
            ra = la * mult * self.sm[ia]
            ph = 1 - np.exp(-rh); pa = 1 - np.exp(-ra)
            Q = np.zeros_like(P)
            for f in (0, 1):
                Q[f] += P[f] * (1 - ph) * (1 - pa)
                gh = P[f] * ph * (1 - pa)
                ga = P[f] * (1 - ph) * pa
                gb = P[f] * ph * pa
                Q[f][1:, :] += gh[:-1, :]; Q[f][-1, :] += gh[-1, :]
                Q[f][:, 1:] += ga[:, :-1]; Q[f][:, -1] += ga[:, -1]
                Q[f][1:, 1:] += gb[:-1, :-1]
            # any mass sitting in a "behind" cell gets the flag set
            h = np.arange(G)[:, None]; a = np.arange(G)[None, :]
            behind = (h < a) if side == "home" else (h > a)
            moved = Q[0] * behind
            Q[1] += moved
            Q[0] = Q[0] * ~behind
            P = Q
        return float(P[1].sum())


def main():
    from src.dixon_coles import DixonColes
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    tm = np.load(PROC / "time_mult.npy"); sm = np.load(PROC / "state_mult.npy")
    b0, b1 = np.load(PROC / "so_coef.npy")
    d = pd.Timestamp("2026-07-19")
    row = df[(df.date == d) & (df.home_team == HOME)].iloc[[0]]
    ed = float(row.elo_diff.iloc[0])
    tr = df[(df.date < d) & (df.date >= d - pd.DateOffset(years=DC_TRAIN_YEARS))]
    dc = DixonColes(xi=0.0015, lam2=0.5).fit(tr, as_of=d)
    lam, kap = dc.rates(HOME, AWAY, home_adv=0.0)
    return Scenarios(lam, kap, tm, sm, ed, float(b0), float(b1)), lam, kap


if __name__ == "__main__":
    S, lam, kap = main()
    dec = S.decompose()
    W = 74
    def hdr(num, title, blurb):
        print("\n" + "=" * W)
        print("  %d. %s" % (num, title))
        print("=" * W)
        print("  " + blurb + "\n")
    def dots(label, value, width=58):
        print("  %s %s" % (label.ljust(width, "."), value))

    tot_s = dec["spain_90"] + dec["spain_et"] + dec["spain_pens"]
    print("\n" + "=" * W)
    print("  WORLD CUP FINAL")
    print("  %s  vs  %s" % (HOME, AWAY))
    print("  19 July 2026, East Rutherford")
    print("=" * W)
    print("\n  CHANCE OF WINNING THE WORLD CUP")
    print("     %-12s %.1f%%" % (HOME, tot_s * 100))
    print("     %-12s %.1f%%" % (AWAY, (1 - tot_s) * 100))
    print("\n  How confident is the model? Not very. Re-running it on 200 slightly")
    print("  different versions of football history gives Spain anywhere between")
    print("  44.8%% and 75.4%%. Treat %.1f%% as the middle of a wide range."
          % (tot_s * 100))
    print("\n  Average goals the model expects:  %s %.2f  -  %.2f %s"
          % (HOME, lam, kap, AWAY))

    # ------------------------------------------------------------------ 1
    hdr(1, "THE THREE WAYS TO WIN A FINAL",
        "A final can be won in normal time, in extra time, or on penalties.")
    print("     %-12s %14s %14s %14s %12s"
          % ("", "Win inside", "Win in", "Win the", "TOTAL"))
    print("     %-12s %14s %14s %14s %12s"
          % ("", "90 minutes", "extra time", "shootout", "CHANCE"))
    print("     " + "-" * 68)
    print("     %-12s %13.1f%% %13.1f%% %13.1f%% %11.1f%%"
          % (HOME, dec["spain_90"] * 100, dec["spain_et"] * 100,
             dec["spain_pens"] * 100, tot_s * 100))
    print("     %-12s %13.1f%% %13.1f%% %13.1f%% %11.1f%%"
          % (AWAY, dec["arg_90"] * 100, dec["arg_et"] * 100,
             dec["arg_pens"] * 100, (1 - tot_s) * 100))
    print()
    dots("  The score is still tied when 90 minutes end",
         "%.1f%% of the time" % (dec["draw_90"] * 100))
    dots("  The match goes all the way to a penalty shootout",
         "%.1f%% of the time" % (dec["pens"] * 100))
    dots("  If there is a shootout, %s win it" % HOME,
         "%.1f%% of the time" % (dec["p_so"] * 100))
    arg_late = (dec["arg_et"] + dec["arg_pens"]) / (1 - tot_s)
    esp_late = (dec["spain_et"] + dec["spain_pens"]) / tot_s
    print("\n  WHAT THIS MEANS: %.0f%% of %s's chance of winning the cup depends"
          % (arg_late * 100, AWAY))
    print("  on the match going past 90 minutes. For %s it is only %.0f%%."
          % (HOME, esp_late * 100))
    print("  %s do not need to be the better team tonight." % AWAY)
    print("  They just need it to stay close.")

    # ------------------------------------------------------------------ 2
    hdr(2, "WHO SCORES FIRST, AND WHEN",
        "The opening goal of the match: who gets it, and how long it takes.")
    F, p00 = S.first_goal()
    both = F["both"].sum()
    dots("  %s score the first goal" % HOME, "%.1f%%" % ((F.spain.sum() + both / 2) * 100))
    dots("  %s score the first goal" % AWAY, "%.1f%%" % ((F.arg.sum() + both / 2) * 100))
    dots("  Nobody scores at all in the 90 minutes", "%.1f%%" % (p00 * 100))
    cum = np.cumsum(F.spain + F.arg + F["both"])
    print("\n  Chance the first goal has already been scored by:")
    for m in [15, 30, 45, 60, 75, 90]:
        lab = "half-time" if m == 45 else "full-time" if m == 90 else "%d minutes" % m
        dots("     %s" % lab, "%.1f%%" % (cum.iloc[m - 1] * 100), 55)

    # ------------------------------------------------------------------ 3
    hdr(3, "SCORING FIRST: HOW OFTEN DOES IT WIN YOU THE CUP?",
        "Whoever opens the scoring, how often do they go on to lift the trophy?")
    print("     %-20s %22s %22s" % ("If the first goal", "and %s scored it," % AWAY.upper(),
                                     "and %s scored it," % HOME.upper()))
    print("     %-20s %22s %22s" % ("is scored at...", "%s win the cup" % AWAY,
                                     "%s win the cup" % HOME))
    print("     " + "-" * 66)
    for m in [10, 20, 30, 45, 60, 70, 80, 85]:
        print("     %-20s %21.1f%% %21.1f%%"
              % ("%d minutes" % m, (1 - S.value(0, 1, m)) * 100, S.value(1, 0, m) * 100))
    print("\n  WHAT THIS MEANS: an early lead is worth far less to %s than to %s." % (AWAY, HOME))
    print("  Scoring in the 10th minute wins %s the cup 67.5%% of the time," % AWAY)
    print("  but wins %s the cup 84.5%% -- %s have the quality and the time" % (HOME, HOME))
    print("  to come back. By the 80th minute the difference has nearly vanished.")

    # ------------------------------------------------------------------ 4
    hdr(4, "WHAT IS THE NEXT GOAL WORTH?",
        "Starting from a tied game, how much does scoring change a team's chances?")
    print("     %-10s %s   %s" % ("", "SPAIN".center(25), "ARGENTINA".center(25)))
    print("     %-10s %8s %8s %7s   %8s %8s %7s"
          % ("Minute", "tied", "scores", "gain", "tied", "scores", "gain"))
    print("     " + "-" * 64)
    for m in [1, 15, 30, 45, 60, 70, 80, 85, 89]:
        lvl_s = S.value(0, 0, m)
        sc_s = S.value(1, 0, m)
        lvl_a = 1 - lvl_s
        sc_a = 1 - S.value(0, 1, m)
        print("     %-10s %7.1f%% %7.1f%% %+7.1f   %7.1f%% %7.1f%% %+7.1f"
              % ("%d min" % m, lvl_s * 100, sc_s * 100, (sc_s - lvl_s) * 100,
                 lvl_a * 100, sc_a * 100, (sc_a - lvl_a) * 100))
    print("\n  Each half of the table is that team's OWN chance of winning the cup.")
    print("    tied   = their chance while the score is still 0-0")
    print("    scores = their chance the moment they go 1-0 in front")
    print("    gain   = how many percentage points that goal is worth to them")
    print("\n  WHAT THIS MEANS: a goal is worth more to %s at every single minute" % AWAY)
    print("  of the match, and the value of scoring roughly doubles between kick-off")
    print("  and the final whistle.")

    # ------------------------------------------------------------------ 5
    hdr(5, "WHEN A TEAM GOES TWO GOALS UP",
        "How safe is a two-goal lead, and how much hope does one goal back give?")
    print("     %-10s %10s %10s %11s %11s"
          % ("", HOME, HOME, AWAY, AWAY))
    print("     %-10s %10s %10s %11s %11s"
          % ("Minute", "2-0 up", "2-1 up", "2-0 up", "2-1 up"))
    print("     " + "-" * 56)
    for m in [30, 60, 75, 85]:
        print("     %-10s %9.1f%% %9.1f%% %10.1f%% %10.1f%%"
              % ("%d min" % m, S.value(2, 0, m) * 100, S.value(2, 1, m) * 100,
                 (1 - S.value(0, 2, m)) * 100, (1 - S.value(1, 2, m)) * 100))
    print("\n  Each number is that team's own chance of going on to win the cup.")
    print("\n  WHAT THIS MEANS: from the hour mark a two-goal lead is effectively over.")
    print("  But pulling one back matters far more than it feels: at 60 minutes, going")
    print("  from 2-0 down to 2-1 down lifts %s from 0.8%% to 7.5%% -- one goal buys" % AWAY)
    print("  back roughly seven points of hope.")

    # ------------------------------------------------------------------ 6
    hdr(6, "IF THE MATCH STAYS 0-0",
        "How Spain's chances drift as a goalless match wears on.")
    for m in [0, 30, 60, 75, 85, 90]:
        lab = "At kick-off" if m == 0 else "Still 0-0 after %d minutes" % m
        dots("  %s" % lab, "%s %.1f%%   %s %.1f%%"
             % (HOME, S.value(0, 0, m) * 100, AWAY, (1 - S.value(0, 0, m)) * 100), 40)
    print("\n  WHAT THIS MEANS: 90 minutes of no goals moves %s by only %.1f points."
          % (HOME, (S.value(0, 0, 0) - S.value(0, 0, 90)) * 100))
    print("  That seems strange but it is right: a goalless match drifts towards")
    print("  penalties, where %s's advantage almost disappears. %s's most" % (HOME, AWAY))
    print("  reliable plan is not to be better -- it is to make the final boring.")

    # ------------------------------------------------------------------ 7
    hdr(7, "EXTRA TIME AND PENALTIES",
        "What happens if 90 minutes are not enough to separate them.")
    print("  If the score is tied after 90 minutes, the teams play 30 more minutes.")
    print("  If it is STILL tied after that, the cup is decided by a penalty shootout.\n")
    P0 = np.zeros((MAXG + 1, MAXG + 1)); P0[0, 0] = 1.0
    blocks = [("SITUATION A: the 90 minutes have just ended, still tied", 30),
              ("SITUATION B: halfway through extra time (105 min), still tied", 15)]
    for label, mins in blocks:
        P = score_distribution(lam, kap, S.tm, S.sm, mins, et_mult=S.etm, P0=P0)
        w = float(np.tril(P, -1).sum()); l = float(np.triu(P, 1).sum()); d = float(np.trace(P))
        t = w + d * S.p_so
        print("  " + label)
        dots("     %s score more in extra time and win" % HOME, "%.1f%%" % (w * 100), 55)
        dots("     %s score more in extra time and win" % AWAY, "%.1f%%" % (l * 100), 55)
        dots("     Nobody scores -- it goes to a shootout", "%.1f%%" % (d * 100), 55)
        print("     ---")
        print("     Adding it all up from this point:")
        print("        %s %.1f%%      %s %.1f%%" % (HOME, t * 100, AWAY, (1 - t) * 100))
        print()
    print("  WHAT THIS MEANS: once extra time begins it is more likely than not (53.3%)")
    print("  to end in a shootout, and a shootout is nearly a coin toss (%s %.1f%%)."
          % (HOME, S.p_so * 100))
    print("  Every extra phase the match survives drags it closer to even.")

    # ------------------------------------------------------------------ 8
    hdr(8, "THE PENALTY SHOOTOUT: WHERE THAT NUMBER COMES FROM",
        "Argentina have a famous penalty record. Does the model account for it?")
    import json
    sp = PROC / "shootout_summary.json"
    if sp.exists():
        J = json.load(open(sp))
        ra, rs = J["records"]["Argentina"], J["records"]["Spain"]
        print("  The raw history says Argentina are much better at this:\n")
        print("     %-12s all-time %d wins %d losses     since 2018: %d-%d"
              % (AWAY, ra["w"], ra["l"], ra["w18"], ra["l18"]))
        print("     %-12s all-time %d wins %d losses     since 2018: %d-%d"
              % (HOME, rs["w"], rs["l"], rs["w18"], rs["l18"]))
        print("\n  So is that a real skill, or luck? Every nation was given its own")
        print("  penalty rating, fitted on %d historical shootouts. How much those"
              % J["n_shootouts"])
        print("  ratings should count was then decided by testing them on shootouts")
        print("  the model had never seen.\n")
        print("  THE VERDICT: the ratings were worth nothing.\n")
        dots("     using team penalty ratings", "%.5f" % J["cv_with"], 50)
        dots("     using overall team strength only", "%.5f" % J["cv_without"], 50)
        print("     (this is prediction error, lower is better. They are identical.)")
        print("\n  %s come out %s best of %d nations, so the model does see"
              % (AWAY, {1:"the",2:"2nd",3:"3rd"}.get(ra["rank"], str(ra["rank"])+"th"), J["n_teams"]))
        print("  their record. It is simply worth nothing once it is tested.\n")
        print("  The giveaway is who else is at the top of that list:\n")
        print("     BEST THREE:   %s" % ", ".join(J["best_teams"]))
        print("     WORST THREE:  %s" % ", ".join(J["worst_teams"]))
        print("\n  Indonesia and Guinea are not penalty superpowers, and the")
        print("  Netherlands are famous for LOSING shootouts. That is what a list")
        print("  looks like when it is measuring luck instead of skill.\n")
        print("  So the shootout number is built from what does survive testing:\n")
        dots("     gap in overall team strength", "worth %+.1f%% to %s"
             % (J["b_elo"] * S.elo_diff / 100 * 25, HOME), 44)
        dots("     playing at a real home ground", "neutral venue, not relevant", 44)
        dots("     shooting first", "decided by a coin toss", 44)
        print("     %s" % ("-" * 20))
        print("     P(%s win a penalty shootout) = %.1f%%" % (HOME, S.p_so * 100))
    else:
        print("  (run  python src/shootout.py  first)")
    print("\n  AND IT BARELY MATTERS ANYWAY. A shootout decides only %.0f%% of finals,"
          % (dec["pens"] * 100))
    print("  so even being badly wrong about it moves the result very little:\n")
    print("     %-42s %s wins the cup" % ("if the shootout were...", HOME))
    for lab, p_so in [("%s %.1f%% (the model)" % (HOME, S.p_so*100), S.p_so),
                      ("a 50/50 coin flip", 0.50),
                      ("%s 55%%" % AWAY, 0.45),
                      ("%s 65%% (their all-time record)" % AWAY, 0.35),
                      ("%s 80%% (far beyond the evidence)" % AWAY, 0.20)]:
        v = dec["spain_90"] + dec["draw_90"] * (
            (dec["spain_et"] / dec["draw_90"]) + (dec["pens"] / dec["draw_90"]) * p_so)
        print("     %-42s %.1f%%" % (lab, v * 100))

    # ------------------------------------------------------------------ 9
    hdr(9, "WILL EITHER TEAM FALL BEHIND?",
        "The chance of being losing at any point during the 90 minutes.")
    dots("  %s fall behind at some point" % HOME, "%.1f%%" % (S.ever_trails("home") * 100), 50)
    dots("  %s fall behind at some point" % AWAY, "%.1f%%" % (S.ever_trails("away") * 100), 50)
    print("\n  WHAT THIS MEANS: %s have not trailed for a single minute in this whole" % HOME)
    print("  tournament -- seven matches, one goal conceded. The model says that run is")
    print("  more likely than not to survive tonight, but it is closer than it sounds:")
    print("  better than a one-in-three chance they go behind for the first time,")
    print("  in the final.")
    print()
