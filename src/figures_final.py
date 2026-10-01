"""Figures for the final."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, FIG

INK, ESP, ARG, MUT = "#1b1b1f", "#c60b1e", "#75AADB", "#8a8a92"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfcfd4", "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9,
})

bs = np.load(PROC / "final_bootstrap.npy")
piv = pd.read_csv(PROC / "win_prob_surface.csv", index_col=0)
tm = np.load(PROC / "time_mult.npy")
sm = np.load(PROC / "state_mult.npy")

POINT, MKT = 0.618, 0.577

# ---------------------------------------------------- fig 1: forecast + uncertainty
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
a.barh([1], [POINT * 100], color=ESP, height=.5)
a.barh([0], [(1 - POINT) * 100], color=ARG, height=.5)
a.text(POINT * 100 - 3, 1, f"{POINT:.1%}", va="center", ha="right", color="white", fontweight="bold", fontsize=13)
a.text((1 - POINT) * 100 - 3, 0, f"{1-POINT:.1%}", va="center", ha="right", color="white", fontweight="bold", fontsize=13)
a.set_yticks([1, 0]); a.set_yticklabels(["Spain", "Argentina"], fontsize=11)
a.set_xlim(0, 100); a.set_xlabel("probability of lifting the trophy (%)")
a.axvline(50, color=MUT, ls="--", lw=1)
a.set_title("The forecast")

a = ax[1]
a.hist(bs * 100, bins=26, color="#d8d8de", edgecolor="white")
a.axvline(POINT * 100, color=ESP, lw=2.4, label=f"point estimate {POINT:.1%}")
lo, hi = np.percentile(bs, [5, 95]) * 100
a.axvspan(lo, hi, color=ESP, alpha=.10)
a.axvline(50, color=INK, ls="--", lw=1.2)
a.axvline(MKT * 100, color="#5a6b7b", lw=2, ls=":", label=f"betting market {MKT:.1%}")
a.set_xlabel("Spain's win probability across 200 refits on resampled history (%)")
a.set_yticks([])
a.set_title(f"How sure is it? 90% interval: {lo:.0f}% to {hi:.0f}%")
a.legend(frameon=False, fontsize=8.5, loc="upper left")
a.text(lo + 1, a.get_ylim()[1] * .45,
       f"on {(bs<0.5).mean():.0%} of resampled\nhistories the model\nwould pick Argentina",
       fontsize=8, color="#42424a")
plt.tight_layout(); plt.savefig(FIG / "05_final_forecast.png", dpi=160); plt.close()

# ------------------------------------------- fig 2: live win probability (the special one)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
minutes = [int(c) for c in piv.columns]
order = ["2-0", "1-0", "0-0", "1-1", "0-1", "0-2"]
cols = {"2-0": "#7a0812", "1-0": ESP, "0-0": "#9a9aa2", "1-1": "#6a6a72",
        "0-1": ARG, "0-2": "#2a5c8a"}
for s in order:
    if s in piv.index:
        ax.plot(minutes, piv.loc[s] * 100, "o-", color=cols[s], lw=2, ms=4.5,
                label=f"{s}  (Spain–Argentina)")
ax.axhline(50, color=MUT, ls="--", lw=1)
ax.set_xlabel("minute"); ax.set_ylabel("Spain's chance of winning the trophy (%)")
ax.set_ylim(-3, 103); ax.set_xticks(minutes)
ax.set_title("Live win probability: what the model thinks at any point in the match")
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="center left")
ax.text(60, 34, "level at 0-0 and 1-1 are the same state:\nwhat matters is the gap, not the score",
        fontsize=7.8, style="italic", color="#42424a")
plt.tight_layout(); plt.savefig(FIG / "06_live_win_probability.png", dpi=160); plt.close()

# ------------------------------------------- fig 3: the confound (raw vs controlled)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
labs = ["2+ behind", "1 behind", "level", "1 ahead", "2+ ahead"]
raw = [2.493 / 1.204, 1.436 / 1.204, 1.0, 0.944 / 1.204, 0.822 / 1.204]
a = ax[0]
a.bar(labs, raw, color="#b0b0b8", width=.6)
a.axhline(1, color=INK, ls="--", lw=1)
a.set_title("Raw data: 'teams that lead score MORE'")
a.set_ylabel("goal rate, relative to level")
a.tick_params(axis="x", labelsize=8)
a.text(.5, .90, "an artefact — the teams who lead\nare mostly good teams beating bad ones",
       transform=a.transAxes, ha="center", fontsize=8.5, color="#8a2a2a", style="italic")
a = ax[1]
cs = [ESP if v < 1 else "#0b6e4f" for v in sm]
a.bar(labs, sm, color=cs, width=.6)
a.axhline(1, color=INK, ls="--", lw=1)
for i, v in enumerate(sm):
    a.text(i, v + .012, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
a.set_ylim(.8, 1.16)
a.set_title("Controlled for team strength: the effect reverses")
a.set_ylabel("multiplier on the team's own baseline rate")
a.tick_params(axis="x", labelsize=8)
a.text(.5, .90, "teams protecting a 1-goal lead score 12% below their own rate",
       transform=a.transAxes, ha="center", fontsize=8.5, color="#0b6e4f", style="italic")
plt.tight_layout(); plt.savefig(FIG / "07_score_effects.png", dpi=160); plt.close()

# ------------------------------------------- fig 4: when goals arrive
fig, ax = plt.subplots(figsize=(9.5, 4.0))
xs = [f"{b*15+1}-{b*15+15}" for b in range(6)]
bars = ax.bar(xs, tm, color=[ESP if v > 1 else "#b0b0b8" for v in tm], width=.62)
ax.axhline(1, color=INK, ls="--", lw=1.2)
for r, v in zip(bars, tm):
    ax.text(r.get_x() + r.get_width() / 2, v + .015, f"{v:.2f}", ha="center",
            fontsize=9, fontweight="bold")
ax.set_ylim(0, 1.62); ax.set_xlabel("minute"); ax.set_ylabel("multiplier on baseline goal rate")
ax.set_title("Goals are not spread evenly: the last 15 minutes produce 1.9× the first 15")
ax.text(.99, .92, "(the 31–45 and 76–90 bars are inflated:\nstoppage-time goals are coded at 45 and 90)",
        transform=ax.transAxes, ha="right", fontsize=7.8, style="italic", color="#42424a")
plt.tight_layout(); plt.savefig(FIG / "08_goal_timing.png", dpi=160); plt.close()

print("wrote:")
for p in sorted(FIG.glob("0[5-8]*.png")):
    print("  ", p.name)
