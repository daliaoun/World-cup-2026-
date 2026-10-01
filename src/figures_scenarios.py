"""Scenario figures for the final."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIG
from src.scenarios import main

INK, ESP, ARG, MUT = "#1b1b1f", "#c60b1e", "#75AADB", "#8a8a92"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfcfd4", "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9,
})

S, lam, kap = main()
mins = list(range(1, 90, 2))
sw = S.swing(mins)
dec = S.decompose()

# ---------------------------------------------- fig 9: what a goal is worth
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
a = ax[0]
a.plot(sw.minute, sw.arg_swing * 100, color=ARG, lw=2.4, label="an Argentina goal")
a.plot(sw.minute, sw.spain_swing * 100, color=ESP, lw=2.4, label="a Spain goal")
a.set_xlabel("minute the goal is scored (from level)")
a.set_ylabel("swing in that team's title chance (pp)")
a.set_title("What is the opening goal worth?")
a.legend(frameon=False, fontsize=9, loc="upper left")
a.annotate(f"{sw.arg_swing.iloc[-1]*100:.0f}pp", (89, sw.arg_swing.iloc[-1] * 100),
           textcoords="offset points", xytext=(-30, 6), color=ARG, fontweight="bold", fontsize=9)
a.annotate(f"{sw.spain_swing.iloc[-1]*100:.0f}pp", (89, sw.spain_swing.iloc[-1] * 100),
           textcoords="offset points", xytext=(-30, -14), color=ESP, fontweight="bold", fontsize=9)
a.text(3, 55, "a goal is worth more to Argentina\nall night — the underdog needs it more",
       fontsize=8.3, style="italic", color="#42424a")

a = ax[1]
lvl = [S.value(0, 0, m) * 100 for m in [0, 15, 30, 45, 60, 75, 90]]
a.plot([0, 15, 30, 45, 60, 75, 90], lvl, "o-", color="#6a6a72", lw=2.4, ms=5)
a.axhline(50, color=MUT, ls="--", lw=1)
a.set_ylim(45, 68)
a.set_xlabel("minute"); a.set_ylabel("Spain's title chance (%)")
a.set_title("If nobody scores, almost nothing happens")
a.annotate("61.8%", (0, lvl[0]), textcoords="offset points", xytext=(6, 6),
           fontweight="bold", fontsize=9)
a.annotate("57.7%", (90, lvl[-1]), textcoords="offset points", xytext=(-38, -16),
           fontweight="bold", fontsize=9)
a.text(12, 48.5, "90 goalless minutes move Spain\nby just 4 points — because level at 90'\nmeans penalties, where they are barely ahead",
       fontsize=8.3, style="italic", color="#42424a")
plt.tight_layout(); plt.savefig(FIG / "09_goal_value.png", dpi=160); plt.close()

# ---------------------------------------------- fig 10: paths to the trophy
fig, ax = plt.subplots(figsize=(10, 3.9))
rows = [("Spain", [dec["spain_90"], dec["spain_et"], dec["spain_pens"]], ESP),
        ("Argentina", [dec["arg_90"], dec["arg_et"], dec["arg_pens"]], ARG)]
shades = [1.0, 0.62, 0.34]
labels = ["won inside 90 minutes", "won in extra time", "won on penalties"]
for i, (nm, parts, base) in enumerate(rows):
    left = 0
    for j, p in enumerate(parts):
        ax.barh(i, p * 100, left=left * 100, color=base, alpha=shades[j], height=.5,
                edgecolor="white", linewidth=1.4)
        if p > 0.04:
            ax.text((left + p / 2) * 100, i, f"{p:.1%}", ha="center", va="center",
                    color="white" if j == 0 else INK, fontweight="bold", fontsize=9.5)
        left += p
    ax.text(left * 100 + 1.2, i, f"{left:.1%}", va="center", fontweight="bold", fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Spain", "Argentina"], fontsize=11)
ax.set_xlim(0, 70); ax.set_xlabel("probability of lifting the trophy (%)")
ax.set_title("Three ways to win a World Cup final")
h = [plt.Rectangle((0, 0), 1, 1, color="#6a6a72", alpha=s) for s in shades]
ax.legend(h, labels, frameon=False, fontsize=8.5, ncol=3, loc="lower right")
ax.text(0.5, -0.42, f"{(dec['arg_et']+dec['arg_pens'])/(dec['arg_90']+dec['arg_et']+dec['arg_pens']):.0%}"
        " of Argentina's title chance requires the match to go past 90 minutes"
        f" (Spain: {(dec['spain_et']+dec['spain_pens'])/(dec['spain_90']+dec['spain_et']+dec['spain_pens']):.0%})",
        transform=ax.get_yaxis_transform(), fontsize=8.5, style="italic", color="#42424a")
plt.tight_layout(); plt.savefig(FIG / "10_paths_to_trophy.png", dpi=160); plt.close()
print("wrote 09_goal_value.png, 10_paths_to_trophy.png")
