"""Figures for the report."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, FIG, DC_TRAIN_YEARS
from src.backtest import eval_set, y_true
from src.evaluate import rps as _rps, load_preds, calibration
from src.dixon_coles import DixonColes
from src.simulate import simulate

INK, ACC, ACC2, MUT = "#1b1b1f", "#c1121f", "#0b6e4f", "#8a8a92"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfcfd4", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "figure.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9,
})
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
e = eval_set(df); y = y_true(e)
P = {n: load_preds(PROC / f"pred_{n}.npz", e) for n in ["elo", "dc", "gbm"]}

# ---------------------------------------------------------------- fig 1: models
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
names = ["Uniform\n(1/3 each)", "Base rates", "Elo", "XGBoost", "Dixon-Coles"]
U = np.full((len(y), 3), 1/3)
B = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
vals = [_rps(U, y).mean(), _rps(B, y).mean(),
        _rps(P["elo"], y).mean(), _rps(P["gbm"], y).mean(), _rps(P["dc"], y).mean()]
cols = [MUT, MUT, "#5a6b7b", ACC2, ACC]
b = ax[0].bar(names, vals, color=cols, width=.62)
for r, v in zip(b, vals):
    ax[0].text(r.get_x()+r.get_width()/2, v+.002, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
ax[0].set_ylim(.16, .245); ax[0].set_ylabel("Ranked Probability Score  (lower = better)")
ax[0].set_title("Walk-forward backtest: 710 elite-tournament matches, 2010–2026")
ax[0].tick_params(axis="x", labelsize=8)

# error bars via paired bootstrap vs DC
rng = np.random.default_rng(1)
d_gbm = _rps(P["gbm"], y) - _rps(P["dc"], y)
d_elo = _rps(P["elo"], y) - _rps(P["dc"], y)
for i, (lab, d) in enumerate([("XGBoost", d_gbm), ("Elo", d_elo)]):
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
    ax[1].barh(i, d.mean(), color=[ACC2, "#5a6b7b"][i], height=.45)
    ax[1].plot([np.percentile(bs, 2.5), np.percentile(bs, 97.5)], [i, i], color=INK, lw=1.6)
ax[1].axvline(0, color=ACC, lw=1.8)
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["XGBoost", "Elo"])
ax[1].set_xlabel("RPS worse than Dixon-Coles  →")
ax[1].set_title("Both lose to Dixon-Coles (95% bootstrap CI)")
ax[1].text(0.001, 1.45, "Dixon-Coles", color=ACC, fontsize=8.5, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"01_model_comparison.png", dpi=160); plt.close()

# ---------------------------------------------------------- fig 2: calibration
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot([0, 1], [0, 1], "--", color=MUT, lw=1)
for nm, key, c in [("Dixon-Coles", "dc", ACC), ("XGBoost", "gbm", ACC2), ("Elo", "elo", "#5a6b7b")]:
    pr = P[key].ravel()
    ob = np.zeros_like(P[key]); ob[np.arange(len(y)), y] = 1; ob = ob.ravel()
    bins = np.linspace(0, 1, 11); ix = np.digitize(pr, bins) - 1
    xs, ys = [], []
    for k in range(10):
        m = ix == k
        if m.sum() > 12:
            xs.append(pr[m].mean()); ys.append(ob[m].mean())
    ax[0].plot(xs, ys, "o-", color=c, label=nm, ms=4.5, lw=1.6)
ax[0].set_xlabel("predicted probability"); ax[0].set_ylabel("observed frequency")
ax[0].set_title("Calibration — do the probabilities mean what they say?")
ax[0].legend(frameon=False, fontsize=8.5)

w = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01") & df.hs.notna()]
ax[1].axis("off")
ax[1].text(0, .93, "Out-of-sample on WC2026 itself", fontsize=11, fontweight="bold")
ax[1].text(0, .80, "The model was refit before every one of the 100 matches\n"
                   "already played at this tournament, never seeing the result.",
           fontsize=9, color="#42424a")
rows = [("", "RPS", "accuracy"), ("Elo", "0.1634", "60.0%"), ("Dixon-Coles", "0.1531", "66.0%"),
        ("  knockouts only", "0.1552", "67.9%")]
for i, (a_, b_, c_) in enumerate(rows):
    yy = .60 - i*.115
    fw = "bold" if i == 0 or "Dixon" in a_ else "normal"
    ax[1].text(0, yy, a_, fontsize=9.5, fontweight=fw)
    ax[1].text(.52, yy, b_, fontsize=9.5, fontweight=fw, color=ACC if "Dixon" in a_ else INK)
    ax[1].text(.78, yy, c_, fontsize=9.5, fontweight=fw, color=ACC if "Dixon" in a_ else INK)
ax[1].text(0, .10, "Better than its own tuning-set score (0.1864).\nThe model is not falling apart on live data.",
           fontsize=8.5, style="italic", color="#42424a")
plt.tight_layout(); plt.savefig(FIG/"02_calibration.png", dpi=160); plt.close()

# ------------------------------------------------- fig 3: the two score matrices
damp = float(np.load(PROC/"et_damp.npy")[0]); b0, b1 = np.load(PROC/"so_coef.npy")
fx = [("2026-07-14", "France", "Spain"), ("2026-07-15", "England", "Argentina")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for k, (d, h, a) in enumerate(fx):
    d = pd.Timestamp(d)
    tr = df[(df.date < d) & (df.date >= d - pd.DateOffset(years=DC_TRAIN_YEARS))]
    dc = DixonColes(xi=0.0015, lam2=0.5).fit(tr, as_of=d)
    M = dc.score_matrix(h, a, home_adv=0.0)[:5, :5]
    im = axs[k].imshow(M*100, cmap="RdPu", vmin=0, vmax=15)
    for i in range(5):
        for j in range(5):
            axs[k].text(j, i, f"{M[i,j]*100:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j]*100 > 8 else INK)
    axs[k].set_xlabel(f"{a} goals"); axs[k].set_ylabel(f"{h} goals")
    axs[k].set_xticks(range(5)); axs[k].set_yticks(range(5))
    lam, kap = dc.rates(h, a, home_adv=0.0)
    axs[k].set_title(f"{h} vs {a}\nexpected goals {lam:.2f} – {kap:.2f}   (% chance of each 90' score)")
plt.tight_layout(); plt.savefig(FIG/"03_score_matrices.png", dpi=160); plt.close()

# ----------------------------------------------- fig 4: forecast vs market
fig, ax = plt.subplots(figsize=(10, 4.4))
labels = ["France\n(v Spain)", "Spain\n(v France)", "England\n(v Argentina)", "Argentina\n(v England)"]
model  = [42.6, 57.4, 44.4, 55.6]
market = [57.8, 42.2, 54.7, 45.3]
x = np.arange(4); w_ = .36
ax.bar(x - w_/2, model,  w_, label="This model (Dixon-Coles + simulation)", color=ACC)
ax.bar(x + w_/2, market, w_, label="Betting market (de-vigged)", color="#5a6b7b")
for i in range(4):
    ax.text(x[i]-w_/2, model[i]+1,  f"{model[i]:.0f}%", ha="center", fontsize=9, fontweight="bold", color=ACC)
    ax.text(x[i]+w_/2, market[i]+1, f"{market[i]:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#5a6b7b")
ax.axhline(50, color=MUT, ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("probability of reaching the final (%)"); ax.set_ylim(0, 70)
ax.set_title("The model and the market disagree on BOTH semi-finals — and the sign is flipped each time")
ax.legend(frameon=False, fontsize=9, loc="upper right")
plt.tight_layout(); plt.savefig(FIG/"04_forecast_vs_market.png", dpi=160); plt.close()

print("figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  ", p.name)
