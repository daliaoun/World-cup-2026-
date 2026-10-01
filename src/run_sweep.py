"""Walk-forward hyperparameter sweep for Dixon-Coles. Writes results incrementally."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC
from src.backtest import eval_set, y_true, backtest_dc, backtest_elo
from src.evaluate import summary

df = pd.read_csv(PROC / "matches_elo.csv", parse_dates=["date"])
e = eval_set(df)
y = y_true(e)

rows = []
# baselines
Pu = np.tile([1 / 3, 1 / 3, 1 / 3], (len(e), 1))
rows.append(dict(model="uniform", xi=np.nan, lam2=np.nan, **summary(Pu, y)))
base = np.array([np.mean(y == 0), np.mean(y == 1), np.mean(y == 2)])
rows.append(dict(model="base_rates", xi=np.nan, lam2=np.nan,
                 **summary(np.tile(base, (len(e), 1)), y)))
rows.append(dict(model="elo", xi=np.nan, lam2=np.nan, **summary(backtest_elo(e), y)))
pd.DataFrame(rows).to_csv(PROC / "dc_sweep.csv", index=False)

grid = [(xi, l2)
        for xi in [0.0000, 0.0005, 0.0010, 0.0020, 0.0035]
        for l2 in [2.0, 8.0, 20.0]]

for xi, l2 in grid:
    P, _ = backtest_dc(df, e, xi=xi, lam2=l2)
    s = summary(P, y)
    rows.append(dict(model="dixon_coles", xi=xi, lam2=l2, **s))
    pd.DataFrame(rows).to_csv(PROC / "dc_sweep.csv", index=False)
    np.save(PROC / f"P_dc_{xi:.4f}_{l2:.0f}.npy", P)
    print(f"xi={xi:.4f} lam2={l2:>5.1f}  RPS={s['RPS']:.4f}  "
          f"ll={s['logloss']:.4f}  acc={s['accuracy']:.1%}", flush=True)

print("DONE")
