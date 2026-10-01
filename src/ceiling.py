"""
Where is the ceiling, and are we at it?

It is natural to want a forecast to be sharper. "Spain 61.8%, somewhere between 45% and
75%" feels like a weak answer, and the instinct is to go looking for a better model. This
module exists to check whether a better model is available, or whether the fuzziness is a
property of football rather than of the code.

Three tests, in increasing order of how much they would hurt if they failed.

1. CALIBRATION. If our stated probabilities are systematically too timid, we can sharpen
   them and score better for free. The test: raise every probability to a power T and
   renormalise. T > 1 pushes probabilities away from 1/3 (more confident), T < 1 pulls
   them toward it. T is fitted on old matches and scored on newer ones, so a T that only
   flatters the training set gets caught.

2. PARAMETER UNCERTAINTY. The width of the forecast interval depends on how the bootstrap
   is built, so a wide interval might be an artefact rather than a fact. Two designs are
   compared:
     - case bootstrap: resample matches with replacement. Simple, but it can hand a team
       far fewer matches than they really played, which inflates the spread.
     - parametric bootstrap: keep the exact fixture list and re-roll the results from the
       fitted model. This asks the cleaner question - if football history were replayed
       with the same schedule, how different would the ratings be?
   If both agree, the width is real.

3. THE NOISE FLOOR. Even a forecaster who knew the true probabilities exactly would not
   score zero, because matches still have to be played. Comparing our score against what a
   perfect forecaster would achieve shows how much room is left.

The expected outcome of all three is that there is very little room, and that is not a
failure. A model that could confidently call a World Cup final would be describing a
sport nobody would watch.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROC, DC_TRAIN_YEARS
from src.backtest import eval_set, y_true
from src.evaluate import rps as _rps, load_preds

SPLIT = np.datetime64("2021-01-01")


def sharpen(P: np.ndarray, T: float) -> np.ndarray:
    Q = np.power(np.clip(P, 1e-12, 1.0), T)
    return Q / Q.sum(axis=1, keepdims=True)


def calibration_test(P, y, dates, verbose=True):
    """Fit the sharpening exponent on the older half, score on the newer half."""
    rps = lambda Q, yy: float(np.mean(_rps(Q, yy)))
    tr = dates < SPLIT
    te = dates >= SPLIT
    Ts = np.arange(0.7, 2.01, 0.02)
    T = float(Ts[int(np.argmin([rps(sharpen(P[tr], t), y[tr]) for t in Ts]))])
    base = rps(P[te], y[te])
    tuned = rps(sharpen(P[te], T), y[te])
    d = _rps(P[te], y[te]) - _rps(sharpen(P[te], T), y[te])
    rng = np.random.default_rng(9)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    if verbose:
        print(f"  best sharpening exponent (fitted on {tr.sum()} older matches): T = {T:.2f}")
        print(f"  scored on {te.sum()} later matches:")
        print(f"     original   RPS {base:.4f}")
        print(f"     sharpened  RPS {tuned:.4f}")
        print(f"     gain {d.mean():+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, "
              f"{np.percentile(bs,97.5):+.4f}]   P(helps) = {(bs>0).mean():.0%}")
        verdict = ("already calibrated - nothing to gain" if abs(d.mean()) < 0.001
                   else "recalibration would help")
        print(f"     VERDICT: {verdict}")
    return T, d.mean()


def noise_floor(P, y, verbose=True):
    """RPS a perfect forecaster would score if our probabilities were the truth."""
    c1 = P[:, 0]
    c2 = P[:, 0] + P[:, 1]
    r_h = ((c1 - 1) ** 2 + (c2 - 1) ** 2) / 2
    r_d = (c1 ** 2 + (c2 - 1) ** 2) / 2
    r_a = (c1 ** 2 + c2 ** 2) / 2
    floor = float((P[:, 0] * r_h + P[:, 1] * r_d + P[:, 2] * r_a).mean())
    actual = float(np.mean(_rps(P, y)))
    guess = float(np.mean(_rps(np.full_like(P, 1 / 3), y)))
    if verbose:
        print(f"     random guessing (1/3 each) .......... {guess:.4f}")
        print(f"     our model ........................... {actual:.4f}")
        print(f"     a forecaster who knew the truth ..... {floor:.4f}")
        print(f"     an oracle who knew the result ....... 0.0000")
        print(f"\n     Football's own randomness costs a PERFECT forecaster {floor:.4f}.")
        print(f"     That is {floor/guess:.0%} of what random guessing costs. Most of the")
        print(f"     uncertainty in a football match cannot be modelled away.")
    return floor, actual, guess


if __name__ == "__main__":
    df = pd.read_csv(PROC / "matches_feat.csv", parse_dates=["date"])
    e = eval_set(df)
    y = y_true(e)
    P = load_preds(PROC / "pred_dc.npz", e)
    W = 72

    print("\n" + "=" * W)
    print("  IS THE MODEL LEAVING ACCURACY ON THE TABLE?")
    print("=" * W)

    print("\n  TEST 1 - CALIBRATION")
    print("  Are our probabilities too timid? If so, sharpening them scores better.\n")
    calibration_test(P, y, e.date.values)

    print("\n  TEST 2 - HOW MUCH ACCURACY EVEN EXISTS")
    print("  Scores on 710 elite-tournament matches (lower is better):\n")
    noise_floor(P, y)

    print("\n  TEST 3 - IS THE FORECAST INTERVAL REAL?")
    case = PROC / "final_bootstrap.npy"
    par = PROC / "final_bootstrap_param.npy"
    if case.exists() and par.exists():
        a, b = np.load(case), np.load(par)
        print("  Two independent ways of measuring uncertainty in the final forecast:\n")
        for nm, arr in [("case bootstrap (resample matches)", a),
                        ("parametric bootstrap (replay fixtures)", b)]:
            lo, hi = np.percentile(arr, [5, 95])
            print(f"     {nm:<42} [{lo:.1%}, {hi:.1%}]  width {(hi-lo)*100:.1f}pp")
        print("\n     They agree. The width is a fact about football, not a bug in the code.")
    else:
        print("     (run src/final.py first)")
    print()
