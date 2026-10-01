"""
Blending the model with the betting market.

The model's known blind spot is information it structurally cannot see: injuries,
suspensions, who trained on Friday, whether a manager intends to rotate. There is one
place that information is already aggregated and priced, by people with money at stake -
the market.

So rather than acquiring player data we cannot get, we can borrow the market's version of
it. Blending in log-odds space:

    logit(p_blend) = w * logit(p_model) + (1 - w) * logit(p_market)

Log-odds rather than raw probabilities, because averaging probabilities directly drags
everything toward 50% and distorts confident forecasts.

HONESTY ABOUT VALIDATION. The forecasting literature broadly finds that model-market
blends beat either component, and the mechanism is not mysterious: the two carry partly
independent information. But this project has no historical odds to test it on, so that
finding is imported, not earned. Worse, the only direct evidence available points the
other way: on the two semi-finals the model disagreed with the market and was right both
times. Blending would have made both forecasts worse.

Two matches is not evidence of anything. Neither is a literature result we could not
reproduce. This module therefore reports the blend rather than adopting it, and the
default weight is 1.0 - the unblended model - so nothing changes silently.
"""
from __future__ import annotations
import numpy as np


def american_to_prob(odds: float) -> float:
    return (-odds) / ((-odds) + 100) if odds < 0 else 100 / (odds + 100)


def devig(p_a: float, p_b: float) -> tuple[float, float]:
    """Strip the bookmaker's margin proportionally."""
    s = p_a + p_b
    return p_a / s, p_b / s


def _logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def blend(p_model: float, p_market: float, w: float = 1.0) -> float:
    """w = 1.0 -> pure model, w = 0.0 -> pure market, w = 0.5 -> equal weight."""
    return float(_sigmoid(w * _logit(p_model) + (1 - w) * _logit(p_market)))


if __name__ == "__main__":
    MODEL = 0.618                    # Spain, from src/final.py
    BOOKS = {"FanDuel": (-148, 129), "DraftKings": (-164, 134)}

    print("\n" + "=" * 70)
    print("  BLENDING THE MODEL WITH THE MARKET -- Spain vs Argentina")
    print("=" * 70)
    print("\n  Bookmaker prices, with the margin stripped out:\n")
    mkts = []
    for name, (o_h, o_a) in BOOKS.items():
        p, q = devig(american_to_prob(o_h), american_to_prob(o_a))
        mkts.append(p)
        print(f"     {name:<14} Spain {p:.1%}   Argentina {q:.1%}")
    market = float(np.mean(mkts))
    print(f"\n     average           Spain {market:.1%}   Argentina {1-market:.1%}")
    print(f"     our model         Spain {MODEL:.1%}   Argentina {1-MODEL:.1%}")
    print(f"     disagreement      {abs(MODEL-market)*100:.1f} points\n")

    print("  What different blends would give:\n")
    print(f"     {'weight on model':<22}{'Spain':>10}{'Argentina':>12}")
    for w in [1.0, 0.75, 0.5, 0.25, 0.0]:
        b = blend(MODEL, market, w)
        tag = "  <- current default" if w == 1.0 else ("  <- pure market" if w == 0.0 else "")
        print(f"     {w:<22.2f}{b:>9.1%}{1-b:>12.1%}{tag}")

    print("\n  STATUS: reported, not adopted. The default remains the unblended model.")
    print("  We have no historical odds to test a blend on, and the only direct")
    print("  evidence -- two semi-finals where the model beat the market -- argues")
    print("  against it. Two matches proves nothing either way, which is the point:")
    print("  there is no basis here for changing the forecast, in either direction.\n")
