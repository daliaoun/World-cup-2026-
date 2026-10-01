"""
Tests for the exact minute-by-minute scoreline engine.

These guard the properties the whole final forecast rests on: that the distribution is
a real distribution (sums to one), and that a stronger attack produces a higher win
probability.
"""
import numpy as np
import pytest

from src.statesim import score_distribution, proba_1x2

# neutral multipliers: no time or score-state effect, so results are easy to reason about
FLAT_TIME = np.ones(6)
FLAT_STATE = np.ones(5)


def test_distribution_sums_to_one():
    P = score_distribution(1.4, 1.1, FLAT_TIME, FLAT_STATE, minutes=90)
    assert P.sum() == pytest.approx(1.0, abs=1e-9)
    assert (P >= 0).all()


def test_1x2_sums_to_one():
    P = score_distribution(1.3, 1.0, FLAT_TIME, FLAT_STATE, minutes=90)
    h, d, a = proba_1x2(P)
    assert (h + d + a) == pytest.approx(1.0, abs=1e-9)


def test_equal_teams_are_symmetric():
    P = score_distribution(1.2, 1.2, FLAT_TIME, FLAT_STATE, minutes=90)
    h, d, a = proba_1x2(P)
    assert h == pytest.approx(a, abs=1e-6)


def test_stronger_attack_wins_more_often():
    strong = proba_1x2(score_distribution(2.0, 0.8, FLAT_TIME, FLAT_STATE))
    even = proba_1x2(score_distribution(1.2, 1.2, FLAT_TIME, FLAT_STATE))
    assert strong[0] > even[0]          # home win prob rises with home scoring rate


def test_zero_rates_force_nil_nil():
    # with (almost) no scoring, essentially all mass sits on 0-0 -> a draw
    P = score_distribution(1e-6, 1e-6, FLAT_TIME, FLAT_STATE, minutes=90)
    assert P[0, 0] == pytest.approx(1.0, abs=1e-3)
