"""Tests for the odds helpers and the model-market blend."""
import pytest

from src.market import american_to_prob, devig, blend


def test_american_to_prob_even_money():
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-100) == pytest.approx(0.5)


def test_american_favourite_above_half():
    assert american_to_prob(-200) > 0.5
    assert american_to_prob(+200) < 0.5


def test_devig_removes_the_margin():
    # two -110 prices imply 52.4% each; de-vigged they must sum to exactly 1
    p, q = devig(american_to_prob(-110), american_to_prob(-110))
    assert (p + q) == pytest.approx(1.0, abs=1e-9)
    assert p == pytest.approx(0.5, abs=1e-9)


def test_blend_endpoints():
    # w = 1 is the pure model, w = 0 is the pure market
    assert blend(0.62, 0.50, w=1.0) == pytest.approx(0.62, abs=1e-6)
    assert blend(0.62, 0.50, w=0.0) == pytest.approx(0.50, abs=1e-6)


def test_blend_is_between_its_inputs():
    b = blend(0.62, 0.50, w=0.5)
    assert 0.50 < b < 0.62
