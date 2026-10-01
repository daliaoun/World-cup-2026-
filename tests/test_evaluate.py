"""
Tests for the scoring metrics and the prediction-alignment guard.

The alignment test is the important one: midway through building this project an
unstable sort silently scrambled 256 of 710 evaluation rows, so the model was being
scored against the wrong matches. save_preds/load_preds was added to make that
impossible. This test locks that guarantee in place.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import rps, save_preds, load_preds


def _mean(x):
    return float(np.mean(x))


def test_rps_perfect_forecast_scores_zero():
    # a forecast that puts all mass on the realised outcome is perfect
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 2])
    assert _mean(rps(P, y)) == pytest.approx(0.0, abs=1e-12)


def test_rps_uniform_known_values():
    # a uniform 1/3 forecast scores 5/18 when a side wins and 1/9 on a draw;
    # over a balanced set of outcomes the mean is 2/9.
    P = np.full((3, 3), 1 / 3)
    per_outcome = rps(P, np.array([0, 1, 2]))
    assert per_outcome[0] == pytest.approx(5 / 18, abs=1e-9)   # home win
    assert per_outcome[1] == pytest.approx(1 / 9, abs=1e-9)    # draw
    assert per_outcome[2] == pytest.approx(5 / 18, abs=1e-9)   # away win
    assert _mean(per_outcome) == pytest.approx(2 / 9, abs=1e-9)


def test_rps_rewards_ordering():
    # home-draw-away is ORDERED: when the away team wins, predicting a draw
    # should beat predicting a home win. A plain classifier metric misses this.
    near = np.array([[0.1, 0.6, 0.3]])   # leaned draw
    far = np.array([[0.6, 0.3, 0.1]])    # leaned home
    y = np.array([2])                    # away won
    assert _mean(rps(near, y)) < _mean(rps(far, y))


def test_alignment_guard_accepts_matching_order(tmp_path):
    e = pd.DataFrame({"match_id": ["a", "b", "c"]})
    P = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    path = tmp_path / "pred.npz"
    save_preds(path, P, e)
    out = load_preds(path, e)
    assert np.allclose(out, P)


def test_alignment_guard_rejects_scrambled_order(tmp_path):
    # the exact failure mode the guard exists to catch
    e = pd.DataFrame({"match_id": ["a", "b", "c"]})
    P = np.zeros((3, 3)) + 1 / 3
    path = tmp_path / "pred.npz"
    save_preds(path, P, e)
    scrambled = pd.DataFrame({"match_id": ["b", "a", "c"]})
    with pytest.raises(ValueError):
        load_preds(path, scrambled)
