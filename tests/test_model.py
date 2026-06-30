"""Tests for the AnomalyDetector wrapper."""

import copy

import numpy as np
import pytest

from model import AnomalyDetector


def test_unsupported_algorithm_raises(config):
    bad = copy.deepcopy(config.model)
    bad.algorithm = "deep_magic"
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        AnomalyDetector(bad)


def test_contamination_auto_passthrough(config):
    cfg = copy.deepcopy(config.model)
    cfg.contamination = "auto"
    det = AnomalyDetector(cfg)
    assert det._parse_contamination() == "auto"


def test_contamination_float_parsed(config):
    cfg = copy.deepcopy(config.model)
    cfg.contamination = "0.05"
    det = AnomalyDetector(cfg)
    assert det._parse_contamination() == pytest.approx(0.05)


def _toy_matrix():
    """Tight normal cluster plus a handful of obvious outliers."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(200, 3))
    outliers = np.array([[50.0, 50.0, 50.0]] * 5)
    X = np.vstack([normal, outliers])
    outlier_idx = np.arange(200, 205)
    return X, outlier_idx


def test_score_length_matches_rows(config):
    X, _ = _toy_matrix()
    scores = AnomalyDetector(config.model).fit_and_score(X, seed=0)
    assert scores.shape == (len(X),)


def test_outliers_score_higher(config):
    X, outlier_idx = _toy_matrix()
    scores = AnomalyDetector(config.model).fit_and_score(X, seed=0)
    # Convention: higher score = more anomalous. Planted outliers
    # should dominate the top of the ranking.
    top5 = set(np.argsort(scores)[-5:])
    assert top5 == set(outlier_idx)


def test_deterministic_for_fixed_seed(config):
    X, _ = _toy_matrix()
    det = AnomalyDetector(config.model)
    a = det.fit_and_score(X, seed=7)
    b = det.fit_and_score(X, seed=7)
    np.testing.assert_array_equal(a, b)
