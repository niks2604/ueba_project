"""
Tests for the Spark FeaturePipeline.

Marked `spark` (and `slow`) because they spin up a local Spark session.
Run just these with:   pytest -m spark
Skip them with:        pytest -m "not spark"
"""

import numpy as np
import pandas as pd
import pytest

from features import FeaturePipeline

pytestmark = [pytest.mark.spark, pytest.mark.slow]

N_DAYS = 15
RAW_COLS = ["bytes", "logins", "hosts", "fails"]


def _small_dataset():
    """
    Two peer groups, 6 users each, 15 days. User 0 has a clean upward
    ramp in bytes so we can check Spearman/ramp signals; everyone else
    is flat-with-noise.
    """
    rng = np.random.default_rng(0)
    rows = []
    user_id = 0
    for group, base in [("engineer", 1000), ("finance", 500)]:
        for _ in range(6):
            ramp_user = user_id == 0
            for day in range(N_DAYS):
                if ramp_user:
                    bytes_ = base + day * 100      # strictly increasing
                else:
                    bytes_ = base + int(rng.normal(0, 20))
                rows.append((
                    user_id, group, day,
                    bytes_,
                    10 + int(rng.normal(0, 1)),
                    20 + int(rng.normal(0, 1)),
                    int(rng.poisson(1)),
                    False, "",
                ))
            user_id += 1
    return pd.DataFrame(rows, columns=[
        "user_id", "group", "day",
        "bytes", "logins", "hosts", "fails",
        "is_anomaly", "anomaly_type",
    ])


@pytest.fixture(scope="module")
def features_pdf(config, spark):
    """Feature table computed once for this module."""
    sdf = spark.createDataFrame(_small_dataset())
    pipeline = FeaturePipeline(config.features, spark)
    out = pipeline.compute_all(sdf)
    return out.toPandas()


def test_row_count_preserved(features_pdf):
    assert len(features_pdf) == 12 * N_DAYS


def test_raw_columns_survive(features_pdf):
    for c in RAW_COLS:
        assert c in features_pdf.columns


@pytest.mark.parametrize("col", [
    "bytes_z_self",        # self_z
    "bytes_rz_self",       # robust_z
    "bytes_z_peer_loo",    # peer_z
    "bytes_roll7_mean",    # rolling
    "bytes_diff1",         # diff
    "bytes_ramp_signal",   # ramp
    "bytes_spearman",      # spearman
    "bytes_spearman_pos",
])
def test_expected_feature_columns_present(features_pdf, col):
    assert col in features_pdf.columns


def test_ramp_user_has_positive_spearman(features_pdf):
    # User 0's bytes strictly increase, so trailing Spearman → 1.0 once
    # the window has enough history.
    ramp = features_pdf[features_pdf["user_id"] == 0].sort_values("day")
    late = ramp[ramp["day"] >= 5]
    assert late["bytes_spearman"].min() > 0.9
    assert (ramp["bytes_spearman_pos"] >= 0).all()


def test_ramp_user_has_positive_slope(features_pdf):
    ramp = features_pdf[features_pdf["user_id"] == 0].sort_values("day")
    late = ramp[ramp["day"] >= 5]
    assert (late["bytes_ramp_slope"] > 0).all()
    # Gated signal equals R^2 (>0) when slope is positive.
    assert (late["bytes_ramp_signal"] > 0).all()


def test_diff_matches_day_over_day(features_pdf):
    ramp = features_pdf[features_pdf["user_id"] == 0].sort_values("day")
    # User 0 increases bytes by exactly 100/day.
    later = ramp[ramp["day"] >= 1]
    assert np.allclose(later["bytes_diff1"], 100.0)


def test_first_day_self_z_is_null_or_nan(features_pdf):
    # Day 0 has no past, so the self z-score has no defined mean/std.
    day0 = features_pdf[features_pdf["day"] == 0]
    assert day0["bytes_mean_self"].isna().all()
