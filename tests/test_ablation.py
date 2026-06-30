"""Tests for AblationRunner's pure helpers (no Spark, no MLflow)."""

import pandas as pd
import pytest

from ablation import AblationRunner


@pytest.fixture
def runner(config):
    return AblationRunner(config)


# ------------------------------------------------------------------
# Feature column resolution
# ------------------------------------------------------------------

def test_baseline_is_raw_columns_only(config, runner):
    cols = runner._resolve_feature_columns("baseline")
    assert cols == list(config.features.raw_columns)


def test_raw_columns_always_included(runner):
    cols = runner._resolve_feature_columns("z_scores")
    for raw in ["bytes", "logins", "hosts", "fails"]:
        assert raw in cols


def test_z_scores_adds_all_three_z_families(runner):
    cols = runner._resolve_feature_columns("z_scores")
    # self_z, robust_z, peer_z each over all 4 raw columns.
    for c in ["bytes", "logins", "hosts", "fails"]:
        assert f"{c}_z_self" in cols
        assert f"{c}_rz_self" in cols
        assert f"{c}_z_peer_loo" in cols


def test_spearman_is_bytes_only(runner):
    cols = runner._resolve_feature_columns("z_plus_spearman")
    assert "bytes_spearman" in cols
    assert "bytes_spearman_pos" in cols
    # bytes-only: no spearman columns for the other raw features.
    for c in ["logins", "hosts", "fails"]:
        assert f"{c}_spearman" not in cols


def test_ramp_is_bytes_only(runner):
    cols = runner._resolve_feature_columns("z_plus_ramp")
    assert "bytes_ramp_signal" in cols
    for c in ["logins", "hosts", "fails"]:
        assert f"{c}_ramp_signal" not in cols


def test_z_plus_spearman_column_count(runner):
    # raw(4) + self_z(4) + robust_z(4) + peer_z(4) + spearman(2, bytes only)
    cols = runner._resolve_feature_columns("z_plus_spearman")
    assert len(cols) == 18


def test_no_duplicate_columns_for_each_config(config, runner):
    for name in config.ablation_configs:
        cols = runner._resolve_feature_columns(name)
        assert len(cols) == len(set(cols)), f"{name} produced duplicate columns"


def test_unknown_family_raises(config, runner):
    # Inject a bogus family into a config name.
    config.ablation_configs["broken"] = ["self_z", "not_a_family"]
    try:
        with pytest.raises(ValueError, match="Unknown feature family"):
            runner._resolve_feature_columns("broken")
    finally:
        del config.ablation_configs["broken"]


# ------------------------------------------------------------------
# Seed aggregation
# ------------------------------------------------------------------

def _fake_runs(config):
    primary = config.evaluation.primary_top_k
    rows = []
    # config "a" averages to higher precision than "b".
    for seed in (0, 1):
        rows.append({
            "config": "a", "seed": seed, "n_features": 10,
            f"precision_at_{primary}": 0.8 + 0.1 * seed,   # 0.8, 0.9 -> 0.85
            f"recall_at_{primary}": 0.5,
            **{f"caught_{t}": 0.5 for t in config.evaluation.attack_types_tracked},
        })
        rows.append({
            "config": "b", "seed": seed, "n_features": 4,
            f"precision_at_{primary}": 0.2 + 0.1 * seed,   # 0.2, 0.3 -> 0.25
            f"recall_at_{primary}": 0.1,
            **{f"caught_{t}": 0.1 for t in config.evaluation.attack_types_tracked},
        })
    return pd.DataFrame(rows)


def test_aggregate_one_row_per_config(config, runner):
    summary = runner._aggregate_seeds(_fake_runs(config))
    assert len(summary) == 2
    assert set(summary["config"]) == {"a", "b"}


def test_aggregate_averages_across_seeds(config, runner):
    primary = config.evaluation.primary_top_k
    summary = runner._aggregate_seeds(_fake_runs(config))
    a_row = summary[summary["config"] == "a"].iloc[0]
    assert a_row[f"precision_at_{primary}"] == pytest.approx(0.85)


def test_aggregate_sorted_by_primary_precision(config, runner):
    summary = runner._aggregate_seeds(_fake_runs(config))
    # Best config first.
    assert summary.iloc[0]["config"] == "a"


def test_aggregate_config_column_first(config, runner):
    summary = runner._aggregate_seeds(_fake_runs(config))
    assert list(summary.columns)[0] == "config"
    assert "seed" not in summary.columns
