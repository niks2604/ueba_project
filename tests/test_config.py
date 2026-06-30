"""Tests for config loading and cross-section validation."""

import copy

import pytest

from config import (
    Config,
    load_config,
    validate,
)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def test_load_returns_config(config):
    assert isinstance(config, Config)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_n_users_total_is_sum_of_groups(config):
    expected = sum(g.n_users for g in config.data.peer_groups.values())
    assert config.data.n_users_total == expected
    # Sanity: the documented 1000-user dataset.
    assert config.data.n_users_total == 1000


def test_attacks_parsed_without_seed_key(config):
    # random_seed must not leak in as an "attack".
    assert "random_seed" not in config.attacks.attacks
    assert set(config.attacks.attacks) == {
        "exfil", "cred_theft", "lateral", "subtle_exfil", "slow_ramp",
    }


def test_evaluation_window_matches_data(config):
    assert config.evaluation.test_window_start == config.data.test_window_start


def test_primary_k_in_top_k_values(config):
    assert config.evaluation.primary_top_k in config.evaluation.top_k_values


# ------------------------------------------------------------------
# Validation — each branch should reject a deliberately broken config
# ------------------------------------------------------------------

def test_validate_passes_on_real_config(config):
    # Should not raise.
    validate(config)


def test_test_window_past_end_rejected(config):
    bad = copy.deepcopy(config)
    bad.data.test_window_start = bad.data.n_days
    with pytest.raises(ValueError, match="test_window_start"):
        validate(bad)


def test_mismatched_eval_window_rejected(config):
    bad = copy.deepcopy(config)
    bad.evaluation.test_window_start = bad.data.test_window_start + 1
    with pytest.raises(ValueError, match="does not match"):
        validate(bad)


def test_primary_k_not_in_top_k_rejected(config):
    bad = copy.deepcopy(config)
    bad.evaluation.primary_top_k = 999
    with pytest.raises(ValueError, match="primary_top_k"):
        validate(bad)


def test_ablation_references_disabled_family_rejected(config):
    bad = copy.deepcopy(config)
    bad.features.enabled_families["spearman"] = False
    # z_plus_spearman references spearman, now disabled.
    with pytest.raises(ValueError, match="ablation_configs"):
        validate(bad)


def test_attack_targets_unknown_group_rejected(config):
    bad = copy.deepcopy(config)
    bad.attacks.attacks["exfil"].target_group = "no_such_group"
    with pytest.raises(ValueError, match="unknown group"):
        validate(bad)


def test_unknown_aggregation_rejected(config):
    bad = copy.deepcopy(config)
    bad.evaluation.aggregation = "median"
    with pytest.raises(ValueError, match="aggregation"):
        validate(bad)
