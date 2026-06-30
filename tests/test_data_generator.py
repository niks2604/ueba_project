"""Tests for the synthetic data generator."""

import pytest

from data_generator import DataGenerator


EXPECTED_COLUMNS = [
    "user_id", "group", "day",
    "bytes", "logins", "hosts", "fails",
    "is_anomaly", "anomaly_type",
]


def test_shape_is_users_times_days(config, generated_df):
    expected_rows = config.data.n_users_total * config.data.n_days
    assert len(generated_df) == expected_rows


def test_columns(generated_df):
    assert list(generated_df.columns) == EXPECTED_COLUMNS


def test_one_row_per_user_day(config, generated_df):
    counts = generated_df.groupby("user_id")["day"].nunique()
    assert (counts == config.data.n_days).all()
    assert generated_df.duplicated(["user_id", "day"]).sum() == 0


def test_day_range(config, generated_df):
    assert generated_df["day"].min() == 0
    assert generated_df["day"].max() == config.data.n_days - 1


def test_group_user_counts_match_config(config, generated_df):
    per_group = generated_df.groupby("group")["user_id"].nunique()
    for name, group_cfg in config.data.peer_groups.items():
        assert per_group[name] == group_cfg.n_users


def test_no_anomalies_before_planting(generated_df):
    # The generator only produces clean data; attacks come later.
    assert not generated_df["is_anomaly"].any()
    assert (generated_df["anomaly_type"] == "").all()


def test_counts_are_non_negative(generated_df):
    for col in ["bytes", "logins", "hosts", "fails"]:
        assert (generated_df[col] >= 0).all()


def test_reproducible_with_same_seed(config):
    a = DataGenerator(config.data).generate()
    b = DataGenerator(config.data).generate()
    # Same seed → identical data.
    assert a.equals(b)


def test_get_user_groups_requires_generate(config):
    gen = DataGenerator(config.data)
    with pytest.raises(RuntimeError):
        gen.get_user_groups()


def test_user_groups_cover_all_users(config):
    gen = DataGenerator(config.data)
    gen.generate()
    groups = gen.get_user_groups()
    assert len(groups) == config.data.n_users_total
    assert set(groups.keys()) == set(range(config.data.n_users_total))
    assert set(groups.values()) == set(config.data.peer_groups.keys())
