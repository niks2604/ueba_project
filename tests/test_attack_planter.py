"""Tests for the attack planter."""

import copy

import pytest

from attack_planter import AttackPlanter


def test_get_victims_requires_plant(config, generator):
    planter = AttackPlanter(
        attacks_config=config.attacks,
        test_window_start=config.data.test_window_start,
        user_groups=generator.get_user_groups(),
    )
    with pytest.raises(RuntimeError):
        planter.get_victims()


def test_all_attacks_have_victims(config, planter_and_df):
    _, planter = planter_and_df
    victims = planter.get_victims()
    assert set(victims) == set(config.attacks.attacks)
    for name, attack in config.attacks.attacks.items():
        assert len(victims[name]) == attack.n_victims


def test_victims_are_disjoint_across_attacks(planter_and_df):
    _, planter = planter_and_df
    victims = planter.get_victims()
    all_ids = [uid for ids in victims.values() for uid in ids]
    assert len(all_ids) == len(set(all_ids)), "a user was attacked twice"


def test_victims_drawn_from_target_group(config, generator, planter_and_df):
    _, planter = planter_and_df
    user_groups = generator.get_user_groups()
    for name, ids in planter.get_victims().items():
        target = config.attacks.attacks[name].target_group
        for uid in ids:
            assert user_groups[int(uid)] == target


def test_total_anomalous_rows_match_expected(config, planter_and_df):
    df_attacked, _ = planter_and_df
    # exfil: 10 victims × 3 days, cred_theft: 10×2, lateral: 10×3,
    # subtle_exfil: 10×2 days, slow_ramp: 10×5 days = 30+20+30+20+50 = 150
    assert df_attacked["is_anomaly"].sum() == 150


def test_anomalies_only_in_test_window(config, planter_and_df):
    df_attacked, _ = planter_and_df
    anomalous = df_attacked[df_attacked["is_anomaly"]]
    assert (anomalous["day"] >= config.data.test_window_start).all()


def test_anomaly_type_matches_victim_assignment(planter_and_df):
    df_attacked, planter = planter_and_df
    anomalous = df_attacked[df_attacked["is_anomaly"]]
    for attack_type, ids in planter.get_victims().items():
        rows = anomalous[anomalous["user_id"].isin([int(i) for i in ids])]
        assert (rows["anomaly_type"] == attack_type).all()


def test_clean_rows_have_empty_type(planter_and_df):
    df_attacked, _ = planter_and_df
    clean = df_attacked[~df_attacked["is_anomaly"]]
    assert (clean["anomaly_type"] == "").all()


def test_exfil_actually_inflates_bytes(config, generated_df, planter_and_df):
    df_attacked, planter = planter_and_df
    victim = int(planter.get_victims()["exfil"][0])
    start = config.data.test_window_start + config.attacks.attacks["exfil"].start_day_offset
    attack_days = range(start, start + config.attacks.attacks["exfil"].duration_days)

    before = generated_df[
        (generated_df["user_id"] == victim)
        & generated_df["day"].isin(attack_days)
    ]["bytes"].values
    after = df_attacked[
        (df_attacked["user_id"] == victim)
        & df_attacked["day"].isin(attack_days)
    ]["bytes"].values

    mult = config.attacks.attacks["exfil"].bytes_multiplier
    # Each attacked day's bytes is the original × multiplier (int-cast).
    assert (after >= before * mult - 1).all()
    assert (after > before).all()


def test_cred_theft_adds_logins_and_fails(config, generated_df, planter_and_df):
    df_attacked, planter = planter_and_df
    victim = int(planter.get_victims()["cred_theft"][0])
    ac = config.attacks.attacks["cred_theft"]
    start = config.data.test_window_start + ac.start_day_offset
    days = list(range(start, start + ac.duration_days))

    def sla(df, col):
        return df[(df["user_id"] == victim) & df["day"].isin(days)].sort_values("day")[col].values

    assert (sla(df_attacked, "logins") - sla(generated_df, "logins") == ac.logins_add).all()
    assert (sla(df_attacked, "fails") - sla(generated_df, "fails") == ac.fails_add).all()


def test_slow_ramp_is_monotonic(config, planter_and_df):
    df_attacked, planter = planter_and_df
    ac = config.attacks.attacks["slow_ramp"]
    start = config.data.test_window_start + ac.start_day_offset
    days = list(range(start, start + ac.duration_days))

    for victim in planter.get_victims()["slow_ramp"]:
        ramp = df_attacked[
            (df_attacked["user_id"] == int(victim))
            & df_attacked["day"].isin(days)
        ].sort_values("day")["bytes"].values
        # Bytes climb each day of the ramp.
        assert (ramp[1:] >= ramp[:-1]).all(), f"user {victim} ramp not monotonic"


def test_not_enough_users_raises(config, generator):
    # Deep-copy so we never mutate the shared session config.
    bad_attacks = copy.deepcopy(config.attacks)
    # Ask for more victims than exist in the smallest group.
    bad_attacks.attacks["slow_ramp"].n_victims = 10_000
    planter = AttackPlanter(
        attacks_config=bad_attacks,
        test_window_start=config.data.test_window_start,
        user_groups=generator.get_user_groups(),
    )
    with pytest.raises(ValueError, match="Not enough users"):
        planter.plant(generator.generate())
