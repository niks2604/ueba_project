"""
Attack planter for the UEBA pipeline.

Injects synthetic attacks into a DataFrame produced by DataGenerator.
Each attack type has its own implementation method, dispatched via
the ATTACK_HANDLERS dict.

Usage:
    from attack_planter import AttackPlanter

    planter = AttackPlanter(
        attacks_config=config.attacks,
        test_window_start=config.data.test_window_start,
        user_groups=generator.get_user_groups(),
    )
    df_attacked = planter.plant(df)
"""

from typing import Callable, Dict, List, Set

import numpy as np
import pandas as pd

from config import AttacksConfig, AttackConfig


class AttackPlanter:
    """
    Injects synthetic attacks into the generated user-day DataFrame.

    Plants attacks based on the attacks: block in the config.
    Maintains attacker assignments so they can be queried later
    (e.g., by the evaluator to compute per-attack-type metrics).
    """

    def __init__(
        self,
        attacks_config: AttacksConfig,
        test_window_start: int,
        user_groups: Dict[int, str],
    ):
        self.config = attacks_config
        self.test_window_start = test_window_start
        self.user_groups = user_groups
        self.rng = np.random.default_rng(attacks_config.random_seed)
        self._victims: Dict[str, List[int]] = {}   # attack_type → user_ids

    def plant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Plant all attacks defined in the config.
        Returns a modified copy of df with attack rows updated.
        """
        df = df.copy()
        used_victims: Set[int] = set()

        for attack_name, attack_config in self.config.attacks.items():
            victims = self._pick_victims(
                group=attack_config.target_group,
                n=attack_config.n_victims,
                exclude=used_victims,
            )
            self._victims[attack_name] = victims
            used_victims.update(victims)

            handler = self.ATTACK_HANDLERS.get(attack_name)
            if handler is None:
                raise ValueError(f"No handler for attack type: {attack_name}")

            df = handler(self, df, attack_config, victims)

        return df

    def get_victims(self) -> Dict[str, List[int]]:
        """
        Return the mapping of attack_type → list of victim user_ids.
        Must be called after plant().
        """
        if not self._victims:
            raise RuntimeError("plant() must be called before get_victims()")
        return self._victims

    # ============================================================
    # Victim selection
    # ============================================================

    def _pick_victims(
        self, group: str, n: int, exclude: Set[int]
    ) -> List[int]:
        """Randomly select n users from the given group, excluding any used."""
        candidates = [
            uid for uid, g in self.user_groups.items()
            if g == group and uid not in exclude
        ]
        if len(candidates) < n:
            raise ValueError(
                f"Not enough users in group {group!r}: "
                f"need {n}, have {len(candidates)}"
            )
        return list(self.rng.choice(candidates, n, replace=False))

    # ============================================================
    # Attack implementations — one method per attack type
    # ============================================================

    def _plant_exfil(
        self, df: pd.DataFrame, ac: AttackConfig, victims: List[int]
    ) -> pd.DataFrame:
        """Big bytes spike for several consecutive days."""
        start = self.test_window_start + ac.start_day_offset
        days = range(start, start + ac.duration_days)
        for uid in victims:
            mask = (df['user_id'] == uid) & df['day'].isin(days)
            df.loc[mask, 'bytes'] = (
                df.loc[mask, 'bytes'] * ac.bytes_multiplier
            ).astype('int64')
            df.loc[mask, 'is_anomaly'] = True
            df.loc[mask, 'anomaly_type'] = 'exfil'
        return df

    def _plant_cred_theft(
        self, df: pd.DataFrame, ac: AttackConfig, victims: List[int]
    ) -> pd.DataFrame:
        """Elevated logins and failed-login spikes."""
        start = self.test_window_start + ac.start_day_offset
        days = range(start, start + ac.duration_days)
        for uid in victims:
            mask = (df['user_id'] == uid) & df['day'].isin(days)
            df.loc[mask, 'logins'] = df.loc[mask, 'logins'] + ac.logins_add
            df.loc[mask, 'fails']  = df.loc[mask, 'fails']  + ac.fails_add
            df.loc[mask, 'is_anomaly'] = True
            df.loc[mask, 'anomaly_type'] = 'cred_theft'
        return df

    def _plant_lateral(
        self, df: pd.DataFrame, ac: AttackConfig, victims: List[int]
    ) -> pd.DataFrame:
        """Many additional host connections."""
        start = self.test_window_start + ac.start_day_offset
        days = range(start, start + ac.duration_days)
        for uid in victims:
            mask = (df['user_id'] == uid) & df['day'].isin(days)
            df.loc[mask, 'hosts'] = df.loc[mask, 'hosts'] + ac.hosts_add
            df.loc[mask, 'is_anomaly'] = True
            df.loc[mask, 'anomaly_type'] = 'lateral'
        return df

    def _plant_subtle_exfil(
        self, df: pd.DataFrame, ac: AttackConfig, victims: List[int]
    ) -> pd.DataFrame:
        """Moderate bytes spike on specific (possibly non-consecutive) days."""
        days = [self.test_window_start + offset for offset in ac.days_offset]
        for uid in victims:
            mask = (df['user_id'] == uid) & df['day'].isin(days)
            df.loc[mask, 'bytes'] = (
                df.loc[mask, 'bytes'] * ac.bytes_multiplier
            ).astype('int64')
            df.loc[mask, 'is_anomaly'] = True
            df.loc[mask, 'anomaly_type'] = 'subtle_exfil'
        return df

    def _plant_slow_ramp(
        self, df: pd.DataFrame, ac: AttackConfig, victims: List[int]
    ) -> pd.DataFrame:
        """Monotonically climbing bytes across multiple days."""
        for uid in victims:
            group = self.user_groups[uid]
            # Base value drawn from this user's typical bytes
            base_bytes = int(df[(df['user_id'] == uid) &
                                (df['day'] < self.test_window_start)]
                              ['bytes'].median())
            for i in range(ac.duration_days):
                day = self.test_window_start + ac.start_day_offset + i
                mult = ac.ramp_start_multiplier + ac.ramp_step * i
                mask = (df['user_id'] == uid) & (df['day'] == day)
                df.loc[mask, 'bytes'] = int(base_bytes * mult)
                df.loc[mask, 'is_anomaly'] = True
                df.loc[mask, 'anomaly_type'] = 'slow_ramp'
        return df

    # ============================================================
    # Dispatcher — maps attack type name → handler method
    # Listed here so adding a new attack type is one entry.
    # ============================================================

    ATTACK_HANDLERS: Dict[str, Callable] = {
        'exfil':        _plant_exfil,
        'cred_theft':   _plant_cred_theft,
        'lateral':      _plant_lateral,
        'subtle_exfil': _plant_subtle_exfil,
        'slow_ramp':    _plant_slow_ramp,
    }