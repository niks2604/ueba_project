"""
Synthetic data generator for the UEBA pipeline.

Generates user-day activity rows based on peer group baselines
defined in the config. Each user gets their own random baseline
drawn from their peer group's mean, then their daily activity
fluctuates around that baseline.

Output is a pandas DataFrame with columns:
  user_id, group, day, bytes, logins, hosts, fails

Usage:
    from config import load_config
    from data_generator import DataGenerator

    config = load_config("config/ueba_config.yaml")
    generator = DataGenerator(config.data)
    df = generator.generate()
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DataConfig, PeerGroupConfig


class DataGenerator:
    """
    Generates synthetic user activity data based on peer group baselines.

    The generation uses three layers of variation:
      1. Group level — each peer group has its own behavioral mean
      2. User level — each user gets a personal baseline drawn from
                      their group's distribution
      3. Day level — each day's activity fluctuates around the
                     user's personal baseline

    This matches how real workforce activity typically looks:
    departments differ, individuals within a department differ,
    and each individual has day-to-day variation.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._user_groups: Dict[int, str] = {}   # populated during generation

    def generate(self) -> pd.DataFrame:
        """
        Generate the full synthetic dataset.

        Returns a pandas DataFrame with one row per (user, day) pair.
        Total rows = n_users_total × n_days.
        """
        rows = self._generate_rows()
        return pd.DataFrame(rows, columns=[
            'user_id', 'group', 'day',
            'bytes', 'logins', 'hosts', 'fails',
            'is_anomaly', 'anomaly_type',
        ])

    def _generate_rows(self) -> List[Tuple]:
        """
        Build the raw row list, group by group, user by user, day by day.
        """
        rows = []
        user_id = 0

        for group_name, group_config in self.config.peer_groups.items():
            for _ in range(group_config.n_users):
                user_baseline = self._draw_user_baseline(group_config)
                self._user_groups[user_id] = group_name

                for day in range(self.config.n_days):
                    row = self._generate_day(
                        user_id=user_id,
                        group_name=group_name,
                        day=day,
                        user_baseline=user_baseline,
                        group_config=group_config,
                    )
                    rows.append(row)

                user_id += 1

        return rows

    def _draw_user_baseline(
        self, group_config: PeerGroupConfig
    ) -> Dict[str, float]:
        """
        Draw a personal baseline for one user, centered on their
        group's mean but with individual variation.

        Returns a dict of feature → personal baseline mean.
        """
        return {
            'log_bytes': self.rng.normal(group_config.log_bytes_mean, 0.3),
            'logins':    self.rng.normal(group_config.logins_mean, 1.0),
            'hosts':     self.rng.normal(group_config.hosts_mean, 2.0),
        }

    def _generate_day(
        self,
        user_id: int,
        group_name: str,
        day: int,
        user_baseline: Dict[str, float],
        group_config: PeerGroupConfig,
    ) -> Tuple:
        """
        Generate one day of activity for one user.

        Distribution choices reflect the natural shape of each feature:
          bytes  — log-normal (heavy-tailed; most days small, occasional big)
          logins — Gaussian, clipped at 0 (counts can't be negative)
          hosts  — Gaussian, clipped at 0
          fails  — Poisson (rare event count)
        """
        bytes_ = int(np.exp(
            self.rng.normal(user_baseline['log_bytes'], 0.5)
        ))
        logins = max(0, int(
            self.rng.normal(user_baseline['logins'], 3.0)
        ))
        hosts = max(0, int(
            self.rng.normal(user_baseline['hosts'], 4.0)
        ))
        fails = int(self.rng.poisson(group_config.fails_mean))

        return (
            user_id, group_name, day,
            bytes_, logins, hosts, fails,
            False, '',   # is_anomaly, anomaly_type — filled by AttackPlanter
        )

    def get_user_groups(self) -> Dict[int, str]:
        """
        Return the mapping of user_id → group_name.
        Used by AttackPlanter to pick victims from specific groups.
        Must be called after generate().
        """
        if not self._user_groups:
            raise RuntimeError(
                "generate() must be called before get_user_groups()"
            )
        return self._user_groups