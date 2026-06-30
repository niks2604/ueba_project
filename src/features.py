"""
Feature engineering for the UEBA pipeline.

Computes seven feature families on a Spark DataFrame:
  1. self_z       — z-score vs user's own past (mean/std)
  2. robust_z     — robust z vs user's past (median/MAD)
  3. peer_z       — z-score vs peer group, leave-one-out
  4. rolling      — rolling 7-day max/mean/std
  5. diff         — day-over-day difference and percent change
  6. ramp         — slope, R², and gated signal for the past N days
  7. spearman     — Spearman rank correlation between day and value

Spearman is the only family computed in pandas (uses scipy); the
rest are pure Spark. Spearman results are joined back to the Spark
DataFrame.

Usage:
    from features import FeaturePipeline
    pipeline = FeaturePipeline(config.features, spark)
    sdf_with_features = pipeline.compute_all(sdf)
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from scipy.stats import spearmanr

from config import FeaturesConfig


class FeaturePipeline:
    """
    Computes all enabled feature families on a Spark DataFrame.

    The class checks the config to decide which families to compute.
    Families are computed in a fixed order so that downstream
    families can reference upstream columns if needed.
    """

    def __init__(self, config: FeaturesConfig, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.raw_cols: List[str] = config.raw_columns
        self.params = config.parameters

    def compute_all(self, sdf: DataFrame) -> DataFrame:
        """
        Compute every enabled feature family on the input DataFrame.
        Returns a new Spark DataFrame with all feature columns added.
        """
        enabled = self.config.enabled_families

        # Dispatcher: family name → handler method
        handlers: Dict[str, Callable[[DataFrame], DataFrame]] = {
            'self_z':   self._add_self_z,
            'robust_z': self._add_robust_z,
            'peer_z':   self._add_peer_z,
            'rolling':  self._add_rolling,
            'diff':     self._add_diff,
            'ramp':     self._add_ramp,
            'spearman': self._add_spearman,
        }

        # Execute each family in the order defined in handlers.
        for family, handler in handlers.items():
            if enabled.get(family, False):
                sdf = handler(sdf)

        return sdf.cache()

    # ============================================================
    # Window definitions
    # ============================================================

    def _self_window(self):
        """All of a user's past days (excludes today)."""
        return (Window
                .partitionBy('user_id')
                .orderBy('day')
                .rowsBetween(Window.unboundedPreceding, -1))

    def _peer_window(self):
        """All users in the same group on the same day."""
        return Window.partitionBy('group', 'day')

    def _rolling_window(self):
        """User's last 7 days (excludes today)."""
        return (Window
                .partitionBy('user_id')
                .orderBy('day')
                .rowsBetween(-self.params.rolling_window, -1))

    def _lag_window(self):
        """User's history ordered by day. Used for lag operations."""
        return Window.partitionBy('user_id').orderBy('day')

    def _ramp_window(self):
        """User's current day + previous (ramp_window - 1) days."""
        # rowsBetween(-N, 0) means today + N preceding rows = N+1 day span
        n = self.params.ramp_window
        return (Window
                .partitionBy('user_id')
                .orderBy('day')
                .rowsBetween(-(n - 1), 0))

    # ============================================================
    # Feature family implementations
    # ============================================================

    def _add_self_z(self, sdf: DataFrame) -> DataFrame:
        """z-score against user's own past: (today - mean) / std."""
        w = self._self_window()
        eps = self.params.epsilon

        for c in self.raw_cols:
            col_d = F.col(c).cast('double')
            sdf = (sdf
                .withColumn(f'{c}_mean_self', F.avg(col_d).over(w))
                .withColumn(f'{c}_std_self',  F.stddev(col_d).over(w))
                .withColumn(f'{c}_z_self',
                    (col_d - F.col(f'{c}_mean_self')) /
                    (F.col(f'{c}_std_self') + F.lit(eps))))
        return sdf

    def _add_robust_z(self, sdf: DataFrame) -> DataFrame:
        """
        Robust z-score using median and MAD.
        MAD = 1.4826 × median(|x - median(x)|)
        The constant 1.4826 makes MAD a consistent estimator of std
        for Gaussian-distributed data.
        """
        w = self._self_window()
        eps = self.params.epsilon
        mad_const = self.params.mad_constant

        # Pass 1: compute median for each column
        for c in self.raw_cols:
            sdf = sdf.withColumn(
                f'{c}_med_self',
                F.percentile_approx(c, 0.5).over(w),
            )

        # Pass 2: compute MAD using the medians, then robust z
        for c in self.raw_cols:
            col_d = F.col(c).cast('double')
            sdf = sdf.withColumn(
                f'{c}_abs_dev',
                F.abs(col_d - F.col(f'{c}_med_self')),
            )
            sdf = sdf.withColumn(
                f'{c}_mad_self',
                F.lit(mad_const) *
                F.percentile_approx(f'{c}_abs_dev', 0.5).over(w),
            )
            sdf = sdf.drop(f'{c}_abs_dev')
            sdf = sdf.withColumn(
                f'{c}_rz_self',
                (col_d - F.col(f'{c}_med_self')) /
                (F.col(f'{c}_mad_self') + F.lit(eps)),
            )
        return sdf

    def _add_peer_z(self, sdf: DataFrame) -> DataFrame:
        """
        Peer z-score with leave-one-out:
        compare today's value to the group's mean and std,
        excluding this user from the group's reference.
        Uses numerically stable variance formulas.
        """
        w = self._peer_window()
        eps = self.params.epsilon

        for c in self.raw_cols:
            col_d = F.col(c).cast('double')

            # Step 1: group-level mean and count
            sdf = (sdf
                .withColumn(f'{c}_grp_n',    F.count(c).over(w))
                .withColumn(f'{c}_grp_mean', F.avg(col_d).over(w)))

            # Step 2: each row's squared deviation from group mean
            dev = col_d - F.col(f'{c}_grp_mean')
            sdf = sdf.withColumn(f'{c}_dev_sq', dev * dev)

            # Step 3: sum of squared deviations across the group
            sdf = sdf.withColumn(
                f'{c}_ssd_grp',
                F.sum(F.col(f'{c}_dev_sq')).over(w),
            )

            # Step 4: leave-one-out mean
            n_grp     = F.col(f'{c}_grp_n')
            mean_grp  = F.col(f'{c}_grp_mean')
            ssd_grp   = F.col(f'{c}_ssd_grp')
            dev_sq    = F.col(f'{c}_dev_sq')

            sdf = sdf.withColumn(
                f'{c}_mean_peer_loo',
                (n_grp * mean_grp - col_d) / (n_grp - F.lit(1)),
            )

            # Leave-one-out sum of squared deviations:
            #   SSD_loo = SSD_grp - (n / (n-1)) * dev_sq_self
            sdf = sdf.withColumn(
                f'{c}_ssd_loo',
                ssd_grp - (n_grp.cast('double') / (n_grp - F.lit(1))) * dev_sq,
            )

            # Standard deviation (clipped at zero for numerical safety)
            sdf = sdf.withColumn(
                f'{c}_std_peer_loo',
                F.sqrt(F.greatest(
                    F.col(f'{c}_ssd_loo') / (n_grp - F.lit(2)),
                    F.lit(0.0),
                )),
            )

            # Final z-score
            sdf = sdf.withColumn(
                f'{c}_z_peer_loo',
                (col_d - F.col(f'{c}_mean_peer_loo')) /
                (F.col(f'{c}_std_peer_loo') + F.lit(eps)),
            )

            # Drop scaffolding columns
            sdf = sdf.drop(
                f'{c}_grp_n', f'{c}_grp_mean',
                f'{c}_dev_sq', f'{c}_ssd_grp', f'{c}_ssd_loo',
            )

        return sdf

    def _add_rolling(self, sdf: DataFrame) -> DataFrame:
        """Rolling stats over the past 7 days (excludes today)."""
        w = self._rolling_window()
        for c in self.raw_cols:
            col_d = F.col(c).cast('double')
            sdf = (sdf
                .withColumn(f'{c}_roll7_max',  F.max(col_d).over(w))
                .withColumn(f'{c}_roll7_mean', F.avg(col_d).over(w))
                .withColumn(f'{c}_roll7_std',  F.stddev(col_d).over(w)))
        return sdf

    def _add_diff(self, sdf: DataFrame) -> DataFrame:
        """Day-over-day diff and percent change."""
        w = self._lag_window()
        eps = self.params.epsilon
        for c in self.raw_cols:
            col_d = F.col(c).cast('double')
            yest = F.lag(col_d, 1).over(w)
            sdf = sdf.withColumn(f'{c}_diff1', col_d - yest)
            sdf = sdf.withColumn(
                f'{c}_pct1',
                (col_d - yest) / (F.abs(yest) + F.lit(eps)),
            )
        return sdf

    def _add_ramp(self, sdf: DataFrame) -> DataFrame:
        """
        Linear regression slope, R², and gated signal over the
        trailing ramp_window days. ramp_signal = R² when slope > 0,
        else 0 — captures upward-trending users specifically.
        """
        w = self._ramp_window()
        for c in self.raw_cols:
            col_d = F.col(c).cast('double')
            day_d = F.col('day').cast('double')

            sdf = sdf.withColumn(
                f'{c}_ramp_slope',
                F.regr_slope(col_d, day_d).over(w),
            )
            sdf = sdf.withColumn(
                f'{c}_ramp_r2',
                F.regr_r2(col_d, day_d).over(w),
            )
            sdf = sdf.withColumn(
                f'{c}_ramp_signal',
                F.when(
                    F.col(f'{c}_ramp_slope') > 0,
                    F.col(f'{c}_ramp_r2'),
                ).otherwise(F.lit(0.0)),
            )
        return sdf

    def _add_spearman(self, sdf: DataFrame) -> DataFrame:
        """
        Spearman rank correlation between day index and value over
        the trailing window. Computed in pandas because scipy is
        not natively available in Spark SQL.

        Returns the Spark DataFrame with Spearman columns added.
        """
        window_size = self.params.spearman_window

        # Pull just what we need into pandas
        pdf = (sdf
               .select('user_id', 'day', *self.raw_cols)
               .orderBy('user_id', 'day')
               .toPandas())

        # Compute trailing Spearman per (user, column)
        for c in self.raw_cols:
            pdf[f'{c}_spearman'] = (
                pdf.groupby('user_id')[c]
                   .transform(lambda s: self._trailing_spearman(
                       s.values, window_size))
            )
            # Gated positive-only version
            pdf[f'{c}_spearman_pos'] = pdf[f'{c}_spearman'].clip(lower=0)

        # Join back to Spark
        spearman_cols = [c for c in pdf.columns if 'spearman' in c]
        spear_sdf = self.spark.createDataFrame(
            pdf[['user_id', 'day'] + spearman_cols]
        )
        return sdf.join(spear_sdf, on=['user_id', 'day'], how='left')

    @staticmethod
    def _trailing_spearman(values: np.ndarray, window: int) -> np.ndarray:
        """
        For each position i, compute Spearman correlation between
        day-index and the trailing window of values ending at i.
        Returns an array of the same length as values, with 0.0 for
        positions that don't have enough history.
        """
        n = len(values)
        out = np.zeros(n)
        for i in range(n):
            start = max(0, i - window + 1)
            slice_ = values[start:i + 1]
            if len(slice_) >= 3 and np.std(slice_) > 1e-9:
                rho, _ = spearmanr(np.arange(len(slice_)), slice_)
                out[i] = 0.0 if np.isnan(rho) else rho
        return out