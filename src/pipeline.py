"""
Top-level entry point for the UEBA pipeline.

Loads config, sets up Spark, runs every pipeline stage in order,
and saves results to the configured output directory.

Usage (from inside src/):
    python pipeline.py
    python pipeline.py --config ../config/ueba_config.yaml
    python pipeline.py --skip-features      # use cached features

Or programmatically:
    from pipeline import UEBAPipeline
    pipeline = UEBAPipeline("../config/ueba_config.yaml")
    summary = pipeline.run()
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from config import Config, load_config
from data_generator import DataGenerator
from attack_planter import AttackPlanter
from features import FeaturePipeline
from ablation import AblationRunner


class UEBAPipeline:
    """
    Top-level orchestrator for the UEBA anomaly detection pipeline.

    Stages:
      1. Load config
      2. Set up Spark
      3. Generate synthetic data (pandas)
      4. Plant attacks (pandas)
      5. Convert to Spark and compute features
      6. Run ablation across configs × seeds
      7. Save outputs (CSV summary, optional model files)
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Config = load_config(config_path)
        self.spark = None

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def run(self, skip_features: bool = False) -> pd.DataFrame:
        """
        Execute the full pipeline end-to-end.

        Args:
            skip_features: if True, load features from the saved
                           parquet instead of recomputing. Fast.

        Returns:
            The ablation summary DataFrame.
        """
        self._log_header()
        self._setup_output_dir()

        # Stages 2-5: get features
        if skip_features and self._has_cached_features():
            pdf, victims = self._load_cached_features()
            self._log("Loaded features from cache")
        else:
            pdf, victims = self._build_features()
            self._maybe_cache_features(pdf, victims)

        # Stage 6: ablation
        summary = self._run_ablation(pdf, victims)

        # Stage 7: save
        self._save_outputs(summary)

        self._log_footer(summary)
        return summary

    # ============================================================
    # STAGES
    # ============================================================

    def _build_features(self):
        """Generate data, plant attacks, compute features. Returns (pdf, victims)."""
        self._setup_spark()

        self._log("Stage 1/3 — generating synthetic data...")
        generator = DataGenerator(self.config.data)
        df = generator.generate()
        self._log(f"  Generated {len(df):,} rows "
                  f"({self.config.data.n_users_total} users × "
                  f"{self.config.data.n_days} days)")

        self._log("Stage 2/3 — planting attacks...")
        planter = AttackPlanter(
            attacks_config=self.config.attacks,
            test_window_start=self.config.data.test_window_start,
            user_groups=generator.get_user_groups(),
        )
        df_attacked = planter.plant(df)
        victims = planter.get_victims()
        n_attackers = sum(len(v) for v in victims.values())
        self._log(f"  Planted {n_attackers} attackers across "
                  f"{len(victims)} attack types")

        self._log("Stage 3/3 — computing features (Spark)...")
        sdf = self.spark.createDataFrame(df_attacked)
        feature_pipeline = FeaturePipeline(self.config.features, self.spark)
        sdf_features = feature_pipeline.compute_all(sdf)
        pdf = sdf_features.toPandas()
        self._log(f"  Computed {len(pdf.columns)} columns "
                  f"across {len(pdf):,} rows")

        return pdf, victims

    def _run_ablation(self, pdf: pd.DataFrame, victims) -> pd.DataFrame:
        """Run the ablation sweep."""
        self._log("\nRunning ablation...")
        runner = AblationRunner(self.config)
        summary = runner.run(pdf=pdf, victims=victims)
        return summary

    # ============================================================
    # CACHING
    # ============================================================

    def _cache_paths(self):
        """Where cached features and victims live on disk."""
        results_dir = Path(self.config.output.results_dir)
        return (
            results_dir / "features.parquet",
            results_dir / "victims.pickle",
        )

    def _has_cached_features(self) -> bool:
        feat_path, vict_path = self._cache_paths()
        return feat_path.exists() and vict_path.exists()

    def _maybe_cache_features(self, pdf, victims):
        """Save features and victims to disk if config asks for it."""
        if not self.config.output.save_features_parquet:
            return
        feat_path, vict_path = self._cache_paths()
        pdf.to_parquet(feat_path)
        import pickle
        with open(vict_path, "wb") as f:
            pickle.dump(victims, f)
        self._log(f"  Cached features to {feat_path}")

    def _load_cached_features(self):
        feat_path, vict_path = self._cache_paths()
        pdf = pd.read_parquet(feat_path)
        import pickle
        with open(vict_path, "rb") as f:
            victims = pickle.load(f)
        return pdf, victims

    # ============================================================
    # SPARK
    # ============================================================

    def _setup_spark(self):
        """Lazy Spark startup — only when actually needed."""
        if self.spark is not None:
            return
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        from pyspark.sql import SparkSession
        self.spark = (SparkSession.builder
                      .appName("UEBA-Pipeline")
                      .master("local[*]")
                      .config("spark.driver.memory", "4g")
                      .config("spark.sql.shuffle.partitions", "8")
                      .config("spark.sql.ansi.enabled", "false")
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel("ERROR")
        self._log(f"  Spark {self.spark.version} started")

    # ============================================================
    # OUTPUT
    # ============================================================

    def _setup_output_dir(self):
        Path(self.config.output.results_dir).mkdir(
            parents=True, exist_ok=True)

    def _save_outputs(self, summary: pd.DataFrame):
        if self.config.output.save_results_csv:
            csv_path = Path(self.config.output.results_dir) / "ablation_summary.csv"
            summary.to_csv(csv_path, index=False)
            self._log(f"\n✓ Saved summary to {csv_path}")

    # ============================================================
    # LOGGING HELPERS
    # ============================================================

    def _log(self, msg: str):
        if self.config.output.verbose:
            print(msg)

    def _log_header(self):
        self._log("=" * 70)
        self._log("UEBA ANOMALY DETECTION PIPELINE")
        self._log(f"Config: {self.config_path}")
        self._log("=" * 70)

    def _log_footer(self, summary: pd.DataFrame):
        primary_k = self.config.evaluation.primary_top_k
        self._log("\n" + "=" * 70)
        self._log(f"FINAL ABLATION SUMMARY (sorted by precision_at_{primary_k})")
        self._log("=" * 70)
        cols = ['config', 'n_features',
                f'precision_at_{primary_k}',
                f'recall_at_{primary_k}']
        cols += [f'caught_{t}' for t in self.config.evaluation.attack_types_tracked]
        self._log(summary[cols].to_string(index=False))


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    """Run the pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the UEBA anomaly detection pipeline."
    )
    parser.add_argument(
        "--config",
        default="../config/ueba_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Load features from cache instead of recomputing.",
    )
    args = parser.parse_args()

    pipeline = UEBAPipeline(args.config)
    pipeline.run(skip_features=args.skip_features)


if __name__ == "__main__":
    main()