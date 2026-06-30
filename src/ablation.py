"""
Ablation runner for the UEBA pipeline.

Runs every named configuration in config.ablation_configs across
every seed in config.model.seeds, logs each run to MLflow, and
returns a seed-averaged summary DataFrame.

This is the main experimental driver — it's what produces the
comparison table that demonstrates feature dilution and identifies
the winning configuration.

Usage:
    from ablation import AblationRunner
    runner = AblationRunner(config)
    summary = runner.run(pdf, victims)
    print(summary)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from model import AnomalyDetector
from evaluator import Evaluator, EvaluationResult


# MLflow is imported lazily so the module loads even if MLflow
# isn't installed (useful for unit testing).
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    import warnings
    warnings.warn(
        "MLflow is not installed — ablation will run but logging will be "
        "skipped. Install with: pip install mlflow",
        RuntimeWarning,
    )


class AblationRunner:
    """
    Runs the full ablation sweep and produces the comparison table.

    For each named config in config.ablation_configs:
      For each seed in config.model.seeds:
        - Build the feature column list from the config's families
        - Fit Isolation Forest with that seed
        - Evaluate
        - Log to MLflow (if enabled)
        - Store result

    After all runs, aggregate per-config (averaged across seeds)
    and return a summary DataFrame.
    """

    def __init__(self, config: Config):
        self.config = config
        self.detector = AnomalyDetector(config.model)
        self.evaluator = Evaluator(config.evaluation)
        self._mlflow_setup_done = False

    def run(
        self,
        pdf: pd.DataFrame,
        victims: Dict[str, List[int]],
    ) -> pd.DataFrame:
        """
        Run the full ablation.

        Args:
            pdf: pandas DataFrame with raw + feature columns
            victims: attack_type → list of victim user_ids

        Returns:
            Summary DataFrame, one row per named config, with
            seed-averaged metrics.
        """
        if self.config.mlflow.enabled and MLFLOW_AVAILABLE:
            self._setup_mlflow()

        # Run every (config, seed) combination
        all_runs = []
        for config_name in self.config.ablation_configs:
            feature_cols = self._resolve_feature_columns(config_name)
            if not feature_cols:
                # Shouldn't happen — baseline still has raw columns.
                continue

            pdf_clean = pdf.copy()
            pdf_clean[feature_cols] = pdf_clean[feature_cols].fillna(0)
            X = pdf_clean[feature_cols].values

            for seed in self.config.model.seeds:
                result = self._run_one(
                    X=X,
                    seed=seed,
                    pdf=pdf_clean,
                    victims=victims,
                    config_name=config_name,
                    n_features=len(feature_cols),
                )
                run_dict = result.to_dict()
                run_dict['config']     = config_name
                run_dict['seed']       = seed
                run_dict['n_features'] = len(feature_cols)
                all_runs.append(run_dict)

                if self.config.output.verbose:
                    summary = result.summary_line(
                        self.config.evaluation.primary_top_k)
                    print(f"  [{config_name:30s} seed={seed}]  {summary}")

        # Aggregate
        all_df = pd.DataFrame(all_runs)
        summary = self._aggregate_seeds(all_df)
        return summary

    # ============================================================
    # SINGLE-RUN LOGIC
    # ============================================================

    def _run_one(
        self,
        X: np.ndarray,
        seed: int,
        pdf: pd.DataFrame,
        victims: Dict[str, List[int]],
        config_name: str,
        n_features: int,
    ) -> EvaluationResult:
        """One (config, seed) combination. Fits, evaluates, logs."""
        scores = self.detector.fit_and_score(X, seed=seed)
        result = self.evaluator.evaluate(
            pdf=pdf, scores=scores, victims=victims,
        )

        if self.config.mlflow.enabled and MLFLOW_AVAILABLE:
            self._log_to_mlflow(
                result=result,
                config_name=config_name,
                seed=seed,
                n_features=n_features,
            )

        return result

    # ============================================================
    # FEATURE COLUMN RESOLUTION
    # Convert a named config (e.g., "z_plus_spearman") into the
    # actual list of column names to use in the feature matrix.
    # ============================================================

    # Family name → list of column-name templates that family produces.
    # Templates use {c} as placeholder for each raw column name.
    FAMILY_COLUMNS = {
        'self_z':   ['{c}_z_self'],
        'robust_z': ['{c}_rz_self'],
        'peer_z':   ['{c}_z_peer_loo'],
        'rolling':  ['{c}_roll7_max', '{c}_roll7_mean', '{c}_roll7_std'],
        'diff':     ['{c}_diff1', '{c}_pct1'],
        'ramp':     ['{c}_ramp_slope', '{c}_ramp_r2', '{c}_ramp_signal'],
        'spearman': ['{c}_spearman', '{c}_spearman_pos'],
    }

    # For "ramp" and "spearman" specifically, we apply only to bytes.
    # This matches the project finding that ramp/Spearman features
    # are noise for non-bytes columns.
    BYTES_ONLY_FAMILIES = {'ramp', 'spearman'}

    def _resolve_feature_columns(self, config_name: str) -> List[str]:
        """
        Build the actual column-name list for a named ablation config.
        Raw columns are always included; the config selects which
        derived families to add on top.
        """
        families = self.config.ablation_configs.get(config_name, [])
        raw_cols = self.config.features.raw_columns

        # Raw columns are always included
        cols = list(raw_cols)

        # For each enabled family, add its columns
        for family in families:
            templates = self.FAMILY_COLUMNS.get(family)
            if templates is None:
                raise ValueError(
                    f"Unknown feature family: {family!r}. "
                    f"Known: {list(self.FAMILY_COLUMNS.keys())}"
                )

            # Decide which raw cols this family applies to
            if family in self.BYTES_ONLY_FAMILIES:
                apply_to = ['bytes']
            else:
                apply_to = raw_cols

            for c in apply_to:
                for template in templates:
                    cols.append(template.format(c=c))

        return cols

    # ============================================================
    # AGGREGATION ACROSS SEEDS
    # ============================================================

    def _aggregate_seeds(self, all_df: pd.DataFrame) -> pd.DataFrame:
        """Average metrics across seeds, one row per named config."""
        # Pick the metric columns to average
        metric_cols = [c for c in all_df.columns
                       if c not in ('config', 'seed')]

        agg = (all_df
            .groupby('config')[metric_cols]
            .mean()
            .reset_index()
            .round(3))

        # Reorder so config appears first, then the headline metrics
        primary_k = self.config.evaluation.primary_top_k
        priority_cols = ['config', 'n_features',
                         f'precision_at_{primary_k}',
                         f'recall_at_{primary_k}']
        priority_cols += [
            f'caught_{t}' for t in self.config.evaluation.attack_types_tracked
        ]
        # Put priority cols first, then anything else
        rest = [c for c in agg.columns if c not in priority_cols]
        agg = agg[priority_cols + rest]

        # Sort by primary precision descending — best at top
        agg = agg.sort_values(
            f'precision_at_{primary_k}', ascending=False
        ).reset_index(drop=True)

        return agg

    # ============================================================
    # MLFLOW INTEGRATION
    # ============================================================

    def _setup_mlflow(self) -> None:
        """One-time MLflow setup at the start of the ablation."""
        if self._mlflow_setup_done:
            return
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        self._mlflow_setup_done = True

    def _log_to_mlflow(
        self,
        result: EvaluationResult,
        config_name: str,
        seed: int,
        n_features: int,
    ) -> None:
        """Log one (config, seed) run as an MLflow run."""
        with mlflow.start_run(run_name=f"{config_name}_seed{seed}"):
            # Params
            mlflow.log_param('config',       config_name)
            mlflow.log_param('seed',         seed)
            mlflow.log_param('n_features',   n_features)
            mlflow.log_param('algorithm',    self.config.model.algorithm)
            mlflow.log_param('n_estimators', self.config.model.n_estimators)
            mlflow.log_param('max_samples',  self.config.model.max_samples)
            mlflow.log_param('aggregation',  self.config.evaluation.aggregation)

            # Metrics (flat dict)
            for k, v in result.to_dict().items():
                mlflow.log_metric(k, v)

            # Tags
            for tag_k, tag_v in self.config.mlflow.tags.items():
                mlflow.set_tag(tag_k, tag_v)