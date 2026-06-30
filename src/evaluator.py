"""
Evaluation for the UEBA pipeline.

Takes scored data and produces user-level rankings and metrics:
  - Aggregates row-level scores to user-level (sum/max/mean)
  - Computes precision@K and recall@K for multiple K values
  - Per-attack-type detection rates
  - Returns a typed EvaluationResult that's both code-friendly
    and dict-able for MLflow logging

Usage:
    from evaluator import Evaluator
    evaluator = Evaluator(config.evaluation)
    result = evaluator.evaluate(
        pdf=pdf,
        scores=scores,
        victims=planter.get_victims(),
    )
    print(result.precision_at_k[50])
    mlflow.log_metrics(result.to_dict())
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import EvaluationConfig


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class EvaluationResult:
    """
    Full evaluation output for one model run.
    
    Holds metrics at every K value defined in the config, plus
    per-attack-type detection rates at the primary K.
    """
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k:    Dict[int, float] = field(default_factory=dict)
    caught_by_type: Dict[str, float] = field(default_factory=dict)
    n_attackers_total: int = 0
    user_rankings:  pd.DataFrame = None     # full user-level table

    def to_dict(self) -> Dict[str, float]:
        """
        Flatten the result into a flat dict, suitable for MLflow
        log_metrics() or printing.
        """
        flat = {}
        for k, v in self.precision_at_k.items():
            flat[f'precision_at_{k}'] = v
        for k, v in self.recall_at_k.items():
            flat[f'recall_at_{k}'] = v
        for attack_type, v in self.caught_by_type.items():
            flat[f'caught_{attack_type}'] = v
        return flat

    def summary_line(self, primary_k: int) -> str:
        """Single-line human-readable summary at primary K."""
        p = self.precision_at_k.get(primary_k, float('nan'))
        r = self.recall_at_k.get(primary_k, float('nan'))
        per_type = ", ".join(
            f"{t}={v:.2f}" for t, v in self.caught_by_type.items()
        )
        return (f"P@{primary_k}={p:.3f}  R@{primary_k}={r:.3f}  "
                f"per-type: {per_type}")


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:
    """
    Aggregates row scores to user level and computes metrics.
    
    The aggregation strategy (sum/max/mean) comes from config.
    The list of K values and attack types comes from config.
    """

    AGGREGATION_FUNCTIONS = {
        'sum':  'sum',
        'max':  'max',
        'mean': 'mean',
    }

    def __init__(self, config: EvaluationConfig):
        self.config = config
        if config.aggregation not in self.AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregation: {config.aggregation!r}. "
                f"Must be one of {list(self.AGGREGATION_FUNCTIONS)}"
            )

    def evaluate(
        self,
        pdf: pd.DataFrame,
        scores: np.ndarray,
        victims: Dict[str, List[int]],
    ) -> EvaluationResult:
        """
        Run the full evaluation.
        
        Args:
            pdf: pandas DataFrame with user_id, day, is_anomaly,
                 anomaly_type columns. Must have len(scores) rows.
            scores: row-level anomaly scores (higher = more anomalous)
            victims: dict of attack_type → list of victim user_ids,
                     from AttackPlanter.get_victims()
        
        Returns:
            EvaluationResult with all metrics populated
        """
        if len(scores) != len(pdf):
            raise ValueError(
                f"Score array length {len(scores)} does not match "
                f"DataFrame length {len(pdf)}"
            )

        user_rankings = self._aggregate_to_user_level(pdf, scores)
        result = EvaluationResult(
            user_rankings=user_rankings,
            n_attackers_total=sum(len(v) for v in victims.values()),
        )

        # precision and recall at each K
        for k in self.config.top_k_values:
            result.precision_at_k[k] = self._precision_at_k(user_rankings, k)
            result.recall_at_k[k]    = self._recall_at_k(
                user_rankings, k, result.n_attackers_total
            )

        # per-attack-type catch rate at primary K
        result.caught_by_type = self._caught_by_type(
            user_rankings, victims, self.config.primary_top_k
        )

        return result

    # ============================================================
    # AGGREGATION — row level to user level
    # ============================================================

    def _aggregate_to_user_level(
        self, pdf: pd.DataFrame, scores: np.ndarray
    ) -> pd.DataFrame:
        """
        Sum (or max/mean) row scores per user across the test window.
        
        Returns a DataFrame with one row per user, sorted by aggregated
        score descending. Columns: user_id, agg_score, is_attacker,
        attack_type, rank.
        """
        pdf = pdf.copy()
        pdf['score'] = scores

        # Restrict to test window — that's where attacks live
        test = pdf[pdf['day'] >= self.config.test_window_start]

        agg_fn = self.AGGREGATION_FUNCTIONS[self.config.aggregation]

        user_level = (test
            .groupby('user_id')
            .agg(
                agg_score=('score', agg_fn),
                is_attacker=('is_anomaly', 'max'),    # any anomaly row?
                attack_type=('anomaly_type',
                             lambda s: next((x for x in s if x), '')),
            )
            .reset_index()
            .sort_values('agg_score', ascending=False)
            .reset_index(drop=True)
        )
        user_level['rank'] = user_level.index + 1
        return user_level

    # ============================================================
    # METRIC COMPUTATIONS
    # ============================================================

    def _precision_at_k(
        self, user_rankings: pd.DataFrame, k: int
    ) -> float:
        """Fraction of the top-K users who are real attackers."""
        if k <= 0:
            return 0.0
        topk = user_rankings.head(k)
        return float(topk['is_attacker'].sum() / k)

    def _recall_at_k(
        self,
        user_rankings: pd.DataFrame,
        k: int,
        n_attackers_total: int,
    ) -> float:
        """Fraction of all real attackers that made it into the top-K."""
        if n_attackers_total == 0:
            return 0.0
        topk = user_rankings.head(k)
        return float(topk['is_attacker'].sum() / n_attackers_total)

    def _caught_by_type(
        self,
        user_rankings: pd.DataFrame,
        victims: Dict[str, List[int]],
        k: int,
    ) -> Dict[str, float]:
        """
        For each attack type, what fraction of its victims appear
        in the top-K?
        """
        topk_ids = set(user_rankings.head(k)['user_id'].tolist())
        out = {}
        for attack_type in self.config.attack_types_tracked:
            type_victims = set(int(v) for v in victims.get(attack_type, []))
            if not type_victims:
                out[attack_type] = 0.0
            else:
                caught = topk_ids & type_victims
                out[attack_type] = len(caught) / len(type_victims)
        return out