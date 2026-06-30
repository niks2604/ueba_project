"""
Anomaly detection model wrapper for the UEBA pipeline.

Currently wraps sklearn's IsolationForest. The class isolates the
model details from the rest of the pipeline, so swapping algorithms
later (LOF, Elliptic Envelope, custom detector) requires changing
only this file.

Usage:
    from model import AnomalyDetector
    detector = AnomalyDetector(config.model)
    scores = detector.fit_and_score(feature_matrix, seed=0)
"""

from typing import Union

import numpy as np
from sklearn.ensemble import IsolationForest

from config import ModelConfig


class AnomalyDetector:
    """
    Wraps an unsupervised anomaly detection model.

    The wrapper exists so the rest of the pipeline doesn't depend
    on sklearn directly. If we ever swap to a different detector,
    only this class changes.

    Currently supports: isolation_forest (sklearn).
    """

    SUPPORTED_ALGORITHMS = {'isolation_forest'}

    def __init__(self, config: ModelConfig):
        self.config = config

        if config.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {config.algorithm!r}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )

    def fit_and_score(
        self,
        X: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """
        Fit the model on X, then return anomaly scores for every row.

        Higher score = more anomalous.

        Args:
            X: feature matrix of shape (n_rows, n_features)
            seed: random seed for this run

        Returns:
            scores: array of length n_rows, higher = more anomalous
        """
        model = self._build_model(seed)
        model.fit(X)
        # sklearn returns the score_samples() in REVERSED convention:
        # higher = more normal. We negate so higher = more anomalous,
        # which matches what the rest of the pipeline expects.
        return -model.score_samples(X)

    def _build_model(self, seed: int):
        """Construct the underlying sklearn model with config params."""
        if self.config.algorithm == 'isolation_forest':
            return IsolationForest(
                n_estimators=self.config.n_estimators,
                max_samples=self.config.max_samples,
                contamination=self._parse_contamination(),
                random_state=seed,
                n_jobs=-1,
            )
        # Future algorithms would dispatch here.
        raise ValueError(f"Unsupported: {self.config.algorithm}")

    def _parse_contamination(self) -> Union[str, float]:
        """
        sklearn accepts contamination as 'auto' OR a float.
        YAML reads 'auto' as a string already; numbers as floats.
        Pass through as-is.
        """
        c = self.config.contamination
        if c == 'auto':
            return 'auto'
        return float(c)