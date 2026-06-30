"""
Configuration loader for the UEBA anomaly detection pipeline.

Reads ueba_config.yaml from the config directory, validates it,
and returns a typed Config object that the rest of the pipeline uses.

Usage:
    from config import load_config
    config = load_config("config/ueba_config.yaml")
    print(config.data.n_users_total)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


# ============================================================
# DATACLASSES — typed structure of the config
# Each section of the YAML becomes one dataclass.
# ============================================================

@dataclass
class PeerGroupConfig:
    """Behavioral baseline for one peer group (e.g., engineers)."""
    n_users: int
    log_bytes_mean: float
    logins_mean: float
    hosts_mean: float
    fails_mean: float


@dataclass
class DataConfig:
    """Data generation parameters."""
    random_seed: int
    n_days: int
    test_window_start: int
    peer_groups: Dict[str, PeerGroupConfig]

    @property
    def n_users_total(self) -> int:
        """Total number of users across all peer groups."""
        return sum(g.n_users for g in self.peer_groups.values())


@dataclass
class AttackConfig:
    """Parameters for one attack type. Optional fields vary by attack."""
    target_group: str
    n_victims: int
    # All attack-specific fields are optional — different attacks use different ones.
    duration_days: int = None
    start_day_offset: int = 0
    bytes_multiplier: float = None
    logins_add: int = None
    fails_add: int = None
    hosts_add: int = None
    days_offset: List[int] = None
    ramp_start_multiplier: float = None
    ramp_step: float = None


@dataclass
class AttacksConfig:
    """All attack configurations."""
    random_seed: int
    attacks: Dict[str, AttackConfig]


@dataclass
class FeatureParameters:
    """Parameters that control how feature families compute."""
    rolling_window: int
    ramp_window: int
    spearman_window: int
    mad_constant: float
    epsilon: float
    std_floor_enabled: bool
    std_floor_fraction: float


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""
    raw_columns: List[str]
    enabled_families: Dict[str, bool]
    parameters: FeatureParameters


@dataclass
class ModelConfig:
    """Isolation Forest parameters."""
    algorithm: str
    n_estimators: int
    max_samples: int
    contamination: str            # "auto" or float as string
    seeds: List[int]


@dataclass
class EvaluationConfig:
    """How model outputs get scored."""
    top_k_values: List[int]
    primary_top_k: int
    attack_types_tracked: List[str]
    test_window_start: int
    aggregation: str              # "sum", "max", or "mean"


@dataclass
class MLflowConfig:
    """Experiment tracking settings."""
    enabled: bool
    tracking_uri: str
    experiment_name: str
    tags: Dict[str, str]


@dataclass
class OutputConfig:
    """Where results get written."""
    results_dir: str
    save_model: bool
    save_features_parquet: bool
    save_results_csv: bool
    generate_plots: bool
    verbose: bool


@dataclass
class Config:
    """Top-level configuration — bundles all sections."""
    data: DataConfig
    attacks: AttacksConfig
    features: FeaturesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    ablation_configs: Dict[str, List[str]]
    mlflow: MLflowConfig
    output: OutputConfig


# ============================================================
# LOADER FUNCTION
# Reads YAML → builds dataclasses → validates → returns Config
# ============================================================

def load_config(path: str) -> Config:
    """
    Load and validate the YAML config file.

    Raises FileNotFoundError if the file is missing.
    Raises ValueError if the config is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Build nested dataclasses from raw dict
    data = DataConfig(
        random_seed=raw["data"]["random_seed"],
        n_days=raw["data"]["n_days"],
        test_window_start=raw["data"]["test_window_start"],
        peer_groups={
            name: PeerGroupConfig(**params)
            for name, params in raw["data"]["peer_groups"].items()
        },
    )

    attacks = AttacksConfig(
        random_seed=raw["attacks"]["random_seed"],
        attacks={
            name: AttackConfig(**params)
            for name, params in raw["attacks"].items()
            if name != "random_seed"
        },
    )

    features = FeaturesConfig(
        raw_columns=raw["features"]["raw_columns"],
        enabled_families=raw["features"]["enabled_families"],
        parameters=FeatureParameters(**raw["features"]["parameters"]),
    )

    model = ModelConfig(**raw["model"])

    evaluation = EvaluationConfig(**raw["evaluation"])

    mlflow = MLflowConfig(**raw["mlflow"])

    output = OutputConfig(**raw["output"])

    config = Config(
        data=data,
        attacks=attacks,
        features=features,
        model=model,
        evaluation=evaluation,
        ablation_configs=raw["ablation_configs"],
        mlflow=mlflow,
        output=output,
    )

    validate(config)
    return config


def validate(config: Config) -> None:
    """
    Cross-section validation checks.
    Catches misconfigurations that individual sections couldn't.
    """
    # The test window should be inside the data range
    if config.data.test_window_start >= config.data.n_days:
        raise ValueError(
            f"test_window_start ({config.data.test_window_start}) "
            f"must be less than n_days ({config.data.n_days})"
        )

    # Evaluation's test_window_start should match data's
    if config.evaluation.test_window_start != config.data.test_window_start:
        raise ValueError(
            "evaluation.test_window_start does not match data.test_window_start"
        )

    # primary_top_k must be in top_k_values
    if config.evaluation.primary_top_k not in config.evaluation.top_k_values:
        raise ValueError(
            f"primary_top_k ({config.evaluation.primary_top_k}) "
            f"must be in top_k_values"
        )

    # Each ablation config should reference only enabled families
    enabled = {n for n, on in config.features.enabled_families.items() if on}
    for name, families in config.ablation_configs.items():
        for fam in families:
            if fam not in enabled:
                raise ValueError(
                    f"ablation_configs[{name!r}] references {fam!r} "
                    f"but it is not enabled in features.enabled_families"
                )

    # Attack targets must reference real peer groups
    valid_groups = set(config.data.peer_groups.keys())
    for attack_name, attack in config.attacks.attacks.items():
        if attack.target_group not in valid_groups:
            raise ValueError(
                f"Attack {attack_name!r} targets unknown group "
                f"{attack.target_group!r}"
            )

    # Aggregation must be one of the supported values
    if config.evaluation.aggregation not in ("sum", "max", "mean"):
        raise ValueError(
            f"evaluation.aggregation must be one of "
            f"['sum', 'max', 'mean'], got {config.evaluation.aggregation!r}"
        )