"""Quick sanity check that config loading works."""
from config import load_config

config = load_config("../config/ueba_config.yaml")

print("✓ Config loaded successfully")
print(f"  Total users: {config.data.n_users_total}")
print(f"  Peer groups: {list(config.data.peer_groups.keys())}")
print(f"  Attack types: {list(config.attacks.attacks.keys())}")
print(f"  Enabled feature families: "
      f"{[k for k, v in config.features.enabled_families.items() if v]}")
print(f"  Ablation configs defined: {list(config.ablation_configs.keys())}")
print(f"  Primary K: {config.evaluation.primary_top_k}")
print(f"  MLflow enabled: {config.mlflow.enabled}")