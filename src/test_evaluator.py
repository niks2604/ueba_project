"""Quick sanity check that evaluation works end-to-end."""

import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import numpy as np
from pyspark.sql import SparkSession

from config import load_config
from data_generator import DataGenerator
from attack_planter import AttackPlanter
from features import FeaturePipeline
from model import AnomalyDetector
from evaluator import Evaluator


# Spark setup
spark = (SparkSession.builder
         .appName("UEBA-EvaluatorTest")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Full pipeline up through model
config = load_config("../config/ueba_config.yaml")

generator = DataGenerator(config.data)
df = generator.generate()
planter = AttackPlanter(
    attacks_config=config.attacks,
    test_window_start=config.data.test_window_start,
    user_groups=generator.get_user_groups(),
)
df_attacked = planter.plant(df)
sdf = spark.createDataFrame(df_attacked)
pipeline = FeaturePipeline(config.features, spark)
sdf_features = pipeline.compute_all(sdf)

pdf = sdf_features.toPandas()

# Use the winning config's features (z + Spearman)
feature_cols = (
    ['bytes', 'logins', 'hosts', 'fails']
    + [f'{c}_z_self'      for c in ['bytes', 'logins', 'hosts', 'fails']]
    + [f'{c}_rz_self'     for c in ['bytes', 'logins', 'hosts', 'fails']]
    + [f'{c}_z_peer_loo'  for c in ['bytes', 'logins', 'hosts', 'fails']]
    + ['bytes_spearman', 'bytes_spearman_pos']
)
pdf[feature_cols] = pdf[feature_cols].fillna(0)

detector = AnomalyDetector(config.model)
scores = detector.fit_and_score(pdf[feature_cols].values, seed=0)

# THE NEW PART: run the evaluator
evaluator = Evaluator(config.evaluation)
result = evaluator.evaluate(
    pdf=pdf,
    scores=scores,
    victims=planter.get_victims(),
)

print(f"✓ Evaluation complete\n")
print(f"  Total attackers: {result.n_attackers_total}\n")

print(f"  Precision and recall at each K:")
for k in config.evaluation.top_k_values:
    p = result.precision_at_k[k]
    r = result.recall_at_k[k]
    print(f"    K={k:3d}    P={p:.3f}   R={r:.3f}")

print(f"\n  Per-attack-type catch rate at K={config.evaluation.primary_top_k}:")
for attack_type, catch in result.caught_by_type.items():
    print(f"    {attack_type:14s}  {catch:.2f}")

print(f"\n  Summary: {result.summary_line(config.evaluation.primary_top_k)}")

print(f"\n  to_dict() output (first few entries):")
flat = result.to_dict()
for k, v in list(flat.items())[:8]:
    print(f"    {k:30s}  {v:.3f}")
# Quick 5-seed averaging check
print("\n\n5-seed average for the same config:")
all_results = []
for seed in range(5):
    scores = detector.fit_and_score(pdf[feature_cols].values, seed=seed)
    result = evaluator.evaluate(pdf=pdf, scores=scores,
                                 victims=planter.get_victims())
    all_results.append(result.to_dict())

import pandas as pd
df_results = pd.DataFrame(all_results)
print(df_results.mean().round(3).to_string())