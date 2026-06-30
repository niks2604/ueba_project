"""Quick sanity check that the model wrapper works."""

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


# Spark setup
spark = (SparkSession.builder
         .appName("UEBA-ModelTest")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Run the pipeline up through features
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

# Convert to pandas for the model
print("Pulling features into pandas for modeling...")
pdf = sdf_features.toPandas()

# Pick a feature set — start with the winning config
feature_cols = (
    ['bytes', 'logins', 'hosts', 'fails']
    + [f'{c}_z_self'      for c in ['bytes', 'logins', 'hosts', 'fails']]
    + [f'{c}_rz_self'     for c in ['bytes', 'logins', 'hosts', 'fails']]
    + [f'{c}_z_peer_loo'  for c in ['bytes', 'logins', 'hosts', 'fails']]
    + ['bytes_spearman', 'bytes_spearman_pos']
)
pdf[feature_cols] = pdf[feature_cols].fillna(0)

# Build the detector and score
detector = AnomalyDetector(config.model)
X = pdf[feature_cols].values
scores = detector.fit_and_score(X, seed=0)

print(f"✓ Model fit and scoring successful")
print(f"  Features used: {len(feature_cols)}")
print(f"  Rows scored: {len(scores):,}")
print(f"  Score range: {scores.min():.3f} to {scores.max():.3f}")
print(f"  Score mean: {scores.mean():.3f}")

# Sanity check: top-scored rows should include attack rows
pdf['score'] = scores
top_rows = pdf.nlargest(50, 'score')
print(f"\n  Top 50 rows by score — how many are anomalies?")
print(f"    {top_rows['is_anomaly'].sum()} / 50")
print(f"  Top 50 rows — anomaly type breakdown:")
print(f"    {top_rows['anomaly_type'].value_counts().to_string()}")