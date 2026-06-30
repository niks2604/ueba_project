"""End-to-end ablation test."""

import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

from config import load_config
from data_generator import DataGenerator
from attack_planter import AttackPlanter
from features import FeaturePipeline
from ablation import AblationRunner


# Spark setup
spark = (SparkSession.builder
         .appName("UEBA-AblationTest")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Build everything up to features
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

# Run the ablation
print("\n" + "="*70)
print("RUNNING ABLATION — all configs × all seeds")
print("="*70)
runner = AblationRunner(config)
summary = runner.run(pdf=pdf, victims=planter.get_victims())

# Print
print("\n" + "="*70)
print("ABLATION SUMMARY (averaged across 5 seeds)")
print("="*70)
print(summary.to_string(index=False))

# Save
out_path = config.output.results_dir + "/ablation_summary.csv"
import os
os.makedirs(config.output.results_dir, exist_ok=True)
summary.to_csv(out_path, index=False)
print(f"\n✓ Saved to {out_path}")