"""Quick sanity check that feature engineering works."""

import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

from config import load_config
from data_generator import DataGenerator
from attack_planter import AttackPlanter
from features import FeaturePipeline


# Start Spark
spark = (SparkSession.builder
         .appName("UEBA-FeatureTest")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark {spark.version} started\n")

# Run the pipeline so far
# Run the pipeline so far
config = load_config("../config/ueba_config.yaml")

# Step 1: generate data
generator = DataGenerator(config.data)
df = generator.generate()

# Step 2: plant attacks (needs user_groups from the generator)
planter = AttackPlanter(
    attacks_config=config.attacks,
    test_window_start=config.data.test_window_start,
    user_groups=generator.get_user_groups(),
)
df_attacked = planter.plant(df)

# Convert to Spark and compute features
sdf = spark.createDataFrame(df_attacked)
pipeline = FeaturePipeline(config.features, spark)
sdf_features = pipeline.compute_all(sdf)

# Force computation by calling count + listing columns
total_rows = sdf_features.count()
all_cols = sdf_features.columns

print(f"✓ Features computed successfully")
print(f"  Rows: {total_rows:,}")
print(f"  Total columns: {len(all_cols)}")
print(f"\n  Columns by family:")

families = {
    'raw':       ['bytes', 'logins', 'hosts', 'fails'],
    'labels':    ['is_anomaly', 'anomaly_type', 'user_id', 'day', 'group'],
    'self_z':    [c for c in all_cols if 'z_self' in c and 'rz' not in c],
    'robust_z':  [c for c in all_cols if 'rz_self' in c or 'med_self' in c or 'mad_self' in c],
    'peer_z':    [c for c in all_cols if 'peer' in c],
    'rolling':   [c for c in all_cols if 'roll7' in c],
    'diff':      [c for c in all_cols if 'diff1' in c or 'pct1' in c],
    'ramp':      [c for c in all_cols if 'ramp' in c],
    'spearman':  [c for c in all_cols if 'spearman' in c],
}
for name, cols in families.items():
    print(f"    {name:10s}: {len(cols)} columns")

# Spot check: a known slow_ramp victim should show climbing
# bytes_spearman over the attack
print(f"\n  Sample slow_ramp victim — Spearman across attack:")
victim = planter.get_victims()['slow_ramp'][0]
sample = (sdf_features
    .filter((sdf_features['user_id'] == victim) &
            sdf_features['day'].between(50, 59))
    .select('day', 'bytes', 'bytes_spearman', 'is_anomaly', 'anomaly_type')
    .orderBy('day'))
sample.show(truncate=False)