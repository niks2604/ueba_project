"""Quick sanity check that attack planting works."""

from config import load_config
from data_generator import DataGenerator
from attack_planter import AttackPlanter


config = load_config("../config/ueba_config.yaml")

# Generate base data
generator = DataGenerator(config.data)
df = generator.generate()

# Plant attacks
planter = AttackPlanter(
    attacks_config=config.attacks,
    test_window_start=config.data.test_window_start,
    user_groups=generator.get_user_groups(),
)
df_attacked = planter.plant(df)

print(f"✓ Attacks planted successfully\n")

# Total anomalous rows
total_anomalous = df_attacked['is_anomaly'].sum()
print(f"  Total anomalous rows: {total_anomalous}")

# Breakdown by type
print(f"\n  Anomalous rows per attack type:")
breakdown = (df_attacked[df_attacked['is_anomaly']]
             .groupby('anomaly_type')['user_id']
             .agg(['count', 'nunique'])
             .rename(columns={'count': 'rows', 'nunique': 'unique_users'}))
print(breakdown.to_string())

# Victims
print(f"\n  Victim assignments (first 5 per type):")
for attack_type, victims in planter.get_victims().items():
    print(f"    {attack_type:14s}: {sorted(victims)[:5]}...")

# Spot check: one exfil victim, see what their bytes look like
print(f"\n  Sample exfil victim — bytes around test window:")
victim = planter.get_victims()['exfil'][0]
sample = df_attacked[
    (df_attacked['user_id'] == victim) &
    df_attacked['day'].between(50, 59)
][['day', 'bytes', 'is_anomaly', 'anomaly_type']]
print(sample.to_string(index=False))