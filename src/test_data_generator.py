"""Quick sanity check that data generation works."""

from config import load_config
from data_generator import DataGenerator


config = load_config("../config/ueba_config.yaml")
generator = DataGenerator(config.data)
df = generator.generate()

print(f"✓ Data generated successfully")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Users: {df['user_id'].nunique()}")
print(f"  Days: {df['day'].nunique()}")
print(f"  Date range: day {df['day'].min()} to day {df['day'].max()}")
print()
print(f"  Per-group user counts:")
print(df.groupby('group')['user_id'].nunique().to_string())
print()
print(f"  Sample rows (user 0, days 0-4):")
print(df[df['user_id'] == 0].head().to_string(index=False))