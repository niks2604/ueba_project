"""
Shared pytest fixtures for the UEBA test suite.

Adds src/ to the import path, loads the real config once, and exposes
the (expensive-ish) data-generation and attack-planting stages as
session-scoped fixtures so they run a single time for the whole suite.

Spark is only started if a test actually requests the `spark` fixture,
so the pure-pandas tests stay fast and Spark-free.
"""

import os
import sys
from pathlib import Path

import pytest

# Make the pipeline modules importable as top-level names (config,
# data_generator, ...), matching how the pipeline imports them.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = REPO_ROOT / "config" / "ueba_config.yaml"


@pytest.fixture(scope="session")
def config_path() -> str:
    """Absolute path to the real project config."""
    return str(CONFIG_PATH)


@pytest.fixture(scope="session")
def config(config_path):
    """The real, validated project config (loaded once)."""
    from config import load_config
    return load_config(config_path)


@pytest.fixture(scope="session")
def generator(config):
    """A DataGenerator that has already produced its data."""
    from data_generator import DataGenerator
    gen = DataGenerator(config.data)
    gen.generate()          # populates user_groups
    return gen


@pytest.fixture(scope="session")
def generated_df(config):
    """The raw synthetic dataset (pandas)."""
    from data_generator import DataGenerator
    return DataGenerator(config.data).generate()


@pytest.fixture(scope="session")
def planter_and_df(config):
    """
    (df_attacked, planter) for the real config.

    Built once and reused; do not mutate the returned DataFrame in a
    test — copy it first if you need to.
    """
    from data_generator import DataGenerator
    from attack_planter import AttackPlanter

    gen = DataGenerator(config.data)
    df = gen.generate()
    planter = AttackPlanter(
        attacks_config=config.attacks,
        test_window_start=config.data.test_window_start,
        user_groups=gen.get_user_groups(),
    )
    df_attacked = planter.plant(df)
    return df_attacked, planter


@pytest.fixture(scope="session")
def spark():
    """
    A local Spark session, started only when requested.

    Session-scoped so the (slow) JVM startup happens at most once.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .appName("UEBA-Tests")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
