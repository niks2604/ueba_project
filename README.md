# UEBA Anomaly Detection Pipeline

A **User & Entity Behavior Analytics (UEBA)** pipeline for detecting insider-threat
and account-compromise activity from daily user-activity logs. It generates synthetic
workforce data, injects realistic attacks, engineers behavioral features on Spark,
scores users with an Isolation Forest, and runs a seed-averaged **ablation study** to
identify which feature families actually improve detection.

The pipeline is built around one question a SOC analyst really faces:
*given a limited investigation budget (top-K users per day), which feature set catches
the most attackers?*

---

## Contents

- [How it works](#how-it-works)
- [Attack scenarios](#attack-scenarios)
- [Feature families](#feature-families)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Interpreting results](#interpreting-results)

---

## How it works

The pipeline runs seven stages, orchestrated by [`src/pipeline.py`](src/pipeline.py):

```
 1. Load & validate config        config.py
 2. Generate synthetic users      data_generator.py   (pandas)
 3. Plant attacks                 attack_planter.py   (pandas)
 4. Compute features              features.py         (Spark)
 5. Score users                   model.py            (Isolation Forest)
 6. Evaluate (precision/recall@K) evaluator.py
 7. Ablation sweep + MLflow log   ablation.py
```

Data is generated in **three layers of variation** that mirror real workforce activity:
each peer group (engineer, finance, sales, exec) has its own behavioral baseline, each
user draws a personal baseline from their group, and each day fluctuates around that
personal baseline. Attacks are then planted only inside the **test window** (the last
days of the series), so the training period stays clean.

Row-level anomaly scores are aggregated to the **user level** (sum / max / mean), users
are ranked, and detection quality is measured as **precision@K** and **recall@K** for a
range of K values, plus a per-attack-type catch rate.

---

## Attack scenarios

Five attack types are injected, each with a distinct behavioral signature
(configured in [`config/ueba_config.yaml`](config/ueba_config.yaml)):

| Attack          | Signal                                   | Target group |
|-----------------|------------------------------------------|--------------|
| `exfil`         | Large multi-day spike in outbound bytes  | engineer     |
| `cred_theft`    | Elevated logins + failed-login spike     | finance      |
| `lateral`       | Many additional host connections         | sales        |
| `subtle_exfil`  | Moderate byte spike on non-consecutive days | finance   |
| `slow_ramp`     | Bytes climbing monotonically over days   | exec         |

`slow_ramp` is the stealth case: each individual day looks unremarkable — the signal
lives in the *trajectory shape*, which is exactly what the ramp and Spearman features
are designed to catch.

---

## Feature families

Seven feature families are computed on Spark (Spearman is computed in pandas via SciPy
and joined back). Each can be toggled independently in the config:

| Family      | What it measures                                          |
|-------------|-----------------------------------------------------------|
| `self_z`    | z-score vs the user's own past (mean / std)               |
| `robust_z`  | robust z vs the user's past (median / MAD)                |
| `peer_z`    | z-score vs the peer group, **leave-one-out**              |
| `rolling`   | rolling 7-day max / mean / std                            |
| `diff`      | day-over-day difference and percent change                |
| `ramp`      | regression slope, R², and a gated upward-trend signal     |
| `spearman`  | trailing Spearman rank correlation of day vs value        |

The ablation runner treats `ramp` and `spearman` as **bytes-only** families, matching
the project finding that they add noise on the other raw columns.

---

## Project layout

```
ueba_project/
├── config/
│   └── ueba_config.yaml      # all tunable parameters (data, attacks, features, model, eval)
├── src/
│   ├── config.py             # typed config loader + cross-section validation
│   ├── data_generator.py     # synthetic user-day data
│   ├── attack_planter.py     # injects the five attack types
│   ├── features.py           # Spark feature engineering (7 families)
│   ├── model.py              # Isolation Forest wrapper
│   ├── evaluator.py          # user-level ranking + precision/recall@K
│   ├── ablation.py           # config × seed sweep, MLflow logging
│   ├── pipeline.py           # end-to-end orchestrator + CLI
│   └── test_*.py             # runnable sanity-check scripts (demos)
└── tests/                    # pytest suite (see Testing)
```

---

## Installation

Requires **Python 3.10+**, Java 8/11/17 (for Spark), and the following packages:

```bash
pip install pyspark pandas numpy scipy scikit-learn pyyaml mlflow pytest pyarrow
```

MLflow is optional — the ablation runner degrades gracefully and skips logging if it
isn't installed.

---

## Usage

Run the full pipeline from inside `src/`:

```bash
cd src
python pipeline.py                                  # default config
python pipeline.py --config ../config/ueba_config.yaml
python pipeline.py --skip-features                  # reuse cached features (fast)
```

Or drive it programmatically:

```python
from pipeline import UEBAPipeline

pipeline = UEBAPipeline("../config/ueba_config.yaml")
summary = pipeline.run()      # returns the seed-averaged ablation summary
print(summary)
```

Outputs are written to the configured `output.results_dir` (default `./results`):
a `ablation_summary.csv`, a cached `features.parquet`, and MLflow runs in `mlflow.db`.

---

## Configuration

Everything is driven by [`config/ueba_config.yaml`](config/ueba_config.yaml) — no code
changes needed to adjust behavior. Key sections:

- **`data`** — number of days, test-window start, and per-peer-group baselines.
- **`attacks`** — victims, duration, and magnitude for each attack type.
- **`features`** — toggle feature families on/off and set window sizes / constants.
- **`model`** — Isolation Forest hyperparameters and the seeds to average over.
- **`evaluation`** — the K values, primary K, and score aggregation (sum / max / mean).
- **`ablation_configs`** — named feature-set experiments to compare.
- **`mlflow` / `output`** — experiment tracking and what artifacts to save.

The loader validates cross-section consistency (e.g. the evaluation window must match the
data window, ablation configs may only reference enabled families) and raises a clear
`ValueError` on misconfiguration.

---

## Testing

The `tests/` directory holds a full **pytest** suite (81 tests). Pure-pandas tests are
fast; Spark-backed feature tests are marked so they can be skipped.

```bash
cd tests
pytest                      # everything (~12s)
pytest -m "not spark"       # fast, no JVM startup (~4s)
pytest -m spark             # only the Spark feature-pipeline tests
```

Coverage spans config loading/validation, data generation, attack planting, the model
wrapper, the evaluator's metrics (checked against analytically known values), the
ablation helpers, and the Spark feature pipeline.

---

## Interpreting results

The ablation summary is one row per named config, sorted by precision at the primary K,
averaged across all model seeds:

- **`baseline`** (raw values only) is the naive floor.
- **`z_scores`** typically gives the best all-round balance.
- **`z_plus_spearman`** is the recommended production config — it recovers `slow_ramp`
  without sacrificing the louder attacks.
- **`everything`** demonstrates *feature dilution*: adding every family degrades ranking
  quality because uninformative columns add noise to the Isolation Forest.

Per-attack-type catch rates (`caught_exfil`, `caught_slow_ramp`, …) show *which* attacks
each config detects, making the trade-offs between feature sets explicit.
