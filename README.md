# UEBA Anomaly Detection Pipeline

A **User & Entity Behavior Analytics (UEBA)** pipeline that detects insider-threat and
account-compromise activity from daily user-activity logs. It generates a synthetic
workforce, injects five realistic attack types, engineers behavioral features on Spark,
scores users with an Isolation Forest, and runs a **seed-averaged ablation study** to
prove *which feature families actually improve detection* — instead of assuming more
features is better.

The whole project is built around one question a real SOC analyst faces:

> Given a limited investigation budget (top-K users per day), which feature set catches
> the most attackers?

---

## Table of contents

- [Why I built it](#why-i-built-it)
- [How I built it (the design story)](#how-i-built-it-the-design-story)
- [What I used, and why](#what-i-used-and-why)
- [Architecture](#architecture)
- [Attack scenarios](#attack-scenarios)
- [Feature families](#feature-families)
- [Results (real numbers)](#results-real-numbers)
- [What I improved along the way](#what-i-improved-along-the-way)
- [What still needs improvement](#what-still-needs-improvement)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## Why I built it

Most anomaly-detection demos stop at "we trained a model and it found outliers." That
tells you nothing about **whether the features earn their keep**, or whether the model
would actually help an analyst who can only investigate a handful of users per day.

I wanted a project that:

1. **Frames detection the way a SOC actually consumes it** — as a ranked list under a
   fixed investigation budget (precision@K / recall@K), not as an abstract AUC.
2. **Treats feature engineering as a hypothesis to be tested**, not decoration. Every
   feature family is toggleable, and an ablation study measures its real contribution.
3. **Is fully reproducible and auditable** — synthetic data with fixed seeds, config-driven
   behavior, MLflow-tracked runs, and a real test suite.

Because real corporate activity logs are sensitive and hard to label, I generate a
**controlled synthetic world** where I know the ground truth (exactly who the attackers
are), which lets me measure detection quality honestly.

---

## How I built it (the design story)

I built this as a **config-driven, stage-based pipeline** so that experiments never
require code changes — everything from dataset size to which feature families are active
lives in [`config/ueba_config.yaml`](config/ueba_config.yaml).

**1. Realistic synthetic data, in three layers of variation.**
Real workforce activity has structure: departments differ, people within a department
differ, and each person varies day to day. So [`data_generator.py`](src/data_generator.py)
models exactly that — a **group baseline** (engineer / finance / sales / exec), a
**per-user baseline** drawn from the group, and **daily fluctuation** around the user's own
baseline. This matters because a flat dataset would make any anomaly trivial to spot;
layered variation is what makes the detection problem realistic.

**2. Attacks planted only in a held-out test window.**
[`attack_planter.py`](src/attack_planter.py) injects attacks *only* into the last days of
the series (`test_window_start`), so the training period stays clean and the model must
detect deviations from a genuinely learned baseline — not memorize the attacks.

**3. Feature engineering as a family of testable hypotheses.**
[`features.py`](src/features.py) computes **seven feature families** on Spark. Each encodes
a different theory of "what an attack looks like": deviation from your own past (`self_z`,
`robust_z`), deviation from your peers (`peer_z`), rate-of-change (`diff`), sustained trend
(`rolling`), and *trajectory shape* (`ramp`, `spearman`) for the stealthy slow-ramp attack.
The pipeline computes only the families enabled in config.

**4. Scoring built to be swappable and stable.**
[`model.py`](src/model.py) wraps sklearn's Isolation Forest behind a thin interface, so a
future detector (LOF, autoencoder) only touches one file. Because Isolation Forest is
randomized, single runs are noisy — so I **fit across five seeds and average**, turning a
jittery score into a stable metric.

**5. Evaluation in the analyst's currency.**
[`evaluator.py`](src/evaluator.py) aggregates row-level scores to the **user level**, ranks
users, and reports **precision@K / recall@K** across a range of budgets plus a
**per-attack-type catch rate** — so I can see not just *how many* attackers were caught but
*which kinds*.

**6. An ablation harness to settle the "more features?" question.**
[`ablation.py`](src/ablation.py) runs every named feature configuration across every seed,
logs each run to MLflow, and produces one seed-averaged comparison table. This is the
centerpiece — it's what turns opinions about features into evidence.

[`pipeline.py`](src/pipeline.py) wires all seven stages together behind a CLI.

---

## What I used, and why

| Tool | Role | Why this choice |
|------|------|-----------------|
| **Python 3.10+** | Everything | Standard for ML tooling; dataclasses give typed config for free |
| **PySpark** | Feature engineering | Window functions express per-user trailing stats cleanly, and it scales to real log volumes without rewriting the feature code |
| **pandas / NumPy** | Data gen + attack planting | Fast and expressive, and the dataset is small before the feature stage |
| **SciPy** | Spearman correlation | The one feature family cleaner to compute in pandas, then joined back to Spark |
| **scikit-learn** | Isolation Forest | The reference unsupervised detector for tabular anomalies; small `max_samples` avoids anomaly masking |
| **MLflow** | Experiment tracking | Every config × seed run is logged to SQLite so results are queryable and auditable across sessions |
| **PyYAML + dataclasses** | Config | One YAML file drives all behavior; a typed loader validates it and fails loudly on misconfiguration |
| **pytest** | Testing | 72-test suite; Spark tests are marked so the fast pandas tests can run without JVM startup |
| **Parquet / PyArrow** | Caching | Feature table is cached so re-runs skip the expensive Spark stage |

---

## Architecture

```
 Stage                              Module              Engine
 ─────────────────────────────────  ──────────────────  ──────────────
 1. Load & validate config          config.py           —
 2. Generate synthetic users        data_generator.py   pandas
 3. Plant attacks (test window)     attack_planter.py   pandas
 4. Compute features (7 families)   features.py         Spark (+SciPy)
 5. Score users                     model.py            Isolation Forest
 6. Evaluate (precision/recall@K)   evaluator.py        pandas/NumPy
 7. Ablation sweep + MLflow log     ablation.py         —
 ─────────────────────────────────  ──────────────────  ──────────────
 Orchestrated end-to-end by         pipeline.py         CLI
```

---

## Attack scenarios

Five attack types, each with a distinct behavioral signature
(all configured in [`config/ueba_config.yaml`](config/ueba_config.yaml)):

| Attack          | Signal                                       | Target group |
|-----------------|----------------------------------------------|--------------|
| `exfil`         | Large multi-day spike in outbound bytes      | engineer     |
| `cred_theft`    | Elevated logins + failed-login spike         | finance      |
| `lateral`       | Many additional host connections             | sales        |
| `subtle_exfil`  | Moderate byte spike on non-consecutive days  | finance      |
| `slow_ramp`     | Bytes climbing monotonically over days       | exec         |

`slow_ramp` is the hard case: **each individual day looks unremarkable** — the signal lives
in the *trajectory shape*. That's exactly what the `ramp` and `spearman` features were
designed to catch, and the ablation results below confirm they're the reason it gets found.

---

## Feature families

| Family     | What it measures                                        |
|------------|---------------------------------------------------------|
| `self_z`   | z-score vs the user's own past (mean / std)             |
| `robust_z` | robust z vs the user's past (median / MAD)              |
| `peer_z`   | z-score vs the peer group, **leave-one-out**            |
| `rolling`  | rolling 7-day max / mean / std                          |
| `diff`     | day-over-day difference and percent change              |
| `ramp`     | regression slope, R², and a gated upward-trend signal   |
| `spearman` | trailing Spearman rank correlation of day vs value      |

`ramp` and `spearman` are treated as **bytes-only** families — applying them to the other
raw columns added noise without adding signal (a finding from the ablation, not an
assumption).

---

## Results (real numbers)

From the seed-averaged ablation
([`src/results/ablation_summary.csv`](src/results/ablation_summary.csv)), at the primary
investigation budget **K = 50**:

| Config                       | Features | Precision@50 | Recall@50 | exfil | cred_theft | lateral | subtle_exfil | slow_ramp |
|------------------------------|:--------:|:------------:|:---------:|:-----:|:----------:|:-------:|:------------:|:---------:|
| **z_plus_spearman** ⭐       | 18       | **0.656**    | 0.656     | 1.00  | 0.82       | 0.40    | 0.32         | 0.74      |
| z_plus_ramp                  | 19       | 0.640        | 0.640     | 1.00  | 0.70       | 0.34    | 0.36         | 0.80      |
| z_plus_ramp_plus_spearman    | 21       | 0.588        | 0.588     | 1.00  | 0.42       | 0.24    | 0.34         | 0.94      |
| z_scores                     | 16       | 0.584        | 0.584     | 1.00  | 0.88       | 0.58    | 0.26         | 0.20      |
| everything                   | 41       | 0.572        | 0.572     | 1.00  | 0.98       | 0.40    | 0.10         | 0.38      |
| z_plus_diff                  | 24       | 0.528        | 0.528     | 1.00  | 0.88       | 0.40    | 0.06         | 0.30      |
| baseline (raw only)          | 4        | 0.340        | 0.340     | 1.00  | 0.44       | 0.22    | 0.02         | 0.02      |

**What the table proves:**

- **Feature engineering nearly doubles detection.** Precision@50 rises from **0.34** (raw
  values) to **0.66** (z-scores + Spearman).
- **`spearman` is what rescues `slow_ramp`** — the stealth attack goes from a **2%** catch
  rate (z-scores alone) to **74%**, without wrecking the louder attacks.
- **More features is *not* better.** The `everything` config (41 features) scores *worse*
  (0.572) than the focused 18-feature winner. This is **feature dilution**: uninformative
  columns add noise the Isolation Forest has to split on. Proving this empirically was a
  core goal of the project.
- **There are real trade-offs.** `z_scores` catches `lateral` best (0.58) but `slow_ramp`
  worst (0.20); adding both `ramp` and `spearman` maxes out `slow_ramp` (0.94) but sinks
  `cred_theft` (0.42). The per-attack columns make these trade-offs explicit instead of
  hiding them inside a single average.

**Recommended production config: `z_plus_spearman`** — the best overall precision and the
one that recovers the stealth attack without sacrificing the loud ones.

---

## What I improved along the way

These are decisions I changed *because the data told me to*, not upfront:

- **From "add every feature" to a curated set.** My first instinct was to enable all seven
  families. The ablation showed `everything` (41 features) underperforming an 18-feature
  config — so the recommendation became a focused set. Feature dilution is now a headline
  finding, not a footnote.
- **Made `ramp` / `spearman` bytes-only.** Computing them across all four raw columns added
  features that never helped and sometimes hurt. Restricting them to `bytes` cut the noise.
- **Switched score aggregation to `sum`.** `max` rewards a single loud day (good for
  `exfil`, blind to `slow_ramp`); `mean` punishes long attacks. `sum` credits sustained
  multi-day patterns and gave the best all-round ranking.
- **Averaged over five seeds.** A single Isolation Forest run produced noticeably jittery
  precision numbers. Seed-averaging turned the ablation table from anecdote into something
  I'd trust.
- **Added leave-one-out to `peer_z`.** Including a user in their own peer statistics let an
  attacker inflate the very baseline they were being compared against. Leave-one-out fixes
  that self-contamination.
- **Hardened the config loader.** It now validates cross-section consistency (e.g. the
  evaluation window must match the data window; ablation configs may only reference enabled
  families) and raises a clear `ValueError` instead of failing deep inside Spark.

---

## What still needs improvement

Honest limitations and the roadmap I'd tackle next:

- **Synthetic-only data.** Everything is validated on generated data where I control the
  ground truth. The real test is labeled (or semi-labeled) production logs — the feature
  code is Spark-native partly to make that transition possible.
- **Single detector.** Only Isolation Forest is wired in. The `model.py` interface is built
  to swap, but LOF, an autoencoder, or a simple ensemble are untested. An ensemble would
  likely help the attacks no single config catches well (`lateral`, `subtle_exfil`).
- **Weak on two attack types.** `lateral` (~0.40) and `subtle_exfil` (~0.32) are still
  poorly detected. They need dedicated features — e.g. host-graph novelty for lateral
  movement, and a non-consecutive-day burst detector for subtle exfil.
- **No temporal drift handling.** The baseline is static. Real behavior drifts (new
  projects, reorgs), which would raise false positives over time. A rolling/decaying
  baseline is the fix.
- **Attack magnitudes are somewhat generous.** I'd like to push the multipliers down toward
  the detection floor to find where each feature family breaks.
- **No cost-sensitive thresholding.** K is fixed by config; a real deployment should tie K
  to analyst capacity and the cost of a miss vs. a false alarm.
- **Single-machine Spark.** The code is cluster-ready but only run locally. Validating on a
  real cluster at production volume is untested.
- **No CI.** The pytest suite is solid but not yet wired into CI, and there's no automated
  lint/type-check gate.

---

## Installation

Requires **Python 3.10+**, **Java 8/11/17** (for Spark), and:

```bash
pip install pyspark pandas numpy scipy scikit-learn pyyaml mlflow pytest pyarrow
```

MLflow is optional — the ablation runner degrades gracefully and skips logging if it isn't
installed.

---

## Usage

Run the full pipeline from inside `src/`:

```bash
cd src
python pipeline.py                                   # default config
python pipeline.py --config ../config/ueba_config.yaml
python pipeline.py --skip-features                   # reuse cached features (fast)
```

Or drive it programmatically:

```python
from pipeline import UEBAPipeline

pipeline = UEBAPipeline("../config/ueba_config.yaml")
summary = pipeline.run()      # returns the seed-averaged ablation summary
print(summary)
```

Outputs land in `output.results_dir` (default `./results`): `ablation_summary.csv`, a
cached `features.parquet`, and MLflow runs in `mlflow.db`.

---

## Configuration

Everything is driven by [`config/ueba_config.yaml`](config/ueba_config.yaml) — no code
changes needed to run a new experiment:

- **`data`** — days, test-window start, and per-peer-group baselines.
- **`attacks`** — victims, duration, and magnitude for each attack type.
- **`features`** — toggle feature families on/off and set window sizes / constants.
- **`model`** — Isolation Forest hyperparameters and the seeds to average over.
- **`evaluation`** — the K values, primary K, and score aggregation (sum / max / mean).
- **`ablation_configs`** — the named feature-set experiments to compare.
- **`mlflow` / `output`** — experiment tracking and which artifacts to save.

---

## Testing

A **pytest** suite (72 tests) lives in `tests/`. Spark-backed tests are marked so the fast
pandas tests can run without JVM startup:

```bash
cd tests
pytest                      # everything
pytest -m "not spark"       # fast, no JVM
pytest -m spark             # only Spark feature-pipeline tests
```

Coverage spans config loading/validation, data generation, attack planting, the model
wrapper, the evaluator's metrics (checked against analytically known values), the ablation
helpers, and the Spark feature pipeline.

---

## Project layout

```
ueba_project/
├── config/
│   └── ueba_config.yaml      # all tunable parameters (data, attacks, features, model, eval)
├── src/
│   ├── config.py             # typed config loader + cross-section validation
│   ├── data_generator.py     # synthetic user-day data (3 layers of variation)
│   ├── attack_planter.py     # injects the five attack types into the test window
│   ├── features.py           # Spark feature engineering (7 families)
│   ├── model.py              # Isolation Forest wrapper (swappable interface)
│   ├── evaluator.py          # user-level ranking + precision/recall@K
│   ├── ablation.py           # config × seed sweep, MLflow logging
│   ├── pipeline.py           # end-to-end orchestrator + CLI
│   └── results/              # ablation_summary.csv, cached features, victims
└── tests/                    # pytest suite (72 tests)
```
