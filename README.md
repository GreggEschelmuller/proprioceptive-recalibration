# Proprioceptive Recalibration Analysis

Analysis code for sensorimotor adaptation and proprioceptive recalibration experiments.
This repository is intentionally structured as a small, configuration-driven analysis
pipeline rather than a collection of notebooks.

Each analysis is defined entirely by a YAML config file. The same code can be reused
across different experimental conditions by changing only the config.

---

## Requirements

- Python 3.11+
- uv (for dependency and environment management)

Install dependencies:

    uv sync

---

## Running an Analysis

Each analysis corresponds to a YAML file in `configs/`.

Run a single analysis like this:

    uv run python scripts/main.py configs/vibration-align.yaml

This will:

- load all participant block-summary CSVs
- apply filtering and invalidation rules
- compute learning-curve summaries (mean and variable error with bootstrap CIs)
- compute recalibration metrics (early-trial block comparisons)
- save figures and summary tables to `outputs/`

---

## Repository Structure

    configs/
        *.yaml                 Analysis configs (one per condition)

    data/
        raw/                   Raw data (not tracked)

    outputs/
        figures/               Saved figures (png + svg)
        *.csv                  Summary tables

    scripts/
        main.py                Run a single analysis config
        concat_summaries.py    Combine summaries across configs

    src/prop_recal/
        io.py                  Data loading utilities
        preprocess.py          Filtering and invalidation logic
        stats.py               Statistical summaries and bootstrap CIs
        plotting.py            Plotting and figure saving
        pipelines/
            run.py              Main analysis pipeline

---

## Configuration (YAML)

All analysis parameters live in YAML, including:

- participant IDs
- block names and ordering
- filtering rules (error thresholds, invalid trial conditions)
- bootstrap parameters
- bin sizes
- figure labels, limits, and output formats

Example snippet:

    recalibration:
      blocks: ["Baseline", "Post"]
      first_n: 5
      min_valid: 5

Changing the config does not require modifying any Python code.

---

## Learning-Curve Analyses

The main pipeline computes:

- Constant error (mean across participants per trial/bin)
- Variable error (within-subject SD per bin, summarized across participants)
- Bootstrap confidence intervals for both metrics

Trials are concatenated across blocks to allow continuous plotting while still
preserving block boundaries.

---

## Recalibration Analysis

The recalibration analysis compares early behavior between two blocks:

- per-participant mean of the first N trials
- difference scores (Block B − Block A)
- bootstrap confidence intervals across participants

Results are saved as a participant-level CSV and visualized as a paired plot with an
embedded difference panel.

---

## Combining Results Across Conditions

Multiple analyses can be combined into a single long-format table for statistics.

Run:

    uv run python scripts/concat_summaries.py configs/concat_summaries.yaml

The config specifies:

- input summary CSVs
- condition labels
- output path

The resulting CSV is suitable for downstream statistical analysis.

---

## Notes

- Raw data are intentionally excluded from version control.
- Figures are saved in both raster (PNG) and vector (SVG) formats.
- Bootstrap methods are used for uncertainty estimation throughout.
- Missing or insufficient participant data raise warnings but do not halt execution.
- The pipeline favors explicit, functional code over hidden state or class-based design.