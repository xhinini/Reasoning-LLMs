# Thinking LLMs’ Reasoning in Code Generation: Quality, Robustness, and Adaptability

This repository accompanies the study on thinking LLMs’ reasoning quality in code generation. It provides datasets, model outputs, and scripts used to evaluate and analyze reasoning traces across diverse coding tasks.

## What’s inside

- **data/**
  - BigCodeBench subsets and task metadata we used in the study.
- **models/**
  - The scripts for running reasoning models.
- **eval/**
  - Evaluation scripts for reasoning quality (efficiency, logical correctness, completeness), stability analysis, and correctness correlations.
- **examples/**
  - Curated example tasks and reasoning traces for quick inspection.
- **model_outputs/**
  - Raw and/or normalized reasoning traces and final code outputs from the evaluated models.


## Quick start

1) Environment

- Python 3.10+ recommended
- Create and activate a virtual environment

2) Data

- Place BigCodeBench subsets and task files under `data/` (we include the exact splits we used where licensing permits). If some files are not included, follow their original dataset instructions to obtain them and mirror the expected structure under `data/`.

3) Evaluate

- The `eval/` folder contains scripts to:
  - parse and score reasoning traces along the three dimensions,
  - compute correlations between reasoning quality and task correctness,
  - perform stability analysis across effort levels,
  - summarize per-model and per-difficulty results.

Example usage pattern (pseudo-CLI; adapt to script names/flags in `eval/`):

```
python eval/evaluate.py \
  --tasks data/<tasks_file>.jsonl \
  --inputs model_outputs/<model_name>/ \
  --metrics efficiency logical_correctness completeness \
  --out results/<model_name>.json
```

