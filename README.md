# Thinking LLMs’ Reasoning in Code Generation: Quality, Robustness, and Adaptability

This repository accompanies the study on thinking LLMs’ reasoning quality in code generation. It provides datasets, model outputs, and scripts used to evaluate and analyze reasoning traces across diverse coding tasks.

## What’s inside

- **data/**
  - BigCodeBench subsets and task metadata we used in the study.
- **models/**
  - The scripts for running reasoning models.
- **eval/**
  - Evaluation scripts for reasoning quality, stability analysis, and correctness correlations.
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

3) Evaluation

- The `eval/` folder contains the evaluation and analysis code used in the paper.
- Prepare the required inputs and run the relevant scripts from `eval/` as needed.
