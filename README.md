# Li-ion Battery RUL Thesis Repository

This repository contains data and code scaffolding for a University thesis on Remaining Useful Life (RUL) prediction for Li-ion batteries.

## Final Project Direction
The project is organized as a reproducible data science workflow:

1. analyze the NASA Li-ion battery aging dataset;
2. build cycle-level features and modeling scenarios;
3. compare direct RUL models against SOH/capacity prediction;
4. derive practical RUL from SOH thresholds;
5. present the workflow in a Streamlit demo.

The main thesis benchmark is `clean_benchmark`; `all_eligible` is kept as a stricter stress benchmark, and `nasa_classic_4` is kept for comparison with public NASA battery RUL repositories.

## Application
- `apps/streamlit_app.py`: guided Streamlit thesis demo over saved predictions, metrics, and battery curves. It opens by stating the final thesis framing: SOH-derived RUL is the main approach, while direct RUL is kept as a comparative benchmark. The first tab presents the SOH -> EOL threshold -> derived RUL story, then keeps model comparisons, methodology, and advanced exploration in separate tabs.

Run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run apps/streamlit_app.py
```

The public repository tracks only lightweight CSV/JSON artifacts required by
the demo and tests, so the Streamlit app can run after cloning and installing
dependencies. Raw datasets, cleaned source data, trained model binaries,
virtual environments, generated figures, and local IDE state are intentionally
excluded.

## Source Code
- `src/config.py`: project paths, scenario labels, and default SOH/EOL thresholds.
- `src/artifacts.py`: reusable loaders for saved CSV, JSON, and joblib artifacts.
- `src/prediction.py`: helpers for scenario/model/battery filtering and current-cycle summaries.
- `src/reporting.py`: model metric summaries and scenario descriptions.

## Notebooks
- `notebooks/01_batteries_5_6_7_18_eda.ipynb`: EDA, battery quality analysis, SOH/capacity curves, and thesis figures.
- `notebooks/02_baseline_group_split_rul_models.ipynb`: direct RUL baselines with SVR and tree models.
- `notebooks/03_cnn_lstm_rul_predictions.ipynb`: LSTM and CNN-LSTM sequence experiments.
- `notebooks/04_soh_capacity_prediction.ipynb`: SOH/capacity prediction and derived RUL analysis.
- `notebooks/05_sequence_soh_prediction.ipynb`: LSTM and CNN-LSTM sequence comparison for SOH prediction.

## Thesis Notes
- `docs/obsidian/`: notes for the final thesis direction, dataset decisions, EOL thresholds, experiment design, results, and demo overview.

## Lightweight Automation
- Local integrity command: `python3 scripts/validate_repo_integrity.py`
- Local tests: `.venv/bin/python -m unittest discover -s tests -v`
- CI uses `VALIDATE_DATASETS=0` so cloud checks validate repo wiring/docs without requiring local datasets.
- CI workflow: `.github/workflows/repo-integrity.yml` (runs on push and pull request)

## Git Hygiene
- `.gitignore` excludes local artifacts, editor state, trained model binaries, and heavy dataset contents.
- `.gitattributes` enforces consistent line endings and marks binary/data artifacts appropriately.

## Dataset Tracking
- `data_test/DATASET_MANIFEST.csv` is the lightweight dataset inventory (source/location/scope/license notes).
- Raw and cleaned datasets are local-only. To regenerate the full experiment
  artifacts, place the NASA data under the documented `data_test/` structure
  and rerun the notebooks/scripts.

## Local Dependencies
Install runtime dependencies into the project virtual environment:

```bash
.venv/bin/pip install -r requirements.txt
```
