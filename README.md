# Li-ion Battery RUL Thesis Repository

This repository contains the source code, lightweight data artifacts, selected
models, and Streamlit demo for a University thesis on Remaining Useful Life
(RUL) prediction for Li-ion batteries.

## Final Project Direction
The project is organized as a reproducible data science workflow:

1. analyze the NASA Li-ion battery aging dataset;
2. build cycle-level features and modeling scenarios;
3. compare direct RUL models against SOH/capacity prediction;
4. derive practical RUL from SOH thresholds;
5. present the workflow in a Streamlit demo with live CSV inference.

The main thesis benchmark is `clean_benchmark`; `all_eligible` is kept as a stricter stress benchmark, and `nasa_classic_4` is kept for comparison with public NASA battery RUL repositories.

## Application
- `apps/streamlit_app.py`: guided Streamlit thesis demo over saved predictions, metrics, battery curves, and selected public SOH models. It opens by stating the final thesis framing: RUL derived from SOH is the main approach, while direct RUL is kept as a comparative benchmark. The app also includes a `Predicție CSV` tab where a model-ready CSV can be uploaded for live SOH/RUL prediction.

Run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run apps/streamlit_app.py
```

The public repository tracks lightweight CSV/JSON artifacts, a B0007 CSV upload
example, and selected SOH model binaries required by the demo and tests. Raw
datasets, the full cleaned source data, local-only generated models, virtual
environments, generated figures, and IDE state are intentionally excluded.

## Reproducibility

For immediate verification after cloning:

```bash
python scripts/validate_repo_integrity.py
python -m unittest discover -s tests -v
streamlit run apps/streamlit_app.py
```

For live inference, open the `Predicție CSV` tab and upload:

```text
data/examples/b0007_soh_inference_example.csv
```

For full regeneration from the public Kaggle source dataset, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The Kaggle dataset reference is:

```text
patrickfleith/nasa-battery-dataset
```

## Source Code
- `src/config.py`: project paths, scenario labels, and default SOH/EOL thresholds.
- `src/artifacts.py`: reusable loaders for saved CSV, JSON, and joblib artifacts.
- `src/inference.py`: public CSV inference helpers for selected SOH models.
- `src/prediction.py`: helpers for scenario/model/battery filtering and current-cycle summaries.
- `src/reporting.py`: model metric summaries and scenario descriptions.

## Notebooks
- `notebooks/01_batteries_5_6_7_18_eda.ipynb`: EDA, battery quality analysis, SOH/capacity curves, and thesis figures.
- `notebooks/02_baseline_group_split_rul_models.ipynb`: direct RUL baselines with SVR and tree models.
- `notebooks/03_cnn_lstm_rul_predictions.ipynb`: LSTM and CNN-LSTM sequence experiments.
- `notebooks/04_soh_capacity_prediction.ipynb`: SOH/capacity prediction and derived RUL analysis.
- `notebooks/05_sequence_soh_prediction.ipynb`: LSTM and CNN-LSTM sequence comparison for SOH prediction.

## Thesis Notes
- The public repository keeps the thesis-facing overview in this README and
  in the folder READMEs. Personal planning notes and local project memory files
  are not part of the submitted source package.

## Lightweight Automation
- Local integrity command: `python scripts/validate_repo_integrity.py`
- Local tests: `python -m unittest discover -s tests -v`
- CI uses `VALIDATE_DATASETS=0` so cloud checks validate repo wiring/docs without requiring local datasets.
- CI workflow: `.github/workflows/repo-integrity.yml` (runs on push and pull request)

## Git Hygiene
- `.gitignore` excludes local artifacts, editor state, heavy dataset contents,
  and non-public trained model binaries. Only selected SOH models needed by the
  public Streamlit demo are tracked.
- `.gitattributes` enforces consistent line endings and marks binary/data artifacts appropriately.

## Dataset Tracking
- `data/processed/` contains the final model-ready feature table and feature
  column definitions used by the experiments.
- `data/splits/` contains the battery split/scenario definitions used by the
  Streamlit demo, tests, and training scripts.
- `data/examples/` contains a B0007 model-ready CSV example for the app upload
  workflow.
- Raw and cleaned NASA source datasets are local-only. The cleaned source copy
  used for regeneration should live at `data/source/cleaned_nasa/`. It is not
  tracked in git because it is large and should be obtained from Kaggle before
  regenerating the processed feature table.

## Local Dependencies
Install runtime dependencies into the project virtual environment:

```bash
.venv/bin/pip install -r requirements.txt
```
