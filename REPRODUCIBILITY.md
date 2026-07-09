# Full Reproducibility Guide

This guide explains how to run the public thesis demo immediately and how to
rebuild the experiment artifacts from the public Kaggle source dataset.

## 1. Clone And Install

```bash
git clone https://github.com/notoctavio/Thesis.git
cd Thesis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project pins the main runtime packages because the public `.joblib` models
are tied to the scikit-learn/joblib versions used during training.

## 2. Run The Streamlit Demo

```bash
streamlit run apps/streamlit_app.py
```

The app loads public CSV/JSON artifacts, saved predictions, and selected
published SOH models. It does not train models at runtime.

Main app flow:

1. `Demo ghidat`: thesis story, SOH prediction, EOL threshold, derived RUL.
2. `Predicție CSV`: live inference on a model-ready CSV.
3. `Rezultate modele`: model comparison tables and metrics.
4. `Metodologie`: dataset, splits, scenarios, and modeling rationale.
5. `Explorare avansată`: technical scenario/model/battery exploration.

## 3. Run Live CSV Prediction

Open the `Predicție CSV` tab.

If no file is uploaded, the app automatically uses:

```text
data/examples/b0007_soh_inference_example.csv
```

That file is model-ready: it already contains cycle-level features used by the
SOH models. The app predicts `pred_soh`, derives an EOL cycle from the selected
SOH threshold, and exports the predictions as CSV.

Important limitation: the upload feature expects model-ready cycle-level CSV.
It does not accept raw NASA `.mat` files or raw per-cycle Kaggle CSV files.

## 4. Validate The Public Repository

```bash
python scripts/validate_repo_integrity.py
python -m unittest discover -s tests -v
```

These checks validate required files, public model artifacts, example CSV
inference, prediction helpers, and Streamlit narrative behavior.

## 5. Download The Full Source Dataset

The source dataset used for regeneration is public on Kaggle:

```text
patrickfleith/nasa-battery-dataset
```

Install and configure the Kaggle CLI with your Kaggle API token, then run:

```bash
kaggle datasets download -d patrickfleith/nasa-battery-dataset -p data/source --unzip
```

The repository expects the cleaned dataset at:

```text
data/source/cleaned_nasa/
  metadata.csv
  data/*.csv
  extra_infos/
```

If Kaggle extracts the folder as `data/source/cleaned_dataset/`, rename it:

```bash
mv data/source/cleaned_dataset data/source/cleaned_nasa
```

Then validate the local source data:

```bash
VALIDATE_SOURCE_DATA=1 python scripts/validate_repo_integrity.py
```

The source dataset is intentionally not committed because it is large and
contains thousands of CSV files.

## 6. Rebuild The Experiment Artifacts

Run the notebooks/scripts in this order:

```text
notebooks/01_batteries_5_6_7_18_eda.ipynb
notebooks/02_baseline_group_split_rul_models.ipynb
notebooks/03_cnn_lstm_rul_predictions.ipynb
notebooks/04_soh_capacity_prediction.ipynb
scripts/train_sequence_soh.py
notebooks/05_sequence_soh_prediction.ipynb
```

Expected outputs:

```text
data/processed/battery_cycle_features_v2.csv
data/processed/*_feature_columns_v2.json
data/splits/*.json
artifacts/metrics/*.csv
artifacts/metrics/*.json
artifacts/predictions/*.csv
artifacts/models/*.joblib
artifacts/models/*.pt
```

The public repository includes only the lightweight processed data, saved
metrics/predictions, and selected SOH models required by the app. Full local
training may create additional models that remain ignored by git.

## 7. Project Notes

- The thesis app demonstrates a reproducible workflow, not an industrial BMS.
- The main thesis output is RUL derived from predicted SOH and an EOL threshold.
- Direct RUL prediction remains a transparent benchmark.
- The `nasa_classic_4` scenario is useful for comparison with literature, but
  broader scenarios must be reported separately.
