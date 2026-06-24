# Lightweight Demo Artifacts

This directory contains generated outputs needed to run the public Streamlit
thesis demo, local tests, and selected CSV inference workflow.

Tracked:

- `metrics/*.csv` and `metrics/*.json`
- `predictions/*.csv`
- `tables/*.csv`
- `models/model_manifest.json`
- selected public SOH `.joblib` models used by the `Predicție CSV` tab

Not stored in `artifacts/`:

- processed/model-ready input data under `data/processed/`
- split/scenario definitions under `data/splits/`

Local-only:

- non-public trained model files under `models/`
- generated figures under `figures/`
- raw or cleaned source datasets

The full artifacts can be regenerated locally from the notebooks and scripts
after obtaining the NASA source dataset and rebuilding the processed feature
table under `data/processed/`.
