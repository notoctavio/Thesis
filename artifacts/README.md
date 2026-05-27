# Lightweight Demo Artifacts

This directory contains the small generated CSV/JSON artifacts needed to run the
public Streamlit thesis demo and local tests.

Tracked:

- `features/battery_cycle_features_v2.csv`
- `features/*_columns_v2.json`
- `metrics/*.csv` and `metrics/*.json`
- `predictions/*.csv`
- `splits/*.json`
- `tables/*.csv`

Not tracked:

- trained model files under `models/`
- generated figures under `figures/`
- raw or cleaned source datasets under `data_test/`

The full artifacts can be regenerated locally from the notebooks and scripts
after placing the NASA dataset in the documented `data_test/` structure.
