# Data Test Directory

This directory contains experimental, raw, and cleaned datasets used during the exploration and preprocessing phases of the Li-ion RUL thesis.

## 📁 Subdirectories

- **Raw Datasets/**:
  - `Nasa website datasets/`: Original zip files directly from the NASA PCoE website.
  - `Datasets raw 5-56/`: Unzipped versions of NASA datasets 5 through 56.
  - `Mat datasets 5-18/`: Kaggle version of NASA datasets 5 through 18 (currently under `results/` with `.mat` files and related artifacts).

- **Cleaned Datasets/**:
  - `Datasets 5-56 cleaned/`: Result of preprocessing NASA datasets 5-56.
    - `cleaned_dataset/`: Contains:
      - `metadata.csv` (the index)
      - `data/` (individual CSVs for each cycle)
      - `extra_infos/` (README files from source groups for traceability)

## 📊 Preprocessing Information
Cleaned data follows a standardized schema:
- **Index:** `metadata.csv` (type, start_time, ambient_temperature, battery_id, test_id, uid, filename, Capacity, Re, Rct).
- **Time-series:** `data/*.csv` (Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Time).
