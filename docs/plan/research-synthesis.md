# Research Synthesis: Li-ion Battery RUL Prediction

This document captures research findings, repository analysis, dataset decisions, and working directions for the thesis. It should be updated whenever a new source, model result, or project decision becomes important.

## Current Thesis Direction

- Topic: Remaining Useful Life (RUL) prediction for Li-ion batteries.
- Main dataset: NASA battery aging dataset, using the cleaned 5-56 version from `data_test/`.
- Main goal: build a reproducible data science workflow from dataset understanding and preprocessing to model evaluation and a Streamlit demo.
- Recommended modeling strategy: compare classical direct-RUL baselines against LSTM/CNN-LSTM, then use SOH/capacity prediction with derived RUL as the strongest practical demo path.
- Final project structure: notebooks generate artifacts; `src/` loads and summarizes them; `apps/streamlit_app.py` presents scenario/model/battery predictions interactively.

## Dataset Decision

### Preferred Dataset

Use:

`data_test/Cleaned Datasets/Datasets 5-56 cleaned/cleaned_dataset/`

Important contents:

- `metadata.csv`: source of truth for cycle metadata and file mapping.
- `data/*.csv`: cycle-level CSV files.
- `extra_infos/`: supporting dataset information.

Reasons:

- It covers 34 batteries, from `B0005` through `B0056` groups.
- It already converts the MATLAB/NASA structure into CSV files.
- It keeps the thesis focused on RUL prediction rather than spending too much effort rebuilding a raw `.mat` parser.
- It is easier to audit, explain, and use in a Streamlit demo.

### Raw Datasets

Use raw `.mat`/NASA archives only for:

- explaining original dataset provenance;
- validating cleaned data assumptions;
- documenting the transformation path from NASA raw data to cleaned CSV data.

Do not make raw `.mat` processing the main implementation path unless the cleaned dataset proves inconsistent.

## Local Notebook Findings

### Current Notebook Workflow

The three active notebooks now follow a single reproducible workflow:

- `notebooks/01_batteries_5_6_7_18_eda.ipynb`: EDA across all useful cleaned NASA 5-56 batteries, with robust capacity/SOH summaries, modeling-quality labels, and thesis figures.
- `notebooks/02_baseline_group_split_rul_models.ipynb`: cycle-level feature extraction, battery-level train/validation/test splits, SVR/RandomForest/HistGradientBoosting baselines, and saved artifacts for two scenarios.
- `notebooks/03_cnn_lstm_rul_predictions.ipynb`: LSTM and CNN-LSTM sequence models using the same feature table and scenario splits as the baseline notebook.

Generated artifacts live under `artifacts/`:

- `artifacts/features/battery_cycle_features_v2.csv`
- `artifacts/splits/battery_split_v1.json`
- `artifacts/splits/modeling_scenarios_v1.json`
- `artifacts/metrics/`
- `artifacts/predictions/`
- `artifacts/models/`
- `artifacts/figures/`

Current modeling scenarios:

- `all_eligible`: strict benchmark using all batteries with at least 20 usable discharge cycles.
- `clean_benchmark`: main thesis benchmark using batteries with enough cycles and coherent capacity/SOH behavior.
- `nasa_classic_4`: classic NASA benchmark using B0005/B0006/B0007/B0018, useful for comparison with many public repositories.

`clean_benchmark` criteria:

- at least 60 useful discharge cycles;
- initial capacity at least 0.5 Ah;
- maximum SOH at most 1.20 after robust capacity cleaning;
- final SOH at most 1.05;
- minimum SOH at most 0.98, so there is visible degradation.

### Baseline Models Notebook

Notebook:

`notebooks/02_baseline_group_split_rul_models.ipynb`

Observed direction:

- Classical models: SVR, RandomForest, HistGradientBoosting.
- Evaluation uses battery-level/group split, which is thesis-grade because it avoids row-level leakage.
- Best baseline observed so far: SVR with RBF kernel.
- Holdout performance is more important than random validation performance.

Use this as the baseline reference for the final project.

Previous single-split verified run:

- Feature rows: 2746
- Batteries used after filtering: 33
- Split:
  - train: 22 batteries, 2084 rows
  - validation: 5 batteries, 362 rows
  - test: 6 batteries, 300 rows
- Best validation model: RandomForest
- Best test RMSE in the first verified run: HistGradientBoosting, RMSE 23.06 cycles, MAE 17.38 cycles, R2 0.123
- Naive train/validation mean baseline on test: RMSE 41.09 cycles, MAE 37.73 cycles

Interpretation: the tabular baseline is strong enough to be a serious thesis benchmark. Model selection should still be based on validation, while test metrics should be reported as final evaluation.

Current two-scenario verified run:

- `all_eligible`:
  - best baseline test result: HistGradientBoosting, RMSE 20.60 cycles, MAE 14.45 cycles, R2 0.494;
  - naive train/validation mean baseline: RMSE 39.36 cycles, MAE 35.44 cycles.
- `clean_benchmark`:
  - best baseline test result: HistGradientBoosting, RMSE 22.56 cycles, MAE 15.88 cycles, R2 0.808;
  - RandomForest: RMSE 24.19 cycles, MAE 16.32 cycles, R2 0.780;
  - SVR_RBF: RMSE 25.37 cycles, MAE 17.14 cycles, R2 0.758;
  - naive train/validation mean baseline: RMSE 52.91 cycles, MAE 43.17 cycles.

Interpretation: `clean_benchmark` gives more thesis-friendly curves and R2 because it removes batteries with short histories, weak degradation, invalid initial capacity, or strong SOH spikes. `all_eligible` remains useful as a stress/generalization benchmark.

Current verified run after robust v2 capacity cleaning:

- `all_eligible`:
  - best baseline test result: ExtraTrees, RMSE 18.48 cycles, MAE 11.58 cycles, R2 0.593.
- `clean_benchmark`:
  - best baseline test result: ExtraTrees, RMSE 22.26 cycles, MAE 13.60 cycles, R2 0.814;
  - HistGradientBoosting remains close: RMSE 23.44 cycles, MAE 16.42 cycles, R2 0.793.
- `nasa_classic_4`:
  - split: train B0005/B0006, validation B0018, test B0007;
  - best diagnostic test result: HistGradientBoosting, RMSE 3.05 cycles, MAE 2.11 cycles, R2 0.996;
  - RandomForest: RMSE 3.69 cycles, MAE 2.34 cycles, R2 0.994;
  - ExtraTrees: RMSE 5.56 cycles, MAE 4.67 cycles, R2 0.987.

Interpretation: the classic NASA setup explains why many public repositories report very high scores: the subset is smaller and more homogeneous. This is still useful, but should be clearly labeled as a classic benchmark, not as performance over the full heterogeneous 5-56 dataset.

### CNN-LSTM Notebook

Notebook:

`notebooks/03_cnn_lstm_rul_predictions.ipynb`

Observed direction:

- Uses sequence windows over engineered battery features.
- CNN-LSTM gave strong results on its current subset/split.
- Current results should not be directly compared to baseline metrics until both are evaluated using the same dataset subset, target definition, and battery-level split.

Use this as the main candidate model, but repeat evaluation under a fair experiment setup.

Previous verified run using the first baseline split:

- Sequence length: 20 cycles
- Features: 31
- Train sequences: 2084
- Validation sequences: 362
- Test sequences: 300
- LSTM test metrics: RMSE 25.51 cycles, MAE 19.97 cycles, R2 -0.073
- CNN-LSTM test metrics: RMSE 29.14 cycles, MAE 18.53 cycles, R2 -0.400
- Naive train mean baseline on test: RMSE 40.99 cycles, MAE 37.64 cycles

Interpretation: in the first fair run, the LSTM is better than CNN-LSTM, but HistGradientBoosting remains the best global test model. This is useful for the thesis because it shows that well-engineered cycle-level features can outperform a more complex sequence model on this split.

Current two-scenario verified run:

- `all_eligible`:
  - LSTM: RMSE 30.46 cycles, MAE 22.01 cycles, R2 -0.107;
  - CNN-LSTM: RMSE 30.95 cycles, MAE 21.49 cycles, R2 -0.143;
  - sequence models underperform tabular baselines on the strict heterogeneous split.
- `clean_benchmark`:
  - CNN-LSTM: RMSE 23.58 cycles, MAE 17.83 cycles, R2 0.791;
  - LSTM: RMSE 25.36 cycles, MAE 18.13 cycles, R2 0.758;
  - CNN-LSTM becomes competitive with the best tabular model on the cleaner benchmark.

Interpretation: the sequence models are sensitive to battery quality and protocol heterogeneity. On `clean_benchmark`, CNN-LSTM gives good and defensible results, but HistGradientBoosting remains the best model by RMSE in the current run.

Current verified run after robust v2 capacity cleaning:

- `all_eligible`: CNN-LSTM improves to RMSE 20.78 cycles, MAE 14.44 cycles, R2 0.485, but ExtraTrees remains best.
- `clean_benchmark`: LSTM is competitive, RMSE 23.75 cycles, MAE 17.43 cycles, R2 0.788, but ExtraTrees remains best.
- `nasa_classic_4`: CNN-LSTM improves to RMSE 24.27 cycles, R2 0.750, but tabular baselines remain far stronger.

Interpretation: for this dataset and current feature setup, LSTM/CNN-LSTM are useful as deep-learning comparisons, but the strongest direct-RUL results come from tree-based tabular models.

### SOH / Capacity Prediction Notebook

Notebook:

`notebooks/04_soh_capacity_prediction.ipynb`

Purpose:

- Predict SOH/capacity ratio instead of direct RUL.
- Exclude current-cycle capacity/SOH features from inputs to avoid direct target leakage.
- Use lagged SOH features from previous cycles plus cycle signal features.
- Derive RUL to an 80% EOL threshold as an exploratory practical output.

Current verified test results:

- `all_eligible`:
  - best SOH result: HistGradientBoosting, RMSE 0.0989 SOH, MAE 0.0406 SOH, R2 0.780.
- `clean_benchmark`:
  - best SOH result after v2 cleaning: RandomForest, RMSE 0.0098 SOH, MAE 0.0057 SOH, R2 0.982;
  - GradientBoosting is very close, R2 0.981;
  - ExtraTrees: R2 0.974.
- `nasa_classic_4`:
  - best SOH result after v2 cleaning: RandomForest, RMSE 0.0064 SOH, MAE 0.0049 SOH, R2 0.994;
  - ExtraTrees: R2 0.993;
  - GradientBoosting: R2 0.992.

Interpretation: SOH/capacity prediction gives the high-quality results expected from the literature because SOH is smoother and more physically meaningful than direct RUL. This should likely become the strongest result section and the basis for the Streamlit prediction demo, with direct RUL retained as a complementary experiment.

### SOH Sequence Models

Notebook:

`notebooks/05_sequence_soh_prediction.ipynb`

Script:

`scripts/train_sequence_soh.py`

Current verified result:

- LSTM/CNN-LSTM have now been added for SOH for a fair comparison.
- The first absolute-SOH formulation made CNN-LSTM unstable on `clean_benchmark`.
- The current sequence formulation predicts `soh_delta_target = soh - prev_soh`, then reconstructs SOH as `prev_soh + pred_delta_soh`.
- `all_eligible`: LSTM RMSE 0.0994 SOH, R2 0.777; CNN-LSTM RMSE 0.1009 SOH, R2 0.770.
- `clean_benchmark`: CNN-LSTM RMSE 0.0100 SOH, R2 0.981; LSTM RMSE 0.0100 SOH, R2 0.981. Both are now competitive with RandomForest.
- `nasa_classic_4`: LSTM RMSE 0.0199 SOH, R2 0.946; CNN-LSTM remains unstable on the very small split, RMSE 0.0865 SOH and R2 -0.028.

Interpretation: residual SOH forecasting is a better fit for sequence models because battery health changes gradually. The thesis should still report this as one-step SOH forecasting from previous health state, not as a fully autonomous long-horizon deployment model.

## External Repository Analysis

### Directly Relevant Battery Repositories

#### XiuzeZhou/RUL

Link: https://github.com/XiuzeZhou/RUL

What it contains:

- Transformer and AttMoE approaches for Li-ion battery RUL.
- NASA and CALCE experiments.
- Associated papers:
  - Transformer network for Li-ion battery RUL prediction.
  - AttMoE: Attention with Mixture of Experts for Li-ion battery RUL prediction.

How to use it:

- Strong literature reference.
- Useful for advanced-model discussion and future work.
- Not recommended as the first final implementation because Transformer/AttMoE would add avoidable complexity.

#### ARYANRAJ1121/battery-diagnostics-NASA

Link: https://github.com/ARYANRAJ1121/battery-diagnostics-NASA

What it contains:

- End-to-end NASA battery pipeline.
- Feature engineering from discharge cycles.
- SoC, SoH, and RUL models.
- RUL LSTM with reported RMSE around 12.49 cycles.

How to use it:

- Best practical reference for final project structure.
- Useful for ideas around feature extraction, result saving, plots, and a complete workflow.

Caveats:

- Some public pipelines risk leakage if scaling is done before train/test split.
- Target definition and split strategy must be checked carefully before comparing results.

### Battery Repositories With Different Dataset Focus

#### MichaelBosello/battery-rul-estimation

Link: https://github.com/MichaelBosello/battery-rul-estimation

What it contains:

- RUL estimation using autoencoders, LSTMs, and CNNs.
- NASA randomized battery usage dataset and UNIBO powertools dataset.
- Clear data processing abstractions.
- Train-only normalization patterns and sequence-generation utilities.

How to use it:

- Strong engineering reference for clean project architecture.
- Useful for separating preprocessing, model data handling, normalization, sequence generation, and experiments.

Caveat:

- Dataset is not the same as the NASA 5-56 aging dataset used in this thesis.

#### OorjaD/iTransformer-based-RUL-prediction-in-Li-ion-batteries

Link: https://github.com/OorjaD/iTransformer-based-RUL-prediction-in-Li-ion-batteries

How to use it:

- Literature/future-work reference for Transformer-style models on battery RUL.
- Not recommended for the first stable thesis implementation.

### Generic RUL / C-MAPSS Repositories

#### dlaredo/NASA_RUL_-CMAPS-

Link: https://github.com/dlaredo/NASA_RUL_-CMAPS-

Use:

- General RUL concept reference only.

Caveat:

- It is about turbofan engines, not Li-ion batteries.

#### ritu-thombre99/RUL-Prediction

Link: https://github.com/ritu-thombre99/RUL-Prediction

Use:

- General sequence-based RUL framing.

Caveat:

- It is also C-MAPSS/turbofan-focused, not battery-focused.

#### jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction

Link: https://github.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction

Use:

- Simple PyTorch LSTM training loop reference.

Caveats:

- Dataset is C-MAPSS, not battery data.
- Some implementation choices are not ideal for a thesis-grade pipeline, including possible scaling inconsistency and hidden-state detachment inside the model.

## Article Directions

Use the local `Aricole` folder for literature support.

Recommended grouping:

- Survey/background:
  - battery degradation and SOH/RUL overview papers;
  - Li-ion aging mechanisms;
  - review papers on machine learning for battery prognostics.
- Classical machine learning:
  - SVR-based RUL prediction;
  - Random Forest / XGBoost approaches;
  - hybrid filtering methods.
- Deep learning:
  - LSTM using voltage/current/temperature profiles;
  - partial charge/discharge LSTM approaches;
  - CNN-LSTM hybrids.
- Advanced/future work:
  - Transformer;
  - attention;
  - mixture of experts;
  - Kalman/particle filtering hybrids;
  - CEEMDAN/KAN-style methods.

## Recommended Final Project Shape

### Reproducible Pipeline

1. Load cleaned metadata and cycle CSV files.
2. Validate dataset integrity.
3. Extract cycle-level features from discharge cycles.
4. Build target variables:
   - RUL in remaining cycles;
   - optionally SOH from capacity divided by nominal/initial capacity.
5. Split by battery, not by random rows.
6. Fit preprocessing only on training batteries.
7. Train baseline models.
8. Train sequence model.
9. Evaluate all models on the same holdout batteries.
10. Save metrics, predictions, plots, and model artifacts.
11. Build Streamlit demo around saved artifacts.

### Recommended Model Set

- Baselines:
  - SVR with RBF kernel;
  - RandomForestRegressor;
  - HistGradientBoostingRegressor or XGBoost if dependency is added intentionally.
- Main sequence model:
  - CNN-LSTM if stable and reproducible;
  - LSTM fallback if CNN-LSTM is too complex or inconsistent.
- Discussion/future work:
  - Transformer/AttMoE.

## Methodological Rules

- Do not compare results from different repositories unless dataset, target definition, and split strategy are equivalent.
- Prefer battery-level holdout over random row splits.
- Fit scalers/imputers only on training data.
- Record battery IDs used in train/validation/test.
- Save metrics and predictions in files, not only in notebook output.
- Keep raw data provenance documented even if the implementation uses cleaned data.

## Streamlit Demo Direction

The demo should show the final workflow, not retrain models interactively.

Current guided-demo direction:

- `Demo ghidat`: default presentation path using `nasa_classic_4`, SOH-derived RUL, B0007 when available, and an 80% practical SOH threshold so RUL is visible in the available curve. The NASA classic 70% threshold remains documented and selectable.
- `Rezultate modele`: comparison tables for SOH/capacity prediction and direct RUL benchmark.
- `Metodologie`: dataset, feature engineering, battery-level split, target definitions, metrics, and limitations.
- `Explorare avansata`: technical controls for switching task, scenario, model, battery, threshold, and inspecting filtered prediction rows.

The first screen should explain data -> SOH prediction -> EOL threshold -> derived RUL. Direct RUL remains available as a benchmark, but the practical thesis story is SOH-derived RUL.

## Current Recommendation

Use the cleaned NASA 5-56 dataset as the main dataset, build a fair baseline-vs-CNN-LSTM comparison using battery-level holdout, and package the final result as a reproducible thesis project plus Streamlit demo.
