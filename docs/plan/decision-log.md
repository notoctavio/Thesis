# Decision Log

Use this file for important thesis engineering decisions.

## Template
### [YYYY-MM-DD] Decision title
- Context:
- Options considered:
- Decision:
- Why:
- Consequences:
- Follow-up actions:

### [2026-05-17] Use cleaned NASA 5-56 dataset as the main modeling dataset
- Context: The project contains raw NASA archives, extracted `.mat` files, a Kaggle-style MATLAB subset, and a cleaned 5-56 CSV dataset under `data_test/`.
- Options considered: rebuild preprocessing from raw `.mat` files; use only the B0005/B0006/B0007/B0018 MATLAB subset; use the cleaned 5-56 dataset as the main source and keep raw files for provenance.
- Decision: Use the cleaned 5-56 dataset as the main modeling dataset.
- Why: It is broad enough for thesis experiments, easier to audit, easier to connect to Streamlit, and keeps the project focused on RUL prediction rather than raw MATLAB conversion.
- Consequences: Raw files remain important for provenance and validation, but the first stable pipeline should load `metadata.csv` and cycle CSV files from the cleaned dataset.
- Follow-up actions: Document raw-to-cleaned provenance, validate cleaned dataset assumptions, and keep all feature extraction traceable to `metadata.csv`.

### [2026-05-17] Use battery-level holdout as the primary evaluation strategy
- Context: RUL prediction can look artificially strong if train and test rows come from the same battery.
- Options considered: random row split; chronological row split after concatenation; battery-level holdout.
- Decision: Use battery-level holdout as the primary evaluation strategy.
- Why: It better measures whether a model generalizes to unseen batteries and is easier to defend in the thesis.
- Consequences: Metrics may be lower than random-split metrics, but they are more credible.
- Follow-up actions: Record train/validation/test battery IDs in every experiment output.

### [2026-05-17] Compare classical baselines with a sequence model
- Context: Existing notebooks already contain baseline ML experiments and a CNN-LSTM experiment.
- Options considered: only use classical ML; only use LSTM/CNN-LSTM; compare both families under a fair split.
- Decision: Compare classical baselines with a sequence model, then select the final model based on reproducible holdout performance and thesis clarity.
- Why: Baselines make the deep learning result meaningful, while the sequence model aligns well with battery degradation over cycles.
- Consequences: The final thesis can explain both simple and advanced approaches without becoming too complex.
- Follow-up actions: Re-run baseline and sequence models using the same dataset subset, target definition, and battery-level split before final comparison.

### [2026-05-17] Store research and planning memory in repo documentation
- Context: Important findings from local notebooks, articles, and GitHub repositories should not live only in chat history.
- Options considered: keep notes in conversation; create external notes; store project memory in versioned repo docs.
- Decision: Store project memory in `docs/plan/research-synthesis.md`, `docs/plan/decision-log.md`, and future implementation plans under `docs/superpowers/plans/`.
- Why: Repo docs are versioned, editable, easy for future assistants to inspect, and aligned with `docs/ai/RULES.md`.
- Consequences: Future changes to direction should update these files in the same PR as related code/workflow changes.
- Follow-up actions: Update `research-synthesis.md` whenever new repositories, papers, dataset facts, or modeling decisions become important.

### [2026-05-17] Use two modeling scenarios: strict benchmark and clean benchmark
- Context: The first all-battery split produced poor-looking prediction curves because several test batteries had short histories, weak degradation, invalid initial capacity, or strong SOH spikes.
- Options considered: keep one all-battery benchmark only; manually cherry-pick batteries for good graphs; define transparent quality criteria and report two scenarios.
- Decision: Use `all_eligible` as a strict stress/generalization benchmark and `clean_benchmark` as the main thesis benchmark.
- Why: This keeps the analysis honest while producing cleaner, more interpretable results for the thesis and Streamlit demo.
- Consequences: Results must always state which scenario was used. Comparisons between models are valid only within the same scenario and split.
- Follow-up actions: Use `clean_benchmark` for primary demo visuals, but report `all_eligible` to discuss limitations and dataset heterogeneity.

### [2026-05-17] Add classic NASA benchmark and SOH prediction path
- Context: Direct RUL prediction on the extended dataset produced moderate R2 values, while public repositories often report much stronger results on smaller NASA subsets or by predicting smoother health indicators.
- Options considered: add Transformer immediately; tune only LSTM/CNN-LSTM; add a classic NASA benchmark and predict SOH/capacity before deriving RUL.
- Decision: Add `nasa_classic_4` and a separate SOH/capacity prediction notebook.
- Why: This gives comparison with common public repositories while keeping the broader `clean_benchmark` and `all_eligible` evaluations honest. SOH prediction is smoother, more physical, and produced R2 above 0.98 on clean/classic scenarios.
- Consequences: The thesis should separate direct RUL prediction from SOH/capacity prediction. High scores from `nasa_classic_4` must be labeled as classic-subset results, not full-dataset generalization.
- Follow-up actions: Use SOH/capacity prediction for the strongest demo visuals, then derive RUL to an EOL threshold as the practical output.

### [2026-05-24] Build final project around saved artifacts and a Streamlit demo
- Context: The notebooks now generate feature tables, metrics, predictions, figures, and trained model files. The final thesis project needs a clean app and reusable code without forcing the Streamlit UI to re-run model training.
- Options considered: keep all logic in notebooks; rebuild a full training pipeline before the demo; build the demo over saved artifacts and refactor reusable loading/reporting logic into `src/`.
- Decision: Use saved artifacts as the interface between notebooks and the Streamlit demo, with helper modules in `src/` for artifact loading, prediction filtering, and result summaries.
- Why: This keeps the project reproducible, fast to demo, easier to explain in the thesis, and safer than retraining models inside the UI.
- Consequences: The first demo explores test-set predictions and selected-cycle summaries. Future work can add live inference for uploaded battery cycles or newly engineered feature rows.
- Follow-up actions: Keep `artifacts/` regenerated by notebooks, keep `apps/streamlit_app.py` lightweight, and update tests when artifact schemas change.

### [2026-05-24] Use robust v2 capacity features and add SOH sequence models
- Context: Some RUL/SOH plots looked wrong because startup capacity anomalies made selected batteries appear below EOL at cycle 1. SOH also lacked LSTM/CNN-LSTM comparisons.
- Options considered: ignore bad-looking plots as dataset noise; manually hide weak predictions; correct preprocessing and add balanced model comparisons.
- Decision: Create `battery_cycle_features_v2.csv`, treat first-cycle startup capacity anomalies before calculating SOH, remove future backfill from SOH lag features, and add SOH LSTM/CNN-LSTM artifacts.
- Why: This fixes a real preprocessing issue while keeping evaluation honest. It also makes the model comparison symmetrical across direct RUL and SOH formulations.
- Consequences: Direct RUL remains a benchmark target based on remaining recorded cycles; SOH-derived RUL is the preferred practical output. Sequence SOH models are reported, but tree models remain stronger on the current split.
- Follow-up actions: Use v2 artifacts for app and final thesis results; label direct RUL carefully in the thesis.

### [2026-05-24] Reframe the Streamlit app as a guided thesis demo
- Context: The first Streamlit version worked technically, but it exposed too many experiment controls before explaining what the commission was looking at.
- Options considered: keep the technical dashboard and add captions; build a full product-style app; reframe the existing app as a guided thesis demo.
- Decision: Reframe the app around a guided SOH-derived RUL story, with `nasa_classic_4` as the default presentation path and advanced controls moved to a separate tab. The guided demo defaults to an 80% practical SOH threshold for B0007 so the derived RUL is visible in the recorded curve; the original NASA classic 70% threshold remains documented and selectable.
- Why: This makes the demo easier to understand in a bachelor-thesis defense while preserving the full experimental detail for questions.
- Consequences: The first app tab explains data -> SOH model -> EOL threshold -> RUL. Model comparisons, methodology, and advanced scenario/model selection remain available but are no longer the first screen.
- Follow-up actions: Use the guided tab for presentation; use the results and advanced tabs for technical discussion.
