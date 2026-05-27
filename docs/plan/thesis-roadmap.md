# Thesis Roadmap

## Goal
Deliver a reproducible and well-documented Li-ion RUL thesis project by end of May.

## Phases
1. Data understanding and preprocessing baseline
2. Baseline model + first quantitative results
3. Main model development (CNN-LSTM / sequence approach)
4. Evaluation, ablations, and thesis figures/tables
5. Streamlit demo and reusable project modules
6. Writing final chapters and polishing submission package

## Current Technical Direction
- Main dataset: cleaned NASA 5-56 dataset under `data_test/`.
- Raw datasets: keep for provenance and validation, not as the first implementation path.
- Evaluation: battery-level holdout is the primary strategy.
- Models: compare classical direct-RUL baselines against LSTM/CNN-LSTM under the same split; use SOH/capacity prediction as the strongest practical path for the demo.
- Demo: Streamlit workflow around saved predictions, metrics, SOH curves, EOL thresholds, and selected-cycle summaries.
- Project memory: maintain Obsidian-compatible notes in `docs/obsidian/`.
- Reusable code: keep artifact loading, prediction filtering, and reporting helpers in `src/`.

## Definition of done (per phase)
- Code/Notebook artifacts are reproducible
- Key assumptions are documented
- Results are saved and traceable
- Documentation is updated (`README`, `docs/ai/RULES.md`, relevant plan logs)
- Streamlit app renders from saved artifacts without re-running training
- Unit tests pass with `python -m unittest discover -s tests -v`

## Memory Files
- Research, repositories, article categories, and modeling directions: `docs/plan/research-synthesis.md`
- Important decisions and rationale: `docs/plan/decision-log.md`
- Execution roadmap: `docs/plan/thesis-roadmap.md`
