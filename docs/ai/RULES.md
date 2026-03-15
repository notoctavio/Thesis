# AI Rules (Canonical)

This file is the **single source of truth** for AI assistant behavior in this repository.

Root adapter files (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`) should stay minimal and point here.

## Project Context
- Thesis topic: Li-ion battery **Remaining Useful Life (RUL)** prediction.
- Current state: data prepared, code scaffolded.
- Main active data path: `data_test/`.

## Repository Reality (must be respected)
- `data/`: placeholder for finalized model-ready artifacts.
- `data_test/Raw Datasets/`:
  - NASA website zip archives
  - extracted `Datasets raw 5-56` `.mat` data
  - Kaggle `Mat datasets 5-18` subset
- `data_test/Cleaned Datasets/Datasets 5-56 cleaned/cleaned_dataset/`:
  - `metadata.csv`
  - `data/*.csv`
  - `extra_infos/`
- `src/`: package skeleton with empty module folders at this stage.

## Working Rules
1. Use `metadata.csv` as index/source-of-truth for cycle file mapping.
2. Validate assumptions against actual folder contents before coding.
3. Keep work reproducible and thesis-grade (clear logic, traceable steps, explicit assumptions).
4. Prefer Python 3.x with `pandas`, `numpy`, `scipy`; use `pytorch` for deep learning unless changed explicitly.

## Documentation Maintenance Rule
When repository structure, workflow, or data inventory changes, update:
- `docs/ai/RULES.md` first
- Root adapters (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`) only if needed
- Relevant folder `README.md` files (`data/README.md`, `data_test/README.md`)

## Automation Baseline
- Integrity script: `scripts/validate_repo_integrity.py`
- CI workflow: `.github/workflows/repo-integrity.yml`
- Issue template: `.github/ISSUE_TEMPLATE/project-task.yml`

## Skills Focus (for assistants)
- Dataset integrity checks (raw vs cleaned consistency)
- Reproducible preprocessing and feature engineering
- Clear experiment organization for thesis reporting
