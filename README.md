# Li-ion Battery RUL Thesis Repository

This repository contains data and code scaffolding for a University thesis on Remaining Useful Life (RUL) prediction for Li-ion batteries.

## Governance and Professional Docs
- `LICENSE`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `CODE_OF_CONDUCT.md`

## Notebooks
- `notebooks/01_batteries_5_6_7_18_eda.ipynb`: step-by-step EDA and visualization notebook for batteries `B0005`, `B0006`, `B0007`, `B0018`.

## AI Assistant Documentation Convention
To keep instructions consistent across assistants:

- Canonical rules live in `docs/ai/RULES.md`.
- Root adapter files (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`) are intentionally thin and point to the canonical file.
- Codex project MCP configuration lives in `.codex/config.toml`.
- AI collaboration memory files:
  - `docs/ai/PREFERENCES.md`
  - `docs/ai/LESSONS_LEARNED.md`
- When workflow, structure, or data inventory changes, update `docs/ai/RULES.md` first, then related adapter/README files if needed.

## Lightweight Automation
- Local integrity command: `python3 scripts/validate_repo_integrity.py`
- CI uses `VALIDATE_DATASETS=0` so cloud checks validate repo wiring/docs without requiring local datasets.
- CI workflow: `.github/workflows/repo-integrity.yml` (runs on push and pull request)
- PR auto-assign workflow: `.github/workflows/auto-assign-pr-author.yml` (assigns PR opener automatically)
- Task template: `.github/ISSUE_TEMPLATE/project-task.yml`
- PR checklist template: `.github/pull_request_template.md`

## Git Hygiene
- `.gitignore` excludes local artifacts and heavy dataset contents.
- `.gitattributes` enforces consistent line endings and marks binary/data artifacts appropriately.

## Planning
- `docs/plan/README.md`
- `docs/plan/thesis-roadmap.md`
- `docs/plan/weekly-log-template.md`
- `docs/plan/decision-log.md`

## Dataset Tracking
- `data_test/DATASET_MANIFEST.csv` is the lightweight dataset inventory (source/location/scope/license notes).
