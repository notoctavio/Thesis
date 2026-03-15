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

## Workflow (must follow)
1. Pick or create the next task as a GitHub Issue (label it `thesis` + `phase:*`, assign the milestone).
2. Move it on the Project board: https://github.com/users/notoctavio/projects/2
3. Work on a branch.
4. Open a PR; ensure required checks pass.
5. Merge the PR.
6. Reflect progress:
   - Weekly log: `docs/plan/*`
   - Decision log when needed: `docs/plan/decision-log.md`

## Documentation Maintenance Rule (agents must do this)
When you change repository structure, workflow, automation, or data inventory, you must proactively update the docs **in the same PR**.

Update order:
- `docs/ai/RULES.md` first (canonical)
- Root adapters (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`) only if the pointer/compatibility needs changes
- User-facing docs:
  - `README.md` (high-level workflow/navigation)
  - Relevant folder READMEs (`data/README.md`, `data_test/README.md`) when their contents/meaning changes
- Process logs:
  - `docs/plan/*` (weekly logs / roadmap updates / decision log)
- Collaboration memory:
  - `docs/ai/PREFERENCES.md` for stable preferences
  - `docs/ai/LESSONS_LEARNED.md` for recurring pitfalls

## Minimal doc update checklist (for every PR)
- If behavior/workflow changed: update `docs/ai/RULES.md` + `README.md`
- If you learned a repeatable preference: update `docs/ai/PREFERENCES.md`
- If you hit a recurring mistake: update `docs/ai/LESSONS_LEARNED.md`
- If work completed a milestone chunk: update `docs/plan/*` (weekly log/roadmap)

## Automation Baseline
- Integrity script: `scripts/validate_repo_integrity.py`
- CI workflow: `.github/workflows/repo-integrity.yml`
- Issue template: `.github/ISSUE_TEMPLATE/project-task.yml`
- Progress tracking: GitHub Issues (milestone `Thesis - End of May`) + narrative logs in `docs/plan/`

## Skills Focus (for assistants)
- Dataset integrity checks (raw vs cleaned consistency)
- Reproducible preprocessing and feature engineering
- Clear experiment organization for thesis reporting

## Continuous Improvement Loop
After each meaningful task:
1. Capture reusable user collaboration preferences in `docs/ai/PREFERENCES.md`.
2. Capture mistakes and prevention rules in `docs/ai/LESSONS_LEARNED.md`.
3. Reflect execution progress in `docs/plan/` files.
