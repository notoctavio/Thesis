# Li-ion Battery RUL Thesis Repository

This repository contains data and code scaffolding for a University thesis on Remaining Useful Life (RUL) prediction for Li-ion batteries.

## Notebooks
- `notebooks/01_batteries_5_6_7_18_eda.ipynb`: step-by-step EDA and visualization notebook for batteries `B0005`, `B0006`, `B0007`, `B0018`.

## AI Assistant Documentation Convention
To keep instructions consistent across assistants:

- Canonical rules live in `docs/ai/RULES.md`.
- Root adapter files (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`) are intentionally thin and point to the canonical file.
- Codex project MCP configuration lives in `.codex/config.toml`.
- When workflow, structure, or data inventory changes, update `docs/ai/RULES.md` first, then related adapter/README files if needed.

## Lightweight Automation
- Local integrity command: `python3 scripts/validate_repo_integrity.py`
- CI workflow: `.github/workflows/repo-integrity.yml` (runs on push and pull request)
- Task template: `.github/ISSUE_TEMPLATE/project-task.yml`

## Git Hygiene
- `.gitignore` excludes local artifacts and heavy dataset contents.
- `.gitattributes` enforces consistent line endings and marks binary/data artifacts appropriately.
