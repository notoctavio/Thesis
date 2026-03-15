# Contributing

Thanks for contributing to this thesis repository.

## Workflow
- Track work as GitHub Issues (milestone `Thesis - End of May`) and use the dashboard: https://github.com/users/notoctavio/projects/2
- Create or use a feature branch (do not push directly to `main`).
- Open a pull request with a clear summary and validation notes.
- Ensure required GitHub checks pass before merge.
- Log weekly progress in `docs/plan/` and record major decisions in `docs/plan/decision-log.md`.

## Required checks before PR
- Run local integrity checks:
  - `python3 scripts/validate_repo_integrity.py`
- If working without local datasets in CI parity mode:
  - `VALIDATE_DATASETS=0 python3 scripts/validate_repo_integrity.py`

## Documentation policy
When changing structure, process, or data inventory:
- Update `docs/ai/RULES.md`
- Update relevant adapter files if needed (`AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, `CODEX.md`)
- Update relevant `README.md` files
- Update `docs/ai/PREFERENCES.md` or `docs/ai/LESSONS_LEARNED.md` when new recurring guidance is discovered

## Notebook guidance
- Keep exploratory work in `notebooks/`.
- Promote stable logic from notebooks into scripts/modules for reproducibility.
- Save thesis-ready figures under `docs/figures/` (create folder when needed).
