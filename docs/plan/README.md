# Planning Hub

This folder is the central place for thesis execution planning and progress tracking.

## Files
- `thesis-roadmap.md`: high-level milestones and deliverables
- `weekly-log-template.md`: weekly execution log template
- `decision-log.md`: important technical decisions and rationale

## GitHub Issues tracking (execution)
We track day-to-day execution in GitHub Issues so progress is visible and auditable.

- Milestone: `Thesis - End of May`
- Labels:
  - `thesis`
  - `phase:data`, `phase:model`, `phase:app`, `phase:writing`, `phase:research`, `phase:review`

The Romanian CSV plan at repo root remains a *planning input*, but issues are the *execution tracker*.

### Create issues from the CSV
This repo includes a lightweight converter (no pandas dependency):

```bash
python3 scripts/plan_csv_to_github_issues.py --create
```

If you only want to preview what would be created:

```bash
python3 scripts/plan_csv_to_github_issues.py --dry-run
```

## Usage rule
When a new sprint/phase starts, update `thesis-roadmap.md` first, then log execution details in weekly logs and close the relevant GitHub issues.
