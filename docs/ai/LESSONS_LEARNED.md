# AI Lessons Learned (Mistake Prevention)

Use this file to capture recurring pitfalls and their prevention rules.

## Current lessons
1. CI cannot assume local datasets are present.
   - Prevention: Run CI integrity with `VALIDATE_DATASETS=0`; run full checks locally.
2. Keep large/raw datasets out of git.
   - Prevention: Use `.gitignore` allowlist strategy and keep only dataset READMEs/manifests in version control.
3. Rename branches carefully when PRs are open.
   - Prevention: Prefer creating a new PR head branch or be ready to recreate PR after rename.

## Add new lesson format
- Date:
- Problem:
- Root cause:
- Prevention rule:
- Where enforced (script/workflow/doc):
