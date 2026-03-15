#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = "Plan Licență_ Termen 31 Mai - Plan Licență_ Termen 31 Mai.csv"
DEFAULT_MILESTONE = "Thesis - End of May"


def _norm_bool(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"1", "true", "yes", "y", "gata"}


def split_numbered_actions(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    # Normalize common formats like: "1. ...2. ...3. ..." into separate lines
    t = re.sub(r"\s*(\d+\.)\s*", r"\n\1 ", t).strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    items: list[str] = []
    for ln in lines:
        ln = re.sub(r"^\d+\.\s*", "", ln).strip()
        if ln:
            items.append(ln)
    return items


def infer_labels(stage: str) -> list[str]:
    s = (stage or "").lower()
    labels = ["thesis"]

    if "baseline" in s or "date" in s or "preproces" in s:
        labels.append("phase:data")
    if "cnn" in s or "lstm" in s or "model" in s:
        labels.append("phase:model")
    if "streamlit" in s or "interfa" in s or "aplica" in s:
        labels.append("phase:app")
    if "scris" in s or "finalizare" in s or "formatare" in s or "introduc" in s or "concluzi" in s:
        labels.append("phase:writing")
    if "bibliograf" in s or "articole" in s:
        labels.append("phase:research")
    if "revizie" in s or "recite" in s:
        labels.append("phase:review")

    # De-dup while preserving order
    out: list[str] = []
    for lb in labels:
        if lb not in out:
            out.append(lb)
    return out


def issue_title(period: str, stage: str) -> str:
    p = (period or "").strip()
    st = (stage or "").strip()
    return f"[Thesis] {p} — {st}".strip()


def issue_body(period: str, stage: str, actions_text: str) -> str:
    items = split_numbered_actions(actions_text)
    checklist = "\n".join([f"- [ ] {it}" for it in items]) if items else "- [ ] (fill in)"

    return (
        "## Period\n"
        f"{(period or '').strip()}\n\n"
        "## Stage / Chapter\n"
        f"{(stage or '').strip()}\n\n"
        "## Actions\n"
        f"{checklist}\n\n"
        "## Done when\n"
        "- [ ] All action items above completed\n"
        "- [ ] Weekly progress logged in `docs/plan/`\n"
        "- [ ] Repo integrity passed (`python3 scripts/validate_repo_integrity.py`)\n"
    )


def gh_json(cmd: list[str]) -> object:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "gh command failed")
    return json.loads(proc.stdout or "null")


def gh_run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert thesis plan CSV to GitHub issues (optionally create them via gh).")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to the plan CSV")
    ap.add_argument("--milestone", default=DEFAULT_MILESTONE, help="Milestone title to assign")
    ap.add_argument("--state", choices=["open", "all"], default="all", help="When checking duplicates")
    ap.add_argument("--create", action="store_true", help="Create issues via `gh issue create`")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be created")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    existing_titles: set[str] = set()
    if args.create:
        # Cache existing issue titles to avoid duplicates
        data = gh_json(["gh", "issue", "list", "--state", args.state, "--limit", "500", "--json", "title"])  # type: ignore[list-item]
        existing_titles = {it.get("title", "") for it in (data or []) if isinstance(it, dict)}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            period = (row.get("PERIOADĂ") or "").strip()
            stage = (row.get("ETAPA (CAPITOLUL)") or "").strip()
            actions = (row.get("ACȚIUNI CONCRETE") or "").strip()
            done = _norm_bool(row.get("STATUS (GATA?)") or "")
            if done:
                continue

            title = issue_title(period, stage)
            body = issue_body(period, stage, actions)
            labels = infer_labels(stage)

            if args.dry_run or not args.create:
                print("\n---")
                print("title:", title)
                print("milestone:", args.milestone)
                print("labels:", ",".join(labels))
                print(body)
                continue

            if title in existing_titles:
                print(f"SKIP (exists): {title}")
                continue

            print(f"CREATE: {title}")
            gh_run(
                [
                    "gh",
                    "issue",
                    "create",
                    "--title",
                    title,
                    "--body",
                    body,
                    "--milestone",
                    args.milestone,
                    "--label",
                    ",".join(labels),
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
