#!/usr/bin/env python3
"""Execute code cells from a notebook in a lightweight local runner."""

from __future__ import annotations

import json
import os
import argparse
import sys
import tempfile
import traceback
from pathlib import Path


def display(*args, **kwargs) -> None:
    for obj in args:
        if hasattr(obj, "head"):
            print(obj.head().to_string(index=False))
        else:
            print(repr(obj))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="Notebook path to execute")
    parser.add_argument(
        "--cells",
        help="Comma-separated code cell indexes to execute. Omit to execute all code cells.",
    )
    args = parser.parse_args()

    cache_root = Path(tempfile.gettempdir()) / "licenta-rul-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    selected_cells = None
    if args.cells:
        selected_cells = {int(part.strip()) for part in args.cells.split(",") if part.strip()}

    notebook_path = Path(args.notebook)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": display}

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if selected_cells is not None and index not in selected_cells:
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"--- executing {notebook_path.name} cell {index} ---", flush=True)
        try:
            exec(compile(source, f"{notebook_path}:cell{index}", "exec"), namespace)
        except Exception:
            print(f"FAILED {notebook_path.name} cell {index}", flush=True)
            traceback.print_exc()
            return 1

    print(f"{notebook_path.name} OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
