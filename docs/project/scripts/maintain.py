#!/usr/bin/env python3
"""
Run maintenance utilities with consistent, minimal output:
- project_structure.py
- project_analytics.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import contextmanager

ROOT_MARKERS = (
    ".git", "pyproject.toml", "package.json", "requirements.txt",
    "setup.cfg", "setup.py", "package-lock.json",
)


def find_project_root(start: Path) -> Path:
    """Prefer a git repo root; otherwise the first parent containing any ROOT_MARKERS."""
    cur = start.resolve()
    p = cur
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    p = cur
    while p != p.parent:
        if any((p / m).exists() for m in ROOT_MARKERS if m != ".git"):
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not determine project root above {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DOCS_DIR = SCRIPT_DIR.parent
REPO_ROOT = find_project_root(SCRIPT_DIR)


@contextmanager
def temp_argv(argv):
    orig = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = orig


def run_project_structure():
    sys.path.insert(0, str(SCRIPT_DIR))
    import project_structure  # type: ignore

    with temp_argv([str(project_structure.__file__), "--outdir", str(PROJECT_DOCS_DIR)]):
        project_structure.main()
    return f"structure: wrote {PROJECT_DOCS_DIR / 'project_structure.txt'}"


def run_project_analytics():
    sys.path.insert(0, str(SCRIPT_DIR))
    import project_analytics  # type: ignore

    os.environ["PROJECT_ANALYTICS_QUIET"] = "1"
    with temp_argv([str(project_analytics.__file__)]):
        project_analytics.main()
    return f"analytics: wrote {PROJECT_DOCS_DIR / 'project_analytics.txt'}"


def main():
    results = []
    results.append(run_project_structure())
    results.append(run_project_analytics())
    for line in results:
        print(line)


if __name__ == "__main__":
    main()
