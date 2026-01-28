#!/usr/bin/env python3
"""
project_structure.py

Generate a directory-tree report (ASCII .txt only), honoring .gitignore
(top-level and nested). No JSON, no logs, no cache paths.

Output defaults to the parent directory of this script (docs/project).

Usage:
    ./project_structure.py [--root /path/to/project]
                          [--outdir /dir/for/output]
                          [--top-level-gitignore-only]
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pathspec  # pip install pathspec

# Folders to always skip by name (in addition to .gitignore rules)
DEFAULT_FOLDER_IGNORES = {
    ".git", "node_modules", ".venv", "venv", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".idea", ".vscode",
    ".firebase", "coverage",
}

# Files/dirs that can mark a project root when walking upward (besides .git/)
ROOT_MARKERS = (
    ".git", "pyproject.toml", "package.json", "requirements.txt",
    "setup.cfg", "setup.py", "package-lock.json",
)


def find_project_root(start: Path) -> Path:
    """Prefer a git repo root; otherwise the first parent containing any ROOT_MARKERS."""
    cur = start.resolve()
    # Try git first
    p = cur
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    # Fallback to other markers
    p = cur
    while p != p.parent:
        if any((p / m).exists() for m in ROOT_MARKERS if m != ".git"):
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not determine project root above {start}")


def _read_gitignore_file(gi_path: Path, rel_prefix: str) -> List[str]:
    """
    Read a .gitignore file and return patterns adjusted to be relative to the
    project root by prefixing with `rel_prefix` (POSIX-style).
    Expand 'dir/' into both 'dir' and 'dir/**'.
    """
    pats: List[str] = []
    for raw in gi_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        def pref(p: str) -> str:
            p = p.lstrip("/")  # anchor to root of repo, then prefix with folder
            return f"{rel_prefix}/{p}" if rel_prefix else p

        if line.endswith("/"):
            d = line.rstrip("/")
            pd = pref(d)
            pats.append(pd)
            pats.append(f"{pd}/**")
        else:
            pats.append(pref(line))
    return pats


def load_gitignore_patterns(root: Path, top_level_only: bool) -> pathspec.PathSpec:
    """Build a gitwildmatch PathSpec from .gitignore files."""
    all_patterns: List[str] = []
    top = root / ".gitignore"
    if top.exists():
        all_patterns.extend(_read_gitignore_file(top, rel_prefix=""))
    if not top_level_only:
        for gi in root.rglob(".gitignore"):
            if gi == top:
                continue
            rel_dir = gi.parent.relative_to(root).as_posix()
            all_patterns.extend(_read_gitignore_file(gi, rel_prefix=rel_dir))
    return pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)


def build_tree(root_dir: Path, ignore_spec: pathspec.PathSpec) -> Dict[str, Any]:
    """Return nested dict similar to `tree -J`, filtered by ignore rules."""
    def node_for(p: Path) -> Optional[Dict[str, Any]]:
        rel = p.relative_to(root_dir).as_posix()
        if p.name in DEFAULT_FOLDER_IGNORES:
            return None
        if rel and ignore_spec.match_file(rel):
            return None

        entry: Dict[str, Any] = {"name": p.name}
        if p.is_dir():
            entry["type"] = "directory"
            children: List[Dict[str, Any]] = []
            for c in sorted(p.iterdir(), key=lambda q: (not q.is_dir(), q.name.lower())):
                child = node_for(c)
                if child is not None:
                    children.append(child)
            entry["contents"] = children
        else:
            entry["type"] = "file"
        return entry

    return node_for(root_dir) or {"name": root_dir.name, "type": "directory", "contents": []}


def count_files_dirs(node: Dict[str, Any]) -> Tuple[int, int]:
    files = 0
    dirs = 0
    def rec(n: Dict[str, Any]):
        nonlocal files, dirs
        if n.get("type") == "directory":
            dirs += 1
            for ch in n.get("contents", []):
                rec(ch)
        elif n.get("type") == "file":
            files += 1
    rec(node)
    return files, dirs


def render_ascii(node: Dict[str, Any], prefix: str = "") -> List[str]:
    lines: List[str] = []
    if node["type"] == "directory":
        lines.append(f"{prefix}{node['name']}/")
        kids = node.get("contents", [])
        for i, ch in enumerate(kids):
            last = (i == len(kids) - 1)
            branch = "`-- " if last else "|-- "
            ext = "    " if last else "|   "
            if ch["type"] == "directory":
                lines.append(f"{prefix}{branch}{ch['name']}/")
                # Skip the header of the recursive call (first line)
                lines.extend(render_ascii(ch, prefix + ext)[1:])
            else:
                lines.append(f"{prefix}{branch}{ch['name']}")
    else:
        lines.append(f"{prefix}{node['name']}")
    return lines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Write a .txt directory tree that respects .gitignore."
    )
    ap.add_argument("--root", type=Path, default=None,
                    help="Project root. If omitted, detect via git or common markers.")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory. Defaults to the script's folder.")
    ap.add_argument("--top-level-gitignore-only", action="store_true",
                    help="Honor only the top-level .gitignore.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = args.root.resolve() if args.root else find_project_root(script_dir)

    ignore_spec = load_gitignore_patterns(project_root, top_level_only=args.top_level_gitignore_only)

    tree = build_tree(project_root, ignore_spec)
    files, dirs = count_files_dirs(tree)

    lines = render_ascii(tree)
    lines += ["", f"Total directories: {dirs}, Total files: {files}"]

    out_dir = args.outdir.resolve() if args.outdir else script_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "project_structure.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
