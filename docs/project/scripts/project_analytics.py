#!/usr/bin/env python3
"""
project_analytics.py

Locate the project root by finding package-lock.json, then:

1. Load ignore rules from the project's .gitignore (plus always ignore .git/).
2. Walk the tree while respecting those rules.
   - Count total folders, files, and lines across all non-ignored files.
   - Break down file counts and line counts by programming language / file type.
3. Display a progress bar as it processes each filesystem entry.
4. Write a single report to:
   <docs/project>/project_analytics.txt

"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Set

try:
    from pathspec import PathSpec
except ImportError:
    print("ERROR: Please install pathspec (`pip install pathspec`).")
    sys.exit(1)

# -------------------------------------------------------------------------------
# Console logging only
# -------------------------------------------------------------------------------
logger = logging.getLogger("project_analytics")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# Files/dirs that can mark a project root when walking upward (besides .git/)
ROOT_MARKERS = (
    ".git", "pyproject.toml", "package.json", "requirements.txt",
    "setup.cfg", "setup.py", "package-lock.json",
)

# Map file extensions to language / category names.
# Updated to reflect a typical mixed codebase.
# Important notes:
# - .js/.ts cover config files like vite.config.ts or firebase config.
# - .env, .env.local are config files we still want counted.
# - .png etc. are static assets; they still count as files, but lines=0.
LANGUAGE_MAP: Dict[str, str] = {
    # Code / app logic
    ".js": "JavaScript",
    ".jsx": "JavaScript/JSX",
    ".ts": "TypeScript",
    ".tsx": "TypeScript/TSX",
    ".cjs": "JavaScript",
    ".mjs": "JavaScript",
    ".json": "JSON",
    ".html": "HTML",
    ".css": "CSS",
    ".py": "Python",
    ".pyc": "Python/Bytecode",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",

    # Config / infra / build
    ".env": "Env",
    ".local": "Env",
    ".space-delivery-877a7": "Env",
    ".ini": "INI",
    ".cfg": "Config",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".gitignore": "GitIgnore",
    ".lock": "Lockfile",

    # Docs / text
    ".md": "Markdown",
    ".txt": "Text",
    ".log": "Log",
    ".tex": "LaTeX",
    ".docx": "Asset/Doc",
    ".pdf": "Asset/PDF",
    ".xls": "Asset/Excel",
    ".xlsx": "Asset/Excel",

    # Data-ish
    ".csv": "CSV",
    ".tsv": "TSV",
    ".sql": "SQL",
    ".xml": "XML",

    # Assets
    ".png": "Asset/Image",
    ".jpg": "Asset/Image",
    ".jpeg": "Asset/Image",
    ".svg": "Asset/Image",
    ".rules": "Firestore Rules",

    # Other languages we might still have around
    ".java": "Java",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".ps1": "PowerShell",
    ".psm1": "PowerShell",
    ".bat": "Batch",
    ".dockerfile": "Dockerfile",
    ".docker": "Dockerfile",
    ".toml": "TOML",
}


def find_project_root(start: Path) -> Path:
    """
    Prefer a git repo root; otherwise the first parent containing any ROOT_MARKERS.
    """
    current = start.resolve()
    p = current
    while p != p.parent:
        if (p / ".git").exists():
            logger.debug(f"Found git project root at {p}")
            return p
        p = p.parent
    p = current
    while p != p.parent:
        if any((p / m).exists() for m in ROOT_MARKERS if m != ".git"):
            logger.debug(f"Found project root at {p}")
            return p
        p = p.parent
    logger.error("Project root markers not found; cannot locate project root.")
    sys.exit(1)


def load_ignore_patterns(root: Path) -> PathSpec:
    """
    Read .gitignore under `root`, including commented lines.

    For each non-blank line:
        - If it begins with '#', strip that '#' and any following spaces -> pattern.
        - Otherwise, strip inline comments after an unescaped '#'.

    If a pattern ends with '/', we:
        (1) strip the trailing slash and add that as a pattern (to ignore the directory itself)
        (2) add pattern+'/**' so that everything under that directory is also ignored.

    We also always add ".git" and ".git/**", so the .git folder is never counted.
    """
    gitignore_path = root / ".gitignore"
    raw_patterns: List[str] = []

    if gitignore_path.exists():
        for raw in gitignore_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if candidate:
                    raw_patterns.append(candidate)
            else:
                if "#" in raw:
                    candidate = raw.split("#", 1)[0].rstrip()
                else:
                    candidate = raw.rstrip()
                if candidate:
                    raw_patterns.append(candidate)

    # Always ignore the .git folder and its contents, even if not in .gitignore
    raw_patterns.append(".git")
    raw_patterns.append(".git/**")

    # Transform "dirname/" -> ["dirname", "dirname/**"]
    final_patterns: List[str] = []
    for pat in raw_patterns:
        if pat.endswith("/"):
            stripped = pat.rstrip("/")
            final_patterns.append(stripped)            # ignore the directory itself
            final_patterns.append(stripped + "/**")    # ignore everything under it
        else:
            final_patterns.append(pat)

    return PathSpec.from_lines("gitwildmatch", final_patterns)


def collect_non_ignored(root: Path, ignore_spec: PathSpec) -> List[Path]:
    """
    Recursively collect all non-ignored entries under `root`, including `root` itself,
    pruning entire directories whose relative path matches ignore_spec.
    Returns a list of Paths (both files and directories).
    """
    non_ignored: List[Path] = []

    # Include the root directory itself if not ignored
    rel_root = Path(".")  # relative path of root to itself
    if not ignore_spec.match_file(str(rel_root)):
        non_ignored.append(root)

    def recurse(path: Path):
        rel = path.relative_to(root)
        # If this path (file or directory) matches ignore patterns, skip it
        # and DON'T recurse into it.
        if ignore_spec.match_file(str(rel)):
            return

        non_ignored.append(path)
        if path.is_dir():
            for child in path.iterdir():
                recurse(child)

    for child in root.iterdir():
        recurse(child)

    return non_ignored


def analyze_tree(
    root: Path,
    ignore_spec: PathSpec,
    quiet: bool = False,
) -> Tuple[int, int, int, Dict[str, Tuple[int, int]], Set[str]]:
    """
    Recursively walk `root`, pruning ignored subtrees at the highest level,
    to count:
        - total_dirs: number of directories (including the root itself)
        - total_files: number of files
        - total_lines: sum of all lines across files
        - lang_stats: mapping language/category -> (file_count, line_count)

    Displays a simple progress bar over the number of non-ignored entries.
    """
    non_ignored = collect_non_ignored(root, ignore_spec)
    total_process = len(non_ignored)
    other_exts: Set[str] = set()

    # Edge case: empty tree
    if total_process == 0:
        return 1, 0, 0, {}, set()

    bar_length = 40
    total_dirs = 0
    total_files = 0
    total_lines = 0
    lang_stats: Dict[str, Tuple[int, int]] = {}

    if not quiet:
        logger.info("Processing non-ignored entries...")
    next_update = 0.0
    for idx, path in enumerate(non_ignored):
        percent = (idx + 1) / total_process
        if not quiet and percent >= next_update:
            filled = int(bar_length * percent)
            bar = "#" * filled + " " * (bar_length - filled)
            print(
                f"\rProcessing:     [{bar}] {percent * 100:6.1f}%",
                end="",
                flush=True,
            )
            next_update += 0.001  # ~every 0.1%

        if path.is_dir():
            total_dirs += 1
        else:
            total_files += 1
            ext = path.suffix.lower()  # e.g. ".jsx"
            language = LANGUAGE_MAP.get(ext, "Other")
            if language == "Other" and ext:
                other_exts.add(ext)

            # Count lines only for text-like files; for binary (e.g. png), this
            # will still 'work' but result in 1 line. We'll guard with a try.
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    count_lines = sum(1 for _ in f)
            except Exception:
                # Binary/unreadable -> treat as 0 logical lines of code/text
                count_lines = 0

            total_lines += count_lines
            prev_files, prev_lines = lang_stats.get(language, (0, 0))
            lang_stats[language] = (prev_files + 1, prev_lines + count_lines)

    # final update at 100%
    if not quiet:
        bar = "#" * bar_length
        print(f"\rProcessing:     [{bar}] 100.0%", flush=True)

    return total_dirs, total_files, total_lines, lang_stats, other_exts


def format_section_header(title: str) -> str:
    """
    Return a formatted section header.
    """
    return f"{'=' * 5} {title} {'=' * 5}\n\n"


def format_report_block(
    total_dirs: int,
    total_files: int,
    total_lines: int,
    lang_stats: Dict[str, Tuple[int, int]],
    other_exts: Set[str] | None = None,
) -> str:
    """
    Build the report block given the metrics.
    Numbers are formatted with thousand separators.
    Align the "Lines:" label in each language line to a fixed column.
    """
    lines = [
        f"Total directories: {total_dirs:,}",
        f"Total files      : {total_files:,}",
        f"Total lines      : {total_lines:,}",
        "",
        "By language / type:",
    ]

    # For alignment of file counts
    formatted_counts = [f"{fcount:,}" for fcount, _ in lang_stats.values()]
    max_fcount_width = max((len(s) for s in formatted_counts), default=0)

    for lang, (fcount, lcount) in sorted(lang_stats.items()):
        fcount_str = f"{fcount:,}".rjust(max_fcount_width)
        line = f"  {lang:<16} Files: {fcount_str}  Lines: {lcount:,}"
        lines.append(line)

    if other_exts:
        lines.append("")
        lines.append("Extensions grouped into Other:")
        for ext in sorted(other_exts):
            lines.append(f"  {ext}")

    return "\n".join(lines) + "\n\n"


def main() -> None:
    """
    Steps:
    - locate project root
    - load .gitignore
    - analyze tree using those rules
    - write one report file in the same directory as this script
    - print summary to console
    """
    script_dir = Path(__file__).parent.resolve()
    project_root = find_project_root(script_dir)
    quiet = os.environ.get("PROJECT_ANALYTICS_QUIET", "0") == "1"
    if quiet:
        logger.setLevel(logging.ERROR)
    logger.info(f"Project root detected: {project_root}")

    logger.info("Loading .gitignore rules...")
    real_ignore_spec = load_ignore_patterns(project_root)

    logger.info("Analyzing project tree (respecting .gitignore)...")
    dirs_wi, files_wi, lines_wi, stats_wi, other_exts = analyze_tree(project_root, real_ignore_spec, quiet=quiet)
    logger.info(
        f"[With .gitignore]  {dirs_wi:,} dirs, {files_wi:,} files, {lines_wi:,} lines"
    )

    output_file = script_dir.parent / "project_analytics.txt"
    try:
        with output_file.open("w", encoding="utf-8") as out:
            out.write(format_section_header("ANALYSIS (With .gitignore)"))
            out.write(format_report_block(dirs_wi, files_wi, lines_wi, stats_wi, other_exts))

        logger.info(f"\nWrote analytics report to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to write report to '{output_file}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
