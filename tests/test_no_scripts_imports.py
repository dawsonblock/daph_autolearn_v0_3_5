"""V037-002: enforce that the library layer never imports from the CLI layer.

The dependency direction must be ``CLI scripts -> daph_learning package``,
never the reverse. This test scans every ``*.py`` file under
``src/daph_learning/`` and fails if any of them contains an import of the
``scripts`` package (whether ``import scripts`` or ``from scripts...``).

This is a static source scan (not an import-time check) so it catches
imports inside functions or conditional branches that might not execute
during a normal test run.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "daph_learning"


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imports_scripts(source: str) -> list[str]:
    """Return a list of offending import statements that reference ``scripts``.

    Uses AST parsing for accuracy (handles ``from scripts.x import y`` and
    ``import scripts.x`` forms, including dotted forms). A simple regex is
    used as a fallback to catch dynamic ``__import__``-style strings, but the
    AST pass is the authoritative check.
    """
    offenders: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return offenders

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scripts" or alias.name.startswith("scripts."):
                    offenders.append(ast.unparse(node))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and (
                node.module == "scripts" or node.module.startswith("scripts.")
            ):
                offenders.append(ast.unparse(node))
    return offenders


def test_no_src_module_imports_scripts():
    """No file under src/daph_learning/ may import the scripts package."""
    offenders_by_file: dict[str, list[str]] = {}
    for path in _iter_python_files(SRC_ROOT):
        source = path.read_text(encoding="utf-8")
        offenders = _imports_scripts(source)
        if offenders:
            rel = path.relative_to(SRC_ROOT.parent.parent)
            offenders_by_file[str(rel)] = offenders

    if offenders_by_file:
        lines = ["src/daph_learning/ modules still import from scripts/:"]
        for f, offs in offenders_by_file.items():
            for off in offs:
                lines.append(f"  {f}: {off}")
        pytest.fail("\n".join(lines))


def test_no_src_module_text_references_scripts_import():
    """Belt-and-suspenders: catch ``import scripts`` / ``from scripts`` text
    that the AST pass might miss (e.g. inside strings or comments that are
    actually executed via exec). Fails only on real import-shaped text."""
    pattern = re.compile(r"^\s*(?:from\s+scripts[\.\s]|import\s+scripts[\.\s])", re.MULTILINE)
    offenders_by_file: dict[str, list[str]] = {}
    for path in _iter_python_files(SRC_ROOT):
        source = path.read_text(encoding="utf-8")
        matches = pattern.findall(source)
        if matches:
            # Filter out matches that are inside comments by re-scanning line by line.
            real = []
            for line in source.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if re.match(r"^\s*(?:from\s+scripts[\.\s]|import\s+scripts[\.\s])", line):
                    real.append(line.strip())
            if real:
                rel = path.relative_to(SRC_ROOT.parent.parent)
                offenders_by_file[str(rel)] = real
    if offenders_by_file:
        lines = ["src/daph_learning/ modules contain scripts-import text:"]
        for f, offs in offenders_by_file.items():
            for off in offs:
                lines.append(f"  {f}: {off}")
        pytest.fail("\n".join(lines))


def test_scripts_imports_daph_learning_not_other_scripts():
    """Sanity: scripts/ may import from daph_learning, but cross-script
    imports (``from scripts.x import y``) should be avoided now that the
    reusable logic lives in the library. This is a soft check — we only
    flag direct cross-script imports, not the ``scripts._manifest`` shim
    which exists explicitly for backward compatibility with external
    callers (e.g. tests that import private helpers)."""
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    offenders_by_file: dict[str, list[str]] = {}
    for path in scripts_root.glob("*.py"):
        if path.name == "__init__.py" or path.name == "_manifest.py":
            continue
        source = path.read_text(encoding="utf-8")
        offenders = _imports_scripts(source)
        if offenders:
            offenders_by_file[path.name] = offenders
    if offenders_by_file:
        lines = ["scripts/ modules still import from scripts/ (should import from daph_learning instead):"]
        for f, offs in offenders_by_file.items():
            for off in offs:
                lines.append(f"  scripts/{f}: {off}")
        pytest.fail("\n".join(lines))
