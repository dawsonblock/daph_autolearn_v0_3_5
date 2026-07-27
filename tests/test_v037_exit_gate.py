"""v0.3.7 exit gate.

Verifies the four exit-gate criteria from ROADMAP §v0.3.7 exit gate:

1. All V037-* tests pass (checked by the rest of the suite; this test
   verifies the specific V037 test files exist and are non-empty).
2. ``pip install -e .`` works on a clean venv (checked by the fact that
   this test runs without ``PYTHONPATH=src`` — the package is installed).
3. No ``src/daph_learning/**`` module imports from ``scripts/`` (re-checks
   the V037-002 guard).
4. Version surfaces agree on ``0.3.7`` (re-checks the V037-004 guards).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "daph_learning"


# --- 1. V037 test files exist ---

V037_TEST_FILES = [
    "tests/test_route_fn_contract.py",          # V037-001
    "tests/test_no_scripts_imports.py",          # V037-002
    "tests/test_exception_fallbacks.py",         # V037-003
    "tests/test_version_claims_discipline.py",   # V037-004
    "tests/test_manifest_environment.py",        # V037-005
    "tests/test_cli_entrypoints.py",             # V037-006
]


@pytest.mark.parametrize("test_file", V037_TEST_FILES)
def test_v037_test_file_exists(test_file):
    path = REPO_ROOT / test_file
    assert path.exists(), f"V037 test file {test_file!r} does not exist"
    assert path.stat().st_size > 0, f"V037 test file {test_file!r} is empty"


# --- 2. Package is installed (no PYTHONPATH required) ---

def test_package_importable_without_pythonpath():
    """If this test runs, the package is installed (pip install -e .)."""
    import daph_learning
    assert daph_learning.__version__ == "0.3.7"


def test_cli_entry_points_on_path():
    """At least one daph-* entry point must be on PATH after install."""
    import shutil
    assert shutil.which("daph-evaluate-routes") is not None, (
        "daph-evaluate-routes is not on PATH; run pip install -e ."
    )


# --- 3. No src→scripts imports (re-check V037-002) ---

def test_no_src_imports_scripts():
    """Re-verify the V037-002 guard: no src/daph_learning/** module
    imports from scripts/."""
    import ast
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith("scripts."):
                        offenders.append(f"{path}: {ast.unparse(node)}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "scripts" or node.module.startswith("scripts.")):
                    offenders.append(f"{path}: {ast.unparse(node)}")
    assert not offenders, (
        "src/daph_learning/ modules still import from scripts/:\n  "
        + "\n  ".join(offenders)
    )


# --- 4. Version surfaces agree (re-check V037-004) ---

def test_version_surfaces_agree():
    """pyproject.toml, __init__.py, README.md, CLAIMS.md all say 0.3.7."""
    import daph_learning
    version = daph_learning.__version__

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert m and m.group(1) == version, (
        f"pyproject.toml version {m.group(1) if m else None!r} != {version!r}"
    )

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^#\s*DAPH AutoLearn v(\d+\.\d+\.\d+)", readme, re.MULTILINE)
    assert m and m.group(1) == version, (
        f"README.md header version {m.group(1) if m else None!r} != {version!r}"
    )

    claims = (REPO_ROOT / "CLAIMS.md").read_text(encoding="utf-8")
    m = re.search(r"^#\s*DAPH AutoLearn v(\d+\.\d+\.\d+)", claims, re.MULTILINE)
    assert m and m.group(1) == version, (
        f"CLAIMS.md header version {m.group(1) if m else None!r} != {version!r}"
    )


# --- Summary: all V037 modules exist ---

V037_SOURCE_MODULES = [
    "src/daph_learning/routing/policy.py",       # V037-001 (RouteDecision)
    "src/daph_learning/routing/errors.py",        # V037-003 (typed errors)
    "src/daph_learning/telemetry.py",             # V037-003 (telemetry)
    "src/daph_learning/environment.py",           # V037-005 (provenance)
    "src/daph_learning/cli/__init__.py",          # V037-006 (entry points)
]


@pytest.mark.parametrize("module_path", V037_SOURCE_MODULES)
def test_v037_source_module_exists(module_path):
    path = REPO_ROOT / module_path
    assert path.exists(), f"V037 source module {module_path!r} does not exist"
    assert path.stat().st_size > 0, f"V037 source module {module_path!r} is empty"
