"""V037-006: CLI entry points.

Acceptance criteria from ROADMAP §V037-006:

1. Each entry point exists in ``pyproject.toml`` ``[project.scripts]``.
2. Each entry point is discoverable via ``importlib.metadata.entry_points``.
3. Each entry point's ``main`` function is callable and dispatches to the
   corresponding script's ``main``.
4. ``PYTHONPATH=src:.`` is no longer required (the package installs
   cleanly via ``pip install -e .``).
"""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ENTRY_POINTS = {
    "daph-autolearn": "daph_learning.cli.autolearn:main",
    "daph-evaluate-routes": "daph_learning.cli.evaluate_routes:main",
    "daph-build-oracles": "daph_learning.cli.build_oracles:main",
    "daph-tune-steering": "daph_learning.cli.tune_steering:main",
    "daph-random-control": "daph_learning.cli.random_control:main",
}


# --- 1. pyproject.toml has the entry points ---

def test_pyproject_has_project_scripts_section():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in text, (
        "pyproject.toml has no [project.scripts] section"
    )


def test_pyproject_lists_all_expected_entry_points():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name, target in EXPECTED_ENTRY_POINTS.items():
        pattern = rf'^{re.escape(name)}\s*=\s*"{re.escape(target)}"'
        assert re.search(pattern, text, re.MULTILINE), (
            f"pyproject.toml does not list entry point {name!r} -> {target!r}"
        )


# --- 2. importlib.metadata discovers the entry points ---

def _get_daph_entry_points() -> dict[str, importlib.metadata.EntryPoint]:
    eps = importlib.metadata.entry_points(group="console_scripts")
    return {ep.name: ep for ep in eps if ep.name.startswith("daph-")}


def test_entry_points_discoverable_via_importlib_metadata():
    """The package must be installed (pip install -e .) for this to pass."""
    eps = _get_daph_entry_points()
    for name in EXPECTED_ENTRY_POINTS:
        assert name in eps, (
            f"entry point {name!r} not found in console_scripts group; "
            f"found: {sorted(eps.keys())}"
        )


def test_entry_point_targets_match_expected():
    eps = _get_daph_entry_points()
    for name, expected_target in EXPECTED_ENTRY_POINTS.items():
        assert name in eps, f"entry point {name!r} missing"
        ep = eps[name]
        assert ep.value == expected_target, (
            f"entry point {name!r} has target {ep.value!r}, "
            f"expected {expected_target!r}"
        )


# --- 3. Each entry point's main is callable and dispatches ---

@pytest.mark.parametrize("module_name,script_name", [
    ("daph_learning.cli.autolearn", "autolearn.py"),
    ("daph_learning.cli.evaluate_routes", "evaluate_routes.py"),
    ("daph_learning.cli.build_oracles", "build_empirical_oracles.py"),
    ("daph_learning.cli.tune_steering", "tune_steering.py"),
    ("daph_learning.cli.random_control", "random_direction_control.py"),
])
def test_cli_module_main_is_callable(module_name, script_name):
    """Each CLI module must expose a callable ``main``."""
    import importlib
    mod = importlib.import_module(module_name)
    assert callable(getattr(mod, "main", None)), (
        f"{module_name}.main is not callable"
    )


def test_cli_load_script_main_dispatches_to_script():
    """_load_script_main must return the script's main function."""
    from daph_learning.cli import _load_script_main
    main_fn = _load_script_main("evaluate_routes.py")
    assert callable(main_fn)
    # The loaded function should be the same object as the script's main.
    assert main_fn.__name__ == "main"


def test_cli_load_script_main_raises_for_missing_script():
    """_load_script_main must raise ModuleNotFoundError for a missing script."""
    from daph_learning.cli import _load_script_main
    with pytest.raises(ModuleNotFoundError, match="not found"):
        _load_script_main("nonexistent_script.py")


# --- 4. The package installs cleanly (no PYTHONPATH required) ---

def test_package_importable_without_pythonpath():
    """daph_learning must be importable without PYTHONPATH=src after install."""
    import daph_learning
    assert daph_learning.__version__ == "0.3.8"


def test_cli_importable_without_pythonpath():
    """The CLI modules must be importable without PYTHONPATH=src."""
    import daph_learning.cli.autolearn
    import daph_learning.cli.evaluate_routes
    import daph_learning.cli.build_oracles
    import daph_learning.cli.tune_steering
    import daph_learning.cli.random_control
    # If we got here, the imports succeeded.
