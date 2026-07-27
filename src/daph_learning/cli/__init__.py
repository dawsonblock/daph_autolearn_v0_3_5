"""CLI entry-point package (V037-006).

Each module in this package is a thin wrapper that loads the corresponding
``scripts/*.py`` by file path (via :mod:`importlib.util`) and calls its
``main()`` function. This avoids requiring ``scripts/`` to be on the Python
path while still reusing the existing CLI argument parsing.

After ``pip install -e .``, the entry points are available as:

- ``daph-autolearn``
- ``daph-evaluate-routes``
- ``daph-build-oracles``
- ``daph-tune-steering``
- ``daph-random-control``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The scripts/ directory relative to the installed package.
# When installed via pip, the scripts/ directory is not included in the
# package (it's not under src/). We locate it relative to the package
# root's parent's parent (the repo root). For editable installs this works
# because the repo root is the install location. For non-editable installs,
# the CLI entry points will not be usable (by design — scripts/ is a
# development convenience, not a shipped component).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _load_script_main(script_name: str):
    """Load ``scripts/<script_name>.py`` by file path and return its ``main``
    callable.

    Raises ``ModuleNotFoundError`` if the scripts directory is not present
    (e.g. in a non-editable install where scripts/ is not shipped).
    """
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise ModuleNotFoundError(
            f"{script_name} not found at {script_path}. The CLI entry points "
            "require an editable install (pip install -e .) so the scripts/ "
            "directory is available."
        )
    module_name = f"_daph_cli_{script_name.replace('.py', '').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Register the module so that relative imports inside the script work.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main
