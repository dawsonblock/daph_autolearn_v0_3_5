"""Entry point for ``daph-evaluate-routes`` (V037-006)."""

from daph_learning.cli import _load_script_main


def main() -> None:
    _load_script_main("evaluate_routes.py")()


if __name__ == "__main__":
    main()
