"""Entry point for ``daph-random-control`` (V037-006)."""

from daph_learning.cli import _load_script_main


def main() -> None:
    _load_script_main("random_direction_control.py")()


if __name__ == "__main__":
    main()
