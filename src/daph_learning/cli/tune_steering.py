"""Entry point for ``daph-tune-steering`` (V037-006)."""

from daph_learning.cli import _load_script_main


def main() -> None:
    _load_script_main("tune_steering.py")()


if __name__ == "__main__":
    main()
