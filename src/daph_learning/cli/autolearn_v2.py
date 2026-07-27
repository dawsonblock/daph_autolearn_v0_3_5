"""Entry point for ``daph-autolearn-v2``."""

from daph_learning.cli import _load_script_main


def main() -> None:
    _load_script_main("autolearn_v2.py")()


if __name__ == "__main__":
    main()
