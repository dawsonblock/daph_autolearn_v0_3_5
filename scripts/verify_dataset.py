from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sha256")
    args = ap.parse_args()
    digest = sha256_file(Path(args.path))
    print(digest)
    if args.sha256 and digest.lower() != args.sha256.lower():
        raise SystemExit(
            f"digest mismatch: expected {args.sha256}, got {digest}"
        )


if __name__ == "__main__":
    main()
