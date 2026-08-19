#!/usr/bin/env python3
"""Fingerprint search-relevant NiaNetVAE source inside any runtime image."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nianetvae

PATTERNS = (
    "config.py",
    "contracts.py",
    "dataloaders/**/*.py",
    "evaluation/calibration.py",
    "evaluation/risk.py",
    "models/**/*.py",
    "search/__init__.py",
    "search/checkpointing.py",
    "search/engine.py",
    "search/genome.py",
    "search/objectives.py",
    "search/storage.py",
    "training/**/*.py",
)


def main() -> None:
    root = Path(nianetvae.__file__).resolve().parent
    sources = {
        path.resolve()
        for pattern in PATTERNS
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not sources:
        raise RuntimeError(f"No architecture-search runtime sources found under {root}.")
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.relative_to(root).as_posix()):
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    print(digest.hexdigest())


if __name__ == "__main__":
    main()
