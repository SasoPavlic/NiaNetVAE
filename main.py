"""Checkout-friendly wrapper for :mod:`nianetvae.cli`."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

if __name__ == "__main__":
    raise SystemExit(import_module("nianetvae.cli").main())
