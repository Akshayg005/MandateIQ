"""
Root conftest.

`python -m pytest` (what `.\\run.ps1 test` runs) already puts the repo root
on sys.path because of how `-m` works, so `from src.core.money import ...`
resolves without this. This is here defensively, for the plain `pytest`
console-script invocation, which does not always do that -- so imports keep
working regardless of which entry point is used to run the suite.
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
