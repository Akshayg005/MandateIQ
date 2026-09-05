"""The only source of "now" in the codebase. Must be freezable, because the
24-hour commitment lag (RBI clause 6(a)) is untestable against a live clock.
Nothing outside this module may call datetime.now() -- see src/core/DESIGN.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_frozen: datetime | None = None


def now() -> datetime:
    """Current time, tz-aware. Returns the frozen value if one is set via
    set_frozen(), otherwise the real wall clock."""
    if _frozen is not None:
        return _frozen
    return datetime.now(timezone.utc)


def set_frozen(dt: datetime | None) -> None:
    """Freeze now() to `dt`. Pass None to unfreeze back to the real clock."""
    global _frozen
    _frozen = dt


def commit_deadline(target: datetime, lead_hours: int = 24) -> datetime:
    """The latest moment an attempt targeting `target` may be committed,
    per RBI clause 6(a)'s pre-transaction notification lead time."""
    return target - timedelta(hours=lead_hours)
