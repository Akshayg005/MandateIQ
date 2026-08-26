"""Integer-paise arithmetic. The only module allowed to format currency for
display (CLAUDE.md, src/core/CLAUDE.md). No float ever leaves this module.
"""
from __future__ import annotations

from fractions import Fraction
from typing import NewType

Paise = NewType("Paise", int)


def _group_indian(rest: str) -> str:
    """Group digits in twos from the right, Indian style. `rest` excludes
    the trailing 3 digits, which the caller joins on separately."""
    if len(rest) <= 2:
        return rest
    groups: list[str] = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    groups.insert(0, rest)
    return ",".join(groups)


def fmt(p: Paise) -> str:
    """Render paise as an Indian-Rupee display string, e.g. ₹15,00,000.00."""
    if int(p) < 0:
        raise ValueError(f"fmt() received negative paise: {p!r}")
    rupees, paise = divmod(int(p), 100)
    rupees_str = str(rupees)
    if len(rupees_str) > 3:
        last3, rest = rupees_str[-3:], rupees_str[:-3]
        rupees_str = f"{_group_indian(rest)},{last3}"
    return f"₹{rupees_str}.{paise:02d}"


def pct_of(p: Paise, frac: float) -> Paise:
    """`frac` of `p`, floored to the nearest whole paise. Rounding is always
    floor -- never round-half-up, never ceil.

    Computed via exact `Fraction` arithmetic rather than `int(p * frac)`:
    plain float multiplication can round the *product* up past the true
    floor for large `p` (e.g. `int(7079410 * 0.7)` gives 4955587, one paise
    over the exact floor of 4955586) even though each input's own float
    representation is fine on its own. Fraction(frac) takes the exact
    IEEE-754 value of the float `frac` -- it does not "fix" the input, it
    just removes the extra rounding error the multiply step would add.
    """
    if int(p) < 0:
        raise ValueError(f"pct_of() received negative paise: {p!r}")
    return Paise((Fraction(int(p)) * Fraction(frac)).__floor__())


def split_floor(p: Paise, n: int) -> list[Paise]:
    """Split `p` into `n` non-negative integer paise parts summing to `p`,
    each part within 1 paise of every other -- floor division with the
    remainder spread one paise at a time."""
    if int(p) < 0:
        raise ValueError(f"split_floor() received negative paise: {p!r}")
    base, remainder = divmod(int(p), n)
    return [Paise(base + 1)] * remainder + [Paise(base)] * (n - remainder)
