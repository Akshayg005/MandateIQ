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


def interpolate_crossing(x0: int, y0: int, x1: int, y1: int) -> Fraction:
    """R3, 2026-09-04 (reports/gates.md, "Post-B16 remediation gates"): the
    exact x-value where a straight line through (x0, y0) and (x1, y1)
    crosses y=0 -- e.g. x = an LTV value in paise, y = (engine's
    recovered_paise - the ladder's) at that LTV, from two adjacent points
    of an LTV sensitivity sweep. This is the break-even point between two
    SWEPT-AND-MEASURED cells, not a claim about what happens between them
    -- report it as an interpolation, never as a third measurement.

    Requires y0 and y1 to have a genuine sign change (or one to be exactly
    zero) -- raises ValueError otherwise, since a crossing is only
    well-defined where the sign actually flips; a caller sweeping several
    points must find the bracketing pair first; this function does not
    search for one.

    Returns an exact Fraction (never a float) so a caller can go on to
    divide it by, e.g., a mean mandate amount without compounding rounding
    error -- money.py's whole reason to exist.
    """
    if y0 == 0:
        return Fraction(x0)
    if y1 == 0:
        return Fraction(x1)
    if (y0 < 0) == (y1 < 0):
        raise ValueError(
            f"interpolate_crossing() requires a sign change between "
            f"y0={y0} and y1={y1} -- no crossing between these two points"
        )
    # Linear interpolation: x = x0 + (0 - y0) * (x1 - x0) / (y1 - y0),
    # kept as one exact Fraction rather than an intermediate float division.
    return Fraction(x0) + Fraction(-y0, 1) * Fraction(x1 - x0, y1 - y0)
