"""
src/core/money.py -- integer-paise arithmetic. This is the only module
allowed to format currency for display (CLAUDE.md, src/core/CLAUDE.md).

Every assertion here is on an int or a formatted string produced from a
known-good int input -- never on a float, and never comparing formatted
strings to anything but a hand-derived expected string.
"""
from __future__ import annotations

import pytest

from fractions import Fraction

from src.core.money import Paise, fmt, interpolate_crossing, pct_of, split_floor

# --- fmt: rupee display, two-decimal paise, Indian digit grouping ----------

FMT_CASES = [
    (0, "₹0.00"),
    (1, "₹0.01"),
    (50, "₹0.50"),
    (100, "₹1.00"),
    (99999, "₹999.99"),  # 3-digit rupee part: no comma at all
    (150000, "₹1,500.00"),  # 4-digit rupee part: one comma
    # 7-digit rupee amount (15,00,000 = 15 lakh) -- the example CLAUDE.md's
    # spec calls out explicitly, and the one place Indian grouping (groups
    # of 2 after the first 3 digits) is distinguishable from Western (groups
    # of 3) grouping.
    (150000000, "₹15,00,000.00"),
    # crore-scale: proves the *second* Indian grouping boundary, not just
    # the first one.
    (1500000000, "₹1,50,00,000.00"),
]


@pytest.mark.parametrize("paise, expected", FMT_CASES)
def test_fmt_produces_exact_rupee_string(paise, expected):
    assert fmt(Paise(paise)) == expected


def test_fmt_returns_a_str():
    assert isinstance(fmt(Paise(12345)), str)


# --- pct_of: paise in, paise out, explicit floor rounding -------------------

def test_pct_of_returns_int_never_float():
    result = pct_of(Paise(10000), 0.5)
    assert type(result) is int


def test_pct_of_basic_half():
    assert pct_of(Paise(10000), 0.5) == 5000


def test_pct_of_zero_fraction_is_zero():
    assert pct_of(Paise(99999), 0.0) == 0


def test_pct_of_full_fraction_is_identity():
    assert pct_of(Paise(12345), 1.0) == 12345


def test_pct_of_rounds_down_not_to_nearest():
    """100 * 0.129 = 12.9. Floor gives 12; round-half-up or ceil would give
    13 -- this is the case that actually distinguishes the three, so it is
    the one that pins 'floor' rather than assuming it."""
    assert pct_of(Paise(100), 0.129) == 12


def test_pct_of_repeating_fraction_floors():
    # 100 * (1/3) = 33.333... -> floor 33
    assert pct_of(Paise(100), 1 / 3) == 33


def test_pct_of_large_amount_does_not_overshoot_the_true_floor():
    """int(p * frac) can round the PRODUCT up past the true floor for large
    p, even when p and frac are each individually fine -- int(7079410 *
    0.7) gives 4955587 via plain float multiplication, one paise over the
    exact floor of 4955586. This is the case that actually distinguishes
    'floor computed via float multiply' from 'floor computed exactly'."""
    assert pct_of(Paise(7_079_410), 0.7) == 4_955_586


def test_pct_of_rejects_negative_paise():
    with pytest.raises(ValueError):
        pct_of(Paise(-1), 0.5)


def test_fmt_rejects_negative_paise():
    with pytest.raises(ValueError):
        fmt(Paise(-1))


def test_split_floor_rejects_negative_paise():
    with pytest.raises(ValueError):
        split_floor(Paise(-1), 3)


# --- split_floor: sum-preserving, near-even split ---------------------------

SPLIT_CASES = [
    (100, 3),
    (10, 4),
    (7, 7),
    (1, 3),
    (0, 5),
    (1_000_000, 7),
    (5, 1),
    (99, 11),
]


@pytest.mark.parametrize("paise, n", SPLIT_CASES)
def test_split_floor_preserves_sum_and_spread(paise, n):
    parts = split_floor(Paise(paise), n)
    assert len(parts) == n
    assert sum(parts) == paise
    assert all(p >= 0 for p in parts)
    assert all(type(p) is int for p in parts)
    if n > 0:
        assert max(parts) - min(parts) <= 1


def test_split_floor_specific_example_matches_floor_division_with_remainder():
    # 100 // 3 == 33, remainder 1 -- exactly one part gets the extra paise.
    # We do not assert *which* index gets it, only the resulting multiset.
    parts = split_floor(Paise(100), 3)
    assert sorted(parts) == [33, 33, 34]


def test_split_floor_single_slot_returns_whole_amount():
    assert split_floor(Paise(4242), 1) == [4242]


def test_split_floor_zero_amount_is_all_zeros():
    assert split_floor(Paise(0), 5) == [0, 0, 0, 0, 0]


# --- interpolate_crossing: exact x where a line through two points hits y=0
# (R3, 2026-09-04). No test existed for this function before R3 became its
# first real caller -- added here rather than assumed correct on the
# strength of the docstring alone.


def test_interpolate_crossing_exact_midpoint():
    # (0, -10) -> (10, +10): straight line crosses y=0 at x=5 exactly.
    assert interpolate_crossing(0, -10, 10, 10) == Fraction(5)


def test_interpolate_crossing_returns_exact_fraction_not_integer():
    # (0, -1) -> (3, 2): crosses at x = 1, an exact integer -- pick a case
    # that is NOT integral to prove the return type carries real precision.
    # (0, -1) -> (1, 2): slope 3, crosses at x0 + 1/3.
    result = interpolate_crossing(0, -1, 1, 2)
    assert result == Fraction(1, 3)
    assert isinstance(result, Fraction)


def test_interpolate_crossing_x0_exactly_zero():
    assert interpolate_crossing(100, 0, 200, 50) == Fraction(100)


def test_interpolate_crossing_x1_exactly_zero():
    assert interpolate_crossing(100, -50, 200, 0) == Fraction(200)


def test_interpolate_crossing_negative_to_positive_and_reverse_agree():
    # Sign convention must not matter -- swapping which endpoint is
    # negative/positive must not change the crossing point.
    a = interpolate_crossing(0, -4, 8, 4)
    b = interpolate_crossing(0, 4, 8, -4)
    assert a == b == Fraction(4)


def test_interpolate_crossing_raises_without_a_sign_change():
    # Both y-values positive: no crossing exists between these two points.
    with pytest.raises(ValueError, match="sign change"):
        interpolate_crossing(0, 5, 10, 15)


def test_interpolate_crossing_raises_both_negative():
    with pytest.raises(ValueError, match="sign change"):
        interpolate_crossing(0, -5, 10, -1)


def test_interpolate_crossing_large_paise_values_stay_exact():
    # Realistic scale (R3's actual use: x = LTV in paise, y = recovered_paise
    # difference) -- confirms no float ever enters the computation, even at
    # values in the tens of millions.
    result = interpolate_crossing(0, -15_047_099, 20_000_000, 5_000_000)
    # Exact rational, hand-verified: x = 0 + 15047099 * 20000000 / 20047099
    expected = Fraction(15_047_099 * 20_000_000, 20_047_099)
    assert result == expected
