"""Cumulative incidence functions and survival curves over a fitted
competing-risks hazard. No dependency on competing_risks.py, person_period.py,
or features.py -- this module is pure array math and is tested that way.

The hazard input `h` has shape (n, 3, 4): axis 0 is mandates, axis 1 indexes
SLOTS 2, 3, 4 in that order (there is no slot-1 entry -- slot 1 is hard-wired
as a structural zero, h_c(1) identically 0 for every cause, per
src/model/person_period.py's `estimable` invariant -- the caller never
supplies it), axis 2 indexes the four Outcome probabilities in Outcome int
order [STILL_PENDING, RECOVERED, DEAD, OPTED_OUT]. Every h[i, s, :] must sum
to 1 within tolerance.

`cif(h)` returns shape (n, 3, 4): axis 1 indexes the 3 NON-REFERENCE outcomes
in Outcome int order [RECOVERED, DEAD, OPTED_OUT], axis 2 indexes SLOT
1, 2, 3, 4 (0-indexed 0..3) -- the cumulative incidence of that cause by that
slot, inclusive. `survival(h)` returns shape (n, 4): S(1), S(2), S(3), S(4).

Recursion (PLAN_DETAIL.md section 2, "CIF"):
    S(1) = 1                                   # hard-wired
    CIF_c(1) = 0                                # hard-wired
    for slot k in (2, 3, 4):                    # h's row index (k - 2)
        S(k) = S(k-1) * (1 - sum_c h_c(k))
        CIF_c(k) = CIF_c(k-1) + h_c(k) * S(k-1)

This is deliberately NOT `1 - KM` computed independently per cause -- that
treats every OTHER cause's event as censoring and overstates incidence,
because it ignores that a mandate resolved by cause A can no longer resolve
by cause B. The correct formula above weights each cause's increment at
slot k by the single, cause-agnostic MARGINAL survival S(k-1), which is
shared across all three causes -- not a per-cause KM curve.
"""
from __future__ import annotations

import numpy as np

_TOL = 1e-6


def _validate_hazards(h: np.ndarray) -> None:
    if h.ndim != 3 or h.shape[1] != 3 or h.shape[2] != 4:
        raise ValueError(
            "hazards array must have shape (n, 3, 4) -- axis 1 indexes "
            f"slots 2/3/4, axis 2 indexes the 4 Outcome probabilities. "
            f"Got shape {h.shape}."
        )
    row_sums = h.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=_TOL):
        bad = float(np.abs(row_sums - 1.0).max())
        raise ValueError(
            f"every hazard row (mandate, slot) must sum to 1 within "
            f"tolerance {_TOL}; max deviation observed: {bad}"
        )


def survival(h: np.ndarray) -> np.ndarray:
    """S(1..4) per mandate, shape (n, 4). S(1) is always exactly 1.0 --
    slot 1 is a hard-wired structural zero, never fit, never supplied."""
    _validate_hazards(h)
    n = h.shape[0]
    s = np.empty((n, 4), dtype=float)
    s[:, 0] = 1.0
    for k in range(2, 5):
        # Sum only the 3 terminal-event causes (RECOVERED, DEAD, OPTED_OUT),
        # never index 0 (STILL_PENDING) -- that's "survives this slot", not
        # a hazard, and every row already sums to 1 by construction, so
        # summing all 4 columns would always give 1.0 regardless of content.
        row_hazard_sum = h[:, k - 2, 1:4].sum(axis=1)
        s[:, k - 1] = s[:, k - 2] * (1.0 - row_hazard_sum)
    return s


def cif(h: np.ndarray) -> np.ndarray:
    """CIF_c(1..4) per mandate per non-reference cause [RECOVERED, DEAD,
    OPTED_OUT], shape (n, 3, 4). Slot-1 column (axis-2 index 0) is
    identically 0 -- see module docstring."""
    _validate_hazards(h)
    n = h.shape[0]
    s = survival(h)
    out = np.zeros((n, 3, 4), dtype=float)
    for k in range(2, 5):
        s_prev = s[:, k - 2]  # S(k-1)
        h_k = h[:, k - 2, 1:4]  # this slot's hazard, causes [RECOVERED, DEAD, OPTED_OUT]
        out[:, :, k - 1] = out[:, :, k - 2] + h_k * s_prev[:, None]
    return out
