"""src/execute/intent_channel.py -- turning an exit-intent score into a
declared likelihood ratio over the three causes.

R5 (reports/gates.md, "Post-B16 remediation gates"). This is the adapter
that lets `src/llm/intent.py` reach `src/policy/belief.py` WITHOUT
`src/policy/` ever importing `src.llm` -- forbidden by
`scripts/guard_invariants.py`'s SRC_LLM_IMPORT, and correctly so. The score
crosses the boundary as a plain float; the mapping from that float to a
likelihood ratio is a DECLARED operating point, stated in one place and
version-stamped, never a calibration this module invents on the fly.

The tests below pin three things the design rests on:
  1. the operating point is DECLARED -- a constant with a version string,
     not a fit;
  2. a high score moves belief toward WONT_PAY and a low score moves it
     away, and neither can reach a degenerate posterior;
  3. the adapter refuses a score outside [0, 1] rather than clamping it
     silently -- `intent_score()` already clamps defensively at its own
     boundary, so an out-of-range value arriving HERE means something
     other than intent_score() produced it, which is exactly when a silent
     clamp would hide the bug.
"""
from __future__ import annotations

import pytest

from src.core.types import Cause


def test_operating_point_is_declared_not_fitted():
    from src.execute.intent_channel import (
        INTENT_CHANNEL_SOURCE_VERSION, INTENT_OPERATING_POINT,
    )

    assert 0.0 < INTENT_OPERATING_POINT.threshold < 1.0
    assert 0.0 < INTENT_OPERATING_POINT.fpr < INTENT_OPERATING_POINT.tpr < 1.0
    assert INTENT_CHANNEL_SOURCE_VERSION


def test_a_high_score_yields_a_wont_pay_dominant_ratio():
    from src.execute.intent_channel import likelihood_ratio_from_intent_score

    lr = likelihood_ratio_from_intent_score(0.95)
    assert lr[Cause.WONT_PAY] > lr[Cause.CANT_PAY_NOW]
    assert lr[Cause.WONT_PAY] > lr[Cause.CANT_PAY_EVER]


def test_a_low_score_yields_a_wont_pay_suppressing_ratio():
    from src.execute.intent_channel import likelihood_ratio_from_intent_score

    lr = likelihood_ratio_from_intent_score(0.05)
    assert lr[Cause.WONT_PAY] < lr[Cause.CANT_PAY_NOW]
    assert lr[Cause.WONT_PAY] < lr[Cause.CANT_PAY_EVER]


def test_the_two_non_wont_pay_causes_are_treated_identically():
    """The channel reads exit INTENT. It carries no information about which
    of the two non-exit causes is at work, and must not pretend otherwise
    -- inventing an asymmetry here would be fabricating evidence."""
    from src.execute.intent_channel import likelihood_ratio_from_intent_score

    for s in (0.0, 0.3, 0.5, 0.8, 1.0):
        lr = likelihood_ratio_from_intent_score(s)
        assert lr[Cause.CANT_PAY_NOW] == lr[Cause.CANT_PAY_EVER]


def test_score_out_of_range_raises_rather_than_clamping():
    from src.execute.intent_channel import likelihood_ratio_from_intent_score

    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            likelihood_ratio_from_intent_score(bad)


def test_the_ratio_feeds_belief_and_moves_it_toward_wont_pay():
    """End to end through the real belief update -- the point of the
    adapter existing at all."""
    from src.execute.intent_channel import (
        INTENT_CHANNEL_SOURCE_VERSION, likelihood_ratio_from_intent_score,
    )
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    out = update_from_likelihood_ratio(
        b, likelihood_ratio_from_intent_score(0.9),
        source_version=INTENT_CHANNEL_SOURCE_VERSION,
    )
    assert out[Cause.WONT_PAY] > b[Cause.WONT_PAY]
    assert f"source={INTENT_CHANNEL_SOURCE_VERSION}" in out.provenance


def test_one_observation_cannot_reach_a_singleton_belief():
    """The declared operating point is deliberately not oracular: ONE
    high-intent message must not be able to slam belief to certainty. The
    off-ramp is the one action a false positive cannot walk back."""
    from src.execute.intent_channel import (
        INTENT_CHANNEL_SOURCE_VERSION, likelihood_ratio_from_intent_score,
    )
    from src.policy.belief import (
        CAUSE_ORDER, REFERENCE_PRIOR, init, update_from_likelihood_ratio,
    )

    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    out = update_from_likelihood_ratio(
        b, likelihood_ratio_from_intent_score(1.0),
        source_version=INTENT_CHANNEL_SOURCE_VERSION,
    )
    assert out[Cause.WONT_PAY] < 0.90


def test_this_module_never_imports_src_llm():
    """The whole reason the adapter exists. src/execute/ is PERMITTED to
    import src.llm -- but this module must not, because doing so would make
    `intent_channel` a live LLM call site inside the decision path rather
    than a pure float -> ratio mapping. The guard cannot catch this (it
    only scopes src/model|policy|core|classify), so it is pinned here."""
    import pathlib

    src = pathlib.Path("src/execute/intent_channel.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "src.llm" not in stripped, f"live LLM import: {stripped}"
            assert "genai" not in stripped, f"live LLM import: {stripped}"
