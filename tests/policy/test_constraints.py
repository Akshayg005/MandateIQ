"""src/policy/constraints.py -- regulatory constants and validation functions.

Design spec: Every constant and function here must cite its RBI clause, as per
root CLAUDE.md's "no unattributed magic numbers" rule. This module is the sole
source of truth for:

- AFA-free limits per clause 8(a) [base] and 8(b) [elevated for specific
  categories].
- The NPCI attempt cap: 1 original + 3 retries = 4 attempts total (MAX_ATTEMPTS),
  with RETRY_SLOTS = (2, 3, 4) capturing policy choices (slot 1 is mandatory,
  never a choice).
- RBI clause 6(a)'s 24-hour pre-transaction notification lead time
  (COMMIT_LEAD_HOURS), setting the earliest slot of any attempt.
- Clause 4(c)'s mandate-ceiling validation: no debit may exceed the customer's
  per-transaction maximum.
- Clause 6(d)'s pre-notification exemptions (FASTag/NCMC auto-replenishment),
  which assert we never enter this system for those categories.

All money is integer paise. All category strings are lowercase with underscores.
Boundary rule (settled at B7): limits are INCLUSIVE. `amount_paise <= limit`
is AFA-free; `requires_afa` returns True strictly above the limit.
"""
from __future__ import annotations

import pytest


# === existing surface (B4) =================================================

def test_afa_free_limit_paise_returns_int_not_float():
    """AFA limit must be an integer in paise, never a float.
    Invariant 2: all money is integer paise."""
    from src.policy.constraints import AFA_FREE_LIMIT_PAISE

    assert isinstance(AFA_FREE_LIMIT_PAISE, int), \
        f"AFA_FREE_LIMIT_PAISE is {type(AFA_FREE_LIMIT_PAISE).__name__}, not int"


def test_elevated_limit_is_higher_than_base_limit():
    """Clause 8(b)'s elevated AFA-free limit (insurance, mutual fund, credit
    card bills) must be strictly greater than clause 8(a)'s base limit."""
    from src.policy.constraints import AFA_FREE_LIMIT_PAISE, AFA_FREE_LIMIT_ELEVATED_PAISE

    assert AFA_FREE_LIMIT_ELEVATED_PAISE > AFA_FREE_LIMIT_PAISE, \
        f"elevated {AFA_FREE_LIMIT_ELEVATED_PAISE} must exceed base {AFA_FREE_LIMIT_PAISE}"


# === AFA boundary tests (boundary rule is INCLUSIVE) ======================

def test_afa_boundary_is_inclusive_at_the_base_limit():
    """Clause 8(a): AFA-free UP TO Rs 15,000. The word "up to" means INCLUSIVE
    at the boundary. At exactly AFA_FREE_LIMIT_PAISE, requires_afa is False.
    One paisa above it, requires_afa is True. This pins the boundary rule that
    eval/frozen/sim_config.yaml left undefined."""
    from src.policy.constraints import AFA_FREE_LIMIT_PAISE, requires_afa

    # At exactly the limit, with a non-elevated category
    assert requires_afa(AFA_FREE_LIMIT_PAISE, "subscription") is False, \
        f"amount exactly at limit {AFA_FREE_LIMIT_PAISE} should be AFA-free"

    # One paisa below: also AFA-free
    assert requires_afa(AFA_FREE_LIMIT_PAISE - 1, "subscription") is False, \
        f"amount {AFA_FREE_LIMIT_PAISE - 1} below limit should be AFA-free"

    # One paisa above: requires AFA
    assert requires_afa(AFA_FREE_LIMIT_PAISE + 1, "subscription") is True, \
        f"amount {AFA_FREE_LIMIT_PAISE + 1} above limit should require AFA"


def test_afa_boundary_is_inclusive_at_the_elevated_limit():
    """Clause 8(b): the elevated AFA-free limit applies to insurance_premium,
    mutual_fund, and credit_card_bill. Boundary is also INCLUSIVE."""
    from src.policy.constraints import AFA_FREE_LIMIT_ELEVATED_PAISE, requires_afa

    elevated_categories = ["insurance_premium", "mutual_fund", "credit_card_bill"]

    for cat in elevated_categories:
        # At exactly the elevated limit
        assert requires_afa(AFA_FREE_LIMIT_ELEVATED_PAISE, cat) is False, \
            f"amount exactly at elevated limit {AFA_FREE_LIMIT_ELEVATED_PAISE} " \
            f"for {cat} should be AFA-free"

        # One paisa below
        assert requires_afa(AFA_FREE_LIMIT_ELEVATED_PAISE - 1, cat) is False, \
            f"amount below elevated limit for {cat} should be AFA-free"

        # One paisa above
        assert requires_afa(AFA_FREE_LIMIT_ELEVATED_PAISE + 1, cat) is True, \
            f"amount above elevated limit {AFA_FREE_LIMIT_ELEVATED_PAISE} " \
            f"for {cat} should require AFA"


def test_subscription_is_not_an_elevated_category():
    """Subscription is the majority category (70% in frozen batch) and stays on
    clause 8(a)'s base limit, never 8(b)'s elevated limit."""
    from src.policy.constraints import AFA_FREE_LIMIT_PAISE, requires_afa

    # Use the base limit boundary
    assert requires_afa(AFA_FREE_LIMIT_PAISE, "subscription") is False, \
        f"subscription at base limit {AFA_FREE_LIMIT_PAISE} should be AFA-free"

    assert requires_afa(AFA_FREE_LIMIT_PAISE + 1, "subscription") is True, \
        f"subscription above base limit should require AFA"


def test_unknown_category_falls_back_to_base_limit():
    """A typo'd or unseen category string (e.g., 'insurance-premium' with a
    hyphen, 'Insurance_Premium' wrong case, 'groceries' not in the taxonomy)
    must fall back to the base AFA-free limit, never the elevated one. A typo
    must not silently unlock the Rs 1,00,000 limit."""
    from src.policy.constraints import AFA_FREE_LIMIT_PAISE, AFA_FREE_LIMIT_ELEVATED_PAISE, requires_afa

    unknown_categories = ["insurance-premium", "Insurance_Premium", "groceries", "SUBSCRIPTION"]

    for cat in unknown_categories:
        # Unknown categories behave like "subscription" (base limit)
        assert requires_afa(AFA_FREE_LIMIT_PAISE, cat) is False, \
            f"unknown category '{cat}' at base limit should be AFA-free"

        assert requires_afa(AFA_FREE_LIMIT_PAISE + 1, cat) is True, \
            f"unknown category '{cat}' above base limit should require AFA"

        # They do NOT get the elevated limit
        assert requires_afa(AFA_FREE_LIMIT_ELEVATED_PAISE - 1, cat) is True, \
            f"unknown category '{cat}' below elevated limit should still require AFA " \
            f"(it uses base limit, not elevated)"


# === NPCI attempt cap (B7 new) ==============================================

def test_max_attempts_is_four_original_plus_three_retries():
    """NPCI rule: 1 original + 3 retries = 4 attempts total. MAX_ATTEMPTS must
    be 4, and RETRY_SLOTS must contain exactly the three slot indices 2, 3, 4
    (slot 1 is mandatory, never a policy choice)."""
    from src.policy.constraints import MAX_ATTEMPTS, RETRY_SLOTS

    assert MAX_ATTEMPTS == 4, \
        f"MAX_ATTEMPTS is {MAX_ATTEMPTS}, expected 4"

    assert len(RETRY_SLOTS) == MAX_ATTEMPTS - 1, \
        f"RETRY_SLOTS has {len(RETRY_SLOTS)} elements, expected {MAX_ATTEMPTS - 1}"

    assert RETRY_SLOTS == (2, 3, 4), \
        f"RETRY_SLOTS is {RETRY_SLOTS}, expected (2, 3, 4)"

    assert 1 not in RETRY_SLOTS, \
        f"slot 1 must not be in RETRY_SLOTS; slot 1 is mandatory, never a policy choice"


def test_max_attempts_agrees_with_paths_horizon():
    """The NPCI cap (1 original + 3 retries = 4) is currently expressed four
    independent times in the codebase: src/model/paths.py:HORIZON,
    schema.sql:44, schema.sql:73, eval/frozen/simulator.py:214. No shared
    constant yet. This test is the mechanical pin that keeps constraints.py and
    paths.py in agreement. src/model/ must NOT import src/policy/, so this
    test is the enforcement point."""
    from src.policy.constraints import MAX_ATTEMPTS
    from src.model.paths import HORIZON

    assert MAX_ATTEMPTS == HORIZON, \
        f"constraints.MAX_ATTEMPTS ({MAX_ATTEMPTS}) must equal " \
        f"paths.HORIZON ({HORIZON})"


# === RBI clause 6(a): 24h pre-transaction notification lead =================

def test_commit_lead_hours_is_24():
    """RBI clause 6(a) mandates a >= 24-hour pre-transaction notification lead
    before every debit. The constant must be 24."""
    from src.policy.constraints import COMMIT_LEAD_HOURS

    assert COMMIT_LEAD_HOURS == 24, \
        f"COMMIT_LEAD_HOURS is {COMMIT_LEAD_HOURS}, expected 24 per clause 6(a)"


def test_commit_lead_hours_agrees_with_clock_commit_deadline():
    """RBI clause 6(a)'s 24h lead time is implemented both in constraints.py
    (the constant) and src/core/clock.py (commit_deadline function, which
    currently hard-codes the default as 24). This test pins them together,
    because src/core/ must not import src/policy/."""
    from datetime import datetime, timedelta, timezone
    from src.policy.constraints import COMMIT_LEAD_HOURS
    from src.core.clock import commit_deadline

    target = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

    # The deadline should be target minus COMMIT_LEAD_HOURS. Using timedelta,
    # not datetime.replace(hour=...) -- target.hour - COMMIT_LEAD_HOURS is
    # negative (12 - 24 = -12), which replace() cannot represent as an hour.
    expected = target - timedelta(hours=COMMIT_LEAD_HOURS)

    result = commit_deadline(target)

    assert result == expected, \
        f"commit_deadline({target}) returned {result}, " \
        f"expected {expected} (target minus {COMMIT_LEAD_HOURS} hours)"


# === Clause 6(d): pre-notification exemptions ===============================

def test_assert_not_pre_notification_exempt_raises_for_fastag_and_ncmc():
    """Clause 6(d) exempts FASTag and NCMC auto-replenishment from the
    pre-transaction notification requirement. These categories must never reach
    this system. assert_not_pre_notification_exempt() must RAISE (not return a
    bool) if encountered."""
    from src.policy.constraints import assert_not_pre_notification_exempt, PRE_NOTIFICATION_EXEMPT_CATEGORIES

    for category in PRE_NOTIFICATION_EXEMPT_CATEGORIES:
        with pytest.raises(ValueError):
            assert_not_pre_notification_exempt(category)


def test_assert_not_pre_notification_exempt_passes_for_every_frozen_batch_category():
    """The four categories in eval/frozen/sim_config.yaml's category_mix
    (subscription, insurance_premium, mutual_fund, credit_card_bill) are all
    subject to the 24h notification requirement. They must pass through
    assert_not_pre_notification_exempt() without raising."""
    from src.policy.constraints import assert_not_pre_notification_exempt

    frozen_batch_categories = [
        "subscription",
        "insurance_premium",
        "mutual_fund",
        "credit_card_bill",
    ]

    for category in frozen_batch_categories:
        # This should not raise
        result = assert_not_pre_notification_exempt(category)
        assert result is None, \
            f"assert_not_pre_notification_exempt('{category}') should return None"


# === Clause 4(c): mandate-ceiling validation ================================

def test_within_mandate_ceiling_is_inclusive():
    """Clause 4(c): variable e-mandates carry a customer-set maximum per
    transaction. No attempt may exceed it. The boundary is INCLUSIVE:
    amount_paise <= ceiling_paise is within; amount + 1 is not."""
    from src.policy.constraints import within_mandate_ceiling

    ceiling = 5_000_000  # Rs 50,000

    # At exactly the ceiling
    assert within_mandate_ceiling(ceiling, ceiling) is True, \
        f"amount exactly at ceiling {ceiling} should be within"

    # Below the ceiling
    assert within_mandate_ceiling(ceiling - 1, ceiling) is True, \
        f"amount {ceiling - 1} below ceiling should be within"

    # Above the ceiling
    assert within_mandate_ceiling(ceiling + 1, ceiling) is False, \
        f"amount {ceiling + 1} above ceiling should not be within"

    # Zero amount
    assert within_mandate_ceiling(0, ceiling) is True, \
        f"amount 0 should be within any positive ceiling"
