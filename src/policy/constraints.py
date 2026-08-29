"""Regulatory constants governing when a debit may bypass Additional Factor
of Authentication (AFA). Every constant here must cite its RBI clause --
root CLAUDE.md's "no unattributed magic numbers" rule.

Source: RBI "Digital Payments -- E-mandate Framework, 2026", circular
RBI/DPSS/2026-27/396, dated 21 April 2026 (see root CLAUDE.md's regulatory
constants table).

This file is created early, at B4, because eval/corpus.py's AFA-cliff
filtering (see its module docstring and DECISIONS.md, 2026-08-27, B4) needs
a real Python constant; AFA_FREE_LIMIT_PAISE existed until now only as a
YAML comment (eval/frozen/sim_config.yaml:41). B7 extends this file with
the remaining clauses (4(c) ceiling, the NPCI attempt cap, 6(a)'s lead time,
6(d)'s exemption) rather than re-creating it.

AFA boundary rule, settled at B7: clause 8(a)/8(b) read "AFA-free UP TO"
their respective limits, so both limits are INCLUSIVE -- an amount exactly
at the limit is still AFA-free; `requires_afa()` returns True only strictly
above it. `eval/frozen/sim_config.yaml:41-46` deliberately left a gap either
side of the cliff "so no mandate lands exactly on the boundary before B7
defines the boundary rule" -- this is that definition. Same inclusive rule
for clause 4(c)'s mandate ceiling (`within_mandate_ceiling`).

Attempt-cost, mandate-LTV, re-auth cost, quiet hours, and contact-frequency
caps carry no RBI clause and do NOT belong in this file -- per
src/policy/CLAUDE.md they are tuning parameters for a config file, not a
regulatory constant. B8 owns where they land.
"""
from __future__ import annotations

# Clause 8(a): AFA-free up to Rs 15,000 per transaction.
AFA_FREE_LIMIT_PAISE = 1_500_000

# Clause 8(b): Rs 1,00,000 specifically for insurance premiums, mutual fund
# subscriptions, and credit card bills -- not a general raise of 8(a)'s limit.
AFA_FREE_LIMIT_ELEVATED_PAISE = 10_000_000

# The category strings eval/frozen/sim_config.yaml's category_mix uses for
# clause 8(b)'s three elevated categories. "subscription" is deliberately
# absent -- it stays on the 8(a) limit.
ELEVATED_AFA_CATEGORIES = frozenset({
    "insurance_premium",
    "mutual_fund",
    "credit_card_bill",
})


def afa_free_limit_paise(category: str) -> int:
    """The AFA-free ceiling that applies to `category` -- 8(b)'s elevated
    limit for the three named categories, 8(a)'s base limit otherwise."""
    if category in ELEVATED_AFA_CATEGORIES:
        return AFA_FREE_LIMIT_ELEVATED_PAISE
    return AFA_FREE_LIMIT_PAISE


def requires_afa(amount_paise: int, category: str) -> bool:
    """True if `amount_paise` exceeds the AFA-free ceiling for `category`
    (8(a)'s base limit, or 8(b)'s elevated limit for the three named
    categories) -- i.e. the attempt must take the re-auth path, never the
    silent one. Boundary is inclusive: an amount exactly at the limit is
    still AFA-free, so this returns False there."""
    return amount_paise > afa_free_limit_paise(category)


# NPCI: 1 original attempt + 3 retries = 4 attempts total, ever. Not itself
# an RBI clause -- NPCI's own attempt-budget rule, cited the same way
# throughout this codebase (root CLAUDE.md's regulatory constants table).
# Currently also expressed independently in src/model/paths.py's HORIZON,
# src/ledger/schema.sql's two `attempt_index` CHECK constraints, and
# eval/frozen/simulator.py's slot-range guard -- src/model/ and src/core/
# must not import src/policy/ (the dependency edge points one way), so
# tests pin these in agreement rather than this module importing them.
MAX_ATTEMPTS = 4

# Slot 1 is the original attempt, given rather than a policy choice --
# retries only ever spend from these three slots.
RETRY_SLOTS = (2, 3, 4)


# Clause 6(a): every attempt must be committed >=24h before the debit it
# notifies. src/core/clock.py's commit_deadline() hard-codes this same 24h
# lead as its default argument (src/core/ must not import src/policy/);
# tests pin the two in agreement.
COMMIT_LEAD_HOURS = 24


# Clause 6(d): pre-notification is exempt only for FASTag / NCMC
# auto-replenishment. Out of scope for this system -- root CLAUDE.md:
# "assert we never hit this path" -- so these categories never reach a
# committable attempt at all, rather than being silently handled correctly.
PRE_NOTIFICATION_EXEMPT_CATEGORIES = frozenset({"fastag", "ncmc"})


def assert_not_pre_notification_exempt(category: str) -> None:
    """Raises if `category` is a clause 6(d) pre-notification exemption
    (FASTag / NCMC auto-replenishment) -- this system never attempts to
    handle that path; it asserts the path is never reached instead."""
    if category in PRE_NOTIFICATION_EXEMPT_CATEGORIES:
        raise ValueError(
            f"category {category!r} is exempt from pre-transaction "
            "notification under clause 6(d) (FASTag/NCMC auto-replenishment) "
            "-- out of scope for this system; it must never reach a "
            "committable attempt."
        )


def within_mandate_ceiling(amount_paise: int, ceiling_paise: int) -> bool:
    """Clause 4(c): a variable e-mandate carries a customer-set maximum per
    transaction. True if `amount_paise` does not exceed `ceiling_paise`.
    Boundary is inclusive: an amount exactly at the ceiling is within it."""
    return amount_paise <= ceiling_paise
