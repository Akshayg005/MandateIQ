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
the remaining clauses (4(c) ceiling, the NPCI attempt cap, stopping rules)
rather than re-creating it.
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
