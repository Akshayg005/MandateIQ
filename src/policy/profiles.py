"""The two RBI compliance interpretations, and what each requires per slot.

RBI circular RBI/DPSS/2026-27/396 never uses the word "retry" -- clause 6(a)
requires a pre-transaction notification >=24h before "every debit", but
whether a *reattempt* within the same mandate cycle needs its own fresh
notification, or is covered by the cycle's original one, is genuinely
unresolved in the text. This project ships both readings and evaluates
under both, per root DESIGN.md: "Never hard-code one interpretation."

  strict     -- every slot (1, 2, 3, 4) needs its own fresh notification.
  permissive -- the cycle's original notification (slot 1) covers its
                retries (slots 2, 3, 4); only slot 1 needs a fresh one.

The `Profile` enum this module dispatches on lives at src/core/types.py and
is imported, never redefined here -- it is already the identity stored in
the `ledger.profile` and `committed_schedule.profile` TEXT columns, and a
second enum would silently diverge from persisted rows.

No module-level default profile is declared. The compliance audit's
check #8 is "Neither interpretation is hard-coded"; the runtime default
arrives from .env's COMPLIANCE_PROFILE at the edge, not from this layer.

`committable_days()` and the day-set shrink `strict` implies for the
allocator's search are src/policy/allocator.py's, at B8 -- not defined here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.core.types import Profile
from src.policy.constraints import MAX_ATTEMPTS


@dataclass(frozen=True)
class ComplianceProfile:
    """profile: which Profile member this is. retries_covered_by_cycle_
    notification: True if the cycle's original notification is read as
    covering its own retries (permissive), False if every slot needs its
    own fresh notification (strict)."""

    profile: Profile
    retries_covered_by_cycle_notification: bool

    def requires_fresh_notification(self, slot: int) -> bool:
        """True if `slot` needs its own fresh pre-transaction notification
        under this profile. Slot 1 always does -- it is the original
        attempt's own notification. Raises for a slot outside the NPCI
        attempt range 1..MAX_ATTEMPTS."""
        if not (1 <= slot <= MAX_ATTEMPTS):
            raise ValueError(
                f"slot must be in 1..{MAX_ATTEMPTS} (NPCI's attempt range); "
                f"got {slot}"
            )
        if slot == 1:
            return True
        return not self.retries_covered_by_cycle_notification


PROFILES: Mapping[Profile, ComplianceProfile] = {
    Profile.strict: ComplianceProfile(
        profile=Profile.strict,
        retries_covered_by_cycle_notification=False,
    ),
    Profile.permissive: ComplianceProfile(
        profile=Profile.permissive,
        retries_covered_by_cycle_notification=True,
    ),
}


def get(profile: Profile) -> ComplianceProfile:
    """The ComplianceProfile for `profile`. Raises TypeError for anything
    that is not literally a Profile member -- Profile is a str Enum, so
    Profile.strict == "strict" is True and a plain dict lookup by the bare
    string would succeed; this rejects that rather than silently coercing,
    matching this codebase's refusal-over-coercion convention."""
    if not isinstance(profile, Profile):
        raise TypeError(
            f"get() requires a Profile member, got {profile!r} of type "
            f"{type(profile).__name__} -- pass Profile.strict or "
            "Profile.permissive, not a bare string."
        )
    return PROFILES[profile]
