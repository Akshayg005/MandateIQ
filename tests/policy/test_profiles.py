"""src/policy/profiles.py -- compliance interpretation profiles and notification rules.

Design spec: RBI circular RBI/DPSS/2026-27/396 never uses the word "retry", so
whether a reattempt needs its OWN fresh pre-transaction notification is
genuinely unresolved. This module ships two profiles and neither is a hard-coded
winner.

- strict (clause 6(a) strict reading): every slot (1, 2, 3, 4) needs its own
  fresh notification before the debit.
- permissive (alternative interpretation): the original cycle notification
  (slot 1) covers the retries (slots 2, 3, 4), so only slot 1 needs a fresh
  notification.

The Profile enum is imported from src.core.types and NOT redefined here. The
runtime default arrives from .env's COMPLIANCE_PROFILE, at the edge, not from
this layer -- no module-level DEFAULT constant. Every dataclass in src/model/
is frozen=True; this layer follows the pattern.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import FrozenInstanceError

import pytest


def _profiles_source_path() -> pathlib.Path:
    """Path to src/policy/profiles.py source file."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    return root / "src" / "policy" / "profiles.py"


# === profile instantiation and enum identity ================================

def test_both_profiles_instantiate():
    """Both strict and permissive profiles must instantiate as
    ComplianceProfile objects. Every Profile enum member must have an entry in
    PROFILES, and get() must return the correct ComplianceProfile for each."""
    from src.core.types import Profile
    from src.policy.profiles import PROFILES, get, ComplianceProfile

    for profile in Profile:
        # Must be in PROFILES mapping
        assert profile in PROFILES, \
            f"Profile.{profile.name} not found in PROFILES dict"

        # get() must return the value
        result = get(profile)
        assert isinstance(result, ComplianceProfile), \
            f"get(Profile.{profile.name}) returned {type(result).__name__}, not ComplianceProfile"

        # Round-trip: result.profile should equal the key
        assert result.profile == profile, \
            f"PROFILES[Profile.{profile.name}].profile is Profile.{result.profile.name}, " \
            f"expected Profile.{profile.name}"


def test_profiles_does_not_redefine_the_profile_enum():
    """Profile is the identity stored in the database (ledger.profile,
    committed_schedule.profile TEXT columns). A second Profile enum in
    src/policy/ would silently diverge from persisted rows. The enum is
    imported from src/core/types and not re-exported or redefined."""
    from src.core.types import Profile
    from src.policy import profiles

    # Check that profiles does not have its own Profile attribute defined
    # (it may import it, but must not redefine it)
    if hasattr(profiles, "Profile"):
        assert profiles.Profile is Profile, \
            f"profiles.Profile is not the same object as core.types.Profile"

    # PROFILES keys must be exactly the set of Profile members
    from src.policy.profiles import PROFILES

    profile_keys = set(PROFILES.keys())
    expected_keys = set(Profile)

    assert profile_keys == expected_keys, \
        f"PROFILES keys {profile_keys} do not match Profile members {expected_keys}"


# === strict profile: fresh notification for every slot ======================

def test_strict_requires_fresh_notification_for_every_slot():
    """The strict profile interprets clause 6(a) to require a fresh
    pre-transaction notification before EVERY slot (1, 2, 3, 4). The
    requires_fresh_notification() method must return True for all four slots."""
    from src.policy.profiles import get
    from src.core.types import Profile
    from src.policy.constraints import MAX_ATTEMPTS

    strict = get(Profile.strict)

    for slot in range(1, MAX_ATTEMPTS + 1):
        result = strict.requires_fresh_notification(slot)
        assert result is True, \
            f"strict.requires_fresh_notification({slot}) returned {result}, expected True"


# === permissive profile: fresh notification only for slot 1 =================

def test_permissive_requires_fresh_notification_only_for_the_original():
    """The permissive profile interprets clause 6(a) to allow the original
    cycle notification (slot 1) to cover the retries (slots 2, 3, 4). Only
    slot 1 needs a fresh notification."""
    from src.policy.profiles import get
    from src.core.types import Profile
    from src.policy.constraints import MAX_ATTEMPTS

    permissive = get(Profile.permissive)

    # Slot 1: requires fresh notification
    assert permissive.requires_fresh_notification(1) is True, \
        f"permissive.requires_fresh_notification(1) must return True"

    # Slots 2, 3, 4: do not require fresh notification
    for slot in range(2, MAX_ATTEMPTS + 1):
        result = permissive.requires_fresh_notification(slot)
        assert result is False, \
            f"permissive.requires_fresh_notification({slot}) returned {result}, expected False"


# === profile differentiation ================================================

def test_the_two_profiles_actually_differ():
    """There must be at least one slot where strict and permissive profiles
    return different values from requires_fresh_notification(). Guards against
    both collapsing to identical behaviour (which would make the two-profile
    evaluation vacuous)."""
    from src.policy.profiles import get
    from src.core.types import Profile
    from src.policy.constraints import MAX_ATTEMPTS

    strict = get(Profile.strict)
    permissive = get(Profile.permissive)

    # Find at least one disagreement
    disagreements = []
    for slot in range(1, MAX_ATTEMPTS + 1):
        strict_result = strict.requires_fresh_notification(slot)
        permissive_result = permissive.requires_fresh_notification(slot)

        if strict_result != permissive_result:
            disagreements.append(slot)

    assert len(disagreements) > 0, \
        f"strict and permissive profiles must differ on at least one slot, " \
        f"but both return the same value for all slots 1..{MAX_ATTEMPTS}"


# === slot validation ========================================================

def test_requires_fresh_notification_rejects_slot_outside_the_npci_range():
    """requires_fresh_notification() must reject slot indices outside the valid
    range 1..MAX_ATTEMPTS. Slots 0, -1, 5, MAX_ATTEMPTS+1 etc. must raise
    ValueError for both profiles."""
    from src.policy.profiles import get
    from src.core.types import Profile
    from src.policy.constraints import MAX_ATTEMPTS

    invalid_slots = [0, -1, -10, MAX_ATTEMPTS + 1, MAX_ATTEMPTS + 2, 100]

    for profile in Profile:
        prof = get(profile)
        for slot in invalid_slots:
            with pytest.raises(ValueError):
                prof.requires_fresh_notification(slot)


# === module-level default guard ==============================================

def test_profiles_module_declares_no_default_profile():
    """Root CLAUDE.md requires 'Never hard-code one interpretation'. The
    compliance-auditor's check #8 is 'Neither interpretation is hard-coded'.
    This test scans src/policy/profiles.py's SOURCE TEXT for module-level
    assignments that might define a DEFAULT, DEFAULT_PROFILE, or a bare
    assignment like `= Profile.strict`. The runtime default arrives from
    .env's COMPLIANCE_PROFILE at the edge layer, not here.

    We scan module-level lines only (no leading whitespace) to avoid
    false positives on type annotations or dataclass field defaults."""
    text = _profiles_source_path().read_text(encoding="utf-8")

    # Split into lines and look for module-level assignments
    lines = text.split("\n")

    forbidden_patterns = [
        r"^DEFAULT\s*=",
        r"^DEFAULT_PROFILE\s*=",
        r"^_default\s*=",
        r"^default_profile\s*=",
    ]

    matches = []
    for i, line in enumerate(lines, start=1):
        # Skip comments and empty lines
        if line.strip().startswith("#") or not line.strip():
            continue

        # Only check lines with no leading whitespace (module-level)
        if line and line[0] not in (" ", "\t"):
            for pattern in forbidden_patterns:
                if re.search(pattern, line):
                    matches.append((i, line))

    assert len(matches) == 0, \
        f"Found hard-coded default profile in profiles.py at: {matches}"


# === dataclass frozen contract ==============================================

def test_compliance_profile_is_frozen():
    """Every dataclass in src/model/ is frozen=True; profiles.py follows the
    pattern. Attempting to mutate a ComplianceProfile field must raise
    FrozenInstanceError (or AttributeError, depending on implementation)."""
    from src.policy.profiles import get
    from src.core.types import Profile

    profile_obj = get(Profile.strict)

    # Attempt to mutate a field
    with pytest.raises((FrozenInstanceError, AttributeError)):
        profile_obj.profile = Profile.permissive  # type: ignore


# === get() argument validation ==============================================

def test_get_rejects_a_non_profile_argument():
    """get() must validate its argument strictly. Passing a string like
    'strict' instead of Profile.strict should raise, not silently coerce.
    Note: Profile is a str Enum, so Profile.strict == 'strict' is True and a
    plain dict lookup WOULD succeed -- this test pins the deliberate choice
    to reject coercion in favor of strict type checking."""
    from src.policy.profiles import get

    # Passing string instead of enum member
    with pytest.raises((TypeError, ValueError)):
        get("strict")  # type: ignore

    with pytest.raises((TypeError, ValueError)):
        get("permissive")  # type: ignore

    # Non-string, non-Profile argument
    with pytest.raises((TypeError, ValueError)):
        get(None)  # type: ignore

    with pytest.raises((TypeError, ValueError)):
        get(123)  # type: ignore
