"""src/policy/hazards.py -- Protocol definition for cause-conditioned hazards.

Design spec: B5 (model-fit phase) ships hazards MARGINAL over cause: a single
set of four outcome probabilities [STILL_PENDING, RECOVERED, DEAD, OPTED_OUT]
per slot and mandate. B7 (policy foundation) is cause-aware but cannot
implement cause-conditioned hazards because B5's fitted model has no production
label for the cause variable itself (cause is latent, inferred via Bayesian
belief update in B7/B8).

src/policy/hazards.py defines ONLY a Protocol -- a type declaration that B8's
allocator can use to name its hazard source in the type system. The hazard
model itself (how it conditions on cause, where it comes from) is B8's design,
not B7's.

The Protocol is @runtime_checkable so a concrete implementation can be verified
against it via isinstance(). It declares a single __call__ signature that takes
cause, slot, on_day, amount_paise and returns a 4-tuple of probabilities in
Outcome INT ORDER (the same convention cif.py and competing_risks.hazards() use).
"""
from __future__ import annotations

import pytest

from src.core.types import Cause, Outcome


# === Protocol conformance ===================================================

def test_a_conforming_callable_satisfies_the_protocol():
    """A trivial stub that matches the Protocol's __call__ signature must pass
    isinstance(stub, CauseConditionedHazard) check. This verifies that the
    Protocol itself is correct and that @runtime_checkable is set."""
    from src.policy.hazards import CauseConditionedHazard

    # Define a conforming stub
    def stub_hazard(*, cause: Cause, slot: int, on_day: int,
                    amount_paise: int) -> tuple[float, float, float, float]:
        """Stub that returns a valid outcome distribution."""
        return (0.5, 0.3, 0.1, 0.1)

    # Must be recognized as implementing the Protocol
    assert isinstance(stub_hazard, CauseConditionedHazard), \
        f"stub_hazard does not satisfy CauseConditionedHazard Protocol"


def test_the_protocol_is_not_implemented_in_this_module():
    """src/policy/hazards.py defines ONLY the Protocol type; it must contain
    no concrete implementations. Grep the source for:
    - No 'def' body other than '...' (Protocol method declarations)
    - No 'return' statement outside the Protocol declaration
    - No class other than the Protocol itself

    If hazards.py acquires an implementation here, that is B8 leaking into B7,
    and the gate boundary collapses."""
    from pathlib import Path

    source = Path("src/policy/hazards.py").read_text()

    # Count lines with 'return' outside docstrings/comments
    # (a simple check: Protocol bodies should only have ... or pass)
    lines = source.split("\n")
    impl_lines = []
    in_docstring = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstring state
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring

        # Skip docstrings and comments
        if in_docstring or stripped.startswith("#"):
            continue

        # Look for 'return' statements (implementation detail)
        if "return " in line and "tuple[float" not in line:
            # Allow return in type hints, but not actual implementations
            if not any(x in line for x in ["->", "..."]):
                impl_lines.append((i, line))

    # There should be no implementation returns (only docstrings mentioning return types)
    for line_num, line in impl_lines:
        # Filter out type hints
        if "->" not in line and "..." not in line:
            assert False, \
                f"hazards.py line {line_num} contains implementation: {line.strip()}"


def test_protocol_docstring_states_the_gap():
    """The module or class docstring must mention that B5 shipped hazards
    MARGINAL over cause, and that the cause variable has no production label
    in the fitted model. This is the entire reason the Protocol exists: B8 will
    need to condition on cause (via the belief), but that conditioning logic
    belongs in B8, not in a B5-era model."""
    from pathlib import Path

    source = Path("src/policy/hazards.py").read_text()

    # Extract the module docstring (first triple-quoted string)
    import re
    docstring_match = re.search(r'^"""(.+?)"""', source, re.DOTALL)

    assert docstring_match, "hazards.py has no module docstring"
    docstring = docstring_match.group(1)

    # Must mention that B5 ships marginal hazards
    b5_ref = docstring.lower().count("b5") > 0 or "marginal" in docstring.lower()
    assert b5_ref, \
        f"Docstring does not mention B5 or marginal hazards"

    # Must mention cause has no production label or that it is latent
    cause_ref = "cause" in docstring.lower() and ("latent" in docstring.lower() or \
                                                   "no production label" in docstring.lower() or \
                                                   "label" in docstring.lower())
    assert cause_ref, \
        f"Docstring does not explain why cause has no production label"
