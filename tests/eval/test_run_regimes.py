"""B13 -- the batch driver and the report renderer.

These are the tests that keep the block's gate honest: that the report cannot
print a coverage claim the run did not earn, that a zero-attempt mandate is
scored rather than crashing, and that the counterfactual used for the error
costs is the real one and not a re-seeded approximation.
"""
from __future__ import annotations

import copy
import json

import pytest

from eval import regimes as R
from eval import report as report_mod
from eval import run as run_mod
from eval.frozen.simulator import Simulator, load_config
from src.core.types import Cause, Outcome, Profile
from src.policy.belief import Belief
from src.policy.gate import ConformalCauseGate, ConformalGate, FullSetGate


# --- the gate ----------------------------------------------------------------


def test_conformal_cause_gate_satisfies_the_protocol():
    gate, kind, _ = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    assert isinstance(gate, ConformalGate)


def test_gate_is_deterministic_for_an_identical_belief():
    """An off-ramp that fires on a coin flip is not a gate. The smoothed
    conformal p-value draws its randomness from the belief itself, so the
    same belief must always produce the same set."""
    gate, kind, _ = run_mod.fit_gate(load_config())
    b = Belief(probs=(0.1, 0.1, 0.8), provenance="test")
    assert gate.pred_set(b) == gate.pred_set(b)


def test_gate_calibration_is_disjoint_from_every_reported_seed():
    """conformal.calibrate() must never see a row it will later be scored
    on. The calibration draw is namespaced by its own seed precisely so this
    is checkable rather than assumed."""
    gate, kind, _ = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    calib_ids = gate.predictor.fit_group_ids
    assert all(i.startswith(f"calib{run_mod.CALIB_SEED}:") for i in calib_ids)
    sim = Simulator("nominal", seed=0)
    reported = frozenset(m.mandate_id for m in sim.mandates)
    assert not (calib_ids & reported)


# --- scoring a mandate the engine never attempted ----------------------------


def test_zero_attempt_mandate_is_preserved_and_still_pending():
    """The frozen scorer raises on an empty attempt list -- reasonably, the
    ladder cannot produce one. REAUTH or OFFER at the first decision point
    produces exactly that, and it is the whole point of having those
    actions."""
    sim = Simulator("nominal", seed=0)
    m = sim.mandates[0]
    r = run_mod._result_for(m, [])
    assert r.final_outcome == Outcome.STILL_PENDING
    assert r.preserved is True
    assert r.amount_recovered_paise == 0
    assert r.attempts == ()


# --- the counterfactual ------------------------------------------------------

def test_counterfactual_runs_on_a_copy_and_does_not_touch_the_real_sim():
    """The error costs are only meaningful if asking the counterfactual
    question cannot change the run that is being measured."""
    sim = Simulator("nominal", seed=3)
    mid = sim.mandates[0].mandate_id
    sim.attempt(mid, slot=2, on_day=1)
    shadow = copy.deepcopy(sim)
    run_mod._counterfactual_recovers(shadow, mid, from_slot=3, last_day=1)
    # the real sim still expects slot 3 -- the copy consumed the slots, not it
    r = sim.attempt(mid, slot=3, on_day=2)
    assert r.slot == 3


def test_counterfactual_respects_the_npci_budget():
    """Four attempts, ever. The counterfactual is 'keep grinding', not
    'grind forever'."""
    sim = Simulator("nominal", seed=5)
    mid = sim.mandates[0].mandate_id
    sim.attempt(mid, slot=2, on_day=1)
    shadow = copy.deepcopy(sim)
    run_mod._counterfactual_recovers(shadow, mid, from_slot=3, last_day=1)
    with pytest.raises(ValueError, match="slot must be 2, 3, or 4"):
        shadow.attempt(mid, slot=5, on_day=99)


# --- the driver end to end ---------------------------------------------------


@pytest.fixture(scope="module")
def small_payload():
    return run_mod.run_all(
        regime_names=["baseline", "retry_storm"],
        arms=["nominal"],
        profiles=[Profile.strict, Profile.permissive],
        seed=0,
        verbose=False,
    )


def test_driver_produces_a_cell_per_policy_regime_profile(small_payload):
    # 2 regimes x 1 arm x 2 profiles x 2 policies
    assert len(small_payload["cells"]) == 8
    assert {c["policy"] for c in small_payload["cells"]} == {"ladder", "engine"}


def test_no_constraint_violations(small_payload):
    viol = [v for c in small_payload["cells"] for v in c["violations"]]
    assert viol == [], viol


def test_every_cell_reports_all_three_bars(small_payload):
    """protocol.md's headline is three bars, never one. A cell missing any of
    them would let the report fall back to recovery-only."""
    for c in small_payload["cells"]:
        assert c["n_mandates"] > 0
        assert c["recovered_paise"] >= 0
        assert c["attempts_spent"] >= 0
        assert 0 <= c["mandates_preserved"] <= c["n_mandates"]


def test_ladder_spends_more_attempts_than_the_engine(small_payload):
    """Not a tuning target -- if this ever flips, the thesis is wrong and the
    report should say so rather than the test passing quietly."""
    by = {}
    for c in small_payload["cells"]:
        by.setdefault((c["regime"], c["arm"], c["profile"]), {})[c["policy"]] = c
    for key, v in by.items():
        assert v["engine"]["attempts_spent"] < v["ladder"]["attempts_spent"], key


def test_money_is_integer_paise_everywhere(small_payload):
    """CLAUDE.md invariant 2. A float that reached a money field would
    survive the JSON round-trip and silently poison the report."""
    for c in small_payload["cells"]:
        for f in ("recovered_paise", "missed_recovery_paise", "false_offramp_paise"):
            assert isinstance(c[f], int), (c["regime"], f, type(c[f]))


def test_artifact_round_trips_through_json(small_payload):
    assert json.loads(json.dumps(small_payload)) == small_payload


# --- the report --------------------------------------------------------------


def test_report_refuses_a_coverage_claim_the_run_did_not_earn(small_payload):
    """The 95%-coverage claim may appear ONLY where the real conformal gate
    was live. Forge a FullSetGate artifact and check the number is withheld."""
    forged = copy.deepcopy(small_payload)
    for c in forged["cells"]:
        c["gate_kind"] = "full_set" if c["policy"] == "engine" else "n/a"
        c["coverage_marginal"] = 1.0
    rendered = "\n".join(report_mod._coverage_table(forged))
    assert "n/a (stub gate)" in rendered
    assert "1.000" not in rendered


def test_report_prints_coverage_when_the_real_gate_was_live(small_payload):
    rendered = "\n".join(report_mod._coverage_table(small_payload))
    assert "conformal" in rendered
    assert "stub gate" not in rendered


def test_report_renders_without_figures(small_payload):
    md = report_mod.render(small_payload, figures=False)
    for required in ("recovered", "attempts", "preserved",
                     "missed recovery", "false off-ramp", "Where we lose"):
        assert required in md, required


def test_report_never_shows_recovery_without_the_other_two_bars(small_payload):
    """The one formatting rule that is actually a design rule."""
    header = report_mod._three_bar_table(small_payload, "strict")[0]
    assert "recovered" in header and "attempts" in header and "preserved" in header
