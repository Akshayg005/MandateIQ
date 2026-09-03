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


def test_gate_refuses_to_answer_without_a_bound_key():
    """The smoothing key MUST be a per-decision id. Deriving it from the
    belief -- the original implementation -- makes the tie-breaking draw a
    deterministic function of the score, which destroys the coverage
    guarantee (stats-reviewer, 2026-08-31). An unbound gate must fail loudly
    rather than silently fall back to anything belief-derived."""
    gate, kind, _ = run_mod.fit_gate(load_config())
    b = Belief(probs=(0.1, 0.1, 0.8), provenance="test")
    with pytest.raises(ValueError, match="without a bound key"):
        gate.pred_set(b)


def test_gate_is_deterministic_for_one_decision_point():
    """Determinism is still required -- an off-ramp that fires on a coin
    flip is not a gate -- but it is keyed per decision, not per belief."""
    gate, _, _ = run_mod.fit_gate(load_config())
    b = Belief(probs=(0.1, 0.1, 0.8), provenance="test")
    g = gate.bind("M0001:C1:s2")
    assert g.pred_set(b) == g.pred_set(b)


def test_gate_randomisation_is_not_a_function_of_the_belief():
    """The regression test for the bug itself: two DIFFERENT decision points
    holding the SAME belief must be able to receive different sets. If they
    cannot, the smoothing draw is pinned to the score again and the coverage
    number is an artifact of a hash.

    The belief used is (0.8, 0.1, 0.1) -- the one the harness actually
    produces after a slot-1 INSUFFICIENT_FUNDS. That matters: smoothing only
    bites on TIES, and this belief's WONT_PAY score of 0.90 lands exactly on
    the single atom that is the whole WONT_PAY calibration pool, which is
    precisely the degenerate case where the old belief-derived key reduced
    the p-value to a constant."""
    gate, _, _ = run_mod.fit_gate(load_config())
    b = Belief(probs=(0.8, 0.1, 0.1), provenance="test")
    sets = {gate.bind(f"M{i:04d}:C1:s2").pred_set(b) for i in range(200)}
    assert len(sets) > 1, (
        "every key produced the same prediction set for one belief -- the "
        "smoothing draw is not varying per row"
    )


def test_gate_can_return_the_wont_pay_singleton():
    """Positive control. The off-ramp never fires in the eval, and the report
    explains that as a property of the proxy decline alphabet rather than of
    the gate. That explanation is only honest if the gate CAN fire when the
    evidence warrants it -- otherwise "never fires" might just be a broken
    gate. Calibrate on a population where the three causes are well separated
    and check that a confident WONT_PAY belief yields the singleton."""
    import numpy as np

    from src.model import conformal

    rng = np.random.default_rng(0)
    probs, y, ids = [], [], []
    for i in range(300):
        c = i % 3
        row = np.full(3, 0.02)
        row[c] = 0.96
        row = row + rng.uniform(0, 0.01, 3)
        probs.append(row / row.sum())
        y.append(c)
        ids.append(f"synthetic:{i}")
    predictor = conformal.calibrate(
        scores=conformal.lac_scores(np.asarray(probs)),
        y=np.asarray(y),
        labels=run_mod.CAUSE_ORDER,
        row_group_ids=ids,
        provenance="calib_conf",
    )
    gate = ConformalCauseGate(predictor)
    wont = Belief(probs=(0.02, 0.02, 0.96), provenance="test")
    sets = [gate.bind(f"k{i}").pred_set(wont) for i in range(50)]
    assert frozenset({Cause.WONT_PAY}) in sets, (
        f"gate never produced the WONT_PAY singleton on a 0.96-confident "
        f"WONT_PAY belief; got {set(sets)}"
    )


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
    # 2 regimes x 1 arm x 2 profiles x 4 policies
    assert len(small_payload["cells"]) == 16
    assert {c["policy"] for c in small_payload["cells"]} == {
        "ladder", "engine", "null", "one_shot"
    }


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


def test_reference_policies_are_present_in_every_cell_group(small_payload):
    """The test that used to live here asserted the engine ALWAYS spends
    fewer attempts than the ladder. payments-domain correctly called that
    out: it pins the confound by test. Every metric in this report is
    monotonically decreasing in attempt count, so asserting the engine
    attempts less is asserting that it must appear to win -- the thing under
    investigation, not an invariant.

    What actually needs guarding is that the reference policies a reader
    needs in order to discount that effect are always emitted."""
    by = {}
    for c in small_payload["cells"]:
        by.setdefault((c["regime"], c["arm"], c["profile"]), {})[c["policy"]] = c
    for key, v in by.items():
        assert set(v) == {"ladder", "engine", "null", "one_shot"}, key


def test_null_policy_preserves_every_mandate_and_spends_nothing(small_payload):
    """The bound that makes the preserved bar interpretable: DEAD and
    OPTED_OUT are reachable only through attempt(), so never attempting
    preserves everything. If this fails, `preserved` no longer means what
    protocol.md says it means."""
    for c in small_payload["cells"]:
        if c["policy"] != "null":
            continue
        assert c["attempts_spent"] == 0
        assert c["recovered_paise"] == 0
        assert c["mandates_preserved"] == c["n_mandates"]


def test_one_shot_spends_exactly_one_attempt_per_mandate(small_payload):
    for c in small_payload["cells"]:
        if c["policy"] == "one_shot":
            assert c["attempts_spent"] == c["n_mandates"]


def test_false_reauth_is_measured_not_assumed(small_payload):
    """issuer_outage pre-registered false-REAUTH as its own falsification
    criterion. A criterion that is never computed is not a criterion."""
    eng = [c for c in small_payload["cells"] if c["policy"] == "engine"]
    assert eng
    for c in eng:
        assert c["false_reauth_count"] <= c["n_reauth"]
    assert any(c["n_reauth"] > 0 for c in eng)


def test_coverage_is_scored_over_actual_gate_queries(small_payload):
    """Coverage was previously replayed over the 200 slot-1 beliefs only,
    which both missed the concentrated post-update queries where the gate
    emits singletons AND ignored arm/profile, printing six numbers as
    thirty-two. It must now reflect every query the gate actually received."""
    for c in small_payload["cells"]:
        if c["policy"] != "engine" or c["gate_kind"] != "conformal":
            continue
        assert c["coverage_n"] > c["n_mandates"], (
            "coverage sample is no larger than one row per mandate -- it is "
            "still measuring slot 1 only"
        )


def test_money_is_integer_paise_everywhere(small_payload):
    """CLAUDE.md invariant 2. A float that reached a money field would
    survive the JSON round-trip and silently poison the report."""
    for c in small_payload["cells"]:
        for f in ("recovered_paise", "missed_recovery_paise", "false_offramp_paise"):
            assert isinstance(c[f], int), (c["regime"], f, type(c[f]))


def test_artifact_round_trips_through_json(small_payload):
    assert json.loads(json.dumps(small_payload)) == small_payload


# --- the artifact must be byte-reproducible ----------------------------------
#
# B13's gate is "every number reproducible by one command", and it was
# verified by deleting the artifact and re-running: identical numbers. True
# of every value that carries meaning -- but each cell also carried a
# `seconds` wall-clock timing, so a reader who checks the claim by comparing
# hashes rather than by reading numbers gets a mismatch and has no way to
# tell a timing jitter from a real divergence. Nothing reads the field. It is
# measured in memory and excluded from the artifact.


def test_no_wall_clock_field_reaches_the_artifact(small_payload):
    for c in small_payload["cells"]:
        assert "seconds" not in c, (
            "a wall-clock field in the artifact makes it non-reproducible "
            "byte-for-byte; see _serialise_cell in eval/run.py"
        )


def test_every_other_cell_field_survives_serialisation(small_payload):
    """The exclusion must be exactly one named field. A blanket filter that
    silently dropped a metric would make the artifact quietly incomplete."""
    from dataclasses import fields

    expected = {f.name for f in fields(run_mod.CellResult)} - run_mod.UNSERIALISED_CELL_FIELDS
    assert run_mod.UNSERIALISED_CELL_FIELDS == {"seconds"}
    for c in small_payload["cells"]:
        assert set(c) == expected


def test_two_runs_of_the_same_seed_serialise_identically():
    """The actual claim, checked directly rather than by proxy: the same
    command twice produces the same bytes."""
    kwargs = dict(regime_names=["baseline"], arms=["nominal"],
                  profiles=[Profile.strict], seed=0, verbose=False)
    first = json.dumps(run_mod.run_all(**kwargs), indent=2)
    second = json.dumps(run_mod.run_all(**kwargs), indent=2)
    assert first == second


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
                     "would have paid", "false off-ramp", "false REAUTH",
                     "Where we lose"):
        assert required in md, required


def test_report_never_shows_recovery_without_the_other_two_bars(small_payload):
    """The one formatting rule that is actually a design rule."""
    header = report_mod._three_bar_table(small_payload, "strict")[0]
    assert "recovered" in header and "attempts" in header and "preserved" in header


# --- multi-seed aggregation ---------------------------------------------------


@pytest.fixture(scope="module")
def two_seed_payload():
    return run_mod.run_all(
        regime_names=["baseline"],
        arms=["nominal"],
        profiles=[Profile.strict],
        seed=0,
        seeds=[0, 1],
        verbose=False,
    )


def test_seeds_multiply_the_cells_and_are_recorded(two_seed_payload):
    assert two_seed_payload["seeds"] == [0, 1]
    # 1 regime x 1 arm x 1 profile x 4 policies x 2 seeds
    assert len(two_seed_payload["cells"]) == 8
    assert {c["seed"] for c in two_seed_payload["cells"]} == {0, 1}


def test_report_averages_across_seeds_and_keeps_money_integral(two_seed_payload):
    """A mean over seeds is a statistic, not a ledger entry -- but invariant 2
    still has to hold of every value the report can emit, so the merged money
    fields must round to whole paise rather than carry a float."""
    pairs = report_mod._paired(two_seed_payload, "strict")
    merged = pairs[("baseline", "nominal")]["engine"]
    assert merged["n_seeds"] == 2
    assert isinstance(merged["recovered_paise"], int)
    assert isinstance(merged["missed_recovery_paise"], int)
    assert merged["recovered_paise__min"] <= merged["recovered_paise"] <= merged["recovered_paise__max"]


def test_single_seed_artifact_still_renders(small_payload):
    """The merge path must not require the seeds key -- a schema-1 artifact
    from before the sweep existed still has to render."""
    md = report_mod.render(small_payload, figures=False)
    assert "Single seed, no error bar" in md


def test_sign_test_counts_per_seed_not_per_averaged_cell(two_seed_payload):
    """The whole point of the sign test is that it can disagree with the
    mean. It must therefore count (regime, arm, profile, SEED) groups."""
    w, l, t = report_mod._seed_win_counts(
        two_seed_payload, "null", "engine", "mandates_preserved"
    )
    # null preserves every mandate, so it wins every seed-level comparison
    assert w == 2 and l == 0


def test_null_beats_everything_on_the_preserved_bar(two_seed_payload):
    """The bound that makes the preserved bar interpretable, asserted rather
    than left to prose: no policy can preserve more than never attempting."""
    for c in two_seed_payload["cells"]:
        if c["policy"] == "null":
            continue
        peer = [x for x in two_seed_payload["cells"]
                if x["policy"] == "null" and x["seed"] == c["seed"]][0]
        assert c["mandates_preserved"] <= peer["mandates_preserved"]
