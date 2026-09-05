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
from src.core.types import Action, Cause, Outcome, Profile
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


# === R8, 2026-09-05: fit_gate()'s calibration pool spans slots 1-4 =========
#
# CRITICAL finding (R5 stats-review pass, DECISIONS.md, 2026-09-05): the
# calibration pool was one row per mandate, at slot 1 only -- 200 rows, 2-3
# distinct nonconformity values per class, maximum calibration confidence
# well below what a real multi-decline trajectory reaches. The fix grinds
# every calibration mandate through its own slot 2/3 (see
# eval/run.py's _calibration_rows), producing more rows and a wider score
# range without depending on the gate it is fitting.


def test_calibration_pool_now_spans_more_than_one_row_per_mandate():
    """200 calibration mandates; if every one contributed only its slot-1
    row, n_calib would be exactly 200 -- the CRITICAL finding's own
    description of the bug. Some mandates now survive to contribute a
    slot-3 and/or slot-4 row too, so n_calib must be strictly more."""
    _, kind, diag = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    assert diag["n_calib"] > 200, (
        f"n_calib={diag['n_calib']} -- calibration pool did not grow beyond "
        "one row per mandate; the slot 2/3 grind is not contributing rows"
    )


def test_calibration_score_range_widens_toward_a_multi_decline_confidence():
    """The support-mismatch half of the CRITICAL finding: the OLD pool's
    per-class scores topped out around 0.90 (p_true <= ~0.10 in the worst
    case), while a real multi-decline trajectory reaches p_true > 0.99 (see
    tests/eval/test_wontpay_channel.py). At least one class's calibration
    pool must now contain a score low enough to prove the pool has actually
    seen a belief that confident, not merely grown in row count."""
    import numpy as np

    gate, kind, _ = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    pred = gate.predictor
    assert pred.calib_scores.min() < 0.10, (
        f"minimum calibration score across all classes is "
        f"{pred.calib_scores.min()} -- no class's pool reached a "
        "multi-decline-level confidence (score < 0.10, i.e. p_true > 0.90)"
    )


def test_calibration_pool_still_meets_the_mondrian_floor_per_class():
    """The grind must not thin out any one class below what Mondrian
    conformal needs (ceil(1/alpha) - 1 = 19 at alpha=0.05) -- fit_gate()
    already raises ConformalUnderpowered if it does, so reaching a
    'conformal' kind here proves the ROW floor holds; the per-class counts
    are checked directly anyway, the same discipline the channel
    diagnostics already use."""
    import numpy as np

    gate, kind, _ = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    pred = gate.predictor
    for c_idx, cause in enumerate(run_mod.CAUSE_ORDER):
        n_c = int(np.sum(pred.calib_labels == c_idx))
        assert n_c >= 19, f"{cause}: only {n_c} calibration rows, below the floor of 19"


def test_the_row_floor_is_not_hiding_too_few_independent_mandates():
    """R8 stats-review pass, 2026-09-05: calibrate()'s floor counts ROWS, and
    _calibration_rows can contribute up to 3 per mandate -- so 19 rows could,
    in principle, come from as few as 7 independent mandates, which is a much
    weaker claim than the floor's own derivation (ceil(1/alpha) - 1
    independent calibration points) assumes. calib_units_per_class counts
    DISTINCT mandate ids instead, so this is checkable rather than assumed.
    Today's margins are wide (measured: CANT_PAY_NOW 105, CANT_PAY_EVER 48,
    WONT_PAY 47 distinct mandates) -- this test pins that the diagnostic
    exists and stays well above the row floor, so a future regression that
    quietly thins the independent-mandate count would be caught even while
    the row count still clears calibrate()'s own check."""
    _, kind, diag = run_mod.fit_gate(load_config())
    assert kind == "conformal"
    units = diag["calib_units_per_class"]
    for cause, n_units in units.items():
        assert n_units >= 19, (
            f"{cause}: only {n_units} DISTINCT calibration mandates -- the "
            "row floor may be clearing on too few independent units"
        )


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


# --- Regression test for R2 terminal-outcome fix ----------------------------
#
# R2 (2026-09-04, reports/gates.md "Post-B16 remediation gates", R2a): After
# a DEAD or OPTED_OUT outcome, the engine must collapse belief to a
# DEGENERATE posterior (observe_terminal), transition the context
# (with_terminal), and re-solve with the updated belief/context. The key
# regressions:
# 1. After a couple of INSUFFICIENT_FUNDS-shaped belief updates (~99%
#    CANT_PAY_NOW), a DEAD outcome should trigger REAUTH, not ATTEMPT.
# 2. An OPTED_OUT outcome should always trigger STOP, not None/skipped.
#
# CORRECTED, same session, before this fix landed: the first version of
# these three tests drove a REAL, uncontrolled Simulator on an arbitrary
# seed/mandate and asserted near-tautological conditions
# (`n_attempt_after_terminal == 0` with no guarantee DEAD was ever reached;
# `total_decisions > 0`, true of nearly any terminating run; `assert True`
# for RECOVERED). All three PASSED even before any fix existed -- proof
# they were not exercising the bug at all, exactly the failure mode R2a's
# gate text warns against ("proven by a test that constructs the exact
# sequence and fails against today's code"). _ScriptedSimulator below (kept
# from that version -- its shape was already right) is now actually USED to
# force the precise DEAD/OPTED_OUT/RECOVERED sequences.


class _ScriptedSimulator:
    """Minimal Simulator stand-in that returns a SCRIPTED sequence of
    outcomes regardless of slot/on_day. _run_engine_mandate calls
    `.attempt(mandate_id, slot=..., on_day=...)` always, and -- since R2b --
    `.effective_cause(mandate_id)` too, but only when a REAUTH is actually
    recorded; this is the whole interface it needs. Used to drive
    _run_engine_mandate through an exact sequence (e.g. STILL_PENDING, then
    DEAD) deterministically, rather than hoping a real Simulator seed
    happens to produce one."""

    def __init__(self, outcomes: list[Outcome], *, effective_cause: Cause = Cause.CANT_PAY_NOW):
        self.outcomes = list(outcomes)
        self.call_count = 0
        self._effective_cause = effective_cause

    def effective_cause(self, mandate_id: str) -> Cause:
        return self._effective_cause

    def attempt(self, mandate_id: str, slot: int, on_day: int):
        if self.call_count >= len(self.outcomes):
            raise AssertionError(
                f"_ScriptedSimulator exhausted: called {self.call_count + 1} "
                f"times, only {len(self.outcomes)} scripted outcome(s) given "
                f"-- the engine attempted more slots than this test scripted, "
                f"which means the terminal outcome did not actually stop the "
                f"ATTEMPT sequence"
            )
        outcome = self.outcomes[self.call_count]
        self.call_count += 1

        class _Result:
            pass

        r = _Result()
        r.outcome = outcome
        r.slot = slot
        r.on_day = on_day
        return r


def _engine_test_mandate(*, amount_paise: int = 500_000, category: str = "subscription"):
    """A SimMandate below the AFA-free limit (so REAUTH's compliance path
    never fires -- only the belief-driven inference path can, which is
    exactly what these tests are checking) and outside any household
    coupling (household_id=None)."""
    from eval.frozen.simulator import SimMandate

    return SimMandate(
        mandate_id="M_r2_test", cycle_id=1, amount_paise=amount_paise,
        ceiling_paise=amount_paise * 2, category=category,
        household_id=None, initial_cause=Cause.CANT_PAY_NOW,
    )


@pytest.fixture(scope="module")
def _real_nominal_hazard():
    """A real fitted hazard, shared across this file's R2 regression tests
    -- fitting is the expensive part; reusing a module-scoped fixture pays
    for it once rather than once per test."""
    from eval.allocator_sweep import fit_nominal_hazard_model, hazard_from_fit
    return hazard_from_fit(fit_nominal_hazard_model())


def test_dead_outcome_yields_reauth_not_attempt(_real_nominal_hazard):
    """R2a's central regression, with the exact sequence actually forced:
    one STILL_PENDING attempt (belief shifts toward CANT_PAY_NOW via the
    ordinary proxy update -- 'after a couple of INSUFFICIENT_FUNDS-shaped
    updates' from the bug's own description), then DEAD.

    Before the fix: DEAD's ordinary belief update (CARD_EXPIRED, prior only
    0.75 toward CANT_PAY_EVER) could not overcome the prior STILL_PENDING
    shift, b.dominant() stayed CANT_PAY_NOW, REAUTH's inference path was
    never entered, permitted() had no rule against ATTEMPT on a dead
    instrument, and the re-solve returned ATTEMPT again --
    n_attempt_after_terminal incremented on a mandate whose instrument the
    issuer had just confirmed dead.

    After the fix: observe_terminal(b, CANT_PAY_EVER) makes belief
    DEGENERATE regardless of what came before, with_terminal(DEAD) makes
    permitted() deny ATTEMPT outright, and REAUTH's inference-path value
    (reauth_success_prob * amount_paise - reauth_cost_paise, comfortably
    positive at this amount) beats STOP's 0.0 floor. FullSetGate is used
    deliberately so OFFER can never fire (never a singleton), isolating the
    REAUTH-vs-ATTEMPT-vs-STOP question this test is actually about."""
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init
    from src.policy.costs import load as load_costs
    from src.policy.gate import FullSetGate

    m = _engine_test_mandate()
    # effective_cause=CANT_PAY_EVER: the mandate genuinely became dead, so
    # this REAUTH is a TRUE positive under R2b's effective-cause scoring
    # too, not just a mechanically-forced one.
    sim = _ScriptedSimulator([Outcome.STILL_PENDING, Outcome.DEAD],
                             effective_cause=Cause.CANT_PAY_EVER)
    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    costs = load_costs()
    cell = run_mod.CellResult(regime="test", arm="test", profile="strict",
                              policy="test", seed=0, gate_kind="full_set")
    trace = []

    run_mod._run_engine_mandate(
        m, sim, profile=Profile.strict, hazard=_real_nominal_hazard,
        costs=costs, gate=FullSetGate(), b=b, cell=cell, trace=trace,
    )

    assert sim.call_count == 2, (
        f"expected exactly 2 attempts (STILL_PENDING, DEAD); the engine "
        f"made {sim.call_count} -- it did not stop attempting after DEAD"
    )
    assert cell.n_attempt_after_terminal == 0, (
        "R2a REGRESSION: the post-terminal re-solve returned ATTEMPT on a "
        "mandate whose instrument was just confirmed DEAD"
    )
    assert cell.n_reauth == 1, (
        f"expected the post-DEAD re-solve to choose REAUTH; cell counters: "
        f"reauth={cell.n_reauth} stop={cell.n_stop} offer={cell.n_offer} "
        f"attempt_after_terminal={cell.n_attempt_after_terminal}"
    )
    final_trace = trace[-1]
    assert final_trace.outcome is None, (
        "the final (post-terminal) decision spends no slot, so its trace "
        "entry must carry no outcome"
    )
    assert final_trace.plan.chosen_action == Action.REAUTH

    # R2b: this REAUTH is via the INFERENCE route (amount_paise=500_000 is
    # below the AFA-free limit, so requires_afa() is False), the mandate's
    # initial_cause is CANT_PAY_NOW (a genuinely wrong belief-independent
    # starting point _engine_test_mandate() sets) so it counts as false
    # against initial_cause, but effective_cause is CANT_PAY_EVER (the
    # scripted DEAD really happened) so it must NOT count as false there --
    # exactly the initial-vs-effective distinction R2b exists to draw.
    assert cell.compliance_reauth_count == 0
    assert cell.false_reauth_count == 1
    assert cell.false_reauth_inference_count == 1
    assert cell.false_reauth_count_effective == 0
    assert cell.false_reauth_inference_count_effective == 0


def test_reauth_via_compliance_path_is_not_counted_as_inference_false(_real_nominal_hazard):
    """R2b's central distinction: an above-AFA-cliff mandate's REAUTH is
    legally mandatory (clause 8(a)/8(b)) regardless of cause -- it must
    increment compliance_reauth_count and must NEVER count toward
    false_reauth_inference_count, even when initial_cause makes the
    pre-registered false_reauth_count increment too. This is exactly the
    conflation measured on the published artifact: 6,784 of 13,354 REAUTHs
    were compliance-route, and the excess of "false" over above-AFA was
    only 790 -- an order of magnitude smaller than false_reauth_count alone
    implied."""
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init
    from src.policy.costs import load as load_costs
    from src.policy.gate import FullSetGate

    # Above the Rs 15,000 AFA-free limit (1_500_000 paise), subscription
    # category (not clause-8(b)-elevated) -- requires_afa() is True here.
    #
    # Padded with 4 STILL_PENDING entries even though REAUTH fires at the
    # FIRST decision point with zero real attempts made: an un-resolved
    # decision with slots_left > 0 also triggers _run_engine_mandate's own
    # error-cost counterfactual (_counterfactual_recovers), which drives a
    # DEEPCOPY of `sim` through up to 3 further scripted attempts asking
    # "would this have recovered if we kept grinding?" -- found by running
    # this test and seeing the stub exhausted one call earlier than
    # expected, not anticipated from reading _run_engine_mandate alone.
    m = _engine_test_mandate(amount_paise=2_000_000, category="subscription")
    sim = _ScriptedSimulator(
        [Outcome.STILL_PENDING] * 4, effective_cause=Cause.CANT_PAY_NOW,
    )
    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    costs = load_costs()
    cell = run_mod.CellResult(regime="test", arm="test", profile="strict",
                              policy="test", seed=0, gate_kind="full_set")

    run_mod._run_engine_mandate(
        m, sim, profile=Profile.strict, hazard=_real_nominal_hazard,
        costs=costs, gate=FullSetGate(), b=b, cell=cell, trace=None,
    )

    assert sim.call_count == 0, (
        "an above-AFA-cliff mandate must route to REAUTH at the FIRST "
        "decision point, before any attempt is ever made -- the scripted "
        "STILL_PENDING outcome above must go unconsumed"
    )
    assert cell.n_reauth == 1, (
        f"an above-AFA-cliff mandate must route straight to REAUTH -- "
        f"cell counters: reauth={cell.n_reauth} attempt={cell.n_attempt}"
    )
    assert cell.compliance_reauth_count == 1
    assert cell.false_reauth_count == 1, "still true against the pre-registered, unredefined criterion"
    assert cell.false_reauth_inference_count == 0, (
        "a compliance-route REAUTH must NEVER count as an inference-path "
        "false positive -- it was never a belief decision to begin with"
    )


def test_opted_out_outcome_yields_stop(_real_nominal_hazard):
    """R2a's second regression: OPTED_OUT on the very first attempt.

    Before the fix: _proxy_decline_class(OPTED_OUT) is None, so the OLD
    code's `if dc is not None:` gate skipped the re-solve ENTIRELY --
    stopped_action stayed None and fell through every counting branch at
    the bottom of _run_engine_mandate, uncounted. Verified directly against
    the pre-fix code: `n_attempt=1` (the one real attempt that was made,
    whose outcome happened to be OPTED_OUT -- unrelated to the bug),
    `n_stop=0, n_offer=0, n_reauth=0, n_attempt_after_terminal=0` -- no
    decision was ever recorded for what happens after the opt-out.

    After the fix: observe_terminal(b, WONT_PAY) plus
    with_terminal(OPTED_OUT) (which sets the EXISTING opted_out field) make
    permitted() deny everything except STOP (clause 6(c), pre-existing
    rule) -- the final re-solve can only return STOP."""
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init
    from src.policy.costs import load as load_costs
    from src.policy.gate import FullSetGate

    m = _engine_test_mandate()
    sim = _ScriptedSimulator([Outcome.OPTED_OUT])
    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    costs = load_costs()
    cell = run_mod.CellResult(regime="test", arm="test", profile="strict",
                              policy="test", seed=0, gate_kind="full_set")
    trace = []

    run_mod._run_engine_mandate(
        m, sim, profile=Profile.strict, hazard=_real_nominal_hazard,
        costs=costs, gate=FullSetGate(), b=b, cell=cell, trace=trace,
    )

    assert sim.call_count == 1
    assert cell.n_stop == 1, (
        f"R2a REGRESSION: the OPTED_OUT re-solve did not record STOP -- "
        f"cell counters: stop={cell.n_stop} attempt={cell.n_attempt} "
        f"offer={cell.n_offer} reauth={cell.n_reauth} "
        f"attempt_after_terminal={cell.n_attempt_after_terminal}"
    )
    assert cell.n_attempt_after_terminal == 0
    final_trace = trace[-1]
    assert final_trace.outcome is None
    assert final_trace.plan.chosen_action == Action.STOP


def test_recovered_outcome_ends_the_loop_without_a_final_resolve(_real_nominal_hazard):
    """RECOVERED is unlike DEAD/OPTED_OUT: the cycle succeeded, so there is
    nothing left to decide. This must stay true after the fix -- confirmed
    by trace length, not just "no exception": exactly one DecisionTrace
    entry (the ATTEMPT that recovered), never a second, no-outcome entry
    for a final re-solve that should not happen."""
    from src.policy.belief import CAUSE_ORDER, REFERENCE_PRIOR, init
    from src.policy.costs import load as load_costs
    from src.policy.gate import FullSetGate

    m = _engine_test_mandate()
    sim = _ScriptedSimulator([Outcome.RECOVERED])
    b = init(dict(zip(CAUSE_ORDER, REFERENCE_PRIOR)))
    costs = load_costs()
    cell = run_mod.CellResult(regime="test", arm="test", profile="strict",
                              policy="test", seed=0, gate_kind="full_set")
    trace = []

    run_mod._run_engine_mandate(
        m, sim, profile=Profile.strict, hazard=_real_nominal_hazard,
        costs=costs, gate=FullSetGate(), b=b, cell=cell, trace=trace,
    )

    assert sim.call_count == 1
    assert len(trace) == 1, (
        f"RECOVERED must not trigger a post-terminal re-solve -- expected "
        f"exactly 1 trace entry (the recovering ATTEMPT), got {len(trace)}"
    )
    assert trace[0].outcome == Outcome.RECOVERED.name
    for field in ("n_offer", "n_reauth", "n_stop", "n_attempt_after_terminal"):
        assert getattr(cell, field) == 0, f"{field} should be untouched by a RECOVERED outcome"
