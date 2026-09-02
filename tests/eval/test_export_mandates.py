"""B14 -- the per-mandate export the dashboard drill-down renders.

The gate wants belief, chosen slot, binding constraint, conformal set and a
ledger trail, per mandate. Those five live together on exactly one object --
policy.allocator.Plan -- and eval/run.py's engine loop discarded every one of
them, keeping only counters. So the dashboard cannot be built over the
existing artifacts; something has to record the Plans the scored run made.

The rule these tests exist to enforce: RECORDING MUST NOT CHANGE THE RUN.
A trace hook that perturbs an RNG stream would silently move every published
number in reports/regimes.json, and the dashboard would then be rendering a
different experiment from the one the report describes.
`test_trace_does_not_change_cell` is the guard, and it compares the whole
CellResult, not a selected field of it.

The second rule: the exported decisions must be THE decisions, not a
re-solve. Re-running solve() outside the scored loop would be a second
experiment that happens to agree today and drift tomorrow, so the trace is
threaded through the same loop that produces the numbers.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

import pytest

from eval import export_mandates as exp
from eval import run as run_mod
from eval.frozen.simulator import load_config
from src.core.types import Action, Cause, Profile
from src.execute.keys import key_for


def _attempted(record: dict) -> bool:
    return any(d["outcome"] is not None for d in record["decisions"])


@pytest.fixture(scope="module")
def engine_bits():
    """The expensive shared setup: one hazard fit and one gate calibration,
    reused across this module exactly as run_all() reuses them across cells."""
    cfg = load_config()
    hazard = run_mod.hazard_from_fit(run_mod.fit_nominal_hazard_model())
    costs = run_mod.load_costs()
    gate, gate_kind, _ = run_mod.fit_gate(cfg)
    return cfg, hazard, costs, gate, gate_kind


# --- the recording must not change the run -----------------------------------


def test_trace_does_not_change_cell(engine_bits):
    """The whole point. Tracing records objects the loop already built; it
    must draw no randomness and take no branch. Compared field-by-field over
    the entire CellResult so a perturbation cannot hide in a column this test
    forgot to name -- `seconds` excluded, being wall-clock."""
    cfg, hazard, costs, gate, gate_kind = engine_bits
    kwargs = dict(regime="baseline", arm="nominal", profile=Profile.strict,
                  cfg=cfg, seed=0, hazard=hazard, costs=costs, gate=gate,
                  gate_kind=gate_kind)

    without = asdict(run_mod.run_engine_cell(**kwargs))
    traces: dict[str, list] = {}
    with_trace = asdict(run_mod.run_engine_cell(**kwargs, traces=traces))

    without.pop("seconds")
    with_trace.pop("seconds")
    assert with_trace == without
    assert traces, "traces requested but nothing was recorded"


def test_trace_holds_one_record_per_solve(engine_bits):
    cfg, hazard, costs, gate, gate_kind = engine_bits
    traces: dict[str, list] = {}
    cell = run_mod.run_engine_cell(
        regime="baseline", arm="nominal", profile=Profile.strict, cfg=cfg,
        seed=0, hazard=hazard, costs=costs, gate=gate, gate_kind=gate_kind,
        traces=traces,
    )
    assert len(traces) == cell.n_mandates

    for mandate_id, decisions in traces.items():
        assert decisions, f"{mandate_id} produced no decision at all"
        for d in decisions:
            assert d.plan.mandate_id == mandate_id
            assert set(d.plan.conformal_set) <= set(Cause)
            assert d.plan.solver_version
            assert d.plan.decision_sha256
            if d.plan.chosen_action is Action.ATTEMPT:
                assert len(d.plan.committed) == 1
                assert 1 <= d.plan.committed[0].slot <= 4
            else:
                assert d.plan.committed == ()

    # A traced ATTEMPT decision that resolved is an attempt the cell counted;
    # both counters come from the same loop and must agree.
    traced_attempts = sum(
        1 for ds in traces.values() for d in ds
        if d.plan.chosen_action is Action.ATTEMPT and d.outcome is not None
    )
    assert traced_attempts == cell.n_attempt


# --- the exported payload ----------------------------------------------------


@pytest.fixture(scope="module")
def records(engine_bits):
    cfg, hazard, costs, gate, gate_kind = engine_bits
    return exp.build_records(regime="baseline", arm="nominal",
                             profile=Profile.strict, seed=0, cfg=cfg,
                             hazard=hazard, costs=costs, gate=gate,
                             gate_kind=gate_kind)


def test_every_record_carries_the_five_gate_fields(records):
    """B14's gate, field by field. Named individually because 'the drill-down
    works' is not checkable and 'these five keys are populated' is."""
    assert records
    for r in records:
        for d in r["decisions"]:
            assert set(d["belief"]) == {c.value for c in Cause}
            assert abs(sum(d["belief"].values()) - 1.0) < 1e-9
            # An EMPTY set is a real conformal answer -- "no cause clears
            # alpha" -- not a missing field, so this asserts membership
            # rather than non-emptiness. One decision in the headline cell
            # produces it, and the dashboard has to render that honestly
            # instead of showing a blank cell.
            assert set(d["conformal_set"]) <= {c.value for c in Cause}
            assert "binding_constraint" in d          # None is a real answer
            assert (d["chosen_slot"] is None) == (d["action"] != "ATTEMPT")
        assert r["ledger"] == [], (
            "build_records must not synthesise a ledger; the trail comes only "
            "from rows the executor actually wrote"
        )


def test_the_wont_pay_singleton_is_the_unreachable_one(records):
    """B13's handoff recorded that the conformal set is 'near-always all three
    causes'. At decision level that is not what happens: roughly a third of
    decisions in this cell are singletons. What is genuinely unreachable is
    the {WONT_PAY} singleton specifically -- cause_map pins P(WONT_PAY) at
    0.10 under both symbols the proxy alphabet can emit -- and that, not a
    degenerate gate, is why OFFER is zero.

    This test pins the distinction so the dashboard cannot restate the
    coarser claim. It fails the day the off-ramp becomes reachable, which is
    exactly when the copy on that panel has to change.
    """
    sets = [tuple(d["conformal_set"]) for r in records for d in r["decisions"]]
    singletons = [s for s in sets if len(s) == 1]

    assert len(singletons) > len(sets) // 10, (
        "the gate produces singletons; if this fails the 'near-always all "
        "three' description has become true and the panel copy must change"
    )
    assert ("WONT_PAY",) not in singletons, (
        "the {WONT_PAY} singleton fired -- the off-ramp is now reachable and "
        "OFFER = 0 is no longer arithmetic"
    )
    assert not any(d["action"] == "OFFER" for r in records for d in r["decisions"])


def test_exported_decisions_rehash_to_their_own_digest(records):
    """The completeness check. allocator._plan() hashes the decision it made;
    if the export dropped a field, reordered the belief or rounded a
    probability, rebuilding that payload from the ARTIFACT stops reproducing
    the digest. A drill-down whose numbers do not rehash to the decision they
    claim to explain is decorative."""
    for r in records:
        for d in r["decisions"]:
            assert exp.recompute_decision_sha256(r, d) == d["decision_sha256"]


def test_ground_truth_is_quarantined(records):
    """initial_cause and household_id are unobservable -- simulator.py says so
    itself. The true cause is useful to a REVIEWER (it is how a false REAUTH
    is visible at all), so it is exported, but only under `ground_truth`,
    never anywhere the engine's own view is rendered from. A dashboard that
    mixed them would be showing the engine knowing something it cannot."""
    for r in records:
        assert "true_cause" in r["ground_truth"]
        assert "household_id" not in json.dumps(r), (
            "household_id must not be exported at all -- a payment aggregator "
            "has no way to know which mandates share a bank account"
        )
        engine_view = json.dumps({k: v for k, v in r.items() if k != "ground_truth"})
        assert "true_cause" not in engine_view


def test_export_is_deterministic(engine_bits, records):
    cfg, hazard, costs, gate, gate_kind = engine_bits
    again = exp.build_records(regime="baseline", arm="nominal",
                              profile=Profile.strict, seed=0, cfg=cfg,
                              hazard=hazard, costs=costs, gate=gate,
                              gate_kind=gate_kind)
    assert json.dumps(again, sort_keys=True) == json.dumps(records, sort_keys=True)


# --- the ledger trail is written by the real executor ------------------------


def test_replay_writes_real_ledger_rows(pg_schema, records):
    """The trail is not hand-built JSON. It is written by commit() and
    execute() into the real schema, so the idempotency keys, the write
    ordering and the plan FK are the production ones. Only the provider's
    ANSWER is simulated -- which is what the simulator already legitimately
    is (see PROXY_SOURCE_VERSION in eval/allocator_sweep.py)."""
    conn = pg_schema.conn
    subset = [copy.deepcopy(r) for r in records if _attempted(r)][:5]
    assert subset, "no record in the export made an attempt"

    exp.replay_to_ledger(conn, subset)

    for r in subset:
        assert r["ledger"], f"{r['mandate_id']} attempted but wrote no ledger row"
        known = {d["decision_sha256"] for d in r["decisions"]}
        for row in r["ledger"]:
            # ledger.decision_sha256 REFERENCES plan (decision_sha256) -- the
            # FK is real, so every row points at a decision the engine made.
            assert row["decision_sha256"] in known
            if row["state"] != "INTENT":
                continue
            assert row["idempotency_key"] == key_for(
                mandate_id=r["mandate_id"],
                cycle_id=r["cycle_id"],
                attempt_index=row["attempt_index"],
                generation=0,
                action=row["action"],
                amount_paise=row["amount_paise"],
            )

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM plan")
        assert cur.fetchone()[0] > 0
        cur.execute(
            "SELECT idempotency_key FROM ledger WHERE state = 'INTENT' "
            "ORDER BY ledger_id"
        )
        keys = [k for (k,) in cur.fetchall()]
    assert len(keys) == len(set(keys)), "ledger_intent_once admitted a duplicate"


def test_an_opt_out_is_not_recorded_as_a_decline(pg_schema, records):
    """Clause 6(c): OPTED_OUT is a distinct outcome, never folded into
    'declined'. The export renders it as a pre-notification revocation, so
    the ledger must show an abort -- never a decline_class, and never a
    debit."""
    conn = pg_schema.conn
    subset = [
        copy.deepcopy(r) for r in records
        if any(d["outcome"] == "OPTED_OUT" for d in r["decisions"])
    ][:3]
    if not subset:
        pytest.skip("no opt-out in this cell")

    exp.replay_to_ledger(conn, subset)

    for r in subset:
        assert r["ledger_note"], "an opt-out rendering must disclose itself"
        terminal = [row for row in r["ledger"] if row["state"] in ("RESULT", "FAILED")]
        assert terminal
        last = terminal[-1]
        assert last["state"] == "FAILED"
        assert last["reason"] == "ABORTED_LIFECYCLE_REVOKED"
        assert last["decline_class"] is None
        assert last["outcome"] != "RECOVERED"


def test_a_post_terminal_resolve_gets_a_plan_row_but_no_schedule_row(pg_schema, records):
    """The engine's disclosed defect (B7/B8): asked 'the instrument is dead,
    now what?', the allocator answers ATTEMPT. 16 of those in this cell.

    The decision is real and belongs in the drill-down, so it gets a plan
    row. It was never committed and never sent, so it must NOT get a
    committed_schedule row -- that table means 'committed >=24h ahead'
    (clause 6a), and a row there would put a debit in the audit trail that
    never happened. Showing the defect is the point; inventing a debit to
    show it would not be."""
    conn = pg_schema.conn
    subset = [
        copy.deepcopy(r) for r in records
        if any(d["action"] == "ATTEMPT" and d["outcome"] is None
               for d in r["decisions"])
    ][:3]
    if not subset:
        pytest.skip("no post-terminal re-solve in this cell")

    exp.replay_to_ledger(conn, subset)

    for r in subset:
        unexecuted = [d for d in r["decisions"]
                      if d["action"] == "ATTEMPT" and d["outcome"] is None]
        for d in unexecuted:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM plan WHERE decision_sha256 = %s",
                            (d["decision_sha256"],))
                assert cur.fetchone()[0] == 1
                cur.execute(
                    "SELECT count(*) FROM committed_schedule WHERE decision_sha256 = %s",
                    (d["decision_sha256"],),
                )
                assert cur.fetchone()[0] == 0
            assert not any(row["decision_sha256"] == d["decision_sha256"]
                           for row in r["ledger"])


def test_replay_is_idempotent(pg_schema, records):
    """Re-running the export against a database that already has it must not
    write a second INTENT row -- that is ledger_intent_once's whole job, and
    an exporter that trips it would be reporting a double-charge that never
    happened."""
    conn = pg_schema.conn
    subset = [copy.deepcopy(r) for r in records if _attempted(r)][:3]

    exp.replay_to_ledger(conn, subset)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ledger")
        first = cur.fetchone()[0]

    exp.replay_to_ledger(conn, subset)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ledger")
        assert cur.fetchone()[0] == first
