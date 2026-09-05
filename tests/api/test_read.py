"""src/api/read.py -- R6's three read endpoints, against a LIVE schema.

R6's gate (reports/gates.md, "Post-B16 remediation gates"): `/plan/{id}`,
`/ledger/{id}` and `/decision/{sha}` return real rows from a live schema,
covered by tests.

**Every populated case here is seeded by running the real R4
`plan_cycle()` / `run_due()` path**, never by hand-writing rows. An
endpoint proven against hand-written fixtures proves it can read a shape
someone imagined; this proves it reads what the engine actually writes. It
is also how the `conformal_set` encoding mismatch these endpoints have to
reconcile stays honest: `commit()` writes a sorted comma-joined STRING,
the JSON exports emit a LIST, and `"".split(",")` is `[""]` -- so the
empty set needs explicit handling, and a hand-seeded row would have hidden
which convention was real.

The fixture pattern is promoted from tests/ingest/test_webhook.py (where
`client` was file-local). tests/ingest/test_deps.py exists BECAUSE an
override once left the real `get_conn` untested and a missing
`autocommit=True` silently discarded every production write while
returning HTTP 200 (POSTMORTEM Incident 2). Read-only endpoints cannot
have that failure mode -- there is nothing to commit -- but the lesson is
recorded here rather than lost with the copied fixture.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.core import clock
from src.core.types import MandateState, Profile
from src.execute.cycle import plan_cycle, run_due
from src.ledger.store import record_lifecycle_event
from src.policy.costs import PolicyCosts

CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

_COSTS = PolicyCosts(
    attempt_cost_paise=50,
    mandate_ltv_paise=180_000,
    reauth_cost_paise=200,
    reauth_success_prob=0.35,
    quiet_hours_start=21,
    quiet_hours_end=8,
    max_contacts_per_cycle=4,
)


def _flat_hazard(p_pending, p_rec, p_dead, p_opt):
    def h(*, slot: int, on_day: int, amount_paise: int):
        return (p_pending, p_rec, p_dead, p_opt)
    return h


# The same hazard tests/execute/test_cycle.py uses to get ATTEMPT out of
# solve() for an otherwise-unconstrained context.
_ATTEMPT_HAZARD = _flat_hazard(0.4, 0.45, 0.1, 0.05)
# Terrible odds and a high opt-out risk: STOP, so no committed_schedule row
# is written and the plan's chosen_action must be DERIVED, not read.
_STOP_HAZARD = _flat_hazard(0.1, 0.01, 0.1, 0.79)


class _FakeClient:
    def __init__(self, charge_response=None):
        self.calls: list[tuple] = []
        self._charge_response = charge_response or {"id": "pay_ok", "status": "captured"}

    def create_order(self, *, amount_paise, receipt, notes):
        raise AssertionError("execute() must call charge(), never create_order()")

    def charge(self, *, amount_paise, receipt, notes):
        self.calls.append(("charge", amount_paise, receipt, notes))
        return self._charge_response

    def pause_subscription(self, subscription_id):
        raise AssertionError("execute() must never call pause_subscription()")

    def find_by_receipt(self, receipt):
        raise AssertionError("execute() must never call find_by_receipt()")


@pytest.fixture(autouse=True)
def _reset_frozen_clock():
    clock.set_frozen(None)
    yield
    clock.set_frozen(None)


@pytest.fixture
def client(pg_schema):
    """Promoted from tests/ingest/test_webhook.py -- same override, same
    reason: the endpoints must run against the test's isolated schema."""
    from src.ingest.app import app
    from src.ingest.deps import get_conn

    app.dependency_overrides[get_conn] = lambda: pg_schema.conn
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _seed_mandate(conn, mandate_id, *, amount_paise=50_000,
                  ceiling_paise=200_000, category="subscription"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO mandate (mandate_id, amount_paise, ceiling_paise, category) "
            "VALUES (%s, %s, %s, %s)",
            (mandate_id, amount_paise, ceiling_paise, category),
        )
    record_lifecycle_event(
        conn, event_id=f"evt-created-{mandate_id}", mandate_id=mandate_id,
        state=MandateState.ACTIVE.value, source="INTERNAL",
        effective_at=CYCLE_START - timedelta(days=1),
    )


def _run_real_cycle(conn, mandate_id, *, hazard=_ATTEMPT_HAZARD, execute=True):
    """Seed by running the ENGINE, not by hand-writing rows."""
    _seed_mandate(conn, mandate_id)
    clock.set_frozen(CYCLE_START)
    committed = plan_cycle(conn, cycle_id=1, cycle_start=CYCLE_START,
                           hazard=hazard, costs=_COSTS, profile=Profile.strict)
    if committed and execute:
        clock.set_frozen(committed[0].scheduled_for)
        run_due(conn, _FakeClient(), costs=_COSTS, owner="worker-a")
    return committed


# === /ledger/{mandate_id} ===================================================

def test_ledger_returns_the_rows_the_engine_actually_wrote(client, pg_schema):
    _run_real_cycle(pg_schema.conn, "M-API-LEDGER")

    r = client.get("/ledger/M-API-LEDGER")
    assert r.status_code == 200
    body = r.json()
    assert body["mandate_id"] == "M-API-LEDGER"
    assert [row["state"] for row in body["rows"]] == ["INTENT", "SENT", "RESULT"]
    assert body["rows"][-1]["outcome"] == "RECOVERED"


def test_ledger_404s_on_an_unknown_mandate(client):
    r = client.get("/ledger/M-NOT-A-MANDATE")
    assert r.status_code == 404
    assert "M-NOT-A-MANDATE" in r.json()["detail"]


def test_ledger_emits_both_paise_and_a_formatted_amount(client, pg_schema):
    """`_row_dict`'s existing convention, reused rather than reinvented:
    the integer is the truth, the string is for humans, and only
    src/core/money.py may produce the string (invariant 2)."""
    from src.core import money

    _run_real_cycle(pg_schema.conn, "M-API-MONEY")
    row = client.get("/ledger/M-API-MONEY").json()["rows"][0]

    assert isinstance(row["amount_paise"], int)
    assert row["amount"] == money.fmt(row["amount_paise"])


# === /plan/{mandate_id} =====================================================

def test_plan_returns_every_plan_for_the_mandate(client, pg_schema):
    committed = _run_real_cycle(pg_schema.conn, "M-API-PLAN")

    body = client.get("/plan/M-API-PLAN").json()
    assert body["mandate_id"] == "M-API-PLAN"
    assert len(body["plans"]) == 1
    plan = body["plans"][0]
    assert plan["decision_sha256"] == committed[0].decision_sha256
    assert plan["chosen_action"] == "ATTEMPT"
    assert plan["committed"]["attempt_index"] == 2
    assert plan["committed"]["amount_paise"] == 50_000


def test_plan_404s_on_an_unknown_mandate(client):
    assert client.get("/plan/M-NOPE").status_code == 404


def test_a_non_attempt_plans_action_is_derived_not_stored(client, pg_schema):
    """`plan` has NO chosen_action column (schema.sql's own shape). The
    action is recoverable only by outer-joining committed_schedule on
    decision_sha256, since only ATTEMPT ever gets a row there -- the
    workaround tests/execute/test_cycle.py::_non_attempt_plan_rows already
    documents, reused here rather than rediscovered."""
    committed = _run_real_cycle(pg_schema.conn, "M-API-STOP", hazard=_STOP_HAZARD)
    assert committed == [], "this hazard must produce a non-ATTEMPT decision"

    plans = client.get("/plan/M-API-STOP").json()["plans"]
    assert len(plans) == 1
    assert plans[0]["chosen_action"] != "ATTEMPT"
    assert plans[0]["committed"] is None
    # The real action is in the sound candidate superset even where the
    # single label cannot name it.
    assert set(plans[0]["chosen_action_candidates"]) <= {"OFFER", "REAUTH", "STOP"}
    assert "ATTEMPT" not in plans[0]["chosen_action_candidates"]


def test_an_opted_out_plan_is_derived_as_stop_and_nothing_else(client, pg_schema):
    """Clause 6(c) is what makes this derivation SOUND rather than likely:
    `permitted()` denies every action but STOP once a mandate has opted
    out, so a plan carrying binding_constraint=OPTED_OUT cannot have been
    anything else. Proved by construction here, not by re-running the
    allocator -- the endpoint reads durable state, so durable state is what
    the test sets."""
    from src.ledger.store import PlanRow
    from src.api.read import _derive_action

    row = PlanRow(
        decision_sha256="x" * 64, mandate_id="M", cycle_id=1, profile="strict",
        belief_json="{}", conformal_set="WONT_PAY",
        binding_constraint="OPTED_OUT", solver_version="v", created_at=None,
    )
    action, candidates, how = _derive_action(row, None)
    assert action == "STOP"
    assert candidates == ["STOP"]
    assert "6(c)" in how


def test_a_non_singleton_conformal_set_rules_out_offer(client, pg_schema):
    """`conformal.should_act()` is the ONLY firing rule in src/ (R5 pins
    that as a test), and it requires the singleton {WONT_PAY} -- so a plan
    recording any other set provably did not offer. A sound negative is
    worth reporting even when no sound positive is available."""
    from src.ledger.store import PlanRow
    from src.api.read import _derive_action

    full_set = PlanRow(
        decision_sha256="y" * 64, mandate_id="M", cycle_id=1, profile="strict",
        belief_json="{}", conformal_set="CANT_PAY_EVER,CANT_PAY_NOW,WONT_PAY",
        binding_constraint=None, solver_version="v", created_at=None,
    )
    _, candidates, how = _derive_action(full_set, None)
    assert "OFFER" not in candidates
    assert "ruled out" in how

    singleton = PlanRow(**{**full_set.__dict__, "conformal_set": "WONT_PAY"})
    _, candidates, _ = _derive_action(singleton, None)
    assert "OFFER" in candidates


def test_plan_carries_the_belief_dict_and_the_verbatim_string(client, pg_schema):
    """`_belief_dict`'s own stated reason: the parsed form DROPS the
    provenance field, which is the auditable part. Both are emitted."""
    _run_real_cycle(pg_schema.conn, "M-API-BELIEF")

    plan = client.get("/plan/M-API-BELIEF").json()["plans"][0]
    assert set(plan["belief"]) == {"CANT_PAY_NOW", "CANT_PAY_EVER", "WONT_PAY"}
    assert "provenance" in json.loads(plan["belief_json"])
    assert "cause_map=" in plan["belief_json"]


def test_conformal_set_is_a_list_not_a_comma_joined_string(client, pg_schema):
    """The DB stores a sorted comma-joined string (commit.py); every JSON
    surface in this repo emits a list (export_mandates.py). The endpoint
    reconciles them."""
    _run_real_cycle(pg_schema.conn, "M-API-CONF")

    plan = client.get("/plan/M-API-CONF").json()["plans"][0]
    assert isinstance(plan["conformal_set"], list)
    assert plan["conformal_set"] == sorted(plan["conformal_set"])
    assert "" not in plan["conformal_set"]


def test_an_empty_conformal_set_reads_back_as_an_empty_list(client, pg_schema):
    '''`"".split(",")` is `[""]`, not `[]` -- a one-element list containing
    the empty string, which would render as a phantom cause. A real
    conformal predictor CAN abstain (SplitConformal.pred_set() returns
    frozenset()), so this row shape is reachable, not hypothetical.'''
    _run_real_cycle(pg_schema.conn, "M-API-EMPTY")
    with pg_schema.conn.cursor() as cur:
        cur.execute("UPDATE plan SET conformal_set = '' WHERE mandate_id = %s",
                    ("M-API-EMPTY",))

    plan = client.get("/plan/M-API-EMPTY").json()["plans"][0]
    assert plan["conformal_set"] == []


# === /decision/{decision_sha256} ============================================

def test_decision_returns_the_plan_and_the_ledger_rows_citing_it(client, pg_schema):
    committed = _run_real_cycle(pg_schema.conn, "M-API-DEC")
    sha = committed[0].decision_sha256

    body = client.get(f"/decision/{sha}").json()
    assert body["plan"]["decision_sha256"] == sha
    assert body["plan"]["mandate_id"] == "M-API-DEC"
    assert [row["state"] for row in body["ledger"]] == ["INTENT", "SENT", "RESULT"]
    assert all(row["decision_sha256"] == sha for row in body["ledger"])


def test_decision_404s_on_an_unknown_hash(client):
    r = client.get("/decision/" + "0" * 64)
    assert r.status_code == 404


def test_a_decision_with_no_ledger_rows_returns_an_empty_list_not_a_404(client, pg_schema):
    """A STOP/REAUTH/OFFER plan writes a `plan` row and no ledger row --
    the decision exists and is auditable, so 404 would be wrong."""
    _run_real_cycle(pg_schema.conn, "M-API-DEC-STOP", hazard=_STOP_HAZARD)
    with pg_schema.conn.cursor() as cur:
        cur.execute("SELECT decision_sha256 FROM plan WHERE mandate_id = %s",
                    ("M-API-DEC-STOP",))
        sha = cur.fetchone()[0]

    body = client.get(f"/decision/{sha}").json()
    assert body["plan"]["decision_sha256"] == sha
    assert body["ledger"] == []


# === serialization conventions =============================================

def test_outcome_is_serialised_by_name_never_by_int(client, pg_schema):
    """`Outcome` is an IntEnum. `.value` would emit 1/2/3 -- a number that
    reads as an id, means nothing to a reader, and silently changes meaning
    if the enum is ever reordered. Every other serializer in this repo
    emits `.name`."""
    _run_real_cycle(pg_schema.conn, "M-API-OUTCOME")

    rows = client.get("/ledger/M-API-OUTCOME").json()["rows"]
    outcomes = [r["outcome"] for r in rows if r["outcome"] is not None]
    assert outcomes and all(isinstance(o, str) for o in outcomes)
    assert "RECOVERED" in outcomes


def test_timestamps_are_isoformat_strings(client, pg_schema):
    _run_real_cycle(pg_schema.conn, "M-API-TIME")

    row = client.get("/ledger/M-API-TIME").json()["rows"][0]
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None


# === the review findings, pinned ============================================

def test_a_voided_attempt_still_derives_as_attempt_and_says_it_is_dead(client, pg_schema):
    """money-auditor, 2026-09-05, and the fix is NOT the one proposed.

    The reviewer suggested filtering `voided_at IS NULL` out of
    `committed_for_decision()`. That would be wrong: `commit()` writes a
    committed_schedule row ONLY for ATTEMPT, so the row's existence PROVES
    the decision was ATTEMPT, and voiding (src/execute/void.py) is a later,
    overtaken-by-events act that cannot change what was chosen. Filtering
    would report NOT_ATTEMPT with candidates [REAUTH, STOP] for a decision
    we can prove was ATTEMPT -- a strictly false answer replacing a
    true-but-incomplete one.

    The real defect was narrower: nothing said the cited row was dead. Both
    halves are pinned here, so neither can regress into the other.
    """
    committed = _run_real_cycle(pg_schema.conn, "M-API-VOID")
    assert committed
    with pg_schema.conn.cursor() as cur:
        cur.execute(
            "UPDATE committed_schedule SET voided_at = now(), void_reason = 'TEST' "
            "WHERE decision_sha256 = %s", (committed[0].decision_sha256,))

    plan = client.get("/plan/M-API-VOID").json()["plans"][0]

    assert plan["chosen_action"] == "ATTEMPT"          # the decision, unchanged
    assert plan["committed"]["is_live"] is False       # and nothing is scheduled
    assert plan["committed"]["void_reason"] == "TEST"
    assert "VOIDED" in plan["chosen_action_source"]


def test_a_live_committed_row_is_flagged_live(client, pg_schema):
    """The control: without it the assertion above would pass on an
    endpoint that reported everything as dead."""
    _run_real_cycle(pg_schema.conn, "M-API-LIVE")
    plan = client.get("/plan/M-API-LIVE").json()["plans"][0]

    assert plan["committed"]["is_live"] is True
    assert "VOIDED" not in plan["chosen_action_source"]


def test_the_module_discloses_that_it_has_no_authentication():
    """compliance-auditor, 2026-09-05: these endpoints serve money amounts,
    decision rationale and customer outcomes with no auth and no tenant
    scoping. That is a stated non-decision, not an oversight -- and a
    stated non-decision that stops being stated is just an oversight, so
    the disclosure is pinned rather than trusted to survive an edit."""
    import pathlib

    src = pathlib.Path("src/api/read.py").read_text(encoding="utf-8")
    head = src[:src.index("Why `src/api/`")]
    assert "NO AUTHENTICATION AND NO TENANT SCOPING" in head
    assert "10(c)" in head

    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "no authentication and no tenant scoping" in readme
