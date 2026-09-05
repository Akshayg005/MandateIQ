"""R6: three read-only endpoints over the ledger and the plan table.

reports/gates.md, "Post-B16 remediation gates", R6: `/plan/{mandate_id}`,
`/ledger/{mandate_id}` and `/decision/{decision_sha256}` return real rows
from a live schema, covered by tests.

=== THESE ENDPOINTS HAVE NO AUTHENTICATION AND NO TENANT SCOPING =========

Stated first because it is the most important thing about this module.
The compliance audit raised it (2026-09-05) and it is correct: anyone who
can reach the port and knows or guesses a `mandate_id` can read that
mandate's debit amounts, the allocator's full decision rationale (belief,
conformal set, binding constraint), its outcome and its decline reasons.
There is no API key, no token, no per-merchant isolation and no access log.
The webhook router next door verifies an HMAC signature; these three routes
verify nothing.

That is a deliberate NON-decision, not an oversight, and the distinction
matters: R6's gate is "return real rows from a live schema, covered by
tests", and inventing an auth scheme nobody specified would be a worse
answer than naming the gap. This project is Razorpay TEST MODE only
(invariant 5) and has no deployment, so the gap costs nothing today and
would be a blocker on day one of anything real.

It also bounds the clause 10(c) claim this project makes elsewhere. An
acquirer-facing audit surface needs tenant scoping and an access log; these
endpoints serve the audit CONTENT and none of the audit CONTROLS. Read the
"acquirer dashboard" story as "here is the trail for a mandate you already
identified", never as "here is a multi-merchant compliance console".

=========================================================================

Why `src/api/` and not `src/ingest/`: **ingest** means events arriving from
the outside. These are reads going out. `src/ingest/app.py`'s docstring
previously asserted that nothing in the planned dependency graph would ever
add a second router -- that sentence has been rewritten rather than left
standing next to the router that disproves it.

=== What these endpoints have to reconcile, and why it is not cosmetic ====

1. **`plan` has no `chosen_action` column.** `src/execute/commit.py`'s
   `_insert_plan_row()` was, until R6, the ONLY code in `src/` that touched
   this table, and it only ever wrote. A plan's action is recoverable only
   by outer-joining `committed_schedule` on `decision_sha256`: commit()'s
   own gate writes such a row for ATTEMPT and for nothing else, so its
   presence IS the action. That workaround is not invented here --
   `tests/execute/test_cycle.py::_non_attempt_plan_rows` already documented
   it at R4, and this reuses its reasoning rather than rediscovering it.

   The derivation is therefore honest but INCOMPLETE, and says so. Three
   rules, each SOUND -- every one of them is a proof from durable state,
   never a likelihood:

     a. a committed_schedule row cites this decision  =>  ATTEMPT.
        commit()'s own gate writes one for ATTEMPT and nothing else. Holds
        even if that row was later VOIDED: voiding is an overtaken-by-events
        event, not a change to what was decided. The response flags such a
        row `is_live: false` and the derivation message says so.
     b. binding_constraint == "OPTED_OUT"             =>  STOP.
        Clause 6(c): `stopping_rules.permitted()` DENIES every action but
        STOP once a mandate has opted out, so no other action was legal.
     c. conformal_set is not exactly {WONT_PAY}       =>  not OFFER.
        `conformal.should_act()` is the single firing rule and requires
        that singleton (R5 pins it as the only call site in src/).

   What remains after those is genuinely undecidable here: a plan with no
   committed row, no opt-out, and a WONT_PAY singleton could be REAUTH,
   STOP or OFFER, and `plan` records nothing that separates them. Such a
   row is reported as `NOT_ATTEMPT` with `chosen_action_candidates` listing
   the actions not ruled out. Naming one of them would be a guess printed
   as a record -- the same failure mode R2's `_binding_constraint()` bug
   was (a hard-forced decision recorded as a free economic choice, across
   thousands of rows).

   The real fix is a `chosen_action` column on `plan`. That is a schema
   change with no migration path in this repo (see the R5-R7 plan's R7
   section: `schema.sql` is only ever applied into throwaway test schemas),
   so R6 does not make it. It is named here rather than left as a surprise.

2. **`conformal_set` has two incompatible encodings in-repo.** The DB
   stores a sorted comma-joined string (`commit.py`); every JSON surface
   emits a list (`eval/export_mandates.py`). `"".split(",")` is `[""]`,
   not `[]` -- a phantom one-element cause -- and an abstaining conformal
   predictor really can produce the empty set
   (`SplitConformal.pred_set()` returns `frozenset()`), so this is a
   reachable row shape, not a hypothetical one.

3. **`belief_json`'s parsed form drops `provenance`**, which is the
   auditable part (B11: "a belief that cannot be traced to a specific
   normaliser version is not auditable"). Both the parsed dict AND the
   verbatim string are emitted -- `eval/export_mandates.py::_belief_dict`'s
   own stated reason, applied here.

4. **Money.** Both `amount_paise` (int) and `amount` (`money.fmt`), exactly
   as `_row_dict` does. `src/core/money.py` stays the only formatter
   (invariant 2). `fmt()` raises on negatives, so it is never called on a
   difference -- these endpoints emit no differences.

5. **`Outcome` is an `IntEnum`.** Ledger rows store its `.name` as TEXT and
   that is what is served. `.value` would emit 1/2/3 -- a number that reads
   as an id and silently changes meaning if the enum is ever reordered.

Read-only by construction: every function below issues SELECTs through
`src/ledger/store.py`'s named functions. No raw SQL lives in this router --
every other DB read in this repo works that way.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.core import money
from src.ingest.deps import get_conn
from src.ledger import store

router = APIRouter(tags=["read"])

# Reported when durable state proves the decision was not ATTEMPT but does
# not single out which of the rest it was. See docstring point 1.
NOT_ATTEMPT = "NOT_ATTEMPT"

# The conformal set OFFER requires. `conformal.should_act()` is the only
# firing rule in src/ (pinned by tests/policy/test_allocator.py), so a plan
# whose recorded set is anything else PROVABLY did not offer.
_OFFER_SET = ("WONT_PAY",)


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _split_conformal_set(raw: str) -> list[str]:
    """The DB's comma-joined string -> a list. `""` -> `[]`, never `[""]`
    (docstring point 2)."""
    return [part for part in raw.split(",") if part]


def _ledger_row(row: store.LedgerRow) -> dict[str, Any]:
    """One ledger row as JSON. Field-for-field the same shape
    `eval/export_mandates.py::_row_dict` already emits -- reused so the
    dashboard and this API cannot describe the same row two ways -- plus
    the identifiers and timestamp a live API caller needs and a static
    export does not."""
    return {
        "ledger_id": row.ledger_id,
        "idempotency_key": row.idempotency_key,
        "mandate_id": row.mandate_id,
        "cycle_id": row.cycle_id,
        "attempt_index": row.attempt_index,
        "action": row.action,
        "state": row.state,
        "amount_paise": row.amount_paise,
        "amount": money.fmt(row.amount_paise),
        "outcome": row.outcome,
        "decline_class": row.decline_class,
        "reason": row.reason,
        "provider_ref": row.provider_ref,
        "profile": row.profile,
        "payload_sha256": row.payload_sha256,
        "decision_sha256": row.decision_sha256,
        "created_at": _iso(row.created_at),
    }


def _derive_action(row: store.PlanRow, committed: dict | None) -> tuple[str, list[str], str]:
    """(chosen_action, candidates, how) -- see docstring point 1.

    Every rule here is SOUND: each one is a proof from durable state, not a
    likelihood. `candidates` is a superset guaranteed to contain the real
    action, so a caller that needs certainty can check its length rather
    than trusting a single label.
    """
    if committed is not None:
        how = ("a committed_schedule row cites this decision, and commit() "
               "writes one only for ATTEMPT")
        if committed["voided_at"] is not None:
            # Sound but incomplete without this: the decision WAS an
            # ATTEMPT -- voiding is a later, overtaken-by-events event
            # (src/execute/void.py) that cannot change what was chosen --
            # but a reader seeing a committed slot and no other signal
            # would reasonably take it for a live commitment. Raised by
            # the money audit, 2026-09-05; see store.committed_for_decision()
            # for why the row is not simply filtered out instead.
            how += (". NOTE: that row is VOIDED and was never sent -- the "
                    "DECISION was still ATTEMPT, but nothing is scheduled")
        return ("ATTEMPT", ["ATTEMPT"], how)
    if row.binding_constraint == "OPTED_OUT":
        return ("STOP", ["STOP"],
                "binding_constraint is OPTED_OUT, and clause 6(c) "
                "(stopping_rules.permitted) denies every action but STOP "
                "once a mandate has opted out -- no other action was legal")
    candidates = ["REAUTH", "STOP"]
    how = ("no committed_schedule row cites this decision, so it was not "
           "ATTEMPT")
    if tuple(_split_conformal_set(row.conformal_set)) == _OFFER_SET:
        candidates.insert(0, "OFFER")
        how += ("; the recorded conformal set IS the {WONT_PAY} singleton "
                "conformal.should_act() fires on, so OFFER cannot be ruled out")
    else:
        how += ("; the recorded conformal set is not the {WONT_PAY} singleton "
                "conformal.should_act() requires, so OFFER is ruled out")
    how += (". `plan` has no chosen_action column, so the remainder cannot "
            "be separated from durable state and is not guessed at")
    return (NOT_ATTEMPT, sorted(candidates), how)


def _plan_row(conn, row: store.PlanRow) -> dict[str, Any]:
    """One plan row as JSON, with `chosen_action` DERIVED from durable
    state alone (docstring point 1)."""
    committed = store.committed_for_decision(conn, row.decision_sha256)
    action, candidates, how = _derive_action(row, committed)
    return {
        "decision_sha256": row.decision_sha256,
        "mandate_id": row.mandate_id,
        "cycle_id": row.cycle_id,
        "profile": row.profile,
        "chosen_action": action,
        "chosen_action_candidates": candidates,
        "chosen_action_source": f"derived: {how}",
        "committed": None if committed is None else {
            # False when the row was voided and not reissued: the decision
            # was ATTEMPT, and nothing is scheduled. Emitted as its own
            # flag rather than left for a caller to infer from voided_at,
            # because "is anything actually going to be charged" is the
            # question a reader of this block is asking.
            "is_live": committed["voided_at"] is None,
            "idempotency_key": committed["idempotency_key"],
            "attempt_index": committed["attempt_index"],
            "amount_paise": committed["amount_paise"],
            "amount": money.fmt(committed["amount_paise"]),
            "scheduled_for": _iso(committed["scheduled_for"]),
            "committed_at": _iso(committed["committed_at"]),
            "voided_at": _iso(committed["voided_at"]),
            "void_reason": committed["void_reason"],
        },
        "belief": _belief_dict(row.belief_json),
        # Verbatim alongside the parsed form: the provenance string is what
        # makes the belief auditable and the parsed form drops it.
        "belief_json": row.belief_json,
        "conformal_set": _split_conformal_set(row.conformal_set),
        "binding_constraint": row.binding_constraint,
        "solver_version": row.solver_version,
        "created_at": _iso(row.created_at),
    }


def _belief_dict(belief_json: str) -> dict[str, float]:
    """The three cause probabilities, parsed out of `plan.belief_json`.
    Deliberately drops `provenance` -- the caller emits the verbatim string
    beside this, exactly as `eval/export_mandates.py::_belief_dict` does.
    Imported from neither: `src/` must never import `eval/`."""
    import json

    from src.policy import belief as belief_mod

    payload = json.loads(belief_json)
    return {c.value: payload[c.value] for c in belief_mod.CAUSE_ORDER}


@router.get("/ledger/{mandate_id}")
def read_ledger(mandate_id: str, conn=Depends(get_conn)) -> dict[str, Any]:
    """Every ledger row for one mandate, in insertion order -- the
    append-only trail `store.replay()` already produces, serialised."""
    rows = store.replay(conn, mandate_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no ledger rows for mandate_id={mandate_id!r}",
        )
    return {"mandate_id": mandate_id, "rows": [_ledger_row(r) for r in rows]}


@router.get("/plan/{mandate_id}")
def read_plans(mandate_id: str, conn=Depends(get_conn)) -> dict[str, Any]:
    """Every plan written for one mandate, oldest first, each with its
    derived action and its committed slot if it has one."""
    rows = store.plans_for_mandate(conn, mandate_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no plan rows for mandate_id={mandate_id!r}",
        )
    return {"mandate_id": mandate_id, "plans": [_plan_row(conn, r) for r in rows]}


@router.get("/decision/{decision_sha256}")
def read_decision(decision_sha256: str, conn=Depends(get_conn)) -> dict[str, Any]:
    """One decision by its hash, plus every ledger row citing it.

    An empty `ledger` list is a real answer, not a 404: a STOP/REAUTH/OFFER
    plan writes a `plan` row and no ledger row, so the decision exists and
    is auditable even though nothing was ever sent for it.
    """
    plan = store.find_plan(conn, decision_sha256)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail=f"no plan row for decision_sha256={decision_sha256!r}",
        )
    rows = store.ledger_for_decision(conn, decision_sha256)
    return {
        "plan": _plan_row(conn, plan),
        "ledger": [_ledger_row(r) for r in rows],
    }
