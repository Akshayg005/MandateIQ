"""B14 -- the per-mandate artifact the dashboard drill-down renders.

    python -m eval.export_mandates            # writes reports/mandates.json

WHY THIS MODULE EXISTS. B14's gate asks the drill-down to show belief, chosen
slot, binding constraint, conformal set and a ledger trail, per mandate.
The first four live together on exactly one object -- policy.allocator.Plan
-- and eval/run.py's engine loop reduced every Plan to a counter and threw
the object away, so `reports/regimes.json` (1024 aggregate cells) cannot
answer a single one of them. run.py now takes an optional trace; this module
turns that trace into an artifact.

ONE CELL, NOT ALL 1024. The export covers `baseline / nominal / strict /
engine`, seed 0 -- the same cell reports/results.json reports on. A drill-down
over 128 cells x 8 seeds would be 200k mandate-cycles of JSON answering a
question nobody asked; the acquirer view gets its breadth from the aggregate
artifact instead.

SEED 0, NOT THE PUBLISHED MEAN. results.json's headline figures are means
over the 8 seeds in its `seeds` field -- preserved 142/200 is 141.875 rounded,
and THIS artifact's seed-0 batch has 135. Both are correct and they are not
the same number. Anything rendering the two together must say which is which;
scripts/dashboard_data.py therefore stages this file for the reviewer
dashboard and withholds it from the landing page, whose counters are means.

=== The ledger trail is written by the real executor ========================

The trail is NOT hand-built JSON that looks like a ledger. Each traced Plan
goes through the same commit() and execute() the production path uses,
against the real schema, so the idempotency keys, the INTENT -> SENT ->
RESULT ordering, the plan foreign key and the lease discipline are the
production ones rather than a plausible imitation of them. This follows
eval/chaos.py, which already drives that path with a provider double.

What is simulated is exactly one thing: THE PROVIDER'S ANSWER. The frozen
simulator decided each attempt's outcome; `_SimProvider` reports that
outcome where Razorpay would report its own. Every decline string it emits
carries `[simulated:` in the text and every mandate is prefixed, so a reader
grepping this database finds simulated evidence rather than mistaking it for
observed evidence -- the same discipline as PROXY_SOURCE_VERSION in
eval/allocator_sweep.py.

=== How an opt-out is rendered, and why it is not a decline ================

CLAUDE.md's clause 6(c) row is explicit that OPTED_OUT is a distinct
outcome, never folded into "declined". So a simulated opt-out is NOT sent as
a decline string. It is recorded as a `mandate_lifecycle` REVOKED row
effective before the attempt's scheduled time, which is what an opt-out
riding the T-24h pre-transaction notification actually is: the customer
leaves BEFORE the debit, so no debit occurs. execute() then takes its own
step-2a abort path and voids the schedule row.

That rendering is a modelling choice and it is visible in the artifact
(`ledger_note` on the affected mandate). The frozen simulator still counts
that slot as spent -- it charges us for the attempt that provoked the
opt-out -- and nothing here changes any scored number; the three bars come
from run.py exactly as before.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from dotenv import find_dotenv, load_dotenv

from eval import regimes as regimes_mod
from eval import run as run_mod
from eval.frozen.simulator import Simulator, load_config
from src.core import clock, money
from src.core.ids import decision_sha256
from src.core.types import Action, Cause, MandateState, Outcome, Profile
from src.execute.commit import _insert_plan_row, commit
from src.execute.executor import execute
from src.execute.keys import ScheduledAttempt
from src.execute.razorpay_client import RazorpayDeclined
from src.ledger.store import record_lifecycle_event, replay
from src.policy import belief as belief_mod
from src.policy.constraints import afa_free_limit_paise
from src.policy.allocator import AllocationContext, CommittedAttempt, Plan

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "mandates.json"

# The designated cell. Named here rather than defaulted in argparse so the
# artifact and the report can be checked against the same constant.
CELL = {"regime": "baseline", "arm": "nominal", "profile": Profile.strict, "seed": 0}

# A fixed, frozen cycle start. The simulator works in day indices; the
# committed_schedule table works in timestamps, and its 24h CHECK is a real
# constraint, so day 1 maps to cycle_start + 1 day. 10:00 UTC sits outside
# the default quiet-hours window either side, so replay exercises the
# executor's real permission check rather than tripping it on an artifact of
# the chosen hour.
CYCLE_START = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

LEASE_TTL_SECONDS = 300

# Decline text for each simulated non-terminal / dead outcome. Chosen to
# classify, under the REAL taxonomy (src/classify/decline_taxonomy.py), to
# the same DeclineClass the engine's own belief update saw via
# _proxy_decline_class -- so the ledger and the belief tell one story. The
# "[simulated:" marker is deliberate and must survive: it is what stops a
# reader taking these for issuer strings.
_DECLINE_TEXT = {
    Outcome.DEAD.name: "card_expired [simulated: eval proxy, not an issuer string]",
    Outcome.STILL_PENDING.name: (
        "insufficient_funds [simulated: eval proxy, not an issuer string]"
    ),
}

_OPT_OUT_NOTE = (
    "OPTED_OUT is rendered as a pre-notification revocation (clause 6c), not "
    "as a decline: the mandate_lifecycle row is written before the attempt, so "
    "execute() aborts and voids rather than debiting. The slot is still counted "
    "as spent by the frozen scorer."
)


# --- building the records ----------------------------------------------------


def _belief_dict(belief_json: str) -> dict[str, float]:
    """The posterior as {cause: p}, read out of Belief.to_json()'s own
    output. That JSON is keyed by cause name and also carries `provenance`
    -- the field B11 added so a belief can be traced to the normaliser
    version that produced it -- which is dropped here and exported verbatim
    alongside instead, never silently discarded."""
    payload = json.loads(belief_json)
    return {c.value: payload[c.value] for c in belief_mod.CAUSE_ORDER}


def _decision_record(index: int, d: run_mod.DecisionTrace) -> dict[str, Any]:
    plan = d.plan
    committed = plan.committed[0] if plan.committed else None
    return {
        "index": index,
        "action": plan.chosen_action.value,
        "chosen_slot": committed.slot if committed else None,
        "chosen_day": committed.on_day if committed else None,
        "amount_paise": committed.amount_paise if committed else None,
        # Formatted HERE, by money.fmt, because money.py is the only module
        # allowed to render currency (CLAUDE.md invariant 2). The dashboard
        # is TypeScript, which guard_invariants.py cannot scan, so a
        # formatter written over there would be the same violation the guard
        # was widened to catch in eval/report.py -- in a language where
        # nothing would catch it. The SPA therefore never divides paise.
        "amount": money.fmt(committed.amount_paise) if committed else None,
        "belief": _belief_dict(plan.belief_json),
        # Verbatim, so replay writes the SOLVER's belief_json into the plan
        # table rather than a re-serialisation of it -- the provenance string
        # is the auditable part and a rebuild would lose it.
        "belief_json": plan.belief_json,
        "conformal_set": sorted(c.value for c in plan.conformal_set),
        "binding_constraint": plan.binding_constraint,
        "solver_version": plan.solver_version,
        "decision_sha256": plan.decision_sha256,
        # R5: the actual pause/downgrade/cancel menu, present iff the
        # decision was OFFER. src/policy/offramp.py was complete and tested
        # from B8 but had no caller anywhere, so a chosen OFFER had never
        # produced an Offer object and nothing downstream could show WHAT
        # was offered -- only that something was. Exported as the ordered
        # step list the customer would see, never as a single ultimatum
        # (invariant 6: the system offers; the customer decides).
        "offer": None if plan.offer is None else {
            "expires_on_day": plan.offer.expires_on_day,
            "steps": [
                {"kind": s.kind, "description": s.description}
                for s in plan.offer.steps
            ],
        },
        "outcome": d.outcome,
    }


# R5, 2026-09-05: the drill-down must show the SAME configuration the
# published grid was run under, or the dashboard and reports/regimes.json
# describe two different experiments while looking like one. Before this,
# build_records() ran with no channel while `eval.run`'s default is the
# pre-registered operating point -- so every exported mandate would have
# shown OFFER as structurally impossible next to a report saying it fires
# 1292 times. Imported from eval.run rather than restated, for the same
# reason eval/run.py imports the operating point from eval.offramp_channel.
DEFAULT_CHANNEL_KIND = run_mod.DEFAULT_CHANNEL_KIND


def _default_channel_spec() -> tuple[str, float, float]:
    from eval.offramp_channel import OPERATING_POINT

    return (DEFAULT_CHANNEL_KIND, OPERATING_POINT[0], OPERATING_POINT[1])


def build_records(*, regime: str, arm: str, profile: Profile, seed: int,
                  cfg: dict | None = None, hazard=None, costs=None,
                  gate=None, gate_kind: str | None = None,
                  channel_spec: tuple | None = ...) -> list[dict[str, Any]]:
    """Run one engine cell with tracing on and return one record per mandate.

    The decisions exported are THE decisions the scored run made, not a
    re-solve: the trace is threaded through run.py's own loop. A second
    solve() outside that loop would be a second experiment that agrees today
    and drifts tomorrow.

    `channel_spec` defaults (via the `...` sentinel, so `None` stays
    meaningful as "no channel") to the SAME pre-registered operating point
    `eval.run` publishes under -- see the comment above. Pass None to export
    the pre-R5 configuration, in which OFFER is structurally unreachable.

    `ledger` comes back empty. It is filled only by replay_to_ledger(), from
    rows the executor actually wrote -- this module never synthesises a
    ledger row.
    """
    if channel_spec is ...:
        channel_spec = _default_channel_spec()
    if cfg is None:
        cfg = regimes_mod.config_for(regime, load_config())
    if hazard is None:
        hazard = run_mod.hazard_from_fit(run_mod.fit_nominal_hazard_model())
    if costs is None:
        costs = run_mod.load_costs()
    if gate is None:
        # Calibrated under the SAME channel the cell will run under: the
        # channel changes the belief distribution, so a gate fitted without
        # it would be calibrated on a pool the cell never draws from.
        gate, gate_kind, _ = run_mod.fit_gate(cfg, channel_spec=channel_spec)

    traces: dict[str, list[run_mod.DecisionTrace]] = {}
    run_mod.run_engine_cell(regime, arm, profile, cfg, seed, hazard, costs,
                            gate, gate_kind or "conformal", traces=traces,
                            channel=run_mod.make_channel(channel_spec, seed))

    # Same arm, same seed, same config -> the same batch, by construction.
    # Rebuilt rather than plumbed out of run_engine_cell so this module adds
    # no second return value to a function 128 cells depend on.
    sim = Simulator(arm, seed=seed, config=cfg)

    records: list[dict[str, Any]] = []
    for m in sim.mandates:
        decisions = [
            _decision_record(i, d) for i, d in enumerate(traces[m.mandate_id])
        ]
        spent = sum(1 for d in decisions if d["outcome"] is not None)
        opted_out = any(d["outcome"] == Outcome.OPTED_OUT.name for d in decisions)
        records.append({
            "mandate_id": m.mandate_id,
            "cycle_id": m.cycle_id,
            "category": m.category,
            "amount_paise": m.amount_paise,
            "amount": money.fmt(m.amount_paise),
            "ceiling_paise": m.ceiling_paise,
            "ceiling": money.fmt(m.ceiling_paise),
            "afa_limit_paise": afa_free_limit_paise(m.category),
            "afa_limit": money.fmt(afa_free_limit_paise(m.category)),
            # Clause 8(a)/8(b): above the category's AFA-free ceiling an
            # attempt is not silently retryable, it goes to re-auth. The
            # dashboard shows which mandates sit on the wrong side of that
            # cliff, so the limit is exported per mandate rather than
            # re-derived in TypeScript from a constant copied out of
            # src/policy/constraints.py.
            "above_afa": m.amount_paise > afa_free_limit_paise(m.category),
            "profile": profile.value,
            "decisions": decisions,
            "attempts_spent": spent,
            "final_action": decisions[-1]["action"] if decisions else None,
            "final_outcome": next(
                (d["outcome"] for d in reversed(decisions) if d["outcome"]), None
            ),
            "ledger": [],
            "ledger_note": _OPT_OUT_NOTE if opted_out else None,
            # Quarantined on purpose: unobservable to any real aggregator
            # (eval/frozen/simulator.py says so itself), exported only
            # because a false REAUTH is not visible without it. Never merged
            # into the engine's own view. household_id is not exported at
            # all -- it is ground truth the dashboard has no use for.
            "ground_truth": {"true_cause": m.initial_cause.value},
        })
    return records


def recompute_decision_sha256(record: dict, decision: dict) -> str:
    """Rebuild allocator._plan()'s hash payload from the EXPORTED fields
    alone and hash it.

    This is the export's own completeness check, and it is why the test
    suite can assert faithfulness rather than assume it: if the artifact
    dropped a field, reordered the belief, or rounded a probability, this
    digest stops matching the one the solver recorded. A drill-down whose
    numbers do not rehash to the decision they claim to explain is
    decorative.
    """
    committed = []
    if decision["chosen_slot"] is not None:
        committed = [{
            "slot": decision["chosen_slot"],
            "on_day": decision["chosen_day"],
            "amount_paise": decision["amount_paise"],
        }]
    return decision_sha256({
        "mandate_id": record["mandate_id"],
        "cycle_id": record["cycle_id"],
        "profile": record["profile"],
        "chosen_action": decision["action"],
        "committed": committed,
        "belief": [decision["belief"][c.value] for c in belief_mod.CAUSE_ORDER],
        "conformal_set": decision["conformal_set"],
        "binding_constraint": decision["binding_constraint"],
        "solver_version": decision["solver_version"],
    })


# --- the provider double -----------------------------------------------------


class _SimProvider:
    """A RazorpayLike double that reports the frozen simulator's outcome.

    Not a mock with canned responses: the answer for each receipt is the
    outcome the scored run already produced for that exact attempt, keyed by
    idempotency key, so the ledger cannot disagree with the evaluation.

    An unknown receipt raises rather than defaulting to success. A double
    that invents an answer for an attempt nobody scheduled would write a
    recovery into the ledger that no run ever earned.
    """

    def __init__(self, outcomes: dict[str, str]) -> None:
        self._outcomes = outcomes
        self.charges: dict[str, int] = {}

    def charge(self, *, amount_paise: int, receipt: str, notes: dict) -> dict:
        if receipt not in self._outcomes:
            raise AssertionError(
                f"no simulated outcome for receipt {receipt!r} -- the replay is "
                "sending an attempt the traced run never made"
            )
        self.charges[receipt] = self.charges.get(receipt, 0) + 1
        outcome = self._outcomes[receipt]
        if outcome == Outcome.RECOVERED.name:
            return {"id": f"pay_sim_{receipt[:12]}", "status": "captured",
                    "amount": amount_paise}
        raise RazorpayDeclined(_DECLINE_TEXT[outcome])

    def find_by_receipt(self, receipt: str) -> dict | None:
        return None

    def create_order(self, **kwargs):
        raise AssertionError("execute() must call charge(), never create_order()")

    def pause_subscription(self, *args, **kwargs):
        raise AssertionError("the executor path must never pause a subscription")


def _plan_from(record: dict, decision: dict) -> Plan:
    """Rebuild the Plan the allocator produced, from the exported record.

    Safe precisely because recompute_decision_sha256() can check it: the
    reconstruction carries the solver's own digest, and the test suite
    asserts that rehashing the reconstruction reproduces it. commit() then
    writes that digest into `plan`, so the ledger's foreign key points at the
    decision the engine actually made.
    """
    committed: tuple[CommittedAttempt, ...] = ()
    if decision["chosen_slot"] is not None:
        committed = (CommittedAttempt(
            slot=decision["chosen_slot"],
            on_day=decision["chosen_day"],
            amount_paise=decision["amount_paise"],
        ),)
    return Plan(
        mandate_id=record["mandate_id"],
        cycle_id=record["cycle_id"],
        profile=Profile(record["profile"]),
        chosen_action=Action(decision["action"]),
        committed=committed,
        belief_json=decision["belief_json"],
        conformal_set=frozenset(Cause(c) for c in decision["conformal_set"]),
        binding_constraint=decision["binding_constraint"],
        solver_version=decision["solver_version"],
        decision_sha256=decision["decision_sha256"],
    )


def _ctx_for(record: dict, *, attempts_used: int,
             committed_days: tuple[int, ...], costs) -> AllocationContext:
    return AllocationContext(
        mandate_id=record["mandate_id"],
        cycle_id=record["cycle_id"],
        profile=Profile(record["profile"]),
        amount_paise=record["amount_paise"],
        ceiling_paise=record["ceiling_paise"],
        category=record["category"],
        plan_day=committed_days[-1] if committed_days else 1,
        attempts_used=attempts_used,
        committed_days=committed_days,
        contacts_sent=attempts_used,
        mandate_state=MandateState.ACTIVE,
        opted_out=False,
        max_contacts_per_cycle=costs.max_contacts_per_cycle,
        quiet_hours_start=costs.quiet_hours_start,
        quiet_hours_end=costs.quiet_hours_end,
    )


def _row_dict(row) -> dict[str, Any]:
    return {
        "idempotency_key": row.idempotency_key,
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
        "decision_sha256": row.decision_sha256,
    }


def replay_to_ledger(conn, records: Sequence[dict], *, costs=None) -> None:
    """Drive each record's traced decisions through commit() and execute(),
    then read the trail back out of the database into `record["ledger"]`.

    Idempotent: commit() dedupes on idempotency_key and execute() dedupes on
    ledger_intent_once, so a re-run over a database that already holds the
    export writes nothing new -- which is the same property that makes crash
    recovery correct, exercised here for free.
    """
    if costs is None:
        costs = run_mod.load_costs()

    for record in records:
        mandate_id = record["mandate_id"]
        clock.set_frozen(CYCLE_START)
        record_lifecycle_event(
            conn,
            event_id=f"evt-created-{mandate_id}",
            mandate_id=mandate_id,
            state=MandateState.ACTIVE.value,
            source="INTERNAL",
            effective_at=CYCLE_START - timedelta(days=1),
        )

        attempts_used = 0
        committed_days: tuple[int, ...] = ()
        for decision in record["decisions"]:
            plan = _plan_from(record, decision)
            clock.set_frozen(CYCLE_START)

            if decision["outcome"] == Outcome.OPTED_OUT.name:
                # Clause 6(c): the customer leaves on the pre-transaction
                # notification, so the debit never happens. Recorded BEFORE
                # the attempt so execute()'s own late read is what stops it.
                record_lifecycle_event(
                    conn,
                    event_id=f"evt-optout-{mandate_id}-{decision['index']}",
                    mandate_id=mandate_id,
                    state=MandateState.REVOKED.value,
                    source="INTERNAL",
                    effective_at=CYCLE_START,
                )

            if decision["outcome"] is None and plan.chosen_action is Action.ATTEMPT:
                # The post-terminal re-solve: the allocator was asked "the
                # instrument is dead, now what?" and answered ATTEMPT. It is a
                # real decision and gets a real plan row -- that is how the
                # drill-down can show it, and it is a disclosed defect (B7/B8,
                # 16 of them in this cell) rather than one to hide. But it was
                # never committed and never sent, so it must NOT get a
                # committed_schedule row: that table means "committed >=24h
                # ahead" (clause 6a), and writing one for an attempt nobody
                # scheduled would put a debit in the audit trail that never
                # existed.
                _insert_plan_row(conn, plan)
                continue

            attempt = commit(conn, plan, cycle_start=CYCLE_START)
            if attempt is None:
                # STOP / OFFER / REAUTH schedule no debit at all; the plan row
                # is written and there is nothing to execute.
                continue

            attempts_used += 1
            committed_days = committed_days + (decision["chosen_day"],)
            provider = _SimProvider({attempt.idempotency_key: decision["outcome"]}
                                    if decision["outcome"] else {})
            clock.set_frozen(attempt.scheduled_for)
            execute(
                conn, provider, attempt,
                _ctx_for(record, attempts_used=attempts_used,
                         committed_days=committed_days, costs=costs),
                owner="b14-export",
                lease_ttl_seconds=LEASE_TTL_SECONDS,
            )

        record["ledger"] = [_row_dict(r) for r in replay(conn, mandate_id)]


# --- entrypoint --------------------------------------------------------------


SCHEMA_SQL = _REPO_ROOT / "src" / "ledger" / "schema.sql"


@contextlib.contextmanager
def _throwaway_schema(name: str = "b14_export"):
    """A disposable schema built from schema.sql, dropped on the way out.

    The export deliberately does NOT write into the default schema, for two
    reasons. First, contamination: that database holds the handful of real
    test-mode rows in `ingested_event` / `webhook_event` that are B3's gate
    evidence, and 200 mandates of simulated provider answers do not belong
    beside them where a later reader could mistake one for the other.
    Second, drift: the running container's schema predates B9's
    `committed_schedule.decision_sha256`, so commit() cannot write to it at
    all. Building from schema.sql means this artifact is reproducible from
    an empty tree and cannot be stale by construction -- the same reasoning
    as tests/conftest.py's pg_schema fixture, which this mirrors.
    """
    import psycopg

    from src.core import db

    conn = psycopg.connect(db.dsn(), autocommit=True, connect_timeout=3)
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            cur.execute(f'CREATE SCHEMA "{name}"')
            cur.execute(f'SET search_path TO "{name}"')
            cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        yield conn
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        finally:
            conn.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=ARTIFACT)
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip the executor replay (no Postgres needed); the "
                         "artifact then carries decisions but an empty trail")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv(find_dotenv())

    records = build_records(regime=CELL["regime"], arm=CELL["arm"],
                            profile=CELL["profile"], seed=CELL["seed"])

    ledger_rows = 0
    if not args.no_ledger:
        with _throwaway_schema() as conn:
            replay_to_ledger(conn, records)
        ledger_rows = sum(len(r["ledger"]) for r in records)

    payload = {
        "schema": 1,
        "cell": {k: (v.value if isinstance(v, Profile) else v)
                 for k, v in CELL.items()},
        "cycle_start": CYCLE_START.isoformat(),
        "ledger_provenance": (
            "written by src/execute/commit.py and src/execute/executor.py into "
            "the real schema; only the provider's answer is simulated"
        ),
        "mandates": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {args.out.name} "
          f"({len(records)} mandates, {ledger_rows} ledger rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
