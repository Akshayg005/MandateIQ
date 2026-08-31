# POSTMORTEM — what broke during the build

The rubric line is *"Failure recovery — what broke, and what you did about
it."* That is asking about this build, not runtime resilience. Entries are
written **at the moment of breakage**, before the cause is known. Do not
backfill a tidy story, and do not delete an entry because it turned out to
be your own mistake — those are the valuable ones.

Use the `/log-incident` skill. Format:

## Incident 1 — coupled arm fabricated money in the B2 freeze

**When:** Block B2, 2026-08-26, minutes after the freeze commit (`8321406`).

**Symptom:** `payments-domain` — dispatched as B2's required gate review,
same session, before any file under `src/policy/` existed — reported that a
real batch run of the `coupled` arm recovered ₹3,91,412, roughly 1.7× the
total liquidity (₹2,30,732) that existed across all 50 simulated households.
Independently re-derived by direct computation before acting on the review:
`total_recovered_paise=39141154` against `total_household_balance_paise=23073171`.

**Root cause:** `Simulator._apply_household_coupling` had a "partial
liquidity" branch — when a household's balance fell below the mandate's
amount, it rolled a probability weighted by `balance/amount` for "recovers
anyway." On success it credited the mandate's **full** `amount_paise` while
only debiting the household down to zero. The gap between what was credited
and what the household actually had was created from nothing. It also
wasn't modeling anything real: UPI AutoPay has no partial-debit semantics —
a debit either succeeds in full or is declined.

**Why it wasn't caught earlier:** the pre-freeze test suite
(`tests/eval/test_simulator.py`) verified the storm *effect* — balance
depletes monotonically as household members are attempted in order, later
members show more iatrogenic failures than earlier ones, balance never goes
negative — but never asserted the money-conservation invariant itself
(total recovered across a household must never exceed that household's
starting balance). That was a gap in what I chose to test, not something
the harness structurally prevented me from testing.

**Fix:** removed the probabilistic branch entirely. A household debit now
either succeeds in full (`balance >= amount`, deducted exactly) or fails
outright (iatrogenic `STILL_PENDING`, balance unchanged) — commit
`d634346`, which supersedes the original freeze commit `8321406`.
`reports/FREEZE_HASH` updated to `d634346`. Verified post-fix on a real
batch run: `coupled` recovers ₹52,556.67, which is `<=` the ₹2,30,731.71
total household balance. The corrected numbers are a *more* dramatic,
and now mathematically defensible, demonstration of the storm effect:
recovered mandates dropped from 102 (under `nominal`) to 22 (under
`coupled`), and iatrogenic failures rose from 102 to 138 out of 200.

While re-touching the frozen files for this fix, also corrected
`protocol.md`'s false claim that the misspecified arm shares "the same base
rates" as nominal (mathematically impossible — cloglog dominates logit for
any shared score, so it can only be "same rates, same shape" [the earlier
no-op bug DECISIONS.md already caught] or "different shape, different
realized rates," never both); added "must not read" docstring warnings on
the two ground-truth-only fields (`household_id`,
`iatrogenic_insufficient_funds`) mirroring the existing warning on
`initial_cause`/`effective_cause`; added `on_day` monotonicity validation
to `attempt()` (a non-increasing day previously clamped silently via
`max(days_since_last, 1)` instead of raising). Full reasoning for each in
`DECISIONS.md`.

**Guard added:** `tests/eval/test_simulator.py::test_coupled_arm_never_recovers_more_than_the_household_ever_had`
— asserts, per household, across 20 seeds, that total recovered never
exceeds the household's starting balance. This is the exact invariant whose
absence let the bug ship undetected by my own tests; it is now a permanent
regression test, not just a review finding.

## Incident 2 — webhook writes silently discarded despite HTTP 200

**When:** Block B3, 2026-08-27, during the plan's manual end-to-end
verification step — after all 235 automated tests and
`guard_invariants.py --all` were already green.

**Symptom:** `POST /webhook/razorpay` against the real running server (not
`TestClient`) returned `HTTP 200 {"status": "ok"}` for a validly-signed
`payment.failed` body. A direct query of `ingested_event` immediately
afterward, against the same real database, found no matching row. The
write reported success and then simply wasn't there.

**Root cause:** `src/ingest/deps.py`'s `get_conn()` opened its connection
via `src.core.db.connect()` with no `autocommit` argument. psycopg3
defaults new connections to `autocommit=False`, so every write inside the
request handler ran inside an implicit transaction that nothing ever
committed. When `get_conn()`'s `finally: conn.close()` ran at the end of
the request, the uncommitted transaction was discarded along with the
connection — silently; closing a connection with a live uncommitted
transaction is not an error in psycopg3, it's just a rollback.

**Why it wasn't caught earlier:** all 235 tests, including the full
8-test `tests/ingest/test_webhook.py` suite (signature verification,
replay window, dedupe, both event-type routing paths, a source guard),
passed cleanly against this exact code — because none of them ever ran
the real `get_conn()`. `test_webhook.py`'s `client` fixture overrides the
dependency entirely (`app.dependency_overrides[get_conn] = lambda:
pg_schema.conn`), and `pg_schema.conn` is opened with `autocommit=True`
directly in `tests/conftest.py`, for an unrelated reason (so a test's own
verification queries see its writes immediately without an explicit
commit call). That override is correct and necessary for test isolation
— it points every write at a scratch schema instead of the real database
— but as a side effect it also bypassed the exact line of production code
that had the bug. Only a manual run against a live server and the real
database could have caught this, and did.

**Fix:** `src/core/db.py::connect()` already leaves autocommit to the
caller, by design (its own docstring says so). Fixed by making the
caller — `deps.py::get_conn()` — decide correctly: `connect(autocommit=
True)`. Verified by restarting the real server and re-running the same
manual check: the row now lands, with the correct `decline_class`,
`mandate_id`, an `amount_paise` that is a plain Python `int`, and a
`cause_prior` JSON summing to 1.0.

**Guard added:** `tests/ingest/test_deps.py::
test_get_conn_yields_an_autocommit_connection` — calls the REAL `get_conn()`
generator, deliberately not overridden (skips if Postgres is unreachable,
matching `pg_schema`'s own skip discipline, rather than mocking around the
exact thing this guard exists to exercise), and asserts `conn.autocommit
is True`. This is the one test in the suite that would have failed before
the fix, because it is the only one that doesn't go through
`dependency_overrides`.

---

## Incident 3 — the crash-recovery interface queried an API that rejects it

**When:** Block B9, 2026-08-30 ~03:15 IST

**Symptom:** The first live test-mode call ever made through
`src/execute/razorpay_client.py` — one `create_order`, then one
`find_by_receipt` to read it back — died on the second call:

```
RazorpayClientError: find_by_receipt('b9-live-1788084714') failed:
  receipt is/are not required and should not be sent
```

`create_order` had succeeded. All 78 B9 tests were green at the time, and
`guard_invariants --all` was clean.

**Root cause:** `find_by_receipt()` called
`self._client.payment.all({"receipt": receipt})`. `receipt` is an **Order**
field; the Payments resource has no such field, and Razorpay rejects the
parameter outright rather than ignoring it. The method could never have
returned anything, for any input, against the real API.

This is not a peripheral helper. Per the B3 spike (DECISIONS.md,
2026-08-27), `receipt` does **not** dedupe `Order.create` — so provider-side
dedup is not available to us, and `find_by_receipt` is the *entire*
"recover by asking, never by resending" path in `recover.py`. Every
`UNCONFIRMED` → `RESULT` resolution runs through it. The B9 gate's third
clause ("`UNCONFIRMED` has a resolution path that is actually reachable")
was, against the real API, false.

**A second finding, from the same probe:** `order.all({"receipt": ...})`
*is* honoured (verified: three known receipts each returned exactly their
own order, count=1). But the list endpoint **lags indexing** — an order
queried by its own receipt at 0s, 3s and 8s after creation returned
count=0, and was absent from the unfiltered recent list too; it appeared
minutes later. So a `None` from a receipt lookup is genuinely ambiguous:
"never created" and "created moments ago" are indistinguishable at the
moment `recover.py` most wants to ask.

**Why it wasn't caught earlier:** every test in
`tests/execute/test_razorpay_client.py` fakes the SDK — deliberately, and
the module docstring says so. Faking the SDK cannot detect that the
*parameter shape* is wrong, because the fake accepts whatever it is handed.
The module docstring flagged `charge()`'s shape as "unverified against live
traffic, recommend a B3-style spike" — but made no such disclosure for
`find_by_receipt`, which was the more load-bearing of the two. The gap was
a disclosure that stopped one method short.

**Fix:** the ATTEMPT path is now anchored to an Order.
`charge()` creates an order carrying the idempotency key as its receipt,
*then* creates the recurring payment against that `order_id` — which is
also how Razorpay itself models recurring debits (the one real Payment in
this test account, `pay_TUqQ25JYjOyNPD` from B3, carries an `order_id`).
`find_by_receipt()` became an indexed two-step: `order.all({"receipt": K})`,
then `order.payments(order_id)`. Ordering matters and is commented in place
— creating the order first means a crash *between* the two calls still
leaves a receipt-addressable record, so recovery finds the order, sees zero
payments, and reports "nothing charged yet" instead of finding nothing at
all. `None` is documented as *unresolved-so-far*, never *never-sent*; the
existing `UNCONFIRMED` backoff already models that correctly, so the lag
finding strengthened the backoff's rationale rather than changing its
design. Chosen over the working alternative (a bounded-window
`payment.all({from,to})` scan matching `notes` client-side, also verified
live) because that one caps at 100 payments per page and degrades on a busy
account exactly when recovery matters most. Committed with this entry.

**Guard added:** `scripts/live_smoke_b9.py` — drives `create_order` →
`find_by_receipt` against real test mode. Kept off the default test path (it
needs network and credentials) and run as a block-level verification step.
This class of bug is invisible to any test that fakes the SDK, because a
fake accepts whatever shape it is handed: the fake-based tests guard
*behaviour*, this guards *wire format*, and they are now documented as
separate risks that do not substitute for each other. Two unit-level guards
were added alongside it: a stub that raises if `payment.all` is ever called
again (`_MustNotCharge.all`), and a test asserting `charge()` sends
`order_id` and no `receipt` in the payment body. The smoke check retries on
a schedule rather than failing on first miss, because the indexing lag is
real — this run resolved on attempt 3, roughly 30s after creation.

**Still not covered, disclosed rather than assumed:** `charge()` itself.
Driving `payment.createRecurring` needs a real saved token / active mandate,
which test mode will not mint on demand, so its exact field shape remains
unverified against live traffic — the same disclosure the module docstring
already carried before this incident, now the *only* remaining one.

---

## Incident 4 — crash recovery keyed on a table its own docstring calls an optimisation

**When:** Block B10, 2026-08-30, on the first chaos run ever executed
(12 uniform kills, seed 0).

**Symptom:** One kill in twelve came back `LOST JOB (last row INTENT)`.
Kill #11 landed in the `INTENT->lease` window; after eight `reconcile()`
passes *and* a re-queue of the same committed attempt, the key's last
ledger row was still `INTENT`. Nothing had been sent, nothing would ever
be sent, and nothing reported that.

```
FAILURES (1)
  #11    INTENT->lease    LOST JOB (last row INTENT)
```

**Root cause:** `recover._dangling_keys()` discovers abandoned work by
iterating `lease.expired(conn)` — that is, by scanning the **lease table**
and then checking each key's ledger row. `executor.execute()` writes the
`INTENT` row (step 1) *before* it claims the lease (step 2), exactly as
the write-ordering protocol requires. A process that dies between those
two steps therefore leaves an `INTENT` row and **no lease row at all** —
not an expired lease, no row. Such a key never appears in
`lease.expired()`, so it is invisible to the dangling scan forever. It is
equally invisible to the `_stuck_keys()` scan, which matches only
`FAILED`/`UNCONFIRMED` rows. And it cannot be rescued by re-running the
attempt either: step 1's `ON CONFLICT DO NOTHING` sees the existing
`INTENT` row, returns it, and never sends. The NPCI slot is consumed
permanently, the customer is never debited, and no metric counts it.

The deeper error is a layering one, and `src/execute/lease.py`'s own
docstring states the rule that was broken: the lease is "an
**OPTIMISATION** over `ledger_intent_once`, not the concurrency control."
Recovery was keyed *solely* on that optimisation, which made a
deliberately non-authoritative table load-bearing for correctness. The
ledger is the source of truth; the scan has to start there.

**Why it wasn't caught earlier:** all 78 B9 tests construct their dangling
keys with a helper (`test_recover._make_dangling`) that writes the `INTENT`
row **and** claims a lease, then advances the clock past the TTL. That is
one specific interleaving — the one where the crash happened *after* the
lease was claimed. It is a perfectly reasonable thing to test and it
passes; it simply never constructs the other interleaving, and no test
that builds its own fixture state can discover a state it does not think
to build. Inducing the kill at an arbitrary point is what produced a state
nobody wrote down.

**Fix:** `_dangling_keys()` now scans the **ledger** for keys whose latest
row is `INTENT` or `SENT`, and excludes only those holding an *unexpired*
lease (someone may still be legitimately mid-attempt). The lease table
becomes what its docstring always said it was — an optimisation that stops
a second worker early — instead of the index recovery depends on. The old
behaviour is a strict subset of the new one: an expired lease with an
`INTENT`/`SENT` latest row still matches.

**Guard added:** `eval/chaos.py` is the guard. It is the first thing in
this project that constructs executor states by interruption rather than
by fixture, and it found this on its first run. `tests/eval/test_chaos.py`
pins the specific regression: a kill placed between the `INTENT` write and
the lease claim must still resolve.

---

## Incident 5 — void-and-reissue can double-charge across generations, invisible to a per-receipt oracle

**When:** Block B10 review (reviewer pass over the chaos harness, requested
separately from the block that built it), 2026-08-30.

**Symptom:** Read-through of `eval/chaos.py`, `src/execute/recover.py`, and
`src/execute/void.py` raised the question the review was specifically asked
to check: `ChaosClient.accepted` (and the real analogue, a receipt-keyed
idempotency check at Razorpay) counts double-charges **per idempotency
key**. But `void.reissue()` mints a **new** key at `generation+1` for the
same `(mandate_id, cycle_id, attempt_index)` slot by design — so a real
double charge that lands on two different generations of the same slot
would show `accepted[key] <= 1` on both keys and pass every oracle B10
built. Written up as `tests/eval/test_chaos.py::
test_lease_expiry_race_lets_a_stalled_live_worker_double_charge_across_generations`,
which reproduces the interleaving by hand (no threads needed — only write
order matters) and **fails**: two real charges, `total_charged == 2`,
`assert total_charged <= 1` raises.

**Root cause:** `recover._resolve_never_sent`'s proof ("no `SENT` row means
no provider call was ever issued") is sound only for a single process's own
local state, observed atomically. It is not sound across a live worker and
a concurrent `reconcile()` pass. Sequence: (1) a worker claims the lease and
then stalls before writing `SENT` — nothing in `executor.py` re-validates
lease ownership between claiming it and calling `client.charge()`, so the
stall can outlast the TTL with the worker still alive; (2) wall-clock time
passes the TTL; (3) a separate `reconcile()` pass sees an expired lease, no
`SENT` row, asks the provider (a true miss, so far), and — correctly, by
its own premises at that instant — voids the slot as `NEVER_SENT`; (4) the
stalled worker, unaware, finally writes `SENT` and charges for real; (5)
`void.reissue()` checks only `voided_at IS NOT NULL`, not whether a `SENT`
row has since appeared, so the now-voided slot is reissued and executed
again under a different key. Two charges, one slot, two keys, zero keys
individually double-charged.

**Why it wasn't caught earlier:** `eval/chaos.py`'s induced-kill mechanism
is single-process and single-threaded — every `_run_one()` call drives
exactly one `execute()` call to completion (or to an induced death) before
recovery ever runs. It can produce a *dead* worker racing recovery, never a
*live-but-slow* one, because a kill signal can only stop a process, not
delay it. That distinction is exactly the class of gap this review was
asked to look for.

**Fix:** none applied. Reported to the block owner per this review's scope
("do not fix, report what the fix should be"). Two candidates, not chosen
between here: (a) `executor.py` re-validates lease ownership (a fencing
token, or a `SELECT ... FOR UPDATE`-style re-check) immediately before step
3, so a lease that has been reassigned or invalidated underneath a stalled
worker aborts its send instead of completing it; (b) `void.reissue()`
re-checks for a `SENT` row at reissue time, not just at void time, closing
the TOCTOU window even if (a) is not done. (a) is the more fundamental fix
— it closes the race at its source rather than at its second-order
symptom.

**RESOLUTION, applied by the block owner after this review: the
`NEVER_SENT` fast path was REVERTED, not patched.**

First, whose regression this was, measured rather than argued. The same
stalled-worker sequence was run twice — once against `recover.py` as
written, once with the `NEVER_SENT` branch forced off to reproduce B9
behaviour:

```
current (NEVER_SENT on):  charges=2  reissue_succeeded=True
pre-change (UNCONFIRMED): charges=1  reissue_succeeded=False
```

So this was **introduced by B10's own slot-recovery optimisation**, not
inherited from B9. Without the void there is no reissue, so there is no
second key and no second charge. The 23-of-60 "wasteful conservatism" that
optimisation removed was partly load-bearing.

Neither candidate fix was applied. (b) — re-checking for a `SENT` row
inside `reissue()` — only narrows the TOCTOU window; the check still
precedes the new attempt's own charge, which by clause 6(a) lands at least
24h later. (a) — lease re-validation before step 3 — is the right fix and
is genuinely a fencing-token problem, i.e. new design work on B9's
executor, on the money path, after two auditors had already signed off on
a different design. Neither is something to bolt on at the end of a block.

What settled it is this project's own refrain, quoted in
`src/execute/recover.py`'s docstring and `PLAN_DETAIL.md` §1: **a
double-charge is worse than ten missed recoveries.** The optimisation
traded exactly that way — ten missed recoveries for one double charge — so
it goes. `recover.py` is back to B9 semantics, keeping only the
independent incident-4 `_dangling_keys` fix. The slot cost returns to
23/60 and is reported by `eval/chaos.py` as a standing measurement rather
than being silently accepted.

**Guard added:** `tests/eval/test_chaos.py::
test_a_stalled_worker_cannot_have_its_slot_voided_and_reissued`, marked
`chaos` and **green**. It asserts the property that makes step 5 of the
sequence impossible — recovery must never void the schedule row of a key
it merely believes unsent — rather than asserting the absence of the
symptom. It is discriminating by construction: with the reverted code
restored, `voided_at` is non-NULL and it fails.

**The generalisable lesson.** A proof about "what my own process did" is
not a proof about "what any process did", and the difference is invisible
until something is slow rather than dead. Two auditors (money, compliance)
cleared this design; both reasoned about crash-safety and concurrency and
both concluded the slot-freeing was safe. It took a reviewer explicitly
hunting for states the harness *cannot construct* to find it — the harness
is single-threaded, and a kill signal can only stop a process, never delay
one.

---

## Incident 6 — UNRESOLVED_FINAL is a permanent dead end even once the charge becomes findable

**When:** Block B10 review, 2026-08-30, same pass as Incident 5.

**Symptom:** The review was asked whether "last ledger row is not
terminal" (`eval/chaos.py`'s `lost_job` definition) could miss a job that
is effectively lost while showing a terminal row. Written up as
`tests/eval/test_chaos.py::
test_unresolved_final_is_a_permanent_dead_end_even_once_the_charge_is_findable`,
which drives a fault-seam charge (the provider genuinely accepts) with a
modelled index lag that outlasts `DEFAULT_MAX_UNCONFIRMED_PASSES`, lets
`recover.py` give up and write `UNRESOLVED_FINAL`, then proves the payment
is findable one lookup later and re-runs `reconcile()` five more times.
**Fails**: the ledger's last row for the key stays `FAILED`/
`UNRESOLVED_FINAL` forever; it never becomes `RESULT`/`RECOVERED` even
though the money is real, confirmed, and was one ask away the whole time.

**Root cause:** `recover._stuck_keys()`'s query is `WHERE state = 'FAILED'
AND reason = 'UNCONFIRMED'`. `UNRESOLVED_FINAL` is a different `reason`
string, so once a key reaches it, no scan in `recover.py` — not
`_dangling_keys` (latest row is terminal, so it's excluded), not
`_stuck_keys` (wrong `reason`) — ever examines that key again, for any
number of future `reconcile()` calls, ever. The design correctly refuses to
free the NPCI slot on an unconfirmed outcome ("a double-charge is worse
than ten missed recoveries," CLAUDE.md), so no money-safety invariant is
violated — but the *reporting* invariant is: a real, later-confirmable
charge is permanently misfiled as unresolved, with no code path that will
ever correct it, even though `razorpay_client.py`'s own docstring already
measured that the exact lag this depends on ("appeared minutes later") is
real and unbounded in principle.

**Why it wasn't caught earlier:** `eval/chaos.py`'s `KillOutcome.lost_job`
checks only whether the last row is non-terminal. `UNRESOLVED_FINAL` *is*
terminal (it's a `FAILED` row), so this state is invisible to the harness's
own oracle by construction — the review had to ask the question the
harness's definition doesn't ask.

**Fix:** none applied — reported, not fixed, per this review's scope.
Candidate: `_stuck_keys()` (or a separate, out-of-band job) should re-ask
about keys at `UNRESOLVED_FINAL` on some bounded, much slower schedule
(not the same backoff budget, or this just becomes an unbounded liability
in the other direction), closing the loop against real settlement data
rather than leaving it permanently open.

**Guard added:** `tests/eval/test_chaos.py::
test_unresolved_final_is_a_permanent_dead_end_even_once_the_charge_is_findable`,
marked `chaos`, currently failing (red) on purpose, for the same reason as
Incident 5's guard.

**Also probed, held:** the same review pass extended B10's kill mechanism
to interrupt `reconcile()` itself (never done by the original 50/14-kill
run — `_run_one` always calls `reconcile(conn, ...)` against the plain
connection, never the killing one), sweeping every statement index a
single `NEVER_SENT` dangling resolution touches
(`tests/eval/test_chaos.py::
test_reconcile_survives_being_killed_between_void_and_its_own_ledger_append`).
This one **passed** — `_resolve_never_sent`'s own documented claim about
surviving a crash between `void()` and its ledger append held under actual
chaos-testing, not just under the hand-traced reasoning in its docstring.
Recorded here because the coverage gap (this scenario was never
kill-tested at all) was real even though the code underneath it was not.

## Incident 7 — the benchmark client dropped the retry logic it was modelled on
**When:** Block B12, 2026-08-31, first full `--n 200 --repeats 5` run

**Symptom:** The benchmark printed its stats arm (`AUC 0.5759`), then died
partway into the first LLM arm with an unhandled
`google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED` —
`quotaValue: '15'`, `GenerateRequestsPerMinutePerProjectPerModel-FreeTier`.
The traceback ran straight out of `InstrumentedGemini.probabilities` with no
retry attempted. Roughly 200 of ~1,200 planned calls had been made. Nothing
was written: `reports/bench.json` is only written after every arm completes,
so the partial run left no half-table to mistake for a real one.

**Root cause:** `bench/llm_vs_stats.py`'s `InstrumentedGemini` is a
deliberate near-copy of `src/llm/client.py`'s `GeminiClient` — the forced
calling config was copied byte-for-byte, and the module docstring says so.
The retry behaviour was not copied. `GeminiClient._call_with_backoff` retries
exclusively on 429 with `_MAX_RETRIES = 6` and `_DEFAULT_RETRY_DELAY_S =
15.0`, precisely because the free tier allows ~15 requests/minute per model;
the bench client had no backoff at all. A benchmark that issues 1,200 calls
as fast as it can is the single most rate-limit-exposed caller in this
repo, and it was the only one with no handling.

**Why it wasn't caught earlier:** The smoke test was `--n 4 --repeats 2
--variance-n 2` — 12 calls, comfortably under the 15/min quota, so it
exercised every line of the call path except the one that matters at scale.
Every unit test in `tests/eval/test_bench.py` is offline by design (no
network, no live calls), which is right for testing metric definitions and
wrong for catching this: the defect lives entirely in the transport layer
those tests deliberately never touch. The class of bug — "reimplemented a
client and silently dropped one of its behaviours" — is invisible to both a
green unit suite and a small live smoke.

**Fix:** Added 429-aware backoff to `InstrumentedGemini`, honouring the
server-supplied `retryDelay` when present and falling back to
`GeminiClient`'s own 15s default. Critically, the retry sleep is excluded
from the recorded latency: only the successful attempt is timed. Sleeping
inside the timed region would have inflated the p95-latency column by tens
of seconds per throttled call and produced a benchmark number measuring this
project's own quota tier rather than the model's response time — a wrong
number that would have looked plausible.

**Guard added:** `tests/eval/test_bench.py` now covers the retry path with a
fake transport that raises a synthetic 429 before succeeding, asserting both
that the call is retried AND that the recorded latency excludes the wait.
The second assertion is the load-bearing one: a retry that silently
poisoned the latency column would still have passed a retry-only test, and
the latency column is one of the four this block exists to produce.
