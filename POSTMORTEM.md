# POSTMORTEM — what broke during the build

The rubric line is *"Failure recovery — what broke, and what you did about
it."* That is asking about this build, not runtime resilience. Entries are
written **at the moment of breakage**, before the cause is known. Do not
backfill a tidy story, and do not delete an entry because it turned out to
be your own mistake — those are the valuable ones.

Use the `/log-incident` skill. Format:

## Incident 1 — coupled arm fabricated money in the B2 freeze

**When:** Block B2, 2026-08-26, minutes after the freeze commit (`8321406`).

**Symptom:** the payments-domain review — dispatched as B2's required gate review,
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
`src/execute/recover.py`'s docstring and the build spec §1: **a
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
than ten missed recoveries," DESIGN.md), so no money-safety invariant is
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

## Incident 8 — a daily quota nobody had read, and 400 paid calls thrown away
**When:** Block B12, 2026-08-31, third full benchmark run

**Symptom:** The run completed the whole `gemini-3.5-flash-lite` accuracy
pass (200 calls) and all five variance repeats at t=0.0 (200 more), then died
on a 429 whose body was **not** the per-minute quota this run had been
carefully paced against:

```
quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'
quotaValue: '500'
```

`reports/bench.json` was not written. Every one of those 400 calls was lost.

**Root cause:** Two independent mistakes that only bite together.

*The budget was never checkable.* Incident 7 fixed the per-minute limit and
the module docstring reasoned confidently about "~15 requests/minute per
model", sizing the run at ~1,200 calls across two models. There is also a
**500/model/day** cap, and 600 calls per model was over it before the first
run started. The arithmetic was done against the only limit that had already
failed, which is a poor way to choose which limits to check.

*Results were written once, at the end.* `main()` builds every arm and then
writes `reports/bench.json`. Any failure at call 400 of 600 discards calls
1–400. This is precisely the pattern `eval/golden_check.py:_persisting()`
was built to avoid at B11 — it flushes each fresh live answer to disk
immediately "so an interrupted run doesn't re-bill" — and that precedent was
read and cited while planning this block, then not applied. The irony is
sharper still: this same run had just been restarted specifically to add
`variance_runs` persistence so raw probabilities would never need re-buying,
and the persistence added was end-of-run persistence, which is no protection
at all against the thing that actually happens.

**Why it wasn't caught earlier:** The smoke tests were 12 and 20 calls. Both
rate limits are invisible below a few hundred calls, and the daily one is
invisible below 500 — so no test short enough to run casually can reach it.
Nothing in the repo asserted a call budget against a documented quota, so
the sizing lived only in a docstring's prose where it could not be wrong out
loud.

**Fix:** An on-disk JSONL cache, flushed after every single live call and
keyed by (model, temperature, repeat, row) plus a hash of the prompt and
tool schema, so a re-run resumes instead of re-billing and a prompt edit
correctly invalidates. Plus an explicit pre-flight budget check that computes
the call count from the actual arguments and refuses to start a run that
cannot fit inside `DAILY_QUOTA_PER_MODEL`, naming the arguments that would.

**Guard added:** `tests/eval/test_bench.py` covers both — that `plan_budget()`
rejects an over-quota configuration before any client is constructed, and
that a second `_score()` pass over identical inputs issues **zero** live
calls. The second is the load-bearing one: a cache that silently missed
would be indistinguishable from no cache at all right up until the next
500-call bill.

**Sequel, same day — the fix was still wrong.** The re-run, resized to 440
calls to fit "the" 500/day cap and pointed at `gemini-3.5-flash`, died after
about 20 calls:

```
quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'
quotaValue: '20'
quotaDimensions: {'model': 'gemini-3.5-flash'}
```

**The daily cap is not the same for every model.** flash-lite allows 500/day;
flash allows **20**. `DAILY_QUOTA_PER_MODEL = 500` — the constant added two
hours earlier specifically to stop this — waved a 440-call run straight
through against a cap of 20, because it encoded the one number that had been
measured as though it were the shape of the world. The same error as the
original, one level up: reasoning about a limit from the single instance that
had already bitten.

Replaced with `DAILY_QUOTA_BY_MODEL`, both values measured from real 429
bodies, and `daily_quota()` defaulting an unmeasured model to the **smallest**
observed cap rather than the largest — guessing high is the whole failure
mode. Tested against the exact configuration that failed.

**What the cache bought, immediately:** 21 flash answers survived on disk and
will be reused rather than re-billed. The incident-7/8 fix paid for itself
inside one run of being written, which is the only reason this sequel cost
20 calls instead of another 400.

**Consequence for B12:** both models' daily quotas are now exhausted
(verified by probe: flash-lite 429/PerDay/500, flash 429/PerDay/20), so the
benchmark's LLM arm could not be completed and **its variance column is
still unmeasured**. The stats and null arms are complete and reproducible
offline. The gate was subsequently ticked on an explicit human scope
decision — the tick records that nothing else was outstanding, not that the
variance number exists. It does not.

---

## Incident 9 — the gate verification and the reader used different browsers
**When:** Block B15, 2026-09-03, hours after B15 was ticked

**Symptom:** The human opened the landing page on their own machine and saw
the no-WebGL fallback — "What the engine does, without the animation" —
where the 3D scene should be. The same machine, same build, renders that
scene at a steady 59.9fps. The page's headline feature was invisible to the
one reader it had been built for, and I had ticked its gate that morning.

**Root cause:** `useWebGLSupport.ts` probed for a context with
`failIfMajorPerformanceCaveat: true` and treated a refusal as "no GPU here".
That flag does not mean that. It means the browser is being conservative,
which it also is with hardware acceleration toggled off, a driver on the
blocklist, a stale GPU process, or a profile simply carrying different flags
from a fresh one. The refusal is a statement about the browser's mood, not
about the hardware, and it was being read as a hardware verdict.

**Why it wasn't caught earlier — and this is the part worth keeping:** it
was not caught *because of how it was verified*. The B15 canvas-failure
criterion was checked by launching Chrome with `--disable-gpu`, and the fps
criterion by launching Chrome against a throwaway `--user-data-dir`. Both
launches produced a **clean profile with default flags**, which is exactly
the configuration that passes the strict probe. Every automated check agreed:
build green, lint green, render-check green, four fps runs at 60fps, the
fallback firing correctly under `--disable-gpu`. The verification was
thorough and it was self-consistent and it was measuring a browser that no
reader has.

Both halves of the probe's contract were tested. What was never tested was
the case in between them — a real GPU that the browser is being cautious
about — because a fresh automation profile is never in that state.

**Fix:** The probe is tiered. The strict ask is now only the first question;
if it is refused, ask again on looser terms and inspect the renderer string
that comes back. Hardware renders the scene at reduced cost (`degraded`);
only a software rasteriser — which would genuinely crawl — reaches the HTML
fallback, which is the case that fallback was written for. The same
hard-coded flag in `Scene.tsx`'s Canvas `gl` config now follows the tier the
probe established, rather than independently re-failing inside the Canvas
where nothing throws.

**Guard added:** the degraded path is now verified by overriding
`HTMLCanvasElement.prototype.getContext` over CDP to refuse every request
carrying `failIfMajorPerformanceCaveat: true`, then asserting the canvas
still mounts. That reproduces the human's machine on demand, which is the
thing no flag combination gave us. A `console.info` naming the tier and the
renderer string is left in the shipped page deliberately: when the next
reader says "I only see the fallback", that one line is the difference
between diagnosing it and guessing.

**Consequence for B15:** the gate's four criteria did hold, and still hold,
re-verified on the shipped build. But the tick was taken on evidence from a
browser configuration that was not representative, and a human found in one
screenshot what four automated runs had missed. The gate note carries a
correction saying so. The lesson generalises past this bug and directly into
B16: **an automation harness that spawns a clean profile is not a witness to
what a reader sees.** The video capture for B16 will run in the same kind of
throwaway browser, and is exposed to the same class of error.

---

## Incident 10 — 132 money-critical tests skipped, and the suite exited 0

**When:** Block B16, 2026-09-03, found by the human while auditing what was
actually deliverable

**Symptom:** `.\run.ps1 test` reported success on a machine where Docker was
not running. The report line was "781 passed, 132 skipped". What skipped was
every test that touches Postgres: `tests/ledger/`, `tests/execute/`
(executor, lease, void, recover, commit), `tests/ingest/` (webhook, dedupe,
lifecycle route) and `tests/eval/test_chaos.py`. That is the entire
idempotency and crash-recovery surface — every test that exists to prove the
ledger write happens before the money action, that an attempt cannot double-
charge, and that a crash mid-flight resolves rather than hangs.

DESIGN.md's definition of done, step 3, is "`.\run.ps1 test` passes before
any commit". That step was satisfiable, green, and signed off, without ever
running the money path.

**Root cause:** `tests/conftest.py`'s `pg_schema` fixture called
`pytest.skip` when Postgres was unreachable, on the reasoning — written down
in its own docstring — that an unreachable database is an environment
problem, not a code one. That reasoning is correct about *whose fault it is*
and wrong about *what to do*. A skip is an assertion that the test did not
need to run. Here it meant the opposite: the tests that most needed to run
were the ones that did not.

**Why it survived this long:** it never produced a red line. Every session
that ran the suite with Docker down saw green, and the skip count sat in a
summary line that reads as noise. The pass/skip counts were even copied into
the status notes at each checkpoint — "781 passed, 132 skipped" — where they were
recorded faithfully and read by nobody as a problem, including by me.

This is the same defect class as the `Invoke-Step` bug found in the B13
end-of-project pass (recorded in run.ps1 itself, 2026-08-31): a bare
`& $Py ...` as the last statement of a switch branch left the script exit
code at 0, so `.\run.ps1 test` reported success on a RED suite. Different
mechanism, identical shape — **a check that passes by not checking** — and
it made the same definition-of-done step unfalsifiable, from the other end.
Incident 9 is a third instance: a verification whose green came from
measuring something other than the thing. Three occurrences in twelve days
is not a coincidence; it is the failure mode to look for first.

**Fix:** `require_pg` in `tests/conftest.py`. An unreachable Postgres now
FAILS every test that needs it, with a message naming the fix
(`.\run.ps1 up`). The skip still exists, because a docs-only machine is a
real situation, but it has to be asked for by name:
`MANDATEIQ_ALLOW_PG_SKIP=1`, and the skip reason then says so, so it appears
in a log as a decision rather than as weather. `tests/ingest/test_deps.py`
had its own hand-rolled copy of the same skip and now takes the shared
`pg_required` fixture.

**Guard added:** `tests/test_pg_guard.py` — twenty-one tests over the opt-out
parsing (`=0` and set-but-empty are refused, not treated as consent) and the
fail-vs-skip branch, plus one test that scans every `test_*.py` in the tree
for a `pytest.skip` mentioning Postgres and fails if it finds one. That last
one is the important one: the way this hole reopens is not by someone
reverting the fix, it is by someone adding a second skip somewhere else.

Verified in both directions rather than asserted: against a dead DSN the
suite produces 22 errors where it used to produce 22 skips; with
`MANDATEIQ_ALLOW_PG_SKIP=1` the same run produces 22 skips, each naming the
variable. With Postgres up, 913 tests pass and nothing skips — so the 132
skips had been hiding no failures, only hiding themselves.

**Two stale skips removed while in there:**
`tests/model/test_conformal.py` guarded two LLM-import invariant tests with
`except FileNotFoundError: pytest.skip("conformal.py does not exist yet")`,
left over from the TDD red state. `src/model/conformal.py` has existed since
B6; deleting it would have turned both invariant tests green. They now read
the file unguarded.


## Incident 11 — fixing the OPTED_OUT belief bug made a previously-unreachable conformal singleton reachable, and a published invariant test caught it

**When:** Post-B16 remediation, block R2, 2026-09-04

**Symptom:** After implementing R2's fix (belief.observe_terminal() collapses
belief on an observed DEAD/OPTED_OUT outcome; eval/run.py's post-terminal
re-solve now actually runs for OPTED_OUT, which it previously skipped
entirely), the full model/eval/policy test suite showed one new failure:
`tests/eval/test_export_mandates.py::test_the_wont_pay_singleton_is_the_unreachable_one`.
Its assertion `("WONT_PAY",) not in singletons` failed —
`[('CANT_PAY_NOW',), ('CANT_PAY_NOW',), ('WONT_PAY',), ...]` — the
`{WONT_PAY}` conformal singleton, which this test's own docstring called
"genuinely unreachable," had appeared.

**Root cause:** Two independent things, confirmed rather than assumed:

1. `src/policy/allocator.py`'s `_build_plan()` computes `gate.pred_set(b0)`
   UNCONDITIONALLY for every `solve()` call, purely for the Plan's own audit
   record (the drill-down's "conformal set" field) — regardless of whether
   OFFER was ever an eligible candidate action for that call.
2. R2's fix makes the OPTED_OUT re-solve actually happen (previously it was
   skipped outright — the exact bug R2a exists to fix), and that re-solve's
   belief is now `observe_terminal(b, Cause.WONT_PAY, ...)` — a DEGENERATE
   (0, 0, 1.0) posterior. Fed through a real, well-calibrated conformal
   predictor, a 100%-confident belief naturally produces a singleton — that
   is the gate working correctly, not a defect in it.

The combination means: a Plan object now exists (for the first time) whose
`conformal_set` is genuinely `{WONT_PAY}` — but it is a RETROSPECTIVE record
on a mandate that has ALREADY opted out, at a decision point where
`permitted(Action.OFFER, ctx)` is unconditionally DENY (clause 6(c): opt-out
denies every action but STOP). Verified directly, not assumed: I ran an
identical 60-mandate diagnostic against the pre-fix code (via `git stash`,
confirmed clean revert and clean restore afterward) and against the fixed
code. Pre-fix: 0 WONT_PAY singletons, `n_stop=0`, `n_attempt_after_terminal=6`
(the bug, reproduced). Post-fix: 11 WONT_PAY singletons across 60 mandates,
`n_stop=11`, `n_attempt_after_terminal=0` — and in both runs, `OFFER` was
chosen exactly 0 times. The singleton is new and real; the actionable claim
("OFFER is never chosen") is untouched.

**Why it wasn't caught earlier:** The original test conflated two different
claims under one assertion: "OFFER is never the chosen action" (an
operational fact about `permitted()` and the Q-value comparison) and "the
{WONT_PAY} singleton never appears in any recorded conformal_set" (an
audit-trail fact about what `gate.pred_set()` was ever asked). R2's fix
changes the second without touching the first, and nothing before this
session had ever exercised a re-solve on a fully-degenerate, post-opt-out
belief — the OPTED_OUT re-solve literally never ran until this fix landed,
so the gate had never been queried on this exact kind of belief before.

**Fix:** `tests/eval/test_export_mandates.py`'s test renamed to
`test_the_wont_pay_singleton_is_unreachable_via_live_inference` and rewritten
to check the precise claim: no LIVE decision (one whose belief provenance
lacks the `;observed=terminal` marker `observe_terminal()` stamps) ever
produces the `{WONT_PAY}` singleton, and — unconditionally, regardless of
what any conformal_set records — `OFFER` is never the chosen action. Also
logged in DECISIONS.md alongside the rest of R2's review-pass findings.
README.md and the dashboard's static copy (`Acquirer.tsx`, `Drilldown.tsx`),
which both currently claim the singleton "never appears" rather than "is
never chosen," are queued for a wording correction once the full 8-seed
sweep is re-run and the real published `singleton_wont_pay_rate` number is
in hand (R8-style republish) — not edited speculatively ahead of that
number.

**Guard added:** The rewritten test itself is the guard: it distinguishes
"singleton on a live decision" (still asserted unreachable, would fail loudly
if inference alone ever reached it — the R5-relevant, actionable case) from
"singleton on a retrospective, definitionally-collapsed belief" (now
correctly permitted). A future change that made the singleton reachable via
ordinary belief inference — the change R5 is actually meant to make — will
still be caught by this same test, at the assertion that matters.


## Incident 12 — a bug fix's own fix asserted false certainty, and review caught it before the gate closed

**When:** Post-B16 remediation, block R2, 2026-09-04, same day as Incident 11

**Symptom:** R2's fix for the DEAD/OPTED_OUT terminal-outcome bug (Incident
11's context) was sent to the payments-domain review for adversarial review before
the gate was ticked. The review's central claim: `belief.observe_terminal()`
collapsed belief to an exact `(1.0, 0.0, 0.0)` posterior on the reasoning
"an observed DEAD outcome means CANT_PAY_EVER -- that is what the cause
label MEANS, not a hypothesis about it" -- and this project's own frozen
`eval/frozen/sim_config.yaml` contradicts it.

**Root cause:** Verified directly, not taken on the reviewer's word: a
200-seed direct simulation (driving `Simulator("nominal", seed=N)` to each
mandate's first DEAD/OPTED_OUT outcome, scored against the privileged
`m.initial_cause` ground truth) measured **P(CANT_PAY_EVER | DEAD) = 0.899**
and **P(WONT_PAY | OPTED_OUT) = 0.904** -- both cross-validated on a disjoint
300-seed range (0.893 / 0.909). Roughly 10% of each terminal outcome has a
different true cause, since `CANT_PAY_NOW`/`WONT_PAY` carry a low but
non-zero `base_dead` rate (0.02, against `CANT_PAY_EVER`'s 0.55), and
similarly for `base_optout`. The degenerate collapse was additionally
IRREVERSIBLE: `cause_map._PRIORS` contains no zeros, so `update()` on an
exact `(0, 1, 0)` belief returns it unchanged forever -- an absorbing state
nothing else in this codebase's belief model can reach, dormant only
because no belief in this eval harness survives past a mandate's own
terminal outcome.

The same review pass found two further, independent bugs while examining
this code: `src/policy/allocator.py`'s `_binding_constraint()` never
checked the new `instrument_dead` field, so a REAUTH forced by that rule
alone recorded `binding_constraint = None` -- the audit trail stating a
hard-forced decision was a free economic choice. And fixing OPTED_OUT's
re-solve meant the conformal gate was queried for the first time on a
belief `observe_terminal()` had already collapsed, contaminating the
coverage/singleton-rate diagnostic the off-ramp's whole safety claim
depends on (this is the mechanism behind Incident 11's nonzero singleton
rate, which that incident's own "confirmed inert" language addressed only
for the ACTION, not the MEASUREMENT).

**Why it wasn't caught earlier:** The docstring's own justification
("that is what the cause label MEANS, not a hypothesis about it") was
written and believed without checking it against the one artifact in this
repo that could confirm or refute it -- the frozen simulator's own
generative process. It reads as principled reasoning about what a Cause
label definitionally means, but it is actually an empirical claim about
this specific DGP, and empirical claims here get checked, not asserted.

**Fix:** `observe_terminal()`'s signature changed from `(b: Belief, cause:
Cause, *, source_version)` to `(cause_probs: Mapping[Cause, float], *,
source_version)` -- no prior-belief parameter, matching `init()`'s own
shape -- and `eval/run.py` now supplies the measured distributions via a
new, fully-cited `_TERMINAL_OBSERVED_CAUSE_PROBS` constant instead of the
module assuming a degenerate one. `_binding_constraint()` now checks
`instrument_dead` first. `_RecordingGate` tags each query live/retrospective
and coverage statistics are computed over live queries only. Re-ran the
full 8-seed sweep after all three fixes: every action count is
byte-identical to the flawed version (REAUTH's economics dominate at 90%
confidence exactly as they did at 100%), `singleton_wont_pay_rate` is back
to exactly 0.000 in every engine cell, and `n_attempt_after_terminal`
remains 0 throughout. Full account in DECISIONS.md, 2026-09-04, "R2 review
pass."

**Guard added:** `tests/policy/test_allocator.py::
test_instrument_dead_denies_attempt_and_names_itself_as_the_binding_
constraint` (the binding-constraint regression). The belief-collapse
question has no single mechanical guard -- the guard here is procedural:
this project's own review-before-gate-closure discipline, which is what
actually caught it. `_TERMINAL_OBSERVED_CAUSE_PROBS`'s derivation is left
fully cited in `eval/run.py` so a future re-measurement (e.g. if
`sim_config.yaml`'s hazard rates ever change) has a documented method to
repeat, not just a number to trust.
