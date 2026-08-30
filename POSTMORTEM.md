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
