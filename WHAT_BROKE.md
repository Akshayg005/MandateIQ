# What broke, and how we fixed it

`POSTMORTEM.md` (12 incidents, ~860 lines) and `DECISIONS.md` (~6,300 lines)
are the complete record. Nobody reads either end to end. **This file is the
short version: the failures that actually taught something, and the review
findings that changed a published conclusion.**

Every item here is reproducible from the repo. Where a number appears, the
file that produced it is named.

---

## The one pattern worth taking away

**A green test suite was never once the thing that caught a real bug.** In
all twelve incidents, the defect was found by a live call, a chaos run, a
manual verification step, a review agent asked an adversarial question, or a
human looking at their own screen — *after* the tests passed. Three of them
were found by the very first real exercise of a path that had only ever been
tested against fakes.

The corollary, which cost more than any single bug: **a check that cannot
fail looks exactly like a check that passes.** Four of those shipped, and
all four were found by asking "what would make this red?" rather than by
running it again.

---

## Bugs in the money path

**The recovery interface was dead on arrival, and 78 green tests said
otherwise** (Incident 3, B9). `find_by_receipt()` — the entire
recover-by-asking path after a crash — filtered *payments* by `receipt`,
which the Razorpay API rejects outright, because `receipt` is an *order*
field. Every test passed because every test used a fake, and a fake accepts
whatever shape it is handed. Found by the first live test-mode call ever
made, run after the block was otherwise complete. Fixed by anchoring the
attempt to an Order that carries the key as its receipt. A second finding
from the same session: the Orders list endpoint lags indexing by ~30s, so
"not found" means *not found yet*, never *never sent*.

**A permanently lost job, invisible to both scans** (Incident 4, B10).
Crash recovery discovered abandoned work by iterating expired *leases* — so
an INTENT row written before the lease was claimed was invisible to it
forever, while the idempotency guard blocked any re-execute. Slot consumed,
customer never debited, nothing reported. `lease.py`'s own docstring names
the rule that was broken: the lease is "an optimisation over
`ledger_intent_once`, not the concurrency control." Fixed by scanning the
ledger and using the lease only to exclude keys a live worker still holds.
The regression test was proven discriminating against a verbatim
reimplementation of the old algorithm, not merely asserted to be.

**A fix that introduced a cross-generation double charge, reverted the same
day** (Incident 5, B10). Recovering NPCI slots burned by never-sent attempts
was approved, built, cleared by the money audit **twice** and by
The compliance audit — then found to double-charge. Its proof ("no SENT row
means no call was issued") is true of one process's own state and false
about a concurrent one: a worker stalled past its lease TTL is alive, not
dead, and is indistinguishable from a crash in durable state. Recovery
voided a live worker's slot; the worker completed its real charge; the freed
slot was reissued under a different key and charged again. Measured directly:
2 charges with the fix, 1 without. Reverted, not patched — a double charge is
worse than ten missed recoveries. **The standing cost is 23 of 60 slots
burned**, reported rather than quietly accepted.

**What found it matters more than the bug.** Three review passes reasoning
explicitly about concurrency all cleared this design. What caught it was a
reviewer asked not *"is this correct?"* but *"what states can this harness
NOT construct?"* — the harness is single-threaded, and an induced kill can
only stop a process, never delay one, so a live-but-slow worker was outside
its reachable state space entirely.

**Webhook writes discarded behind HTTP 200** (Incident 2, B3). A validly
signed `payment.failed` returned `{"status": "ok"}` and wrote nothing:
psycopg3 defaults to `autocommit=False`, so every write ran in an implicit
transaction that nothing committed, and closing the connection discarded it
silently. All 235 tests and the guards were green. Found by querying the real
database after a real request.

---

## Bugs in the evaluation

**The frozen simulator fabricated money** (Incident 1, B2 — minutes after the
freeze commit). The `coupled` arm recovered ₹3,91,412 against ₹2,30,732 of
total household liquidity — 1.7× the money that existed. A "partial
liquidity" branch credited the mandate's *full* amount while debiting the
household only down to zero. It also modelled nothing real: UPI AutoPay has
no partial-debit semantics. Caught by the gate review that block required,
before any policy code existed.

**132 money-critical tests were skipping, and the suite exited 0**
(Incident 10). Windows had reserved the TCP range containing 5432, so the
Postgres container could not bind, so every ledger/executor/lease/recover
test skipped — quietly, while the suite reported success. The whole money
path was unverified and looked verified. The database now moves to port
15432, and a missing database **fails** the suite instead of skipping it.

**The verification browser and the reader's browser were different machines**
(Incident 9, B15). Four consecutive green automated runs confirmed the 3D
landing page rendered. On the human's own Chrome, same laptop, same build, it
served the no-WebGL fallback instead: the probe asked for a context with
`failIfMajorPerformanceCaveat: true` and read the refusal as "no GPU," when
it only means the browser is being conservative. A clean automation profile
is never in that state, so no run here could have caught it. Found in one
screenshot by a human. **This is why the video pre-flight insists on
recording in a real everyday browser profile.**

**A quota nobody had read, and 400 completed calls thrown away** (Incidents
7 and 8, B12). The benchmark's Gemini client was a deliberate near-copy of
the production one — the forced-calling config copied byte-for-byte, the
retry/backoff logic not copied — so the first 429 killed the run. The rerun
then hit a *daily* cap nobody had looked up, and because results were written
only after every arm completed, ~400 already-paid-for calls were discarded.
Both fixed (retry restored; per-model measured quota table defaulting an
unknown model to the smallest observed cap; a call cache that banks partial
work).

---

## Findings that changed a published conclusion

These were not crashes. Each one made a claim in the README false, and each
was corrected rather than left standing.

**"The off-ramp never fires" was arithmetic, not a measurement** (B13
review). `OFFER` was chosen 0 times in every published run — and the reason
was not that the model declined to offer. The proxy decline alphabet gave
`WONT_PAY` an identical likelihood under both symbols it could emit, so
P(WONT_PAY) was pinned at 0.10 and the `{WONT_PAY}` singleton the gate
requires was unreachable **for any alpha, seed or regime**. The lane was
untested, not tested-and-negative. Fixed at R5 by adding a
`CUSTOMER_DECLINED` class and an evaluation channel that emits it — and that
channel reads the simulator's privileged ground truth, which is disclosed
everywhere the resulting number appears.

**The headline comparison could not be falsified** (B13 review). The engine
beat the fixed ladder on mandates preserved in every cell — but `null` (never
attempt) preserves 200/200 everywhere, and `one_shot` (one attempt, no model,
no belief, no gate) preserved more than the engine while spending fewer
attempts. Every metric was monotonically decreasing in attempt count by
construction, so "preserves more" followed from "attempts less" and was not
evidence of cause inference at all. Both baselines are now first-class arms
in the table, the figures and the README. **A test asserting the engine
always spends fewer attempts than the ladder was deleted: it pinned the
confound by test.**

**The conformal gate's reported coverage was an artifact — twice.** First
(B13): the smoothing key was derived from the belief itself, which collapsed
the WONT_PAY p-value to a hash of a constant, and coverage was scored over
200 slot-1 beliefs instead of the ~4,700 queries the gate actually receives.
The cited 0.980 was meaningless. Second (R5→R8, this is the deepest one): the
calibration pool held only 2–3 distinct values per class and topped out well
below the confidence a real multi-decline trajectory reaches — a **support
mismatch**, so the fitted threshold barely moved across an 8× range of alpha.
Fixed at R8 by calibrating across each mandate's own slot 2/3 trajectory
(200 → 333 rows). Effect on the published numbers: `OFFER` 1292 → **300**,
false-off-ramp rate 15.5% → **1.3%**, per-class coverage 0.795–0.986 →
**0.836–1.0**. Still short of the 0.95 target on one class, and said so.

**A widened model made things measurably worse, and shipped as a finding**
(R1a). Adding amount and category to the design matrix was expected to help.
Pooled out-of-fold log loss, clustered by mandate: mean difference +0.00103,
t = +2.88, p = 0.0040 — **worse**, not merely no better, and 0 of 18 new
coefficients had a CI excluding zero. Published as the result rather than
reframed as a failure. The first version of that test was itself wrong (a
normal approximation on 4 degrees of freedom), and that is disclosed too.

---

## The four checks that could not fail

1. **`run.ps1 test` returned 0 on a red suite** — a bare call in a switch
   branch does not set the script exit code, which made "tests pass before
   any commit" unfalsifiable. Proven with a repro, then fixed.
2. **The golden-set freshness check tested file existence**, so a
   quota-killed run leaving 1 of 30 rows cached reported "current."
3. **A coverage test asserted `0.0 <= coverage <= 1.0`** — true of any
   probability, so a total calibration collapse would have passed silently.
4. **The B8 gate said "zero constraint violations"** — trivially satisfied by
   an allocator that never attempts anything. Amended *before* any allocator
   code existed, with an attempt-rate floor and a discrimination margin
   derived from the simulator's own parameters, both proven to reject a real
   failing case by test.
