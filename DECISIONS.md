# DECISIONS

Where a model was used, where one deliberately was not, and why.

The Buildathon rubric scores *"AI judgment — the right tool in the right
place, and where you chose not to use one."* An assertion scores nothing.
Every "no" row below should point at a measurement.

## Where we chose NOT to use a model

| Subtask | What we used | The alternative | Why not the alternative |
|---|---|---|---|
| Retry timing | Discrete-time competing-risks hazard model | LLM as classifier | *(fill from `make bench` — AUC, p95 latency, cost/1k, run-to-run variance)* |
| Attempt allocation | Exact backward induction over 4 slots | RL / heuristic search | State space is small enough to solve optimally. An approximation here is a choice to be wrong for no reason |
| Decline → cause mapping | Deterministic taxonomy | LLM classification | Deterministic and auditable. The LLM only normalises *unseen string variants* into this taxonomy — it never assigns the cause |
| Money arithmetic | Integer paise | — | No model touches arithmetic. Enforced by `scripts/guard_invariants.py` |
| Constraint checking | Hard-coded clauses with citations | Prompted compliance check | A regulatory constraint that can be talked out of is not a constraint |

## Where we DID use a model

| Subtask | Model | Why a model is right here |
|---|---|---|
| Decline-string normalisation | Haiku | Issuer strings are unstandardised free text; new variants appear weekly. This is genuinely a language task |
| Cancellation-intent extraction | Haiku | Support tickets, including Hinglish. No feasible rule set |
| Merchant root-cause narrative | Sonnet | Once per batch, not per transaction. Writing, not deciding |

## The benchmark

*(paste the table from `make bench` here — and if the LLM wins on AUC, say
so and explain why it still does not ship.)*

## Decisions log

*(append dated entries as design choices are made and reversed)*

### 2026-08-26 · B0 · Toolchain moved 3.11 → 3.13

All 22 dependencies import clean on 3.13; `.\run.ps1 verify` passes 5/5 against
live test mode. No reason to build on an older interpreter than the one the
statsmodels / lifelines / scikit-learn stack already supports.

### 2026-08-26 · B0 · Work keyed to dependency blocks, not calendar days

`PLAN.md`'s `Day N` headings are superseded by `B0`–`B16` in
`reports/gates.md`. A block is sized by what it can *prove*, so two may close
in one sitting and one may take three. Progress is gates passed, never blocks
started. Calendar days encouraged ticking a gate because the day was over.

### 2026-08-26 · B1 · The frozen config carries three arms, not two

`nominal` + `misspecified` + **`coupled`** (mandates share one household
balance; our own debit consumes it).

*Why.* `payments-domain` found that `solve()` is per-mandate with no batch
capacity term, so every `CANT_PAY_NOW`-dominant mandate computes the same
argmax day and the allocator builds a debit storm. Worse, when our own first
debit consumes the balance the second returns `INSUFFICIENT_FUNDS`, which
`cause_map` reads as evidence *for* `CANT_PAY_NOW` — the system misreads its
own success as the customer's liquidity failure. The `misspecified` arm varies
functional form while holding independence fixed, so it cannot catch this.

*Why now.* B2 is the only window. After the freeze commit the arm cannot be
added without invalidating the pre-registration, and a de-risk added at B13
is an excuse rather than a measurement.

*Consequence.* `baseline_ladder` runs under all three arms. `gates.md` B5 was
edited from "both frozen arms" to "all three" at plan time — before the freeze
commit, before any file under `src/policy/` existed, before any result was
seen. The reason is recorded inline in `gates.md`.

### 2026-08-26 · B1 · `committable_days` returns ~4 structural candidates

Pre-salary · salary window (days 1–5) · post-salary · month-end, each carrying
its replenishment-rhythm rationale in a docstring.

*Why.* `payments-domain` correctly noted that "a few thousand nodes" ignored
date branching: the tree is `(|days|·|DeclineClass|)^4`, so ~20 arbitrary days
gives `≈10^8` and the exact-solve claim would have to be dropped. Partially
pushed back — the fix is to restrict the day set and *say so*, not to abandon
exact solving for RL or beam search. At 4 candidates the tree is `(4·8)^4 ≈
10^6` before memoisation, and the 2-slot brute-force equivalence test at B8
stays tractable. "Solved exactly" then has a test behind it.

### 2026-08-26 · B1 · The off-ramp ships behind a Protocol, with a safe stub

`src/policy/gate.py` defines the `ConformalGate` Protocol. `FullSetGate`
returns the full prediction set, so the singleton-`{WONT_PAY}` rule never fires
and the policy degrades to "never offer an off-ramp".

*Why.* It decouples B8 from B6 without weakening the safety claim: the stub's
failure mode is declining to offer, which is the safe direction. A B6 slip then
costs the off-ramp lane and the coverage claim, not the allocator.

*Consequence.* `eval/run.py` records which gate was active, and B13's report may
make the 95%-coverage claim only for runs where the real `conformal.py` gate was.

### 2026-08-26 · B1 · money-auditor review — two real bugs fixed, one fix rejected

`money-auditor` reviewed the six B1 files before the gate was ticked (required
by the definition of done: any diff touching money or the ledger). Findings,
and what was done with each:

**Fixed — `pct_of` could overshoot the true floor by 1 paise on large
amounts.** `int(p * frac)` computes the floor of a float *product*, and float
multiplication itself can round that product up past the true mathematical
floor once `p` is large enough — independent of any imprecision in `frac`'s
own representation. Confirmed by execution: `pct_of(7_079_410, 0.7)` returned
`4955587` via the old code; the exact floor (computed via `Fraction`) is
`4955586`. Fixed by computing `(Fraction(p) * Fraction(frac)).__floor__()`
instead — this takes the exact IEEE-754 value of the float `frac` (it does not
"fix" the input) and removes only the extra rounding error the multiply step
was adding. `frac: float` stays the signature; the fraction being computed is
a rate, not a money value, and `pct_of`'s own money argument (`p`) was never a
float. (One of the auditor's two example inputs, `pct_of(100, 0.29)`, turned
out on inspection to already be correct — `float(0.29)` is not exactly decimal
0.29, and `28` is the true floor of the actual double being multiplied. Worth
recording so a future session doesn't "fix" a case that was never broken.)

**Fixed — `idempotency_key` did not coerce numeric inputs.** `500` and `500.0`
derived different keys, because the fields were `str()`-concatenated without
an `int()` cast first. Two code paths that agree numerically but differ in
type (plausible once B8's allocator does arithmetic outside `money.py`) would
then collide as "the same attempt" while deriving different keys —
`ledger_intent_once` would not catch the duplicate, and a retry could reach
the provider a second time. Fixed by coercing every numeric field with `int()`
before stringifying. Also added a delimiter-safety check (`mandate_id` /
`action` may not contain `|`) — unreachable today since `action` is a closed
enum and `mandate_id` is a Razorpay token, but cheap, and `ids.py` is the one
place responsible for guaranteeing collision-free keys.

**Fixed — NPCI's attempt cap and non-negative amounts were comments, not
constraints.** `schema.sql` had `-- 1..4` as an inline comment on
`attempt_index` with nothing enforcing it, and no `CHECK` stopped a negative
`amount_paise` from being persisted. Added `CHECK (attempt_index BETWEEN 1 AND
4)` and `CHECK (amount_paise >= 0)` to both `ledger` and `committed_schedule`,
mirroring the precedent the 24h `CHECK` on `committed_schedule` already set
for clause 6(a). Required bumping the pinned tests' dummy `attempt_index` from
`0` to `1` (7 call sites in `tests/ledger/test_schema.py` and
`tests/ledger/test_store.py`) — `0` was never a valid domain value under the
1-indexed spec, so this is a correction to an arbitrary filler value, not a
change to any tested behaviour.

**Rejected — `committed_schedule.committed_at DEFAULT now()`.** The auditor's
underlying concern is real: nothing stops an application bug from backdating
`committed_at` to launder a schedule row past the 24h `CHECK` while the true
notification lead time is much shorter. But the suggested fix works against
this project's own invariant: nothing outside `src/core/clock.py` may call
`datetime.now()`, specifically so tests can freeze time. A DB-level
`DEFAULT now()` invites future code to omit the column and let Postgres's own
un-freezable clock write it instead, which would make a frozen-clock test lose
control over what gets persisted here. The correct owner of "was this commit
honestly made ≥24h ahead" is the B9 executor, which must derive
`committed_at` from `src.core.clock.now()` and gets reviewed for exactly that
at the B9 gate — not something `schema.sql` can prove without a `BEFORE
INSERT` trigger, which is out of scope by the same reasoning that kept
`ledger` append-only-by-convention rather than by trigger. Recorded in
`schema.sql` as a comment so a future session does not "fix" this the wrong
way.

### 2026-08-26 · B1 · Schema and key derivation land at B1; behaviour at B9

`schema.sql` and `ids.py` ship `mandate_lifecycle`, the `plan` table,
`committed_schedule.generation` / `voided_at` / `void_reason` / `profile`,
`ledger.reason` / `ledger.profile`, and `generation` in the idempotency key —
now, at B1. B3 adds the revocation event route; B9 implements the void path and
the `UNCONFIRMED` resolution against DDL that already exists.

*Why.* `payments-domain` filed the opt-out-inside-the-window gap as a B9
concern, but most of it is DDL, and `schema.sql` plus `ids.py` are what B1's
gate certifies. Doing it at B9 means reopening a passed gate and rewriting
`schema.sql`, `store.py`, `ids.py` and their tests after they were signed off.

*The sharpest edge.* Because the key excluded `scheduled_for`, void-and-reissue
derived the *same* key and collided with `ledger_intent_once` — so the void path
was unimplementable without a key change, not merely unimplemented. `generation`
is read from the immutable `committed_schedule` row like `amount_paise` is,
never incremented in-process, and does **not** spend a slot: the NPCI budget
counts distinct `attempt_index`, never distinct keys.

### 2026-08-26 · B1 · `SENT` is written, but is forensic — never control flow

`SENT` was in the state enum and never written by the §3 protocol. It is now
written immediately before the provider call, and `recover.py` treats
INTENT-no-RESULT and SENT-no-RESULT **identically** — both must ask the
provider.

*Why the caveat is the decision.* A crash between the `SENT` commit and the
socket write is indistinguishable from a crash after the provider accepted. So
"INTENT without `SENT` ⇒ safe to resend" is a double-charge waiting to be
written by a future session reasoning from the enum instead of from the
protocol. Recorded as a "must NOT" on `recover.py` rather than as a comment.

### 2026-08-26 · B2 · The freeze: three-arm simulator, protocol, baseline ladder

`eval/frozen/sim_config.yaml`, `eval/frozen/simulator.py`, `eval/frozen/protocol.md`
written and frozen (via `guard_frozen.py`'s unconditional deny on
Edit/Write under `eval/frozen/` — these files were created with Bash/`cp`,
which the hook does not intercept, exactly as the scaffold's own hook design
requires). `eval/baseline_ladder.py` (not frozen — its behaviour is fixed
and externally specified, nothing to tune) drives the incumbent T+1/T+2/T+3
ladder against all three arms.

**Bug caught and fixed before the freeze commit — the cloglog link was a
no-op.** The first implementation of `_cloglog_probs` derived the terminal
probability by round-tripping through `_softmax` (compute the logit-implied
probability, then invert it through cloglog's inverse). That round-trip is a
mathematical identity — `1 - exp(-exp(log(-log(1-p)))) == p` for any `p` —
so the "misspecified" arm was silently reproducing nominal's exact
probabilities under a different function name. A statistical test comparing
the two links directly caught it (`nominal_probs["survive"] ==
misspecified_probs["survive"]` to 1e-16). Fixed by applying the cloglog
link to the raw combined logit score (`log-sum-exp` of the three non-reference
logits) instead of to an already-softmax-computed probability — logit and
cloglog now act as genuinely different functions of the same underlying
score, diverging by design (verified: ~0.40 under logit vs. ~0.49 under
cloglog for CANT_PAY_NOW's base rates).

*Why this matters beyond one arm.* This is exactly the failure mode the
freeze exists to prevent, caught one step early: a config number
(`link: cloglog`) that looks like it pre-registers a real difference, backed
by an implementation that quietly doesn't produce one. Pre-freeze iteration
on the generative mechanism itself is expected and healthy — CLAUDE.md's
"immutable after the Day-1 freeze commit" is about after the commit, not
about never revising the design while `src/policy/` is still empty. Found
via `tests/eval/test_simulator.py::test_misspecified_uses_cloglog_not_softmax_directly`,
before any git commit existed for `eval/frozen/`.

**What `coupled` is verified to reproduce, empirically, before payments-domain
review:** a controlled scenario (`tests/eval/test_simulator.py`,
`_tight_coupled_config`) shows household balance depleting monotonically as
members are attempted in order, later-scheduled members showing a
significantly higher `iatrogenic_insufficient_funds` rate than earlier ones,
and the effect vanishing under an effectively-unlimited balance — isolating
the coupling mechanic from ordinary hazard noise. A second test drives the
*actual* `baseline_ladder` (not a hand-rolled scenario) over the real frozen
config and confirms `coupled` produces iatrogenic failures under the
incumbent's real fixed-cadence schedule while `nominal` produces zero (it has
no coupling mechanic at all).

### 2026-08-26 · B2 · Freeze corrected — money-fabrication bug found by required gate review

`payments-domain` (B2's required review, dispatched in the same session,
before the working tree had a single file under `src/policy/`) found that
`coupled`'s household-balance coupling fabricated money: a below-balance
attempt had a chance to "recover anyway," crediting the mandate's full
amount while only debiting the household to zero. A real run recovered 1.7x
the total liquidity that existed across every household — independently
re-derived before acting. Full incident writeup: `POSTMORTEM.md` incident 1.

*Why this got a human decision rather than a silent patch.* `guard_frozen.py`
denies every Edit/Write under `eval/frozen/` unconditionally, and its own
docstring says a genuinely-necessary change to frozen content is "a decision
logged in DECISIONS.md and made by a human... not something an agent does
mid-task." A serious, unambiguous correctness bug found minutes after the
freeze commit, before any policy code exists, is exactly the scenario that
guidance anticipates — so the fix was not applied until the user explicitly
chose "fix critical + high, re-freeze now" from an enumerated set of options
that also included "keep this freeze, log everything as known gaps" and
"stop and let me look first."

*What changed, and why each was in scope for this pass:*

- **The fabrication itself** — the probabilistic branch is gone. A household
  debit now either succeeds in full or fails outright, matching real UPI
  AutoPay semantics (no partial debits) as well as fixing conservation.
- **`protocol.md`'s "same base rates" claim for `misspecified`** — checked
  against the math and found false: `cloglog(s) >= sigmoid(s)` for every
  real `s`, so a link that produces different realized probabilities from
  the same score *cannot* also preserve the same base rates — those are
  mutually exclusive, not two independent knobs. Forcing "same rates" would
  have meant round-tripping through the already-fitted probability again,
  reproducing the exact no-op bug the pre-freeze cloglog fix already caught.
  Corrected the claim to describe what is actually true (a uniform upward
  shift in every hazard) rather than change the behavior to match a false
  claim.
- **Oracle-field warnings** (`household_id`, `iatrogenic_insufficient_funds`)
  — cheap, directly requested, and closes the same class of leak the
  existing `initial_cause`/`effective_cause` warning already covers.
- **`on_day` monotonicity validation** — a caller passing a non-increasing
  day previously got a silently-clamped heavy-tail exponent instead of an
  error; now rejected outright.

*What was NOT fixed, and why — logged as protocol.md limitations instead:*
no exogenous household-balance competition (real households compete with
more than our own debits; modeling that belongs in a B13 stress regime, not
a retrofit onto this arm); coupling by call-order rather than calendar
distance (consistent with the stated "no mid-cycle top-up" design, just
cruder than an ideal time-aware model); `cause_switch_prob`'s marginal
invariance (real, but its intended target was always B7/B8's belief-update
stationarity assumption, not B5's population-level MNLogit fit);
`replenishment_exponent` being inert under the ladder's fixed 1-day cadence
(real, and expected — it activates once a policy chooses variable retry
spacing, which the ladder never does). Also added: a pre-registered
multi-seed comparison protocol (seeds 0-19, mean ± 1 pooled SD) for any
"beats the ladder" claim from B5 onward, since a 40-seed sweep showed ~19%
CV on recovered paise under `nominal` — a single-seed comparison was never
a defensible claim.

*Deferred, not forgotten:* mechanically enforcing the oracle-field
warnings via `guard_invariants.py` (matching the existing LLM-import /
float-money / hard-cancel checks) was raised but not implemented here —
`src/policy/` and `src/model/` are still empty, so there is nothing to test
the guard against yet. Add it when B7/B8 land.

`reports/FREEZE_HASH` now points to `d634346` (was `8321406`). The original
commit is not rewritten — git history shows the freeze was corrected, which
is more honest than hiding that it happened.

### 2026-08-27 · B2 · B5's gate rebound per metric, per arm — coupled cannot demonstrate money-recovered headroom

`eval/oracle_policy.py`'s privileged timing oracle (built end of the last
session, 20-seed sweep) found `coupled` does not discriminate on money
recovered even under perfect knowledge of timing and cause history: 9/11/0
wins against the ladder, mean/SE = −0.22 — a coin flip. `misspecified`
discriminates decisively (20/20, mean/SE = 7.66); `nominal` shows zero
headroom by design (20/20 exact ties — it is timing-invariant by
construction, so this is expected, not a finding).

*Why this doesn't mean loosening the arm.* The household balance/demand
ratio (~9%) could be raised until `coupled` starts discriminating on money
recovered too, but that is tuning the evaluation until our policy can win —
the exact failure mode the freeze exists to prevent, and it would not read
as anything else to a reviewer. `eval/frozen/` stays as re-frozen at
`d634346` (the entry above); this decision does not touch it.

*What `coupled` was actually built to show, restated.* Independence, not
scheduling skill, is what `coupled` varies. A per-mandate allocator that
never checks batch-level contention builds its own debit storm and
misreads the result as customer illiquidity (PLAN_DETAIL.md finding 1) —
that is a claim about **wasting less under contention**, not about
**recovering more**. Money recovered was never the right axis to hold
`coupled` to a "beats the ladder" bar on.

*New evidence: a cause-aware oracle.* The timing oracle above attempts
every mandate regardless of cause — it can only ever show timing headroom.
A second oracle (`run_cause_aware`, same file) additionally acts on the
true cause: skips `CANT_PAY_EVER` entirely (real action: stop, request
re-auth) and never attempts `WONT_PAY` (real action: offer an exit),
spending attempts only on `CANT_PAY_NOW`. 20-seed sweep
(`eval/cause_aware_headroom.py`), paired per seed against the real
fixed-cadence ladder over the real frozen config:

| arm | attempts: ladder | attempts: oracle | diff mean/SE | iatrogenic: ladder | iatrogenic: oracle | diff mean/SE |
|---|---|---|---|---|---|---|
| nominal | 329.8 ± 9.6 | 154.9 ± 12.3 | 174.90 / 3.09 | 0.0 | 0.0 | n/a — arm has no coupling mechanic |
| misspecified | 274.5 ± 7.7 | 122.3 ± 10.3 | 152.20 / 3.05 | 0.0 | 0.0 | n/a — arm has no coupling mechanic |
| coupled | 435.6 ± 10.4 | 251.5 ± 20.7 | 184.10 / 4.17 | 128.1 ± 13.5 | 119.0 ± 14.9 | 9.15 / 1.62 |

The attempts-spent gap is mechanically guaranteed by construction (an
oracle with ground-truth cause will always skip non-`CANT_PAY_NOW`
mandates — it is a wiring sanity check, not evidence of skill) and is
reported for completeness, not as the finding. The `coupled` iatrogenic-
failures gap (mean/SE ≈ 5.6, ~7% relative reduction, 128.1 → 119.0) is the
substantive result: a second-order effect through the shared household
balance — skipping dead/resistant siblings leaves more balance for
`CANT_PAY_NOW` members — that the timing-only oracle's money-recovered
metric could not surface. `coupled` **can** demonstrate this project's
thesis; it demonstrates it on the iatrogenic-failures axis, not the
recovered-money axis.

*Decision.* B5's gate ("beats the ladder on all three frozen arms") is
amended to bind different metrics on different arms, reflecting what each
arm is actually built to test:
- **recovered money** — binds on `misspecified` only.
- **attempts spent and iatrogenic count** — binds on `coupled` only.
- **mandates preserved** — binds on all three arms.

`reports/gates.md` and `PLAN_DETAIL.md`'s B5 section updated to match, both
carrying this entry's date, the same pattern as the "both" → "all three"
amendment already on that line.

*Deferred, not forgotten.* `protocol.md`'s Known Limitations section should
record the cause-aware-oracle finding, the fixed contention order
(`simulator.py:181-183` — households assigned by generation-order slicing,
`f"H{i // household_size}"`, not randomized or interleaved), and this
arm-by-metric split, the same way the misspecified-arm asymmetry is already
documented there. That file is under `eval/frozen/`, and `guard_frozen.py`
denies every agent edit to it unconditionally — per its own docstring, this
needs a human edit outside a Claude session, the same path the 2026-08-26
corrected freeze used. Not yet applied as of this entry — pending the
user's choice of how to route a second frozen-file amendment.

### 2026-08-27 · B3 · Gate rebound: "lands in the ledger" → "lands in `ingested_event`"

`reports/gates.md`'s B3 line read "a real test-mode `payment.failed` lands in
the **ledger** with a classified cause." It cannot: `ledger.decision_sha256`
is `NOT NULL REFERENCES plan`, and no `plan` row can exist before B8's
allocator produces one. A bare observed decline is not a decision of this
system's to move money — forcing it through `ledger` would be a category
error, not a simplification.

*Resolution.* A new, additive table, `ingested_event` (plus a second,
`webhook_event`, backing dedupe only), is the classified landing zone
instead. Gate text amended to name it explicitly, before any webhook had
been received and before this block's code existed beyond the plan file —
the same pre-result timing as the B5 per-arm rebinding earlier this block.
Does not touch `eval/frozen/`.

*What still needs `ledger` eventually.* Nothing from B3. `ledger` stays
exactly what B1 built it to be: this system's own authorised decisions to
move money, each traceable to a `plan` row. B8 is where a decline observed
here starts feeding a belief that eventually produces one.

### 2026-08-27 · B3 · Razorpay has no dedicated mandate-revocation decline reason

Independently verified against Razorpay's own error-reason documentation
(the UPI and card payment error-reason pages) while building
`decline_taxonomy.py`: there is no `error_reason` value meaning "the
customer revoked their mandate." Further research (Razorpay's subscription-
states documentation) found the actual behaviour: when a customer cancels a
UPI AutoPay mandate at their bank, the *next* debit attempt does not surface
a distinct decline reason for it at all — the subscription's own status
moves to `pending`, and **Razorpay's own auto-retry keeps attempting it
blindly the following day.** This is the incumbent behaviour this project
exists to replace, showing up as a documented fact about the platform
itself, not a hypothesis.

*Consequence for this block.* `decline_taxonomy.classify()`'s
`MANDATE_REVOKED` class is therefore a weak, best-effort match against
free-text revocation language only (e.g. a bank narration that happens to
say "mandate revoked") — never a confident classifier, and its docstring
says so. The reliable channel for clause 6(c) signal is
`src/ingest/lifecycle_route.py` reading the subscription entity's own
`status` from a `subscription.cancelled`-family webhook, mapped to
`MandateState.REVOKED` — a structurally different, more trustworthy signal
than anything decline text can offer.

*The gap this leaves, found by `compliance-auditor`'s B3 review.* If that
subscription webhook is delayed beyond the 48h replay window or genuinely
never arrives (Razorpay retries failed delivery on backoff for 24h, then
gives up), the only signal remaining is the weak decline-text match — and
today, nothing reads it as a fallback. `mandate_lifecycle` would show no
`REVOKED` row, `latest_state()` would still report the mandate `ACTIVE`, and
a future executor consulting only that table could attempt again.

*Why this is disclosed, not fixed, here.* The fallback the auditor describes
— treat a repeated pattern of `MANDATE_REVOKED`-classified `ingested_event`
rows with no corresponding lifecycle row as secondary revocation evidence —
is executor-side interpretation logic, which is B9's `pre-call re-read`
concern (PLAN_DETAIL.md §1 B9's "late-read principle"), not B3's ingest
concern. Designing it now, before B9's belief/executor machinery exists,
would be guessing at an interface that doesn't exist yet. What B3 *has* done
is make the fallback possible: every weakly-classified `MANDATE_REVOKED`
decline is captured in `ingested_event` with its `mandate_id`, so the raw
signal exists for B9 to build on. **Flagging for B9:** consult
`ingested_event` as well as `mandate_lifecycle` before concluding a mandate
is still retriable.

### 2026-08-27 · B3 · Provider idempotency spike — `receipt` does NOT dedupe Order.create

`scripts/idempotency_spike.py`, run for real against the live Razorpay
test-mode API (not simulated, not assumed from docs): called `Order.create`
twice with an identical body and a fixed `receipt`. **Observed:
`DOUBLE_CREATED`** — two distinct orders, `order_TUlHyAjGj0hWzK` and
`order_TUlHyP6FTZCKYY`, both created from the same request body and the same
`receipt` value.

*This contradicts the previous session's documentation-only claim* (recorded
in this block's plan: "Orders' `receipt` field is 'treated as an idempotency
key' but rejects a second create call with the same value") — Razorpay's own
prose describes `receipt` as *"treated as"* an idempotency key, and the
empirical result says it is not enforced as one on Orders, at least not
under the conditions this spike used. A follow-up search found Razorpay
does document per-endpoint idempotency-key HTTP headers (`X-Payout-
Idempotency`, `X-Refund-Idempotency`) for Payouts and Refunds specifically —
no equivalent was found documented for Orders, consistent with what the
spike observed, though absence from search results is suggestive, not
exhaustive proof no such mechanism exists.

*Why this doesn't weaken this project's own safety case — if anything it
sharpens it.* This system was never designed to lean on provider-side
dedup as a backstop: invariant 3 (ledger write before money action) and
`ledger_intent_once`'s partial unique index exist specifically so that a
second `Order.create`/`Payment` call for the same attempt is *never issued
in the first place* — the lease-claim-before-send step in PLAN_DETAIL.md
§3's write-ordering protocol is the actual safety mechanism, not a
courtesy. This result proves that assumption was load-bearing correctly:
had `razorpay_client.py` been designed around "the provider will catch a
duplicate for us," this finding would have been a shipped double-charge
path, not a spike result.

*Consequence for B9.* `PLAN_DETAIL.md` §1 B9 already names
`find_by_receipt` as `razorpay_client.py`'s recovery interface rather than
"trust the key" — this result confirms that was the right call, for a
stronger reason than originally written (not "belt and braces," but "the
belt is the only thing holding anything up"). `find_by_receipt`'s
implementation should use `receipt` as a *lookup filter* on a fetch/list
call after a crash (querying "did an order with this receipt get created?"),
never rely on `receipt` to have prevented a duplicate `create` from
happening — B9 must not call `Order.create` again for an attempt whose
INTENT row already exists, full stop; recovery is by asking, never by
resending. Implementation deferred to B9, per plan.

### 2026-08-27 · B3 · payments-domain review — taxonomy fixed, cause_map's safe-default corrected, two gaps disclosed

`payments-domain`'s required B3 review (decline-taxonomy coverage) ran
adversarially rather than confirmatory, and found real, demonstrated
failures — not speculation — by feeding `classify()` the actual verified
Razorpay `error_description` strings this session's own research had
already collected, with `code` stripped. What was found, and what was done
with each:

**Fixed — text-only input collapsed to UNKNOWN for classes whose real
description doesn't contain the enum token.** Demonstrated concretely: the
real `insufficient_funds` description never contains the word
"insufficient"; `invalid_vpa`'s never says "vpa"; `debit_instrument_blocked`'s
says "card being blocked", not "instrument blocked"; `card_not_enrolled`'s
says "not activated for online", not "not enrolled". Every keyword list had
been built mostly from the underscored `error_reason` token, with only
partial prose coverage — a real gap given `classify()`'s own signature
accepts `code: str | None`, meaning free-text-only input (exactly what
issuers/NPCI narration or a future non-Razorpay source would supply) was
already a designed-for case, just a badly-handled one. Fixed by adding the
actual verified description phrases as additional keywords (`decline_taxonomy.py`,
`TAXONOMY_VERSION` bumped implicitly — see below); pinned with 6 new
text-only parametrized test cases plus a compound `"account" in haystack
and "closed" in haystack` check for the "account has been closed" case
specifically (real prose isn't the contiguous phrase "account closed" the
original keyword matched).

**Fixed — `payment_cancelled` could satisfy the MANDATE_REVOKED heuristic by
accident.** The mandate-revoked check requires "mandate" + ("revoked" or
"cancelled"); on UPI AutoPay, "mandate" is the ordinary product noun, so a
per-attempt cancel's own free text routinely names it (constructed example:
`payment_cancelled` / *"The customer cancelled the UPI AutoPay mandate
approval request."*) — exactly the conflation this file's one hard
invariant exists to prevent (a per-attempt decline is not evidence the
whole mandate was revoked). Fixed by excluding `"payment_cancelled"` from
the check explicitly, since it's a real, specific Razorpay code for exactly
this case; pinned with a named regression test.

**Fixed — `cause_map.py`'s ambiguous-class priors violated documented
project policy.** `.claude/skills/new-failure-class/SKILL.md` already says,
for a genuinely ambiguous class: *"map it to CANT_PAY_NOW (the safe
default: we retry rather than offer an exit)."* `UNKNOWN` (exact uniform)
and `ISSUER_DECLINE` (near-uniform) both violated this — a design choice
made last session (PLAN_DETAIL.md, before this skill file was consulted)
reasoning from statistical honesty ("no signal → no opinion") rather than
this project's actual safety framing. The reviewer's sharper point: the
three causes are not symmetric in consequence — `CANT_PAY_NOW` costs a
cheap, reversible retry slot; `WONT_PAY` routes toward an off-ramp offer —
so an abstention that spreads mass evenly *is* a bet, and the skill had
already settled which way it should fall. Both classes changed to 0.60 /
0.20 / 0.20 (`CANT_PAY_NOW` / `CANT_PAY_EVER` / `WONT_PAY`); `PRIOR_VERSION`
bumped `v1` → `v2`; the one test that hard-coded exact uniformity
(`test_unknown_is_exactly_uniform`) rewritten to assert the skew instead of
weakened to pass — its replacement, `test_unknown_skews_cant_pay_now_not_uniform`,
also asserts the skew stops short of 0.75 (an abstention, not overconfidence).

**Added — classification is now versioned.** Neither `decline_class` nor
`cause_prior` written to `ingested_event` carried any record of which
ruleset produced it — the same gap B11's gate exists to close for the LLM
normaliser ("normaliser output is versioned in the ledger before it can
touch a belief"), just for the keyword matcher instead, and arguably more
urgent given `new-failure-class/SKILL.md` states outright that *"the
taxonomy will grow all week."* Two nullable columns, `taxonomy_version` /
`prior_version`, added to `ingested_event`; `decline_taxonomy.TAXONOMY_VERSION`
/ `cause_map.PRIOR_VERSION` module constants added, bumped by hand,
threaded through `store.record_ingested_event` and `webhook.py`.

**Disclosed, not fixed — raw NPCI/NACH response codes.** ("51", "U17",
similar) arrive, if at all, entirely outside Razorpay's normalised
`error_reason` vocabulary, and no substring rule can safely reach short
numeric/alphanumeric codes without risking a false match against an amount
or an id fragment. Recorded in `decline_taxonomy.py`'s docstring as a named
gap, explicitly the free-text problem B11's LLM layer exists for — not
attempted here, since it needs either a dedicated code table or the B11
normaliser, both out of B3's scope.

**Disclosed, not fixed — a decline whose real cause is "amount exceeds the
mandate ceiling"** (clause 4(c) territory) has no dedicated `DeclineClass`
among the 7 and lands in `ISSUER_DECLINE`, the least-wrong available
bucket. `DeclineClass`'s members are a B1 artifact; adding an 8th is out of
scope for B3.

**Disclosed, not fixed — `ISSUER_DECLINE` is a grab-bag bin, not a
phenomenon.** The reviewer's deeper critique: the class currently bundles
`incorrect_cvv` (a typo), `risk_check_failed` (an issuer fraud flag),
`transaction_limit` (a structural cap), and `payment_declined` (whose own
description reads as insufficient-funds-flavoured) under one shared prior,
purely as an accident of which strings happen to contain "declined".
Splitting it into better-fitting sub-bins is a real improvement but a
larger redesign than a review-response pass should attempt unilaterally —
noted for whoever next touches this file, not actioned here.

**Independently corroborates the compliance-auditor's finding, same
session:** both reviews, working from different files and different
reasoning paths, converged on the same real gap — `lifecycle_route.py`
declines to map `pending`/`halted` because the taxonomy said the reliable
signal is the subscription status, and the taxonomy declines to confidently
classify revocation because it expects `lifecycle_route.py` to catch it —
each individually well-reasoned, but a bank-side cancellation can fall
through both. See the entry above ("Razorpay has no dedicated
mandate-revocation decline reason") for the full writeup and why it's
disclosed rather than mitigated in B3.

**Also found, unrelated to the taxonomy — a tooling gap.** `run.ps1 lint`
/ `guard_invariants.py --all` scans `git ls-files *.py`, i.e. only
git-*tracked* files. Every file this block created was untracked at review
time, so `--all` silently checked none of them; the guard had to be re-run
against explicit paths to actually cover this block's own diff. Not fixed
here — pre-existing tooling behaviour, unrelated to B3's deliverables, and
worth its own deliberate look rather than a bolt-on fix.

### 2026-08-27 · B3 · Live-tunnel phase: gate closed with a real webhook

`cloudflared` tunnel + `run.ps1 serve` equivalent, webhook registered on the
real Razorpay test-mode dashboard, a real order created via the API with
`notes.mandate_id` set (per this block's own plan — the mitigation for "the
one thing most likely to go wrong"), payment attempted against it. Result:
a genuine `payment.failed` webhook, delivered from `52.66.76.63` (AWS
Mumbai — consistent with Razorpay's own infrastructure, not a replay),
landed in `ingested_event`:

- `mandate_id` resolved correctly via `notes` — **not NULL**, confirming
  the plan's mitigation actually worked, not just in theory.
- `decline_class = ISSUER_DECLINE` (not `UNKNOWN`) for decline text this
  session had never seen before ("Your payment didn't go through due to a
  temporary issue...") — matched via the `payment_failed` keyword, one
  layer of coverage this same session added in response to
  payments-domain's review.
- `cause_prior = {"CANT_PAY_NOW": 0.6, "CANT_PAY_EVER": 0.2, "WONT_PAY":
  0.2}` — the corrected safe-default skew, also confirmed live.
- `taxonomy_version = "v1"`, `prior_version = "v2"` both stamped.

*Deviation from plan, and why it doesn't weaken the result.* The intended
path was UPI, VPA `failure@razorpay` — Razorpay's documented deterministic
test-mode failure address. Two S2S JSON API endpoints
(`/v1/payments/create/json`, `/v1/payments/create/upi`) both returned a
genuine 404 from Razorpay's real API (confirmed via direct URL inspection,
not an SDK bug) — this account does not have server-to-server payment
creation enabled. Fell back to Razorpay's primary, definitely-supported
mechanism: a real Checkout.js popup in a real browser. **No UPI option
appeared in that popup either** — the user completed the flow via Wallet
→ Ola Money → failure instead. The gate's requirement ("a real test-mode
`payment.failed`") is about the webhook being genuine, not about which
payment method produced it, so this still satisfies it — but it's the same
decline text a wallet failure produces, not a UPI one, so it doesn't
exercise the UPI-specific vocabulary this session researched as thoroughly.

**Flagging for the user, not this session to fix:** this test account is
missing both **Subscriptions** (no `subscription.*` category in the
webhook event picker) and, now confirmed, **UPI** as a Checkout payment
method. `SETUP_GUIDE_WINDOWS.md` Stage B already anticipated the
Subscriptions gap ("if it is missing or greyed out, request activation via
support right now") — worth doing that request now, since B9's
`pause_subscription` and any real e-mandate registration testing need it,
and this project is specifically about UPI AutoPay mandates.

Two throwaway test-mode orders/payments were created in the process
(`order_TUlxvAtuEEvLGb`, `order_TUlydHC25aE5Nk` — both abandoned after the
404s, no payment attached; `order_TUlz2ij96t6mZ4` — the one actually paid,
test mode, no real money moved). No cleanup needed; Razorpay does not
charge for unused or failed test-mode orders.

### 2026-08-27 · B2 · protocol.md correction applied — FREEZE_HASH updated again

The Known-limitations bullets and per-arm restatements drafted in the entry
above were pasted into `eval/frozen/protocol.md` by hand, outside a Claude
session (one paste-seam bug — a missing line break between the last
existing bullet and the first new one, which would have silently merged
into the preceding paragraph rather than rendering as its own bullet — was
caught in review before commit and fixed), then committed as `4daf9ec`
(`FREEZE (corrected): document coupled's arm-by-metric split and fixed
contention order in protocol.md`).

`reports/FREEZE_HASH` now points to `4daf9ec` (was `d634346`). No
simulator or scoring behaviour changed — this correction is disclosure
only, so unlike the 2026-08-26 correction (an actual money-fabrication
bug), no `POSTMORTEM.md` incident is logged for it.

### 2026-08-27/28 · B4 · Person-period frame: corpus design, UNSOURCED features, split proportions, and a critical review-caught contamination bug

**What was built.** `src/policy/constraints.py` (AFA constants, 8(a)/8(b),
created early since `eval/corpus.py` needs one and there is no source of
truth elsewhere), `eval/corpus.py` (exploring-behaviour-policy training
corpus), `src/model/person_period.py` (`build`/`validate`),
`src/model/features.py` (`featurize`, `SPEC_COLUMNS`, `UNSOURCED`,
`FORBIDDEN`), `src/model/splits.py` (`split`). Tests written first via
`test-writer`, four new files under `tests/model/` and `tests/eval/`.

**Why training data is not the frozen ladder's own episodes.** The fixed
T+1/T+2/T+3 cadence (`sim_config.yaml:26-33`) attempts on days 1/2/3 —
entirely inside the salary window (`1 <= on_day <= 5`) — with
`days_since_last_attempt` always exactly 1. A model trained on that would
see zero variance on two of the three real hazard signals the simulator's
`_draw_outcome` actually depends on; it would fit cleanly, `validate()`
would pass, and B8's allocator would then choose `on_day` by extrapolating
outside its own training support. `eval/corpus.py` instead drives the
frozen `Simulator` under an exploring policy across 10 seeds
(`TRAIN_SEEDS`) disjoint from `sim_config.yaml`'s frozen seed (asserted at
import time). `nominal` arm only — training on `misspecified` or `coupled`
would void their purpose as held-out stress arms.

**AFA-cliff mandates are excluded from the corpus, not clipped or
attempted**, because the frozen simulator has no re-auth path. This is not
a training gap to backfill: a compliant above-cliff mandate should never
reach the hazard model's retry-timing decision at all — clause 8(a)/8(b)
routes it to `Action.REAUTH` before any retry-timing choice. **B8's
allocator must apply this identical filter before ever consulting the
hazard model** (flagged inline at `reports/gates.md`'s B8 entry). No hazard
signal is lost by the exclusion (`amount_paise`/`category` never enter
`_draw_outcome` under `nominal`), only sample size and, per the review
below, 9% of the frozen batch's own training analog.

**Seven-plus-two `SPEC_COLUMNS` features have no source, and are omitted +
declared rather than emitted null or fabricated** (the user's explicit
choice): `last_decline_class`, `decline_class_slot1..3` (simulator emits
`Outcome` only); `mandate_age_days`, `prior_cycles_ok`, `prior_cycles_failed`
(`cycle_id` hard-coded to 1); `issuer_id`, `instrument_type` (never
generated); `notification_lead_hours` (a policy output); `afa_limit_paise`,
`above_afa_cliff` (constant, per the AFA exclusion). `features.UNSOURCED`
names each with its reason; a test asserts `SPEC_COLUMNS == (emitted ∩
SPEC_COLUMNS) | set(UNSOURCED)`.

**`featurize()` physically strips outcome/censoring columns from its own
output**, not merely excludes them from `FORBIDDEN`. B5 reads the fit
target from `person_period.build()`'s own frame, rejoined by `row_id`,
rather than from `featurize()`'s output — making the `y = (df.outcome ==
"RECOVERED")` anti-pattern structurally harder to commit, since the leakage
column is simply never in the same frame as X.

**`profile` is a small, additive extension** of the stated `featurize(df)`
signature: `featurize(df, *, profile: Profile = Profile.strict)`, stamped
as a constant column. Note for B5: being constant within any one call makes
it perfectly collinear with the intercept once dummy-encoded — drop it
from the design matrix, or only include it when a batch genuinely mixes
both profiles' rows.

**Reviews, round 1.** `compliance-auditor`: all four items VERIFIED (AFA
paise conversion exact; category gating matches `sim_config.yaml` exactly;
AFA-cliff exclusion is the compliant response; no `rzp_live_`, no
cancellation calls). One recommendation applied: `assert_legal()`'s
docstring now states explicitly that its clause 6(a) check is a
training-data artifact (day granularity only, no hour component), not real
enforcement, naming the exact failure mode (a schedule committed at 23:59
and attempted at 00:01) a future B9 session must not assume this already
covers. `money-auditor`: clean, no defects — every money value confirmed
Python `int` end to end from `SimMandate` through `.astype("int64")`, no
float/division, no unit mismatch, no fabrication/duplication across
`build()`/`featurize()`.

**Review, round 2 — `stats-reviewer`, NOT clean on the first pass.**
Verified clean, with hard evidence: censoring (319/1,769 episodes censored,
all kept as `STILL_PENDING`/`event_code=0`, none dropped or relabeled,
worked example B round-trips exactly); `featurize()` leakage (every
column at slot k checked against slot <= k only; `days_since_last_attempt`
verified against the simulator's own internal day tracking across all
4,801 rows, 0 mismatches); split disjointness (200-seed brute-force,
maximum pairwise mandate overlap = 0); seed/id namespacing; `assert_legal()`
enforcement. But found four real defects, all now fixed:

1. **CRITICAL — slot-1 rows contaminated every hazard coefficient.**
   Slot 1 is always `STILL_PENDING` by construction (P=1, no variation to
   explain), so `h_c(1) ≡ 0` is a structural zero, not a parameter to
   estimate. Fitting slot 1 into the same likelihood as slots 2-4 let the
   MLE "explain" a deterministic outcome using whichever covariates happen
   to separate slot-1 rows from the rest. Measured concretely, same data,
   only difference being inclusion of slot-1 rows: `days_since_last_attempt`
   → RECOVERED went from a correct ≈0 to a fabricated **+0.10 logit/day**
   (a real hazard model would invent this project's own timing thesis out
   of a frame artifact); `in_salary_window` → OPTED_OUT went from ≈0 to
   **+1.38**; `prior_failures` → RECOVERED went from ≈0 to **+0.75**.
   *Fix:* `person_period.build()` now emits `estimable: bool` (`slot >= 2`),
   asserted by `validate()` (`estimable == (slot >= 2)` on every row, raising
   `FrameError` on any drift). B5 must filter `df[df.estimable]` before
   fitting anything, and must not independently reconstruct this as
   `df.slot >= 2` — one flag, not two definitions that can drift apart.
   `on_day = 0` for slot 1 is kept, NOT nulled: it is the mandate's true
   cycle-start anchor, and `features.featurize()`'s
   `days_since_last_attempt` computation for slot 2 (a real, estimable row)
   depends on it being a real number — nulling it would silently corrupt
   slot 2's gap to a wrong constant, a worse bug than the one being fixed.
   `estimable` is dropped by `featurize()` the same way `event_code` is
   handled: B5 consults it via `build()`'s frame, not via `featurize()`'s
   output.

2. **HIGH — the calibration split was asked to do two jobs that break each
   other.** `calib` was fitting isotonic calibration AND supplying the
   conformal quantile. Split conformal validity requires the quantile's
   scores to be exchangeable with the test-time score; once isotonic has
   been fit on a row, that row's score is no longer an honest out-of-sample
   residual. The failure mode is prediction sets narrower than the stated
   95%, which fire the singleton `{WONT_PAY}` off-ramp *more* often than
   the guarantee permits — the exact harm the conformal gate exists to
   prevent — while a reliability diagram fit and read on the same rows
   would still look diagonal. *Fix:* `src/model/splits.py`'s `split()` now
   returns **four** frames — `(train, calib_iso, calib_conf, test)` at
   **70/10/10/10** (changed from PLAN_DETAIL.md's literal 3-tuple
   interface; the guarantee could not be delivered otherwise). `calib_iso`
   fits isotonic (B6); `calib_conf`, a disjoint mandate set, supplies the
   conformal quantile. At ~1,769 mandates each lands near 175-180, clearing
   conformal's n≥19 bare validity floor with margin. Bare `assert`s (which
   `python -O` strips) replaced with a real `SplitIntegrityError`.

3. **HIGH — the exploring policy didn't explore late-slot timing, and the
   sizing diagnostic couldn't see it.** Slot 4 had **zero** in-salary-window
   rows across the whole corpus (slot 3 had 21/898). Mechanically forced:
   with `on_day` strictly increasing and the salary window a one-time,
   absolute, cycle-start range (never recurring), `day4 <= 5` requires
   `day2`, `gap_2_3`, and `gap_3_4` all tiny simultaneously — vanishingly
   rare under independent wide-range draws. Separately, `cell_counts()`
   only created a dict key for a cell that occurred at least once, so
   `thin_cells()` — which just filters `counts.items()` — could never
   report the one case that actually mattered: a cell with a true zero
   count never became a key to filter. *Fix:* `_draw_schedule()` is now a
   two-component mixture — with probability `COMPRESSED_FRAC=0.30`, the
   whole three-attempt schedule is drawn as three distinct days within the
   first 7 days (the only way slot 3/4 can land in-window at all), otherwise
   the original wide independent-gap draw (preserving `days_since_last_attempt`'s
   broad variety). `cell_counts()` now initializes the full 18-cell
   (3 causes × 3 slots × 2 buckets) grid to 0 before counting, so an
   uncovered cell is visible. `generate()` now calls `thin_cells(..., threshold=1)`
   on its own output and **raises** if any cell is truly empty
   (`check_coverage: bool = True`, opt-out only for small deliberately-partial
   `seeds` subsets used to test something else, e.g. namespacing).
   *Residual, disclosed, not fixed further:* slot-4-in-window is real but
   thin even after the fix — 7 (CANT_PAY_EVER), 14 (CANT_PAY_NOW), 11
   (WONT_PAY) observations, all clearing zero but none clearing
   `MIN_CELL_COUNT=20`. This is a structural consequence of strictly-
   increasing `on_day` against a non-recurring window, not a remaining bug;
   pinned by `tests/eval/test_corpus.py::test_generate_default_corpus_residual_thin_cells_are_disclosed`
   so a future change that reduces it further is caught, not silently
   accepted.

4. **MEDIUM — 9% of the frozen evaluation batch has no training analog.**
   The AFA-cliff exclusion is correct on its own terms (verified: under
   `nominal`, `_draw_outcome` never reads `amount_paise`/`category`, so the
   exclusion is MCAR with respect to outcome), but it means 18/200
   mandates in the frozen batch (`subscription` above ₹15,000) never appear
   in training, while `eval/baseline_ladder.py` scores them with no filter.
   *Resolution, not a code change to this block:* the correct fix is at
   B8 — the allocator must apply the identical `afa_free_limit_paise()`
   filter before ever consulting the hazard model, routing these straight
   to `Action.REAUTH`. `baseline_ladder.py` not filtering them is faithful
   to the real, documented incumbent (no AFA-aware routing either), not a
   bug to fix here. Flagged inline at `reports/gates.md`'s B8 entry.
   *Least-confident assumption, stated plainly:* the "exclusion doesn't
   bias hazards" argument is a property of `nominal` specifically and does
   NOT hold for `coupled`, where recovery depends on `household_balance`
   versus `mandate.amount_paise` directly — training on `coupled` would
   make this exclusion a real selection-on-outcome bias. Training on
   `nominal` only is what currently keeps this safe; documented in
   `eval/corpus.py`'s module docstring so a future session doesn't extend
   training to `coupled` without re-deriving this.

**Minor findings, also fixed:** `person_period.py`'s `_apply_dtypes`
docstring claimed `outcome` keeps enum MEMBERS as category values; verified
false (`Outcome` is an `IntEnum`, pandas/numpy silently unbox it to
`numpy.int64` when building the categorical — `df.outcome ==
Outcome.RECOVERED` still works via int equality, but `.name` access would
raise `AttributeError`). `censor_reason` (`str, Enum`) genuinely does keep
enum members. Docstring corrected to state the actual, verified mechanism
for each. Dead-code note added: the zero-attempt (`WINDOW_CLOSED` before
slot 2) branch is unreachable under `generate()`'s current defaults
(`day2 <= 20 < MAX_DAY = 40`) — exercised only by hand-built tests, which
is fine (a future `max_day`/schedule change could make it reachable, and
`build()` must keep handling it correctly regardless).

**Test-writer bugs found and fixed by hand, not routed around.** Three
genuine defects in the generated test suite, found by actually running it
rather than trusting the subagent's summary: (1)
`test_spec_columns_equals_emitted_plus_unsourced` computed "emitted" as a
set-difference against `build()`'s columns, which excludes any column
`featurize()` legitimately carries through unchanged (`amount_paise`,
`ceiling_paise`, `category`) — unconditionally false for any correct
implementation; fixed to intersect with `SPEC_COLUMNS` instead. (2)
`test_no_forbidden_columns_survive_featurize` asserted ALL nine `FORBIDDEN`
members are present in `build()`'s output — four are simulator-internal
oracle fields that never become person-period columns at all; loosened to
a non-vacuous, non-universal check. (3) A keyword-argument typo
(`Outcome=` instead of `outcome=`) would have raised `TypeError` before the
test body ran. Also closed two coverage gaps the generated suite missed:
`validate()`'s missing-required-column branch, and a rejection shape it
never exercised (a group whose last row is not marked `is_terminal` at
all, distinct from "terminal row followed by another row").

**Direct empirical re-verification of finding 1's fix**, not just asserted:
fit the same kind of pooled multinomial logit on the real corpus two ways
— all rows, and `df[df.estimable]` only. All-rows reproduced the
reviewer's measurement closely (`days_since_last`→RECOVERED +0.121 vs
their +0.101; `prior_failures`→RECOVERED +0.735 vs their +0.750), and
critically showed the actual separation *signature*: `in_salary_window`'s
coefficient was nearly uniform across RECOVERED/DEAD/OPTED_OUT alike
(2.03/1.87/1.52) — nonsensical, since only `CANT_PAY_NOW`'s recovery
should see a salary-window effect. Filtered to `estimable` rows, the same
coefficient dropped to 0.47/0.13/0.12 — concentrated on RECOVERED
specifically, consistent with the true DGP diluted by pooling across
causes (this toy check has no per-cause hazard, which is B5's job). Both
the magnitude and the qualitative pattern change exactly as the fix
predicts.

**Verification.** 311/311 tests pass (245 pre-B4 + 66 new/updated), 98%
coverage (the ~8 uncovered lines are defensive belt-and-braces checks in
`features.py`/`splits.py` that cannot be reached by any current valid or
invalid input — they guard against a future bug in those functions' own
logic, not a gap in test design). `guard_invariants.py` clean against every
changed file, checked explicitly by path (not `--all`, per the known B3
tooling gap). Full pipeline verified at real scale: `generate(TRAIN_SEEDS)`
→ 1,769/2,000 mandates survive the AFA filter → 4,785 person-period rows
(1,769 non-estimable slot-1 + 3,016 estimable) → `featurize()` → `split()`
→ 1238/177/177/177 mandates (70.0/10.0/10.0/10.0% almost exactly) → target
cleanly re-joinable by `row_id`, filtered to `estimable` rows, matching
B5's intended usage pattern exactly.
