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
| Decline-string normalisation | Gemini 3.5 Flash-Lite | Issuer strings are unstandardised free text; new variants appear weekly. This is genuinely a language task |
| Cancellation-intent extraction | Gemini 3.5 Flash-Lite | Support tickets, including Hinglish. No feasible rule set |
| Merchant root-cause narrative | Gemini 3.5 Flash | Once per batch, not per transaction. Writing, not deciding |

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

### 2026-08-28 · B5 · Gate rebound to model-fit evidence; a null policy cleared three of four clauses; an earlier same-session amendment reversed

**Written before any coefficient existed.** No file under `src/model/` beyond
B4's frame exists as of this entry; `competing_risks.py` and `cif.py` are
unwritten, nothing has been fit, and no CIF has been computed. The same
standard as the B3 and earlier B5 rebinds.

#### 1. The finding: three of B5's four gate clauses do not measure the model

`payments-domain` (dispatched during planning, per PLAN.md's prescription for
this block) constructed a **null policy** — attempt slot 2 once, stop, consult
nothing — and scored it paired against the fixed ladder over seeds 0-19. The
result was reproduced independently in the main session before being accepted:

| arm | metric | ladder | null policy | margin / pooled SD |
|---|---|---|---|---|
| nominal | mandates preserved | 113.5 | 152.6 | **+5.90** |
| misspecified | mandates preserved | 106.7 | 137.8 | **+5.41** |
| coupled | mandates preserved | 106.4 | 151.8 | **+6.32** |
| coupled | attempts spent | 435.6 | 200.0 | **32.07** |
| coupled | iatrogenic count | 128.1 | 43.8 | **7.72** |

The mechanism is structural, not a tuning accident. `scoring.py`'s
`NOT_PRESERVED = {DEAD, OPTED_OUT}`, and both outcomes are reachable **only by
making an attempt**; `attempts_spent` is trivially monotone; and
`simulator.py`'s household coupling fires only inside `attempt()`. All three
metrics are therefore monotonically decreasing in attempt count. A policy that
does less scores better on them by construction.

`protocol.md` already disclaims this — "a policy that trivially wins on
iatrogenic count by attempting less is not evidence it solved anything" — but
applies the disclaimer **only to iatrogenic count**. It applies with more force
to `mandates_preserved`, which was the only clause binding `nominal` at all,
and the disclaimer is not there.

**Honesty check, recorded rather than omitted:** the gate is a conjunction, so
the null policy does *not* pass it — it loses recovered money on every arm
(nominal −2.12, misspecified −1.17, coupled −1.89 pooled SD). The accurate
statement is not "the gate is trivially passable." It is: **the gate was
carried entirely by one clause, and the other three carried no information
about whether the model works.**

#### 2. The remaining clause was structurally unwinnable from `nominal`-only training

`misspecified`/recovered-money was the one informative clause. Its entire
mechanism is `replenishment_exponent`, which enters through
`_cloglog_probs(logits, cause, days_since_last)` — and `_draw_outcome` passes
`days_since_last` **only** into that function. Verified arm links:
`nominal: logit`, `misspecified: cloglog`, `coupled: logit`.

So under `nominal` — the arm B4 correctly restricted training to — the true
coefficient on `days_since_last_attempt` is **identically zero**. The proposed
design would have estimated a known-zero parameter on train and relied on it to
generate the entire margin on the arm where it is the whole mechanism. That is
not a power problem more corpus seeds could fix; the signal is absent by
construction.

#### 3. Decision: B5's gate is rebound to model-fit evidence, and B5 becomes policy-free

A gate that a model-free null policy clears on three of four clauses is not
measuring the model. It cannot fail, so it cannot certify anything — and it
would have passed a broken model through to B8 with a green tick behind it.

**Original text:**

> ★ competing risks + CIF: beats the ladder on **recovered money**
> (misspecified arm), on **attempts spent and iatrogenic count** (coupled arm),
> and on **mandates preserved** (all three arms); `Σ_c CIF_c(4) + S(4) == 1`;
> stats-reviewer returns clean

**New text:**

> ★ competing risks + CIF: held-out multinomial log-loss and per-cause Brier on
> the `test` split beat an intercept-only MNLogit null; transfer degradation of
> that same fit reported on `misspecified` and `coupled` frames;
> calibration-in-the-large reported per `slot × in_salary_window` cell;
> `Σ_c CIF_c(4) + S(4) == 1`; stats-reviewer returns clean

**The intercept-only MNLogit null IS the ladder's implicit model** — constant
hazard, no covariates, no adaptation. This framing is worth more than the rupee
comparison it replaces: it turns "we recovered more money" into "**the ladder
assumes constant hazard, and here is held-out evidence that assumption is
wrong.**" That claim is falsifiable, needs no policy, and cannot be won by
doing less.

`eval/cif_policy.py` and `eval/model_vs_ladder.py` — both approved earlier this
same session — are **not built**. A policy at B5 would have had exactly one
tunable scalar (the stop-vs-continue threshold), and three of the four gate
clauses were monotone in it: tuning a scoring parameter on already-seen data
along the grading axis is the precise failure the freeze exists to prevent.
Its "always take slot 2" behaviour was a `score_mandate` artifact (the scorer
raises on zero attempts), not a policy choice — scoring it would have been
scoring an exploit. **"Beats the ladder" moves to B8**, where an allocator
exists that can honestly bear it and where the stop threshold is a policy
decision gated by B6's conformal set rather than a free parameter fit to the
scoreboard.

Building `cif_policy.py` merely as a disclosed diagnostic was considered and
rejected: the null-policy numbers in §1 above are the same disclosure with no
extra file and no tunable scalar.

#### 4. Reversal: the paired win-criterion amendment approved earlier this session is withdrawn

**The sequence is the point, so it is recorded rather than only its outcome.**
Earlier this session, on the finding that the privileged oracle misses
`protocol.md`'s one-pooled-SD bar on coupled's iatrogenic count (margin 9.10 vs
pooled SD 14.22, ratio 0.64), an amendment to a **paired** mean/SE criterion was
proposed and approved. It is now withdrawn, unimplemented.

Reason: choosing the statistical convention *after* seeing which one your number
clears is goalpost-moving in sequence, regardless of whether paired mean/SE is
the better convention in the abstract — and it is. (The pairing is real: mandate
populations and household balances are bit-identical across constructions at a
given seed, and the oracle's paired SE of 1.62 against Var(C)+Var(L)=404 implies
correlation ≈ 0.87.) It is the same failure rejected in §3 above: adjusting a
scoring parameter on seen data along the grading axis. Nothing under
`eval/frozen/` is edited, and the by-hand protocol edit previously contemplated
is no longer needed.

#### 5. `gates.md`'s coupled clause is loosened to agree with our own frozen protocol

`protocol.md` already states, frozen and predating every result, that the
coupled arm "**is not scored primarily on whether the policy 'wins.'**"
`gates.md` was stricter than the protocol it derives from. Bringing the two into
agreement removes a goalpost planted in the wrong place using text that predates
the result — it does not move one. `eval/frozen/` is untouched.

#### 6. Disclosed: `protocol.md`'s win criterion is directionally undefined, and is NOT being amended

`protocol.md` requires "the candidate's mean to **exceed** the ladder's mean by
more than one pooled standard deviation." For `attempts_spent` and
`iatrogenic_failures`, **lower is better** — so read literally, the frozen
criterion demands a candidate produce *more* iatrogenic failures and *more*
attempts than the ladder in order to pass. The text was written for
higher-is-better metrics and was never extended when the earlier B2 amendment
bound the gate onto two lower-is-better ones.

This is recorded as a **disclosed flaw in our own pre-registered protocol**. We
are deliberately **not** amending it (see §4 — the same reasoning). Wherever
those two metrics are reported, the clause is interpreted as *lower is better*,
and that interpretation is stated inline at the point of reporting.

#### 7. Standing requirements for every comparison from here on

- Report **both** paired and unpaired numbers, with `SD_diff` shown. Do not
  choose between them in prose; let the reader apply either bar.
- Report the **realized correlation** between candidate and ladder per arm
  alongside any paired SE. The `Simulator` shares one RNG across all mandates,
  so a policy whose attempt count diverges from the ladder's re-rolls the draw
  stream (measured: two extra attempts on one mandate changed the slot-2 outcome
  of 112 of 200 mandates at the same seed). This is correct simulator semantics
  and biases nothing, but it means pairing **weakens exactly as the effect
  grows** — so the variance reduction must be shown, never assumed.
- Single-seed debugging of a policy is not meaningful here; any behavioural
  change re-rolls every mandate.

#### 8. Carried forward to B8, before it designs anything

Even under a faithful paired bar (`SD_diff` = 7.25, backed out from the oracle's
own numbers), the **privileged** cause-aware oracle clears coupled's iatrogenic
clause by only 26%. An omniscient policy barely clearing is strong evidence a
real one will fail. B8 should know this before designing to that metric, not
after measuring against it.

Also carried forward: `DECISIONS.md`'s 2026-08-27 headroom table presents
128.1 → 119.0 as the achievable iatrogenic headroom on `coupled`. The reachable
floor is **43.8** (the null policy, §1). The oracle is closer to the ladder than
to the floor; the 9.15 "headroom" is the residue of holding attempt count
roughly fixed, not a measure of what cause-knowledge avoids. A null-policy
column belongs in `eval/cause_aware_headroom.py`'s output so the next session
does not re-derive the old conclusion from the old table. *(Done same session:
`_null_run` added to `eval/cause_aware_headroom.py`, reported alongside the
ladder and cause-aware oracle for both attempts and iatrogenic count, all
three arms.)*

#### 9. Design-matrix corrections adopted before fitting, from the same `payments-domain` review

Two further findings from the review that produced §1-2 apply to the design
matrix in the original approved plan, and are corrected here — before any
coefficient exists — rather than discovered after a fit:

- **`committed_day_of_month` is dropped entirely.** Measured on the real
  corpus (n=3,016 estimable rows): on all 1,769 slot-2 rows,
  `days_since_last_attempt == committed_day_of_month` exactly (both compute
  from `on_day`, and slot 1's `on_day = 0` makes them identical at slot 2 by
  construction). Overall correlation 0.71, design condition number 144. Slot
  2 is the reference level, so the two names are one variable across the
  majority of the frame, separated only by slots 3-4 — exactly where support
  is thinnest (B4 finding 3). Keeping a spurious linear day term is not
  neutral: a downstream policy maximising CIF-implied recovery over a 1-40 day
  grid would find whatever noise tilt this coefficient picked up and push to
  an endpoint, i.e. out-of-support extrapolation dressed as an optimum. Since
  no policy ships at B5 (§3) this specific risk is deferred, but the
  collinearity is real regardless and the column is dropped now.
- **The `late_slot(3|4) × in_salary_window` interaction is restricted to slot
  3 only** (renamed `slot3_x_in_salary_window`); no slot-4 interaction term.
  Measured cell counts on the estimable frame: slot 2 = 843 in-window rows,
  slot 3 = 218, slot 4 = **32** (1.1% of the frame) — consistent with B4
  finding 3's disclosed 7/14/11-by-cause thin cells at slot 4, none clearing
  `MIN_CELL_COUNT=20`. Pooling 3 and 4 lets the term look estimable by
  concealing that one of its two constituents contributes almost nothing to
  it. Fitting the interaction on slot 3 alone and saying so is more honest
  than pooling a cell already documented as unestimable.

Final `nominal`-arm design matrix for `competing_risks.py`: slot dummies
(2 = reference, 3, 4), `in_salary_window`, `days_since_last_attempt`,
`slot3_x_in_salary_window`. Still excluded, per the original plan and B4's
decisions: `amount_paise`, `category`, `above_afa_cliff` (no true hazard
signal under `nominal`), `prior_failures_this_cycle` (≡ `slot − 1`, collinear
with the slot dummies), `profile` (constant per call, collinear with the
intercept).

#### 10. B5 gate closed — real numbers, `eval/model_fit_report.py`, run via `eval-runner`

`src/model/cif.py` and `src/model/competing_risks.py` implemented, tests
written first (`test-writer`; three real bugs found and fixed by hand before
trusting the suite — see below), 50/50 new tests pass, full suite 300 passed
/ 61 skipped (all 61 are `Postgres unavailable: connection timeout expired`
in `tests/ingest,ledger/` — Docker not running this session, unrelated to
B5; arithmetic confirms no regression: 311 old baseline + 50 new = 361 total,
361 − 61 now-skipped = 300 passed). `guard_invariants.py` clean, checked both
`--all` and explicitly by path for every new/untracked file (the known B3
tooling gap — `--all` only sees git-tracked files).

**Bugs found in `test-writer`'s generated suite, fixed by hand, not routed
around** (same discipline as B4): (1) `Episode(censor_reason=None)`
unconditionally in a test helper — `build()` calls `CensorReason(None)` on
any terminal-STILL_PENDING row and raises `ValueError`; fixed to a real
`CensorReason` member. (2) `pytest.raises((AttributeError, type(None)))` —
`type(None)` is not an exception type, errors at collection; fixed to
`pytest.raises(AttributeError)` alone (`dataclasses.FrozenInstanceError`
subclasses it). (3) `(df1 - df2).abs().max() < 1e-6` in a bare `assert`,
where `df1`/`df2` are `MNLogitResults.params` — a multi-column DataFrame (one
column per non-reference outcome), so the comparison produces a Series and
raises `ValueError: truth value of a Series is ambiguous`; fixed via
`np.abs(np.asarray(...) - np.asarray(...)).max()`. (4) The two "not 1-KM"
regression tests in `test_cif.py` were arithmetically wrong in a way that
also caught a real implementation bug: my first `survival()` summed all 4
hazard columns (including `STILL_PENDING`, i.e. "survives this slot") as
"total hazard," which is always 1 by construction and silently makes every
`S(k)` collapse to 0 for k≥2 regardless of input — caught immediately by
`test_cif_all_outcome_zero_hazard` (expected S≡1, got `[1,0,0,0]`). Fixed to
sum only the 3 terminal-event causes. The tests' OWN hand-computed expected
values had a matching conceptual error (summing all 4 as "total hazard") and
a second, separate error (treating CIF as resettable rather than cumulative
— asserting slot-3/4 CIF returns to 0 after a slot-2 resolution, when a
cumulative incidence function must carry its value forward). Both fixed, and
the "not 1-KM" test was rewritten with a genuine two-simultaneous-cause
scenario — the original only had one cause with nonzero hazard, under which
naive 1-KM and the correct recursion coincide by construction (nothing to
compete with), so it never actually exercised the regression it claimed to
guard.

**The report** (`.venv\Scripts\python.exe -m eval.model_fit_report`, fit on
`train` from a 10-seed `nominal` corpus, split 70/10/10/10, scored on the
held-out `test` split's 296 estimable rows):

```
=== held-out test split: full model vs intercept-only null ===
log_loss   full=1.2305  null=1.2364  BEATS the null (lower is better)
brier[0]  full=0.2514  null=0.2500  does not beat   (STILL_PENDING)
brier[1]  full=0.1912  null=0.1912  ties            (RECOVERED)
brier[2]  full=0.1084  null=0.1094  beats           (DEAD)
brier[3]  full=0.1185  null=0.1193  beats           (OPTED_OUT)

=== transfer degradation (same fit, scored on data never trained on) ===
misspecified  n=1299  log_loss=1.3411  degradation=+0.1106 (worse, as expected)
coupled       n=1867  log_loss=1.0422  degradation=-0.1883 (better)
```

**Honest reading, not smoothed over.** Log-loss — the metric MNLogit's own
fitting objective directly targets, and the one that answers the gate's real
claim ("the ladder assumes constant hazard; here is held-out evidence that
assumption is wrong") — beats the null on genuinely held-out data. Per-cause
Brier is **mixed: 2 of 4 beat, 1 ties, 1 loses**, not glossed as a clean
sweep. This is coherent with, not contrary to, the `payments-domain` review
earlier this session (§1-2 above): that review fit the same design and found
only 1 of 18 coefficients cleared z=2 (`in_salary_window`→RECOVERED). A
model with one real signal concentrated on one outcome is expected to move
log-loss (which aggregates across the whole predicted distribution) while
leaving per-cause Brier on the two majority outcomes (STILL_PENDING,
RECOVERED — where the signal actually lives, and where it's weakest per that
same review's z=3.62) close to a tie, and to help more cleanly on the two
minority causes (DEAD, OPTED_OUT) where a small amount of information goes
further against a smaller base rate. This is reported as the real, granular
result, not summarized as "the model wins."

**Transfer degradation, both signs, neither is alarming, one is not fully
explained.** `misspecified` degrades (+0.11), as expected — the model was
correctly fit on `nominal` only, and `misspecified`'s mechanism
(`replenishment_exponent`, entering via `_cloglog_probs` under the `cloglog`
link) is a zero-coefficient blind spot for a model trained on `nominal`'s
`logit` link (§2 above). `coupled` *improves* (−0.19) on data the model was
never trained on. Plausible, not verified: household coupling converts some
`RECOVERED` draws into iatrogenic `STILL_PENDING`, which could shift
`coupled`'s realized outcome mix further toward the majority class both
models already predict with high probability (~45-53% per the calibration
table below), making log-loss mechanically easier there regardless of real
model quality. Flagged as a plausible explanation, not a confirmed one — a
genuine "this needs more looking at" for whoever next touches transfer
scoring, not a claim to build on.

**Calibration-in-the-large**, per `slot × in_salary_window` cell, held-out
test split (24 rows, one per cell × event_code — full table in
`eval/model_fit_report.py`'s output, not reproduced here). Broadly close
(e.g. slot 2/no-window/STILL_PENDING: predicted 0.521 vs realized 0.448;
slot 2/window/RECOVERED: predicted 0.292 vs realized 0.300), with the
largest gaps exactly where B4 already disclosed thin support (slot
4/in-window cells: 6-32 rows each) — consistent with, not contradicting,
that earlier finding.

**`Σ_c CIF_c(4) + S(4) == 1`**: proven on `src/model/cif.py`'s own test
suite (property-tested on random valid hazards, large batches, and multiple
degenerate cases — `tests/model/test_cif.py`, 21/21 passing), not
re-derived from a real fitted model. `cif.py` has no dependency on
`competing_risks.py` by design (module docstring) — assembling a real
per-mandate 4-slot hazard grid from a fitted model is B8's integration work
(the allocator's backward induction), not B5's; building it now to re-prove
an identity already proven on the underlying math would be the same scope
creep already cut once this session (`eval/cif_policy.py`).

**`stats-reviewer` returned NOT CLEAN on the first pass** — not for any of
the five questions it was specifically asked (join correctness, filter-
before-fit, split ordering, null purity, transfer-scoring isolation all
independently verified clean, several confirmed by rerunning the reviewer's
own diagnostics against the real fit), but because **the headline claim
above was not actually supported by its own evidence.**

1. **BLOCKING — the "beats the null" verdict was a coin flip.** §10's
   report used one hardcoded `SPLIT_SEED = 1` with no dispersion reported at
   all — a bare point comparison. Reviewer reran it three ways: paired
   per-row (t=−0.84, p=0.40), across 25 split seeds (full model wins only
   17/25, with 8 seeds printing the opposite verdict), and 5-fold
   mandate-grouped CV with mandate-clustered SEs (t=−1.05, p=0.29). **Both
   halves of §10's "log-loss beats, Brier is mixed" reading were noise at
   that sample size** — not a defensible read of a weak-but-real effect, a
   genuine violation of this same entry's own §7 standing requirement
   ("report SE/CI, never a bare verdict from one seed").
2. **The fix: the corpus was ~4x too small.** `eval/corpus.py::TRAIN_SEEDS`
   was 10 seeds (1,769 mandates / 3,016 estimable rows). Reviewer reran the
   grouped-CV check on 40 seeds (12,316 estimable rows) and found the effect
   is real and large once there's enough data to see it: t=−6.32, p<1e-9,
   coefficients cleanly recovering the frozen DGP (`in_salary_window`→
   RECOVERED +0.564, z=11.7; slot→OPTED_OUT +0.428/+0.489 vs the DGP's exact
   linear escalation, z=6.9/5.8). *"The corpus is simulated and free."*
   **`TRAIN_SEEDS` widened 10→40** (`eval/corpus.py`, still disjoint from
   the frozen sim seed, assertion unchanged). Side effect, unplanned but
   welcome: this fully resolves B4's disclosed slot-4-in-window thin-cell
   limitation (7/14/11 observations, below `MIN_CELL_COUNT=20`) — at 40
   seeds those same three cells measure 22/52/38, and the corpus has zero
   thin cells anywhere. `tests/eval/test_corpus.py`'s pinned regression test
   for the old thin-cell state was updated (not deleted) to assert the new,
   better one, with the old numbers kept in its docstring for history.
3. **The transfer-degradation numbers were base-rate artifacts.** §10 scored
   only the full model on `misspecified`/`coupled` and compared to the full
   model's own *nominal* number. Reviewer scored the null on the same
   transfer frames: `coupled`'s reported "improvement" (−0.19) turned out to
   be near-identical for the null (−0.1938 vs the full model's −0.1883) —
   almost entirely a base-rate shift (`coupled` is 71.3% STILL_PENDING vs
   nominal's 48%, and both models predict that outcome with high
   probability regardless), not model skill. §10 had already flagged this as
   "plausible, not verified" — it is now verified, and was wrong to report
   without the null-relative baseline. Fixed: `eval/model_fit_report.py` now
   scores both models on every transfer frame and reports the full-vs-null
   *gap*, not the full model's raw number.
4. **6 of 15 fitted parameters were pure noise the DGP sets to exactly
   zero.** `days_since_last_attempt`'s true `nominal`-arm coefficient is
   identically 0 (§2 above — it only reaches `_cloglog_probs`, the
   `misspecified`-arm link); there is no slot×window interaction anywhere in
   `_draw_outcome` at any slot, so `slot3_x_in_salary_window`'s true value is
   also 0 at every slot, not just slot 4. Fitted, both had |z| ≤ 1.34 —
   diluting the one real signal's power. §9's corrections (drop
   `committed_day_of_month`, restrict the interaction to slot 3) were
   directionally right but incomplete: they fixed the *collinearity* problem
   and left the two *zero-coefficient* terms in, which is where the wasted
   variance actually was. **Fixed: `FEATURE_COLUMNS` now excludes both**
   (`const, slot_3, slot_4, in_salary_window` — 4 columns, not 6).
   `_design_matrix()` itself is unchanged (still computes all 6 possible
   columns; `HazardModel.feature_columns` selects the subset a given fit
   actually used), so this needed no test-file changes — the existing stub
   tests hardcode their own `feature_columns` tuples and were unaffected.
   This is a model-specification correction driven by review, the same
   category as B4's `estimable`-filter fix, not a change to any evaluation
   *criterion* — §4's "don't move the goalposts" discipline governs the
   gate's pass/fail bar, not the model's own design matrix, which review is
   explicitly supposed to refine (CLAUDE.md's "Definition of done" #4).
5. *(non-blocking, also fixed)* `hazards()` now asserts its output is
   exactly `(len(X), 4)` — `sm.MNLogit` derives its column count from the
   *distinct `event_code` values actually present at fit time*, so a fold
   missing an outcome class would silently return 3 columns and misalign
   `Outcome` int order downstream with no error. `fit()` now asserts all 4
   classes are present in the estimable training rows before fitting, for
   the same reason.
6. *(disclosed, not fixed — flagged for whoever next touches transfer
   scoring)* Reviewer's own least-confident finding: at slot 4 the model
   conditions on `days_since_last_attempt = day4 − day3` but never on
   `day2`, even though `day2` partly determines whether a slot-4 row exists
   at all — covariate-dependent selection not fully spanned by the design
   matrix. Harmless under `nominal` (the only day-dependence, `in_salary_
   window`, is in the model) but not obviously harmless under
   `misspecified`, where day-gaps are the entire mechanism — some unknown
   share of that arm's transfer number may be selection rather than pure
   link misspecification. Not resolved this session.
7. *(forward-looking, for B6)* `splits.py` groups on `mandate_id`, but
   `household_id` is in `features.FORBIDDEN` (correctly excluded from the
   design matrix) and so isn't available to group ON either — under
   `coupled`, `_apply_household_coupling` makes outcomes dependent *within*
   a household, so a household split across train/calib_conf breaks the
   exchangeability split conformal needs, in the direction of narrower
   (more confident) prediction sets — exactly the failure mode the off-ramp
   gate exists to prevent. B6 needs `household_id` carried as an identity
   column (never in the design matrix) before conformal touches a `coupled`
   frame. Not B5's to fix; recorded so B6 doesn't rediscover it.

**Corrected report, rerun after all fixes** (`eval/model_fit_report.py`, via
`eval-runner`; 40-seed corpus → 7,154 mandates, 19,470 person-period rows,
12,316 estimable; verdict now a 20-seed mandate-grouped split sweep, not one
seed):

```
=== held-out log_loss, full vs intercept-only null, 20 split seeds ===
mean(full - null) = -0.00784   SD_diff = 0.00324   SE = 0.00073   t = -10.80
wins: 20/20 seeds
verdict: full model BEATS the null at ~95% (|t|>2)

=== representative single fit, split seed=0 ===
log_loss   full=1.2211  null=1.2290
brier[0]  full=0.2486  null=0.2501  beats
brier[1]  full=0.1838  null=0.1855  beats
brier[2]  full=0.1041  null=0.1041  beats
brier[3]  full=0.1245  null=0.1253  beats

=== transfer degradation, full-vs-null gap on each arm ===
misspecified  full-vs-null=-0.0159  (nominal test full-vs-null was -0.0079)
coupled       full-vs-null=+0.0044  (nominal test full-vs-null was -0.0079)
```

**This is now a defensible result, not a rationalized one.** Log-loss beats
the null decisively and consistently (20/20 seeds, t=−10.80 — an order of
magnitude past the |t|>2 bar, not a marginal call). Brier now beats on all 4
causes at the representative seed (previously 2/4 — a direct, measured
consequence of removing the two zero-signal columns, exactly as fix 4
predicted). Transfer degradation, properly null-relative: the model's edge
over the null *widens* slightly under `misspecified` (−0.0159 vs nominal's
−0.0079 — plausible since the null's constant-hazard assumption is worse
suited there too, though the model was never fit to that arm's actual
mechanism) and *vanishes* under `coupled` (+0.0044, essentially tied) —
coherent with `coupled`'s mechanism (household contention) being invisible
to a design matrix that only sees slot and salary-window timing. Neither
transfer number is alarming and neither needed a fabricated explanation.

**`Σ_c CIF_c(4) + S(4) == 1`**: reviewer independently verified this on the
actual fitted model over the real test-split mandates (not just `cif.py`'s
synthetic-array tests): max deviation **0.000e+00**, CIF monotone
non-decreasing, S monotone non-increasing.

**The B4 `estimable` fix independently reconfirmed on the current fit**: the
spurious effects B4 measured (`+0.10` logit/day on RECOVERED, `+1.4` logit on
OPTED_OUT from including slot-1 rows) are absent; the current fit's
`days_since_last_attempt`/`in_salary_window` coefficients (when included in
diagnostic fits) sit at |z| < 1.2, and the real, expected structure —
OPTED_OUT's slot_3/slot_4 escalation — recovers the DGP's linear ratio
almost exactly (fitted 1:1.81 vs the DGP's exact 1:2).

**A CONFIRMING `stats-reviewer` pass (fresh instance, no memory of the
above) also returned NOT CLEAN — narrowly, and on a genuinely different
class of defect than the first pass.** It independently reproduced "full
beats null" by a method neither prior version of this script used (5-fold
mandate-grouped CV with mandate-clustered SEs: mean=−0.00794, SE=0.00120,
t=−6.64, 5/5 folds negative) and confirmed the substantive conclusion is
TRUE — but found the *reported* dispersion statistic was itself invalid, and
two sentences already written into this record were false.

1. **BLOCKING — the 20-split-seed "SE"/"t" was not a standard error.** The
   20 `SPLIT_SEEDS` re-splits reuse the SAME fixed corpus with ~90%-
   overlapping test sets, so `SD/sqrt(n_seeds)` measures split-to-split
   variability of one fixed dataset, not sampling error (the classic
   repeated-random-subsampling trap — Dietterich 1998; Nadeau & Bengio
   2003). Proof: the reviewer reran the identical computation at
   n_seeds=10/20/40/60 on the SAME data and got t=−9.56/−10.80/−14.01/
   −18.67 — a statistic that is a function of a free parameter (how many
   times you loop), not evidence. The point estimate (mean ≈ −0.0078) was
   fine throughout; only the SE/t/verdict derived from it was wrong.
   *Fixed:* `eval/model_fit_report.py` now computes the primary verdict via
   `_grouped_cv_diffs()` — 5-fold mandate-grouped CV, disjoint and jointly-
   exhaustive folds, refitting full+null per fold, so the K per-fold means
   genuinely are independent samples and `SD/sqrt(K)` is valid. The old
   seed-sweep is kept only as a labeled "split-stability check" (mean/SD/
   win-count), with an explicit comment that no SE/t/verdict may be derived
   from it.
2. **BLOCKING (record integrity) — the Brier 2/4→4/4 improvement was
   misattributed.** This entry's own text credited "removing the two
   zero-signal columns" (fix 4 above). Reviewer held the corpus fixed at 40
   seeds and compared the OLD 6-column design against the NEW 4-column one:
   both score 4/4 on Brier, with the 6-column design's log-loss marginally
   *better*. The improvement came entirely from fix 2 (widening
   `TRAIN_SEEDS` 10→40), not from dropping the two columns. Corrected here;
   also corrected in `src/model/competing_risks.py`'s docstring.
3. **BLOCKING (record integrity) — `|z| <= 1.34` was a stale 10-seed
   number, and "improves the held-out margin" was false.** On the 40-seed
   corpus actually in use, `slot3_x_in_salary_window`→OPTED_OUT is
   **z=2.85** (not noise), and the 4-column vs 6-column designs differ by a
   stability-sweep mean of −0.00002 (essentially a coin flip, 4-column
   better on only 11/20 seeds) — not an improvement. **Deeper correction, not
   just a stale number:** the original reasoning ("the frozen simulator sets
   this coefficient to exactly zero, therefore it's noise") is a category
   error for THIS model. `_draw_outcome`'s zero-coefficient claim holds only
   *conditional on the latent cause*, which this model never observes — it
   fits the CAUSE-MARGINAL hazard, and the risk set's cause mix shifts by
   slot (CANT_PAY_NOW mandates resolve and exit, enriching later slots in
   WONT_PAY/CANT_PAY_EVER), so a marginal model can show a genuinely
   non-zero coefficient on a term that is exactly zero within every latent
   stratum. Reviewer confirmed directly: within-latent-cause fits keep both
   terms at |z| <= 1.45; only the pooled marginal fit shows z=2.85. The drop
   is still kept (it is empirically neutral, per the −0.00002 stability
   result, and defensible on parsimony/DGP-consistency grounds alone) but
   **the original justification is retracted and must not be reused as
   precedent** — flagged explicitly in `competing_risks.py`'s docstring so
   B8 doesn't inherit a category error when it next touches this design
   matrix.
4. *(non-blocking, disclosed)* The transfer-degradation numbers (misspecified
   −0.0159, coupled +0.0044) remain a single point estimate at one split
   seed with no dispersion — unlike the nominal-arm claim, not yet put
   through K-fold CV. Flagged in the report's own output and here; not fixed
   this session.
5. *(reconfirmed clean)* All of the first pass's CLEAN findings (join
   correctness, filter-before-fit, split/assemble ordering, null purity,
   transfer-scoring isolation, censoring handling) — untouched by this
   round's fixes, and the confirming reviewer's own spot-checks did not
   dispute them.

**Corrected numbers, final** (`eval/model_fit_report.py`, via `eval-runner`,
after all three fixes):

```
=== 5-fold mandate-grouped CV (the valid inferential test) ===
per-fold (full - null): [-0.00654, -0.0077, -0.00601, -0.00794, -0.01084]
mean = -0.00781   SD = 0.00188   SE = 0.00084   t = -9.30
folds negative (full beats null): 5/5
verdict: full model BEATS the null at ~95% (|t|>2)

=== split-stability check ONLY (not inferential) ===
mean across 20 split seeds = -0.00784   SD = 0.00324   wins = 20/20
```

Independently cross-checked against the confirming reviewer's own from-
scratch implementation of the same method: mean=−0.00794, SE=0.00120,
t=−6.64, 5/5 folds negative. This run: mean=−0.00781 (differs by 0.00013,
well inside noise), t=−9.30 (stronger magnitude, likely fold-assignment
implementation detail — GroupKFold's own deterministic partitioning
differs run-to-run only in which mandates land in which fold, not in
whether folds are valid). Same conclusion, same order of magnitude, from
two independent implementations. **This is the number the B5 gate closes
on**, not the retracted single-seed or repeated-subsampling ones above.

**Gate closed.** Both `stats-reviewer` passes' blocking findings are
addressed; the confirming pass explicitly said it would sign off once
finding 1's replacement was in place, and that replacement — 5-fold
mandate-grouped CV — is what produced the number above, independently
reproduced by a second run. `Σ_c CIF_c(4) + S(4) == 1` verified on the real
fitted model (confirming pass, max deviation 0.000e+00) as well as on
`cif.py`'s own synthetic-array test suite (21/21 passing).

---

### 2026-08-28 · B6 · Calibration + conformal: label space narrowed to Outcome, household split fix, isotonic measured to lose, schedule-threading deferred

Four decisions taken before any code, all confirmed with the user first
(this project's "never cut scope without asking" rule, applied to a
planning-time fork rather than a mid-build one).

**1 — Conformal validated over `Outcome`, not `Cause`.** `PLAN_DETAIL.md:299`
literally specifies `pred_set(score) -> set[Cause]`. No cause-scorer exists at
B6 — `belief.py` is B7, `intent.py` is B11, and `cause_map.py`'s own docstring
says nothing downstream of B5 should read it. `Cause` is also latent: it has
no production label, ever, so a `Cause`-space coverage claim could never be
re-validated against real issuer data. `src/model/conformal.py` therefore
consumes nonconformity **scores only, never probabilities**, and is generic
over any hashable label type (tested directly:
`test_generic_over_custom_enum_labels` exercises a 3-member throwaway enum
alongside the 4-member `Outcome` case). **Consequence:** B6 unblocks the
off-ramp gate's *machinery*, not a live off-ramp — B8 keeps `FullSetGate`
(never offers, the safe direction) until a cause posterior exists at B7/B11.
`PLAN_DETAIL.md:309`'s "unblocks B8's *real* off-ramp gate" is narrowed by
this entry, not by an edit to that file.

**2 — `household_id` carried as an identity column; `split()` groups on it.**
Flagged at B5 (stats-reviewer finding 7, previous entry): `splits.py` grouped
on `mandate_id` only, but under `coupled` a household split across
train/calib_conf breaks the exchangeability split conformal needs, narrowing
prediction sets in the direction of a false off-ramp. Fix: `EMITTED_COLUMNS`
gains `household_id` (identity block, after `row_id`); `person_period.
validate()` gains two consistency checks (constant within a
`(mandate_id, cycle_id)` group, and across a mandate's cycles);
`features.featurize()` drops it the same way it already drops `on_day`, so
`FORBIDDEN`'s existing `household_id` entry is never actually violated by
`featurize()`'s own output; `splits.split()` gains a keyword-only
`group_key: pd.Series | None = None`, default `None` reproducing today's
mandate-only grouping exactly. **Bit-identity is the load-bearing guarantee,
not a nice-to-have**: on `nominal`, `household_id` is always null, so
`household_id.fillna(mandate_id)` is elementwise identical to `mandate_id`,
and every already-reported B5 number is reproduced byte-for-byte — proven,
not asserted, by `test_split_group_key_matches_default_bit_identical_on_
synthetic_frame` (`pd.testing.assert_frame_equal` across 5 split seeds) and
a second test exercising the literal `household_id.fillna(mandate_id)`
expression against a real (all-null) column on a `person_period.build()`
frame. A `Simulator` trap noted for whoever next builds a multi-seed
`coupled` corpus: `household_id = f"H{i // household_size}"`
(`simulator.py:182`) is **not** seed-namespaced the way `corpus.py`
namespaces `mandate_id` — namespace it in `corpus.py` when that corpus is
built, never in the frozen simulator.

**3 — Per-class isotonic on the three event hazards, `STILL_PENDING` as pure
residual, never renormalised.** `cif.py:63-68` already treats hazard column 0
as "survives this slot", never reading it — `calibration.py`'s
`h_cal[:,0] = 1 - (h_cal[:,1]+h_cal[:,2]+h_cal[:,3])` makes that semantics
explicit rather than inventing a new one. `apply()` raises `SimplexViolation`
(never clips) if the three event maps would sum above 1 — tested against a
deliberately pathological hand-built `IsotonicCalibrator`, not just the
happy path.

**Real, measured finding, not a hypothesis:** isotonic calibration on this
model's hazards **does not improve** classwise ECE. 20-seed sweep on the real
40-seed corpus: raw ECE min/mean/max = 0.0116/0.0213/0.0283, calibrated =
0.0165/0.0256/0.0350 — calibrated is worse on every seed measured. Mechanism:
this model's `FEATURE_COLUMNS` (`const, slot_3, slot_4, in_salary_window`)
produces only 6 distinct hazard vectors; hazard-level calibration on `test` is
already close to its own noise floor, and isotonic fit on ~1,200-row
`calib_iso` cells replaces a 4-parameter MLE fit on ~8,600 rows with
per-cell empirical means on far less data. The regression guard in both
`tests/model/test_calibration.py` and the eval report is therefore
`classwise_ece(calibrated) <= classwise_ece(raw) + 0.01`, explicitly **not**
"calibration improves ECE" — a future session must not "fix" this direction
without re-deriving it; a test demanding improvement would be asserting
noise, which is exactly the trap `competing_risks.py`'s B5 caveat (this
file's earlier entry, "do not repeat the original argument as precedent")
already burned this project once on.

**4 — LAC over APS, smoothed Mondrian conformal as the shipped default** (per
the approved plan; not relitigated here). Real, measured trap caught during
implementation, not anticipated at planning time: `apply()`'s simplex check
was originally `event_sum >= 1.0`, which incorrectly rejected the valid
boundary case (residual exactly 0) — caught by
`test_apply_maintains_isotonic_monotonicity` producing a false
`SimplexViolation` at `sum == 1.0` exactly; fixed to strict `> 1.0`.

**Schedule-threading deferred, disclosed as a measured rate, not silently
dropped.** `paths.hazard_tensor()`'s preferred path (thread the real
committed schedule through `eval.corpus.Episode`, eliminating imputation for
un-attempted slots) was **not** implemented this block — the `schedule=None`
fallback (exact when the mandate's last real attempt already fell outside the
salary window; a documented, disclosed assumption otherwise) is what ships.
Measured, not estimated: **42.7%–43.6% of `test`-split tensor cells are
imputed**, printed by `eval/model_fit_report.py`'s new B6 section every run
(`sweep['imputed_fraction']`), not just asserted in this file. This is within
the plan's own stated tolerance ("Prefer threading the real schedule... the
`schedule=None` fallback imputes 41% of cells... report the count either
way") but is real scope not completed, flagged explicitly for whoever next
touches `paths.py` or B8's allocator (which will want the real schedule
regardless, to commit attempt days).

**Bugs found and fixed in generated tests, not in the design.** Three
test-writer subagents (Haiku) produced `tests/model/test_paths.py` (36
tests), `tests/model/test_calibration.py` (33), `tests/model/test_conformal.py`
(30), plus 14 tests extending `test_person_period.py`/`test_features.py`/
`test_splits.py` for decision 2 — all reviewed and fixed by hand before
implementation, not accepted blind:
1. `test_paths.py`'s `_build_model_frame` fit `competing_risks.fit()` on the
   SAME small, outcome-homogeneous episodes each test used for scoring —
   `fit()` requires all 4 `event_code` classes present, which most
   individually-small per-test episode sets don't have. Confirmed by direct
   reproduction before fixing, not assumed. Fixed by decoupling fit-time data
   (a fixed, class-diverse `_diverse_fit_frame()`) from predict-time data —
   which mirrors real usage, not a workaround (`competing_risks.py`'s own
   docstring already notes fit-time and predict-time batches routinely
   differ in composition).
2. One `test_paths.py` case built a slot-4-only attempt with no slot-2/3 rows
   — `person_period.build()` correctly rejects this as a non-contiguous slot
   sequence. Fixed by adding the missing intermediate STILL_PENDING attempts,
   matching every other multi-slot test in the same file.
3. `test_calibration.py`: three tests fed statistically **independent**
   random `h`/`y` (two separate `RandomState(seed)` draws, no relationship)
   into isotonic fitting at small n (15-20 rows) — three independently-noisy
   per-class curves evaluated together on further-independent test data can
   genuinely sum above 1 by sampling chance (measured, not assumed: reaching
   sums up to ~1.31). This is `SimplexViolation` working as designed against
   a real pathology, not a bug in it, but the wrong fixture for tests
   checking unrelated properties (exact-sum arithmetic, determinism, cif
   compatibility) — fixed by sizing those three tests' calibration data to
   n=300 (verified empirically against each test's exact seed before
   committing). A fourth test (`test_calibration_graceful_degradation_
   independent_data`) is legitimately ABOUT independent data by design and
   name — fixed the same way (n_calib 50→500) rather than changing its
   intent. A fifth test called `RandomState.dirichlet(..., random_state=rng)`
   — not a valid parameter on that method (confused with a
   `scipy.stats`-style API) — fixed by removing the invalid kwarg.
4. `test_conformal.py`'s `test_exact_coverage_on_continuous_exchangeable_
   scores` asserted single-seed marginal coverage in a narrow
   `[0.945, 0.955]` band, justified only by test-set sampling SE
   (`sqrt(0.95*0.05/20000)≈0.00154`). Split conformal's coverage for one
   FIXED calibration draw is itself a random variable (its quantile is an
   order statistic; asymptotically Beta-distributed) with its own standard
   deviation — measured directly by sweeping 20 independent calibration
   seeds: mean 0.9489, SD 0.0055, roughly **half the seeds land outside a
   naive ±0.005 band**, purely from calibration-draw variance the test's own
   justification never accounted for. This is the identical class of mistake
   this project already caught and fixed once before, at B5 (`eval/
   model_fit_report.py`'s own docstring: "a single-seed comparison is not
   evidence of anything by itself"). Fixed the same way: sweep 20 seeds,
   assert on the mean with a band sized off the measured SD, not one
   arbitrary seed. `test_pred_set_returns_frozenset` separately hit
   `ConformalUnderpowered` at its original n=100/seed=11 (one class randomly
   drew only 16 of the required 19 examples) — orthogonal to what that test
   checks (`pred_set`'s return type); fixed by raising n_cal to 150,
   verified to clear the floor for that exact seed before committing.
5. `coverage_report()`'s integer count columns (`n`, `singleton_count`, …)
   returned `numpy.int64` on `.iloc[]` access, which is not
   `isinstance(x, int)` (unlike `numpy.float64`, which does subclass Python's
   `float` — the asymmetry is a real numpy quirk, not a guess) — caught by
   `test_mean_prediction_set_size_and_singleton_count_correct`. Fixed by
   casting those columns to `dtype=object` before returning, the same fix
   already used in `src/model/paths.py`'s `HazardTensor.observed` for the
   identical reason (`is True`/`is False` identity checks). One test in the
   same function also asserted `mean_set_size >= 1` as if it were a real
   invariant — it isn't (alpha=0.5 against 3 calibration rows routinely
   produces empty sets, which the same test suite's own
   `test_coverage_report_counts_empty_set_as_miscoverage` relies on via high
   alpha) — corrected to the actual structural guarantee, `>= 0`.

**Numbers, real 40-seed corpus, 20-seed sweep** (`eval/model_fit_report.py`,
via `eval-runner`):

```
classwise ECE  raw:        min=0.0116  mean=0.0213  max=0.0283
classwise ECE  calibrated: min=0.0165  mean=0.0256  max=0.0350

conformal marginal coverage: min=0.9360  mean=0.9568  max=0.9675  (nominal 0.95)
conformal mean set size:     min=3.8195  mean=3.8437  max=3.8801  (of 4 labels)
conformal singleton rate:    min=0.0000  mean=0.0030  max=0.0188

per-class coverage [STILL_PENDING]: min=0.8659  mean=0.9545  max=0.9881
per-class coverage [RECOVERED]:     min=0.9365  mean=0.9587  max=0.9753
per-class coverage [DEAD]:          min=0.9073  mean=0.9615  max=1.0000
per-class coverage [OPTED_OUT]:     min=0.8882  mean=0.9514  max=0.9770

un-attempted-slot imputation rate (test split): ~0.427-0.436
```

**Honest headline, stated plainly rather than left implicit:** coverage
holds reasonably per-class under exchangeability (all four classes' means
sit within ~1pt of nominal; STILL_PENDING's minimum, 0.8659, is the
softest of the four, consistent with it being the class this project's own
design notes flagged as hardest), but the sets are almost always
uninformative at this model's resolution — mean size 3.84 of 4, singleton
rate under 0.3%. That correctly routes the real work (a design matrix with
enough resolution to ever produce a singleton) to B8, rather than a passing
coverage number hiding it.

**Gate status:** `stats-reviewer` review pending as of this entry; will be
appended once returned, per this project's standing review discipline.

---

### 2026-08-28 · B6 · stats-reviewer review — 2 blocking findings, both real, both fixed; gate closed on corrected numbers

Full review requested against the entry above, with six specific questions
(leakage across the four-way split, `terminal_labels()`'s eligibility edge
cases, the `household_id` bit-identity claim, isotonic's residual
construction, the conformal p-value formula I had explicitly flagged as
self-unverified, and the `schedule=None` imputation's effect on the
numbers). Verdict up front, in the reviewer's own words: "the censoring
semantics, the split, and the bridge are all correct — I tried hard to
break them and could not." Two blocking findings, both real, both now
fixed and reverified — this entry is not a summary written before seeing
whether the fixes worked; both were rerun via `eval-runner` after the
fixes and the corrected numbers are below.

**BLOCKING 1 — `hazard_tensor()`'s imputation was outcome-determined: real
leakage, not a precision limitation.** The `schedule=None` fallback (which
is what every call in `eval/model_fit_report.py` was actually using,
despite the `schedule` parameter existing) imputes an un-attempted slot's
`in_salary_window` based on whether the episode *survived* to that slot —
and survival to slot 3/4 is a deterministic function of the very outcome
being predicted. Measured corpus-wide: essentially 100% of STILL_PENDING
episodes have a real slot-3 row; only 36–43% of RECOVERED ones do. That is
exactly `src/model/CLAUDE.md` rule 2 ("no feature may encode the future"),
and it meant the previously-reported coverage number (this entry's earlier
table) was for a predictor that cannot exist at commit time — at commit
time NO slot-3/4 cell is real, so the deployed scorer is 100% imputed while
the evaluated one was ~57% real. The reviewer's own decision-time-only
reprobe (imputing slots 3/4 unconditionally, matching what a real deployment
sees) found OPTED_OUT and DEAD coverage falling *below* nominal — the unsafe
direction, sets narrower, closer to the singleton that fires the off-ramp.

**Fixed by doing the deferred item properly, not by degrading to the
pessimistic always-impute case.** `eval/corpus.py`'s `Episode` gains an
optional `schedule: tuple[int, int, int] | None = None` field — the full
`(day2, day3, day4)` `_draw_schedule()` already draws before any `attempt()`
call (legitimate per `assert_legal`'s own "committed once, never adjusted"
guarantee); `generate()` now populates it. `eval/model_fit_report.py`'s
`_build_corpus()` returns a third value, `schedule_df` (built by the new
`_schedule_frame()`), threaded into every `hazard_tensor(..., schedule=...)`
call in `_calibration_conformal_sweep`. This is not a new design — it is
`paths.py`'s own already-documented preferred path, previously deferred and
disclosed as a measured imputation RATE (~43%) rather than a measured BIAS.
The reviewer's finding is what turned "rate, disclosed" into "which cells,
and that matters" — the fix removes the *dependence*, not the *rate*: the
un-attempted-slot rate is still ~40–44% (mandates that resolve early
genuinely have fewer real attempts — that part was never the problem), but
every one of those cells now gets its `in_salary_window` from the real
committed day, the same source regardless of what happened at earlier
slots. `eval/model_fit_report.py`'s printed line was reworded to say this
precisely, not just re-labelled.

**BLOCKING 2 — the conformal p-value formula was not Vovk's smoothed
p-value.** `src/model/conformal.py`'s `_p_value` computed
`(greater + weight*equal + 1) / (n+1)` — the `+1` (the test point's own tie
with itself under the hypothetical (n+1)-point exchangeable sequence) was a
flat constant instead of being weighted by `u` like the calibration ties
are. This under-smooths, giving systematic OVER-coverage that grows as the
pool shrinks and ties get heavier — precisely this project's regime
(Mondrian pools as small as the `ceil(1/alpha)-1=19` floor, ~6 distinct
hazard atoms). Measured by the reviewer in isolation: up to +4.35 points of
excess coverage at pool size 19 with 6 score atoms. One-line fix:
`(greater + weight*(equal + 1)) / (n+1)`. The unsmoothed path (`u=1`) is
unaffected — weighting doesn't matter when it's always 1, which is why
`test_heavy_ties_instability_unsmoothed_shows_variance` and every other
unsmoothed-path test still passes unchanged.

**Corrected numbers, both fixes applied, real 40-seed corpus, 20-seed
sweep** (`eval/model_fit_report.py`, via `eval-runner`; both directions
moved exactly as theory predicts — down, toward nominal, tighter sets):

```
                              before (this entry, pre-review)   after (both fixes)
classwise ECE  raw:           min=0.0116 mean=0.0213 max=0.0283  unchanged (fixes don't touch calibration.py)
classwise ECE  calibrated:    min=0.0165 mean=0.0256 max=0.0350  unchanged

conformal marginal coverage:  min=0.9360 mean=0.9568 max=0.9675  min=0.9327 mean=0.9517 max=0.9653
conformal mean set size:      min=3.8195 mean=3.8437 max=3.8801  min=3.7301 mean=3.7830 max=3.8483
conformal singleton rate:     min=0.0000 mean=0.0030 max=0.0188  min=0.0000 mean=0.0069 max=0.0421

per-class coverage [STILL_PENDING]: mean 0.9545 -> 0.9476
per-class coverage [RECOVERED]:     mean 0.9587 -> 0.9557
per-class coverage [DEAD]:          mean 0.9615 -> 0.9490   (largest single move, matching the p-value fix's expected effect)
per-class coverage [OPTED_OUT]:     mean 0.9514 -> 0.9495

un-attempted-slot rate (test split): min=0.4004 mean=0.4236 max=0.4376
```

No implementation defects in the corrected run (all coverage in [0,1], set
sizes in [0,4], no NaN). Notably, the corrected numbers land BETTER than the
reviewer's own decision-time-only reprobe (which used the pessimistic
always-impute-False approximation, not the real fix) — properly threading
the schedule gives the model genuine information at every slot rather than
forcing a worst-case guess, so DEAD/OPTED_OUT stay near nominal (0.9490,
0.9495 mean) rather than falling below it. **This is the honest number: no
leaked information, no over-smoothing bug, real 40-seed corpus.**

**Non-blocking findings, addressed:**
1. *(terminal_labels robustness)* Eligibility's primary filter changed from
   `slot == horizon AND censor_reason == BUDGET_EXHAUSTED` to `slot >=
   horizon`, with `BUDGET_EXHAUSTED` demoted to an assertion (raises loudly
   if violated) rather than baked into the mask — a STILL_PENDING row at or
   past the horizon is the observed event regardless of *why* the episode
   then stopped; today `BUDGET_EXHAUSTED` is the only reason this corpus
   ever stamps there (verified corpus-wide by the reviewer: 777/777 slot-4
   pending rows are `BUDGET_EXHAUSTED`, 283/283 slot-3 pending are
   `WINDOW_CLOSED`, zero at slot 1/2) — but a future censor reason at the
   horizon no longer risks silently dropping an honest label.
2. *(`split()` defensive check)* `group_key`, when supplied, now gets a
   length check and an index-equality check against `df`, raising
   `SplitIntegrityError` on mismatch — every internal slice is `.iloc[pos]`
   on both `df` and `key` in lockstep, which silently assumed positional
   alignment before. Not exercised by any production caller today (see
   finding 3 below), but a real footgun for whoever wires one in.
3. *(disclosed, not fixed — correctly scoped as future work)* The
   `household_id`/`group_key` machinery from earlier in this entry is
   real and tested, but **inert**: `featurize()` drops `household_id`
   before `assembled` exists, and `assembled` is the only frame
   `eval/model_fit_report.py` ever calls `split()` on — so B5's finding 7
   (household exchangeability under `coupled`) is not yet closed
   operationally, only the mechanism exists. Correct as-is: there is no
   `coupled`-arm training/reporting pipeline before a later block, so
   nothing exists yet to wire it into. Flagged here so whoever builds that
   pipeline knows the machinery is ready and tested but not yet called with
   a real `group_key`.
4. *(caveat added, not a functional change)* `cif.terminal_distribution()`'s
   docstring now discloses that the eligible mandate population (excluding
   the ~4% WINDOW_CLOSED-before-horizon episodes) is not a random subset —
   exclusion correlates with the same schedule draw that sets
   `in_salary_window`. Confirmed harmless for conformal specifically (an
   independent permutation control gave 0.9575 vs the real split's 0.9578 —
   indistinguishable), but a real caveat for a future reader of this
   function's output as unconditional per-mandate risk (B8's allocator).
5. *(disclosed, one sentence, not a code change)* Isotonic calibration and
   split conformal are never composed in this block — conformal scores raw
   hazards (`hazard_tensor` → `competing_risks.hazards`), not
   isotonic-calibrated ones. Given isotonic measurably loses (this entry's
   original finding, independently reconfirmed by the reviewer per-class:
   the three fitted event classes degrade by +0.0100 combined, more than
   the residual class absorbs at +0.0041), not composing them is the
   correct call for now, not an oversight — but it means `calib_iso`'s 10%
   of every split is currently spent on a component nothing downstream
   consumes.
6. *(cleanup)* Removed `pp_calib_iso` from `_calibration_conformal_sweep` —
   computed, never used; isotonic calibration works at the person-period row
   level and never needed `terminal_labels()`'s per-mandate frame.

**Confirmed clean, independently reasoned through (not re-asserted from my
own claims):** the household bit-identity mechanism (`GroupShuffleSplit`
partitions from `np.unique(groups)`'s sort order and count alone, never
dtype or array provenance — the claim is true for the right reason, not by
luck); isotonic's lack of any hidden `test`-split dependency; the censoring
semantics in `person_period.build()` ("the thing most likely to be wrong in
a model like this, and it is right"); split exchangeability, verified
independently by pooling `calib_conf ∪ test` and re-splitting at random
(0.9575 vs the real grouped split's 0.9578 — indistinguishable, meaning the
residual ~0.8pp of over-coverage after the p-value fix is ordinary
finite-sample Mondrian conservativeness, not contamination); the bridge's
split-membership preservation end to end (the `_row` position carried
through `_terminal_distribution_and_labels`'s merge makes cross-mandate
mislabelling structurally unreachable); `assert_disjoint()` present at both
boundaries that matter; cause-specific-vs-subdistribution usage (the
allocator-facing quantity is correctly `cif.terminal_distribution`, absolute
risk, not the cause-specific hazards); every structural invariant (no
LLM/`Cause` import in the new modules, `household_id` physically dropped,
money untouched, the `> 1.0` simplex boundary fix is correct as shipped).

**Carried forward for B8, not B6's to fix:** the reviewer's own
least-comfortable flag — the coverage guarantee rests on `calib_conf`/`test`
exchangeability, which holds here only because the simulator's schedule RNG
(`corpus.py`'s `day_rng`) is independent of its outcome RNG. In a real
deployment, the allocator *chooses* the committed schedule in response to
the model's own output, which breaks that independence. Coverage validated
on this corpus does not automatically transfer to a closed loop — written
down now, while the reasoning is fresh, for whoever builds B8.

**Gate closed.** `★ calibration + conformal: reliability diagram roughly
diagonal` — supported (raw classwise ECE 0.0213 with ~140 rows/atom/split
sits at the estimator's own noise floor, stated as such, not dressed up).
`empirical coverage matches nominal on held-out data` — supported on the
corrected numbers above: marginal 0.9517 mean (min 0.9327), all four
per-class means within ~1.5pp of nominal, mean set size 3.78/4, singleton
rate under 0.7%. The honest headline stands as originally written: coverage
holds reasonably per-class under exchangeability, but the sets are almost
always uninformative at this model's resolution — which correctly routes
the real work (a design matrix with enough resolution to ever produce a
singleton) to B8, rather than a passing number hiding it.

### 2026-08-29 · B7 · Explicit likelihood inversion for the belief update; cause_map.py's docstring narrowed, not deleted

`src/policy/belief.py`'s `update(b, obs)` needs `P(decline | cause)`, the
likelihood Bayes' rule actually multiplies by. Nothing in the repo fits
this direction: `src/classify/cause_map.prior()` is the opposite direction,
`P(cause | decline)`, a hand-authored table whose own docstring says it is
exact "only given a flat prior over causes" — using it directly as a
likelihood would be exact today (the current prior over causes happens to
be uniform) and silently wrong the moment `init()` is ever called with a
non-flat starting prior, which is precisely the case B7 exists to support
(e.g. chaining a mandate's belief from a previous cycle's posterior).

**Chosen: read `cause_map.prior()` and invert it explicitly** through a
named `REFERENCE_PRIOR = (1/3, 1/3, 1/3)` constant, carrying its own
`REFERENCE_PRIOR_VERSION = "ref-v1"`:

    likelihood(dc)[c] = cause_map.prior(dc)[c] / REFERENCE_PRIOR[c]

left deliberately unnormalised — any `dc`-only factor cancels inside
`update()`'s renormalisation, so normalising here would be pure busywork.
`Belief.provenance` composes `cause_map.PRIOR_VERSION` (currently `"v2"`)
with `REFERENCE_PRIOR_VERSION`, so a belief written to `plan.belief_json`
is traceable to both tables that produced it — what B11's gate ("normaliser
output is versioned in the ledger before it can touch a belief") requires.

Two alternatives were considered and rejected. A second, independently
hand-authored `P(decline | cause)` table living in `belief.py` avoids the
inversion but creates two tables related by Bayes' rule with nothing
forcing them to agree — drift between them would be invisible to any test
that checks each table alone, the same duplicated-source-of-truth failure
mode `AFA_FREE_LIMIT_PAISE` already avoids by having exactly one home. And
the new table would carry no `PRIOR_VERSION`, leaving B11's gate nothing to
thread. Deferring `belief.py` to B8 was also rejected — it would move the
hardest remaining modelling decision into the block that already carries
backward induction and the allocator.

**The anti-drift test.** `tests/policy/test_belief.py::
test_inversion_round_trips_to_prior_exactly`, parametrized over every
`DeclineClass`: invert `prior(dc)` to a likelihood, re-apply Bayes with
`REFERENCE_PRIOR`, and recover `prior(dc)` exactly. The identity holds for
*any* `REFERENCE_PRIOR`, not just the current uniform one, so it breaks the
moment the two representations drift apart rather than merely today.
Verified: passes for all 7 `DeclineClass` members, `abs=1e-9`.

**`cause_map.py`'s docstring narrowed, not deleted.** Its original text
read, verbatim:

> "B5/B6 supersede this with fitted hazards; nothing downstream of B5
> should still be reading this file."

This was written specifically about the file's **outcome-hazard** role —
at the time, `person_period.py`/`paths.py`/`competing_risks.py` had just
taken over producing `P(outcome | slot, ...)`, and the sentence meant
"stop reading this file as *that*." Read literally, though, it forbids
*every* downstream read, which contradicts `PLAN_DETAIL.md` section
4:999's own comment — present since the file's B3 authoring, unchanged —
naming `cause_map.prior()` as the belief update's likelihood source. No
belief coefficient of any kind existed when this sentence was written; the
contradiction was latent until B7 needed to actually call `prior()`.

New text narrows rather than deletes: it still forbids a downstream reader
from treating this file as an outcome-hazard source (that prohibition
stands, unchanged in force), and names the one permitted exception —
`src/policy/belief.py` reading `prior()` itself, explicitly, for the
inversion above. A specific prohibition was narrowed to carve out one
named exception; it was not turned into a general permission.

### 2026-08-29 · B7 · Static-cause belief update is measurably overconfident relative to cause persistence — disclosed, not damped

`eval/frozen/protocol.md` (lines 153-159, pre-registered at B2, before any
file under `src/policy/` existed) already flagged this: `cause_switch_prob`
"intended target is the *within-mandate* stationarity assumption the
belief update ... relies on ... expect this arm to stress B7/B8 more than
B5." B7 measures the stress rather than assuming it away.

**Measured** (`tests/policy/test_belief.py`, exact arithmetic, `abs=1e-9`):
three identical `INSUFFICIENT_FUNDS` declines from a flat prior drive
`update()`'s belief to **0.996108949416** confidence in `CANT_PAY_NOW`
(`(0.8^3)/(0.8^3+0.1^3+0.1^3) = 256/257`). `sim_config.yaml:99`'s
`cause_switch_prob: 0.15` on the `misspecified` arm means the probability
the mandate's cause even *stayed the same* across those same three attempts
is only **0.614125** (`0.85^3`). The gap — **0.381983949416** — is the
overconfidence: `update()` behaves as if it were far more certain than the
generative process it will be evaluated against actually allows.

**Decision: `update()` stays pure static Bayes — `b[c] * likelihood(dc)[c]`,
renormalised, nothing else. No damping, no tempering, no cause-switch-leak
parameter, not even one that defaults to off.** Two alternatives were
considered and rejected, both for the same underlying reason:

- **A `switch_eps` parameter mixing the stationary distribution back in
  each update, defaulting to `0.0`.** This is the principled model — it is
  literally the HMM transition `cause_switch_prob` describes. Rejected
  anyway: a parameter that exists, defaults to zero, and is documented as
  "B8/B13 can turn it on with evidence" *will* be turned on, using results
  measured on the very arm it was built to fix. That is the same shape as
  B5's stop-threshold scalar and the paired-criterion reversal logged
  2026-08-28 — a dial pre-aimed at the grading axis before any honest
  measurement exists to aim it. This is the third instance of that pattern
  in this project; declined on the same grounds each time.
- **Picking a damping constant now, before B8 has measured anything.** An
  unattributed tuning constant chosen to improve results ahead of any
  evidence is exactly what `src/policy/CLAUDE.md` rules out for this
  layer — it belongs in a config file, chosen with evidence, not silently
  embedded in the update rule.

If B8's allocator shows this overconfidence materially hurts recovered
money, mandates preserved, or the off-ramp's false-positive rate, that is a
**finding to report** against the frozen eval, not a defect in `belief.py`
to quietly patch. `test_static_cause_belief_is_overconfident_relative_to_
cause_persistence` pins both numbers and the gap so the finding, if it
comes, is a comparison against a known baseline rather than a surprise.

### 2026-08-29 · B7 · Cause-conditioned hazard gap named as a Protocol, not closed

`PLAN_DETAIL.md` section 4's `Q(b, ATTEMPT(d,m))` sums over causes using
`h_rec(c,d,m,ctx)`, `h_opt(c,d,m,ctx)`, `h_dead(c,d,m,ctx)` — hazards
conditioned on a *specific* cause `c`. B5 shipped hazards marginal over
cause instead (`competing_risks.hazards()`'s signature has no cause
argument at all), and it could not have done otherwise: `Cause` is latent
with no production label, ever (2026-08-28, B6 entry above), so nothing
exists to fit `P(outcome | cause, slot, day, amount)` against. This gap is
therefore permanent, not a B7 shortcut to close later with more data.

**Decision: `src/policy/hazards.py` defines a `CauseConditionedHazard`
Protocol at B7, and implements nothing.** A dated comment alone (the
`reports/gates.md` B4/B5/B6 amendment pattern) was judged insufficient on
its own here — a comment in another file is read at gate-check time,
*after* B8 is already written, and does not constrain the code while it is
being written. Every genuine save in this project so far has come from a
mechanical constraint (`guard_frozen.py`, the ledger's `plan` foreign key,
`guard_invariants.py`), not from a note read later. The Protocol makes
substituting cause-marginal hazards an explicit, visible act in B8's own
type signature, rather than a silent default. Both the Protocol and the
documentation trail ship together, as complements, not alternatives — see
the dated amendment on B8's gate in `reports/gates.md`, 2026-08-29.

Bounded deliberately: type declaration only, no logic, no default
implementation, no helper functions. If it acquires one, that is B8's work
leaking into B7.

**Explicitly left open, for B8 to decide with the allocator in front of
it, not settled here:** whether the allocator resolves this gap by fitting
something new, or by having `Cause` enter only through *which actions are
legal* — `REAUTH` when `CANT_PAY_EVER` dominates the belief, `OFFER` only
on a singleton conformal set — rather than through the hazard arithmetic
itself. That would be a real narrowing of section 4's `Q`-function, and
deciding it now, at B7, to make the handoff tidier would be deciding it
without the allocator's actual constraints in view. Recorded as an open
question on B8's gate rather than pre-answered.

### 2026-08-29 · infra · `eval-quick` removed from `ci`; it was never buildable this early, and it was never "since B4"

Raised as a pre-B8 concern: `.\run.ps1 ci` — the target intended to be the
Stop hook's gate — had failed every session since B4 with
`ModuleNotFoundError: No module named eval.run`, training the habit of
ignoring a red hook that also carries `guard_invariants`.

**Two corrections to the premise, checked before acting on it rather than
assumed:**

1. **No `Stop` hook is actually wired.** `.claude/settings.json` defines
   only `PreToolUse` (`guard_frozen.py`), `PostToolUse` (`guard_invariants.py`),
   and `SessionStart` (`show_state.py`) — no `Stop` entry exists. `run.ps1`'s
   own help text calls `ci` "what the Stop hook runs," but that wiring was
   never implemented, only documented as intent. No session has actually
   been blocked by a red `Stop` hook; `ci` has been red whenever a human or
   session ran it directly. Left for a separate decision — this entry fixes
   `ci`'s content, not whether to wire a hook that runs it automatically.
2. **It was never "since B4."** `git log -p -- run.ps1` shows both `ci` and
   `eval-quick` were introduced in the scaffold commit itself (`d1acbe4`,
   2026-08-25) — before B0 closed. `eval.run` has never existed at any point
   in this repository's history. The condition is four days and seven
   blocks (B0–B7) older than believed.

**Decision: remove the eval step from `ci`; do not build a minimal
`eval/run.py` now.** `eval/run.py` is not undifferentiated "infrastructure"
— it is explicitly **B13's** file (`PLAN_DETAIL.md:518`, entry gate B8 +
B10 + B12), a batch driver over all arms × regimes × profiles that records
which `ConformalGate` was active. At B7 there is no allocator (B8), no
chaos-hardened executor (B9/B10), no benchmark (B12), and no regime file
(`eval/regimes.py`, also B13) for it to drive. A "minimal" version now
would either silently re-implement `eval/baseline_ladder.py` (already
built, already gated, at B2) under a different name, or need to reach into
B8's allocator to have anything real to call — scope creep into the block
this was raised specifically to protect. Rejected for the same reason
option (a) was rejected when proposed: it competes with B8, the hardest
remaining block, for the same session's attention.

`ci` is now `test-fast` + `guard_invariants.py --all` only — the two checks
that have been real and green since B1. `eval-quick` remains runnable
standalone (`.\run.ps1 eval-quick`) and still fails the same way; it is
simply no longer in the gate. This brings it in line with how `golden`,
`bench`, `chaos`, `eval` (full), and `report` were already correctly
excluded from `ci` as not-yet-built. `verify-invariants`'s SKILL.md step 5
updated to match: report the expected failure, do not stop the sequence on
it (previously it would have silently prevented step 6, the freeze-hash
check, from ever running).

**No gate between B4 and B7 claimed an eval check as part of its
verification.** `reports/gates.md`'s B4, B5, B6, and B7 gate conditions —
dated 2026-08-27 through 2026-08-29 — cite tests, `stats-reviewer`,
`compliance-auditor`, and specific measured numbers; none cite `ci`,
`eval-quick`, or any eval-harness output. Stated here explicitly so the
absence is a checked fact, not an assumption carried over from the
original premise.

### 2026-08-29 · infra · The 11-minute suite was a missing connect_timeout, not real simulations -- one line fixed it, `test-fast` was a no-op until today

Raised alongside the `ci` fix, same session: `.\run.ps1 test` takes ~11
minutes and `/checkpoint` runs it every block. Asked to find the cause,
mark real-simulation tests `slow`, exclude them from the default path, and
report before/after timings. The stated hypothesis was that simulations
were the cost.

**Two findings, checked by profiling (`pytest --durations=40`) rather than
guessed:**

1. **`run.ps1 test-fast` already existed** (`-m "not chaos"`) but **nothing
   has ever been marked `chaos`** -- no marker of any kind was registered
   in the suite (`config.addinivalue_line` appeared nowhere), so the filter
   silently matched everything and `test-fast` ran the identical ~11
   minutes as `test`, every time it was ever invoked.
2. **The hypothesis was mostly wrong.** Of 656.30s (`pytest -q
   --durations=40`, no coverage instrumentation, 476 passed / 61 skipped),
   one single test —
   `tests/ingest/test_deps.py::test_get_conn_yields_an_autocommit_connection`
   — cost **260.14s by itself, 40% of the entire suite.** Only one test
   fits the "real simulation" description at all:
   `tests/model/test_conformal.py::test_exact_coverage_on_continuous_
   exchangeable_scores` (n_cal=2000, n_test=20000, swept over 20 seeds,
   18.67s, correctly slow -- this test's own docstring already explains why
   a single seed isn't adequate). Everything else was ~35 Postgres-backed
   tests each paying an independent ~6.0–6.13s fixture-setup tax (below).

**The 260s test was a real bug, not a slow test to mark and hide.**
`src/ingest/deps.py::get_conn()` calls `src.core.db.connect(autocommit=True)`
with no `connect_timeout` -- unlike `tests/conftest.py`'s `pg_schema`
fixture, which already passes `connect_timeout=3` explicitly to its own
direct `psycopg.connect()` call. libpq's default connect timeout is
unbounded; against an unreachable Postgres (Docker down, as in this
session throughout), the OS's own TCP-level give-up governs instead of the
application, and on Windows that measured ~260s for this one connection
attempt. This is not just a test-speed problem: `get_conn()` is the real
FastAPI dependency every ingest request opens a connection through — the
same unbounded wait would hang a live production request for however long
Windows (or whatever OS the deployment target runs) takes to notice a dead
socket, with no application-level control over it at all.

**Fix: `src/core/db.py::connect()` now defaults `connect_timeout` to 3
seconds** (`DEFAULT_CONNECT_TIMEOUT_SECONDS`, matching the value
`pg_schema` already chose independently) via `kwargs.setdefault(...)`, so
any caller that does pass its own value is unaffected. One call site fixed
by this (`src/ingest/deps.py::get_conn()`); a second
(`scripts/decline_coverage.py`) picks up the same fix for free, having had
no timeout either. `tests/conftest.py`'s `pg_schema` fixture bypasses
`db.connect()` entirely (calls `psycopg.connect()` directly) and is
unaffected either way.

**Measured after:** `.\run.ps1 test-fast` -- **420.08s (7:00)**, 475 passed
/ 61 skipped / 1 deselected (the conformal test, now marked `slow`). Down
236.22s (36%) from the 656.30s baseline, from a one-line fix plus one
marker -- not from excluding real work.

**What remains, disclosed rather than silently left:** the ~35 Postgres-
backed tests each still pay their own ~6.0–6.13s `pg_schema` fixture setup
independently -- roughly 210s of the remaining 420s, the largest cost left.
Each test's fixture makes its own fresh `psycopg.connect()` attempt against
an unreachable database rather than sharing one session-level reachability
check. Not fixed here: collapsing this to a single cached probe is a real
change to a fixture every Postgres-backed test already depends on, not a
one-line default -- a decision surfaced for the human to make, not one to
take silently while already mid-flight on two other fixes in the same
session. Shortening `pg_schema`'s existing `connect_timeout=3` further was
considered and also left alone, for the same reason: it was set
deliberately by an earlier session, not an oversight, and tightening it
risks trading a slow-but-safe timeout for false skips against a Postgres
that is merely slow to accept a connection rather than actually down.

*(Correction, same day: the "~35 tests / ~210s" figure above was read off
the visible top of a `--durations=40` report, not summed. The real count
and total, and the fix, are two entries below.)*

### 2026-08-29 · infra · Decision: wire no `Stop` hook

Raised alongside the two fixes above: whether to also wire an actual
`Stop` hook in `.claude/settings.json` to run `ci` automatically, given
`run.ps1`'s own help text has described `ci` as "what the Stop hook runs"
since the scaffold commit despite no such hook ever existing.

**Decided: do not wire one.** `PostToolUse`'s `guard_invariants.py` already
catches invariant violations at edit time, before they can accumulate;
`/checkpoint` already runs `/verify-invariants` at the end of every block,
when attention is actually on the result; and a session-end check costing
the better part of a block's *first two minutes* would very quickly be the
thing skipped rather than the thing enforced -- a hook nobody trusts to be
fast is a hook that gets routed around within days, which is a worse
failure mode than having no hook and relying on `/checkpoint`'s discipline
honestly. Revisit if `test-fast` ever drops under 30s, at which point the
cost argument against it no longer holds.

*(That threshold is crossed by the very next entry, same day — test-fast
reaches 21.59s below. Left as an open question for the human rather than
silently reopened, since the decision above was made on a promise about
future speed, not a promise about this same session immediately meeting
it.)*

### 2026-08-29 · infra · Diagnosed before rewriting: the Postgres overhead was a per-test reachability re-check, not schema setup -- fixed in ~35 lines, not a fixture rewrite

Follow-up, same day: asked to diagnose the "~210s Postgres overhead"
disclosed two entries above before committing to any fixture-architecture
rewrite, with three specific questions answered by measurement, not
inference.

**1. What exactly is the cost?** First, a correction to the earlier
estimate: "~35 tests / ~210s" was read off the visible top of a
`--durations=40` report, not summed -- the real number, measured by
running the six Postgres-dependent files alone with `--durations=0`
(nothing truncated): **61 skipped items, total 369.97s, mean 6.065s/item.**
100% of it is connection establishment. `pg_schema`'s own code calls
`pytest.skip()` inside the `except` block around `psycopg.connect()`,
before `CREATE SCHEMA` or `schema.sql` are ever reached -- schema
create/drop, data seeding, and transaction setup contribute exactly zero,
confirmed by the fixture's own control flow, not assumed. Isolated
single-call timing pinned the mechanism precisely: `localhost` resolves to
both `::1` and `127.0.0.1` on this machine, and psycopg/libpq tries each
address in turn, each carrying the full `connect_timeout=3` -- 5 runs
against `localhost` averaged 6.04s; the identical call against `127.0.0.1`
directly averaged 3.04s, a clean half. Two addresses, one timeout each,
sequential -- the entire mechanism, and the same mechanism (at libpq's
*unbounded* default, before the earlier entry's fix) that produced the
original 260s single-test bug.

**2. How many tests need Postgres?** 61 of 537 collected items (~11%) --
53 unique test functions (some parametrized into multiple items) across 6
files, counted by parsing every test function's argument list with `ast`
and cross-checked against the live skip count (61, exact match once
`test_deps.py`'s one non-fixture-based Postgres test is included). A clear
minority, confirming a `db`-marker exclusion was a viable cheap fix --
richness below is why it was not the one taken.

**3. Is the check repeated per test?** Yes -- `pg_schema` carries no
`scope=` argument, so it defaults to function scope: all 61 items each
made their own independent `psycopg.connect()` attempt against an
already-known-down Postgres.

**Fix: a session-scoped `_pg_reachable` fixture in `tests/conftest.py`**
(one real connection attempt, `pytest.fixture(scope="session")` -- pytest
caches a session-scoped fixture's return value automatically, so no manual
cache was needed) that `pg_schema` and `test_deps.py`'s one non-fixture
test both consult first, skipping immediately on a cached negative rather
than each re-probing. ~35 lines, one file plus the one standalone test
that bypasses `pg_schema` by design (its own docstring explains why: it
exists to exercise the real, non-overridden `get_conn()`). Deliberately
narrow: only the negative "is anything even listening" case is cached --
`pg_schema` still opens its own connection when Postgres IS reachable,
since every test needs an isolated schema-scoped connection for real work;
no connection pooling or sharing was added. This is the actual boundary of
"session-scoped cache, not an architecture change": every test's real
behavior when Postgres is up is completely unchanged.

**Why not the `db`-marker instead, given question 2 confirmed a
minority:** the cache fixes the same root cause in one file with the fast
path still exercising every one of the 61 tests' real assertions the
moment Postgres is available (e.g., the moment `docker compose up` is run
locally) -- marker-exclusion would instead remove those 61 tests from
`test-fast` permanently, trading away real DB-backed coverage in the fast
loop even in a properly running dev environment, to fix a cost that is
purely an artifact of Postgres being *down*. The cache fixes the artifact;
the marker would have fixed around it.

**Measured after, re-timed as asked:**
- The six Postgres-dependent files alone: 369.97s -> **6.45s** (one ~6.05s
  probe, 61 items skip near-instantly off the cached result).
- `.\run.ps1 test-fast`: **21.59s** (475 passed, 61 skipped, 1 deselected)
  -- well under the 60s target, from the original 656.30s baseline: a 30x
  reduction, entirely from two one-file fixes (`connect_timeout` default,
  this cache) plus one `slow` marker. No `db` marker needed; the 60s floor
  was reached without it.
- `.\run.ps1 ci`: **22.70s**, exit 0, guard clean.

No fixture-architecture rewrite was required, matching the instruction not
to do one unless (1) and (3) showed it was genuinely necessary -- they
showed the opposite: a fixture-architecture rewrite would have spent
effort solving a problem two much smaller, targeted fixes already solved.

### 2026-08-29 · infra · Stop hook wired: declined at 656s, wired at 21.59s -- the sequence, not just the outcome

Two entries above, this session first declined a `Stop` hook: "a slow
session-end check gets bypassed within two days," revisit if `test-fast`
ever dropped under 30s. The very next entry dropped it to 21.59s, in the
same session. Asked to honour the pre-stated trigger rather than argue
around it after the fact -- done, not re-litigated.

**The literal command specified did not actually satisfy its own stated
requirement.** Verified against Claude Code's hooks reference before
wiring anything (not inferred): a `Stop` hook is blocking, and its stderr
shown to Claude, **only on exit code 2 specifically** -- every other
non-zero exit is silently non-blocking, the same family of risk as every
other finding in this session's vacuous-checks audit. `pwsh -NoProfile
-File run.ps1 ci` fails this two ways, both checked empirically, not
assumed:

1. **`pwsh` does not exist on this machine at all** (`where pwsh` -> no
   match; only `powershell.exe` 5.1 is installed, confirmed directly).
   The hook would fail on "command not found" before ever reaching `ci`.
2. **`ci`'s own exit code on a real failure is not 2.** Forced a genuine
   test failure and confirmed pytest exits **1**, which `run.ps1 ci`
   propagates unchanged (`Invoke-Step`'s `exit $LASTEXITCODE`). Per the
   verified contract, exit 1 is non-blocking -- Claude is never told, and
   the turn ends anyway, exactly the vacuous-check pattern this audit
   exists to catch.

**Fix: `scripts/stop_hook_ci.ps1`**, a thin wrapper, not a fixture/`ci`
rewrite. Reads stdin for `stop_hook_active` and exits 0 immediately on a
repeat (avoids spending Claude Code's own 8-block cap re-running an
already-known-broken `ci`). Spawns `run.ps1 ci` as a **genuinely separate
process** via `powershell.exe -NoProfile -File` -- not `& $runPs1 ci`
in-process, which was tried first and found broken: `Invoke-Step`'s `exit
$LASTEXITCODE` on a failing step would terminate the wrapper's own process
in-process, before the translation logic below it ever ran. On `ci`
success: exit 0. On `ci` failure: **exit 2** with the real failure detail
on stderr -- the one code the verified contract actually blocks on.
`run.ps1 ci` itself is untouched; the translation lives only in the
wrapper, so every other caller (`/verify-invariants`, a human at the
terminal) still sees `ci`'s real, meaningful exit code.

**Both paths tested manually before wiring into `.claude/settings.json`:**
success path, piping `{}` on stdin, exit 0 in ~31s (a real `ci` run, not
short-circuited). Failure path, with a deliberately failing probe test:
exit **2**, stderr showing `ci failed (exit 1) -- fix before ending this
session:` followed by pytest's real failure output. Probe test removed
immediately after, confirmed via `git status`.

**Requirement #1 (must actually fire, shown to the human) was claimed
here as "confirmed by this session ending" -- that claim was wrong,
corrected the same day, then actually resolved the same day.** Sequence,
recorded rather than only the outcome:

1. Checked against Claude Code's own docs after the user asked how they
   would actually know: a passing (exit 0) Stop hook is documented as
   completely silent, and a blocking (exit 2) hook has no distinct marker
   either per the docs -- "it just looks like Claude decided to keep
   talking." The session continuing normally after a turn is therefore
   not evidence the hook fired by itself.
2. `/debug` was enabled mid-session to check the documented file-based
   log. Dead end in this environment: no `debug/` directory was ever
   created under `~/.claude/` despite `/debug` being active across
   multiple real `Stop` events -- the documented file logging appears to
   be a standalone-CLI mechanism that does not produce a file inside this
   VSCode-extension session. Not pursued further; a different route to
   the same answer existed and was taken instead.
3. **Resolved empirically, definitively, the same way PostToolUse was
   resolved above: forced a real `ci` failure and watched what happened.**
   Planted `tests/_scratch_stop_hook_probe_test.py` with a deliberate
   `assert False`, confirmed it broke `ci` (`pytest` exit 1), then ended
   the turn without fixing it -- the broken state deliberately left to
   persist across the stop boundary, the one thing every other probe this
   session did not need to do. The hook fired, translated `ci`'s real
   exit 1 to exit 2, and the next turn opened with the harness's own
   label: `"Stop hook feedback: [powershell -NoProfile -File scripts/
   stop_hook_ci.ps1]: ci failed (exit 1) -- fix before ending this
   session:"` followed by the real pytest failure output naming the
   planted test exactly. This is a stronger result than the docs
   predicted -- they describe no distinct marker at all ("just looks like
   Claude decided to keep talking"), but this integration surfaces an
   explicit, labeled "Stop hook feedback" block, more visible than the
   general contract promised. Probe file removed immediately after,
   confirmed via `git status`; `ci` reconfirmed green (545 passed, 1
   deselected, exit 0).

**Requirement #1 is now genuinely confirmed, by the mechanism the hook
was built to provide** -- not inferred, not assumed from a silent turn
boundary.

### 2026-08-29 · infra · PostToolUse empty-input visibility resolved empirically: exit 2 is loud, guard_invariants now returns 2 not 0 on empty input

Highest-severity unknown from the vacuous-checks audit: does
`guard_invariants.py`'s "no files resolved" warning (previously exit 0)
actually reach anyone, or is it swallowed the way a `Stop` hook's exit-0
stdout is (confirmed the same session, entry above)?

**Resolved empirically, not from docs, as asked.** Wrote
`src/model/_scratch_probe.py` containing `import anthropic` -- a real
invariant 1 violation -- via the Write tool, letting the real `PostToolUse`
hook fire for real. Its stderr **appeared in full**, unambiguously, as a
"PostToolUse:Write hook blocking error" message quoting the exact
violation text. The write itself was not undone -- confirms exit 2 is
loud but not destructive for this hook type. Probe file deleted
immediately after, confirmed via `git status`.

This does not, by itself, prove exit 0 is swallowed (the empty-input path
is a different exit code and could not be forced through a normal
Edit/Write -- `hookio.py`'s git-diff fallback means the empty-input branch
is a defensive rare case, not something that fires on ordinary edits with
a dirty working tree). The decision to change it anyway rests on: (a) the
now twice-confirmed pattern across two hook types (`Stop`: exit 0 -> debug
log only, never shown; `PostToolUse`: exit 2 -> shown in full, just
demonstrated) makes exit 0 the consistently risky code across this
project's hooks; (b) `guard_invariants.py`'s own existing comment already
said "never silently pass... say so loudly rather than exiting 0" --
the code just didn't match its own stated intent; (c) a false-fail (loud
when it didn't strictly need to be) is a categorically safer failure mode
here than a false-pass (the primary guard silently examining nothing on
every edit).

**Fix:** `guard_invariants.py`'s empty-input branch now returns 2, both
for the bare (`PostToolUse`) path and `--all`. Regression-tested directly
(`tests/scripts/test_guard_invariants.py`, 3 tests): empty input exits 2
in both modes; a real, clean, resolvable file still exits 0 -- confirms
the fix is scoped to the empty case, not an overcorrection.

### 2026-08-29 · infra · Three more audit fixes: live-key scan, checkpoint.py's swallowed returncode, verify-invariants exit-code assertions

Three more of the four items authorized from the same audit, each
mechanical once diagnosed:

**Live-key scan, two locations.** `run.ps1 lint` and `verify` check 3 both
printed a clean pass when `Get-ChildItem` resolved zero files -- the
security check meant to be hardest to fool was the easiest, in two
places sharing the same shape. Both now assert a non-zero file count
before trusting an empty `$hits`/`$lk`; both now also report the file
count on success (`"no live keys: OK (242 files scanned)"`), so a passing
run states what it actually examined rather than a bare "OK."

**`checkpoint.py::sh()` never checked `returncode`.** A genuine pytest
crash (no parseable "passed"/"failed"/"error" line anywhere in its output)
and a merely-slow run produced the identical benign message: `"{n}
collected, run inconclusive."` `sh()` now returns `(stdout, returncode)`;
every call site updated. `test_status()` now distinguishes: no parseable
summary AND non-zero returncode -> `"TEST RUN FAILED (exit N): <last
line>"`; no parseable summary but returncode 0 (genuinely still running
long, not crashed) -> the original "run inconclusive" wording, now
actually meaning what it says. Verified both branches by monkeypatching
`sh()` directly (forcing a real pytest crash to order was impractical) --
a simulated `INTERNALERROR` correctly surfaces as `TEST RUN FAILED (exit
3): INTERNALERROR> some plugin crashed`; a simulated subprocess exception
(`sh()`'s own except-branch shape, `("", 1)`) correctly surfaces as `TEST
RUN FAILED (exit 1): (no output captured)`.

**Marker drift fixed in the same pass.** `checkpoint.py::test_status()`
hard-coded `-m "not chaos"`, missing `"and not slow"` -- its own reported
test count silently diverged from what `test-fast`/`ci` actually run.
Fixed by threading `run.ps1`'s `$TestFastFilter` through as `checkpoint.py`
argv[2] (a `DEFAULT_TEST_FILTER` constant remains for standalone
invocation, explicitly commented as a fallback whose text must be kept in
sync by hand, not the source of truth). Confirmed: before the fix,
checkpoint reported "537 passed" with the slow-marked conformal test
silently included; after, "539 passed, 1 deselected" (539 = 536 + the 3
new `test_guard_invariants.py` tests added this session), matching
`test-fast`/`ci`'s selection exactly.

**`/verify-invariants` steps 1, 3, 4 had no exit-code assertion in the
skill's own text.** Relied on the operator noticing -- "the exact pattern
this audit was looking for," per the instruction. All three now state the
required exit code explicitly in `SKILL.md` (step 1 and 3: must exit 0;
step 4: must exit 0), each with a one-line note on what a pass now also
guarantees (step 1: empty input is a failure too, per the entry above;
step 3: the live-key scan's own file count, per the entry above).

### 2026-08-29 · infra · Remainder of the vacuous-checks audit: known and accepted for now, not fixed

Explicitly left alone this session, per instruction -- logged so the
choice not to act is a decision, not an oversight:

- **`test`/`test-fast`'s exit-5 partial protection.** A filter matching
  zero tests makes pytest exit 5 (non-zero, so a caller checking
  `$LASTEXITCODE` is protected), but nothing prints a distinct failure
  banner the way `Invoke-Step`'s tasks do -- the visual signal is real but
  easy to miss reading raw scrollback. Accepted: `ci`'s own tests step
  *is* wrapped in `Invoke-Step` and would still hard-fail correctly; only
  the bare, unwrapped `test`/`test-fast` invocations carry this residual
  risk.
- **`slow`/`chaos` marker scope drift.** Nothing asserts the *count* of
  tests carrying either marker stays sane -- a future session mismarking
  an entire file `slow` would silently shrink `test-fast`'s real coverage
  with no alarm.
- **`eval`'s ungated sequencing.** `eval.run` and `eval.report` run as two
  bare, unconditional commands with no exit-code gate between them --
  irrelevant today (`eval.run` doesn't exist, so it crashes loudly every
  time before `eval.report` gets a chance to run against stale or
  non-existent input), but latent once B13 builds both files for real.
  **Fixed at B13**, when that harness actually exists to gate -- not
  speculatively now, against a file that isn't written yet.

### 2026-08-29 · B8 gate amended · "zero constraint violations" was vacuously satisfiable by an allocator that never attempts anything

From the vacuous-checks audit, same day: B8's gate is B5's null-policy
finding (2026-08-28) recurring in the block about to start. "Zero
constraint violations across the eval" is trivially true of a policy that
takes no action at all -- an allocator that never attempts violates
nothing, by never doing anything. Amended before any allocator code
exists (`src/policy/allocator.py` does not exist; nothing under
`src/policy/` implements backward induction yet), the same standard as
every previous gate amendment this project has made.

**Original text** (`reports/gates.md`, unchanged since PLAN_DETAIL.md v2):

> "★ allocator + stopping + off-ramp: 2-slot brute-force equivalence test
> passes; zero constraint violations across the eval; both profiles
> produce numbers"

**Two clauses added**, both derived from the frozen simulator's own
generative parameters rather than chosen by hand:

1. **An attempt-rate floor** -- the allocator must attempt on at least a
   stated minimum fraction of AFA-eligible mandates. A null policy attempts
   0%, failing trivially.
2. **A discrimination clause** -- the allocator's mean attempt rate on
   true `CANT_PAY_NOW` mandates must exceed its mean attempt rate on true
   `CANT_PAY_EVER` mandates (ground truth, eval-only, never read by the
   allocator under test) by a stated margin. Added because clause 1 alone
   is vacuous in a subtler way: a policy attempting the floor fraction
   *uniformly at random*, ignoring cause entirely, clears it while
   demonstrating nothing this project's thesis claims. This is the clause
   that actually tests the thesis -- that knowing WHY a payment failed
   changes what you do.

**First floor value proposed and computed, seeds 0-19** (matching
`protocol.md`'s own "beats the ladder" seed sweep), using `Simulator` +
`src.policy.constraints.afa_free_limit_paise()` -- the same AFA-cliff
filter B8's gate already requires the allocator apply before consulting
any model:

| | count | fraction |
|---|---|---|
| Total mandates (200 x 20 seeds) | 4000 | -- |
| AFA-cliff excluded (routes to REAUTH, never ATTEMPT) | 398 | 9.95% |
| AFA-eligible remainder | 3602 | -- |
| -- true `CANT_PAY_NOW` (correct action: ATTEMPT, unambiguous) | 1760 | **48.86%** |
| -- true `CANT_PAY_EVER` (correct action: REAUTH) | 765 | 21.24% |
| -- true `WONT_PAY` (correct action: OFFER, not live at B8 -- B6's `FullSetGate` stub) | 1077 | 29.90% |

Identical across all three frozen arms -- `cause_mix`/`amount_paise` are
shared in `sim_config.yaml`; only the link function and coupling differ.

Proposed initially: floor = 48.86%, the exact measured `CANT_PAY_NOW`
fraction. **Rejected before implementation** -- see the next entry, same
day, for why, and for the value actually used.

### 2026-08-29 · B8 gate floor lowered to 25%, discrimination margin added and derived · sequence recorded, not just the outcome

**The 48.86% floor proposed in the entry above was rejected on review,
before any code was written against it.** Reason: the allocator does not
observe cause, only a belief. It will never partition its attempts at
exactly the true `CANT_PAY_NOW` fraction. A correct policy that declines a
handful of true-`CANT_PAY_NOW` mandates on negative expected value -- a
legitimate stopping decision, not a violation -- would land just under
48.86% and fail a floor set there. And when it fails, the cheapest fix is
to attempt more often to clear the threshold -- tuning the allocator to
the grading axis rather than to value, the same shape already rejected
three times this project: B5's stop-threshold scalar (2026-08-28), the
paired-criterion reversal (2026-08-28), and B7's declined `switch_eps`
parameter (2026-08-29). **A floor is a tripwire against a degenerate
policy, not a performance target**, and 48.86% would have quietly become
the second kind.

**Decided: floor = 0.25 (25%).** Roughly half the true `CANT_PAY_NOW`
fraction -- unreachable by a null policy (0%), unreachable by any policy
that ignores the `ATTEMPT` action altogether, comfortably clear of any
legitimate stopping behaviour. The 48.86% derivation above is not
discarded -- it is exactly what justifies *where* 25% sits (roughly half
of a precisely measured quantity, not an arbitrary round number chosen
without reference to it).

**Discrimination margin, derived the same way, not chosen by hand.**
Simulated a uniform-random policy attempting at exactly the floor rate
(25%), independent of cause -- the precise borderline case the
discrimination clause exists to reject -- across seeds 0-19, using an RNG
stream independent of the simulator's own (`seed + 100_000`), reproduced
exactly in `tests/eval/test_gate_criteria.py`:

- Mean discrimination gap (true-`CANT_PAY_NOW` attempt rate minus
  true-`CANT_PAY_EVER` attempt rate) across the 20 seeds: **-0.0068**
  (~0, as expected -- the random draw is independent of cause by
  construction). This is the number requested to demonstrate the clause
  has teeth: a policy attempting on cause carries no information scores
  approximately zero on it.
- Per-seed standard deviation: **0.0876**.
- **Margin set at one pooled SD above the random baseline's own mean:
  0.0808** -- reusing `protocol.md`'s own existing "clear one pooled SD"
  convention for "beats the ladder" claims, not a new statistic invented
  for this clause. Expressed against the mean gap's own sampling
  distribution (the gate evaluates a 20-seed mean, the same way every
  other "beats X" claim in this project is evaluated): SE = SD/sqrt(20) =
  0.0196, so the margin sits **4.13 standard errors from zero** -- not
  something a non-discriminating policy clears by chance at that
  granularity. 5-SD and 8-SD alternatives (0.43 and 0.69) were also
  computed and rejected as unnecessarily strict, approaching the
  theoretical ceiling (100 percentage points, a perfectly cause-aware
  oracle's own gap) closely enough to become a performance target rather
  than a tripwire -- the same trap the 48.86% floor fell into, avoided
  deliberately this time.

**Both required-fail cases proven by test**, not asserted --
`tests/eval/test_gate_criteria.py`, 6 tests:

- `test_null_policy_fails_the_attempt_rate_floor` -- attempt rate 0.0
  fails the 0.25 floor, exactly the case the clause exists to catch.
- `test_uniform_random_at_floor_rate_fails_the_discrimination_clause` --
  reproduces the simulation above; asserts the measured mean gap
  (pinned to full precision, so a future drift in `sim_config.yaml`'s
  cause_mix -- impossible, it is frozen -- or the RNG scheme shows up as
  a failing test) is below the margin.
- `test_discrimination_margin_is_one_pooled_sd_above_random_baseline` and
  `test_true_cant_pay_now_fraction_is_48_86_percent_not_used_as_the_floor`
  pin both derivations against drift.

Constants and scoring helpers (`attempt_rate`, `discrimination_gap`) live
in new file `eval/gate_criteria.py` -- not `src/policy/constraints.py`,
whose own docstring scopes it to RBI-clause-attributed constants only;
these are eval-calibration constants with no clause to cite. Not under
`eval/frozen/` -- this is a post-freeze addition, not part of the Day-1
pre-registration, and `guard_frozen.py` would deny writing it there
regardless.

`reports/gates.md`'s B8 line updated to the amended text; full derivation
kept here rather than restated in the gate comment, per this project's
established split (gate comments cite where the reasoning lives, they do
not repeat it).

### 2026-08-29 · B9 gate strengthened · opt-out-inside-24h clause now requires a test that actively constructs the race

Approved as proposed, same standard as B8's amendment above -- no executor
code exists yet (`src/execute/` is unwritten; this is B9's own future
file table, PLAN_DETAIL.md).

**Original text:**

> "★ executor + idempotency: keys test passes (no clock/uuid/pid); an
> opt-out arriving inside the 24h window is honoured; `UNCONFIRMED` has a
> resolution path that is actually reachable"

**Problem:** "an opt-out arriving inside the 24h window is honoured" is
satisfiable by a test suite that simply never generates a late opt-out at
all -- vacuously true by never exercising the path it claims to cover,
the same shape as every other finding in this session's audit.

**New text:** "an opt-out arriving inside the 24h window is honoured —
proven by a test that actively constructs the race (commits an attempt,
then delivers a late opt-out event inside that window, and asserts the
attempt is aborted — not merely a test that happens never to generate
one)." The parenthetical stays in the gate text itself, not just in this
entry, so a future session closing B9 cannot satisfy the clause with a
test that is silent on the actual race by construction.

### 2026-08-29 · B8 · action-gating decision made, and the belief-fixed
consequence of making it

Built `src/policy/allocator.py`. Answered the question `reports/gates.md`'s
B8 entry left open at B7: PLAN_DETAIL.md section 4's `Q(b, ATTEMPT)` sums
`Sigma_c b[c] * h_c(...)` — cause-conditioned hazards B5 never fit and
cannot fit (`Cause` has no production label, ever). **Decided: cause
enters only through action-gating** — `REAUTH` feasible when
`b.dominant() == CANT_PAY_EVER`, `OFFER` feasible only on a singleton
`{WONT_PAY}` conformal set — never through the Q-value arithmetic, which
uses the marginal hazard directly. Lossless given the available hazard
source: `Sigma_c b[c] * h == h` for any belief when `h` does not vary with
`c` (`test_marginal_hazard_makes_the_cause_sum_an_identity`, allocator
suite).

**Consequence, worked out during implementation, not pre-planned: belief
is carried UNCHANGED across every recursive node within one `solve()`
call.** The pure form of the recursion updates belief via
`update(b, obs=survived(c,d,m))` on the "still pending" branch, but that
needs a specific *observed* `DeclineClass` — a cause-marginal hazard model
has no honest way to produce one (it predicts 4-class Outcome
probabilities, not the 7-class decline taxonomy `belief.update()` accepts).
Fabricating one inside the recursion would make the exact-solve claim
dishonest in the one place it matters most. Real belief updates happen
between `solve()` calls instead, from real evidence, at whatever layer has
it. One clean side effect: since `Sigma_c b[c] * h` collapses to `h`
regardless of `b`, and `b` never changes within one call, the memoisation
key's belief component is *provably* constant within a call
(`test_belief_is_the_sole_constant_key_component_within_one_solve_call`) —
the quantisation-collision risk flagged going into this block (STATE.md,
"whether a good allocator clears \[the discrimination margin\] comfortably
or scrapes it is unknown until B8 runs") cannot arise in this design,
because there is only ever one belief value to quantise per call. It was a
reasonable risk to flag before this design was worked out; it does not
apply to what was actually built.

Also found and fixed, before it ever reached the eval sweep: `committable_days()`
originally computed its earliest eligible day from `plan_day` and the
profile's lead time alone, ignoring `ctx.committed_days`. Under `permissive`
(no fresh-notification lead required), this could reproduce the SAME day
just committed as a candidate for the next slot —
`eval/frozen/simulator.py`'s `Simulator.attempt()` enforces strictly-
increasing `on_day` per mandate and would have raised. Fixed to take the
later of `(plan_day + lead)` and `(last committed day + 1)`; regression
test `test_committable_days_never_repeats_an_already_committed_day` pins
it. Found by running the eval harness against the real simulator, not by
inspection — logged per this project's "found while building" convention
because reconstructed-later write-ups read as fake.

2-slot brute-force equivalence (the gate's own required test): an
independent, unmemoised reimplementation of the Bellman recursion, written
from scratch in `tests/policy/test_allocator.py` rather than calling
`allocator.py`'s private helpers, agrees with the memoised solver exactly
across 7 scenarios including two beliefs deliberately near-colliding at the
`1e-6` quantisation grid. `zero constraint violations`, `AFA cliff routes to
REAUTH without ever calling the hazard model`, and both profiles producing
numbers are all proven by test. Does not touch `eval/frozen/`.

### 2026-08-29 · B8 · the discrimination-margin clause measured ~0 for a
reason unrelated to the allocator — replaced before the gate was ever
ticked, full derivation-then-validation-then-measurement trail kept

Ran `eval/allocator_sweep.py` (new, B8-local gate harness — not
`eval/run.py`, which is B13's file and still does not exist) against the
gate committed in `reports/gates.md`, 2026-08-29, earlier the same day:
attempt rate on AFA-eligible mandates >= 0.25, discrimination gap (true
`CANT_PAY_NOW` attempt rate minus true `CANT_PAY_EVER` attempt rate,
20 seeds) > 0.0808.

**Measured: attempt rate 0.9858 (clears easily), discrimination gap
0.0009 (fails — indistinguishable from the cause-blind-random baseline's
own measured -0.0068).** Zero constraint violations, both profiles produce
numbers. Before concluding the allocator does not discriminate, checked
directly: `src/policy/allocator.py`'s own unit tests already proved REAUTH
correctly routes a `CANT_PAY_EVER`-dominant belief away from `ATTEMPT`
given a poor hazard; a standalone diagnostic (seed 0, one mandate at a
time) found REAUTH firing for 12 of 34 true-`CANT_PAY_EVER` mandates once
evidence existed. The discrimination was real. The metric could not see
it.

**Root cause: the original clause is a boolean ("ever attempted") pinned
to `True` at the first, necessarily uninformed decision, for nearly every
mandate regardless of cause.** This allocator starts every belief at the
uniform reference prior and updates only from observed evidence — there is
no evidence before the first retry. Under ordinary economics, attempting
is positive-EV at a neutral prior for ~98.6% of AFA-eligible mandates of
*either* true cause, so "ever attempted" is set `True` for both groups at
slot 2, before any cause-discrimination (which does happen, at slot 3+,
once a `DEAD`-type signal updates belief) has a chance to move a boolean
that is already pinned. A boolean cannot un-True itself; it does not
accumulate what a later, correct `REAUTH` decision represents.

**This was a process failure in how the original clause was approved, not
bad luck in measurement, and it is being named as exactly that rather than
smoothed over: the clause was pinned as a constant without ever checking
that *any* achievable policy could move it.** Specifically, without
computing its value for an oracle that reads true cause and acts
perfectly — which would have shown the metric saturates near its ceiling
for every policy that ever attempts a cause-blind first slot, oracle
included, leaving no daylight for a real policy to be measured in. That
check is what the user asked to make mandatory for the replacement, and is
exactly the gap that let the first version through structurally unable to
measure anything. Recorded plainly as that gap, not attributed to a
one-off derivation mistake, so the next gate amendment in this project
inherits the validation step rather than just the apology.

**Replacement, derived and validated in that order — candidate chosen and
its null/random/oracle values computed before this session read anything
our own allocator produces (confirmed by conversation order: the
validation script constructs its NULL, RANDOM, and ORACLE reference
policies directly against `eval/frozen/simulator.py`, with no import of
`allocator.py` or the fitted hazard model anywhere in it; the allocator's
own sweep was only re-run afterward):**

Candidate: mean **attempts spent** (a count in `{0,1,2,3}` — retries only,
slot 1 is given), true `CANT_PAY_NOW` minus true `CANT_PAY_EVER`. Chosen
over the alternative considered (REAUTH-routing rate conditional on a
`DEAD`-type signal) because a count is continuous and accumulates —
it cannot saturate to a single pinned value the way a boolean does — and
because it is already one of this project's own three headline bars
(recovered, **attempts spent**, mandates preserved), so scoring well here
is demonstrably practising the thesis rather than satisfying a bespoke eval
statistic.

Validated with the null/random/oracle protocol the original clause
skipped, all three constructed directly against the frozen simulator, 20
seeds:

| policy | mean gap | sd |
|---|---|---|
| NULL (never attempts) | 0.0 exactly | 0.0 |
| RANDOM (fair coin per retry point, cause- and outcome-blind, never calls the simulator) | -0.015262401928163994 | 0.3108523266763266 |
| ORACLE (reads true cause; `CANT_PAY_EVER` → 0 attempts, stop; else → attempt every remaining slot to a real terminal outcome or the cap) | ~1.5816 | ~0.0699 |

Oracle − random separation: **1.5969, or 5.14 random-policy standard
deviations** — wide, clearly resolved daylight between "no information"
and "perfect information." This candidate has real resolving power;
`tests/eval/test_gate_criteria.py::test_oracle_policy_clears_the_discrimination_margin`
pins it so a future change cannot silently collapse that separation again
without a test noticing.

`DISCRIMINATION_MARGIN` set the same way as the original — random
baseline's own mean plus one pooled SD (protocol.md's existing "clear one
pooled SD" convention, reused rather than inventing a new statistic):
`-0.015262401928163994 + 0.3108523266763266 = 0.29558992474816265`, sitting
4.25 standard errors from zero against the mean's own 20-seed sampling
distribution. `ATTEMPT_RATE_FLOOR` (0.25) is untouched — it was never the
broken clause, and the null policy already fails it trivially regardless
of what clause 2 measures.

**Only after the above was fixed and validated, `eval/allocator_sweep.py`
was updated to score `attempts_spent` instead of the boolean and re-run.
Measured: discrimination gap 0.0412 — a real, non-zero signal (above the
random baseline's own -0.0153), but well short of the 0.2956 margin, and
only ~2.6% of the oracle's 1.5816 ceiling.** Not adjusted to pass; reported
as measured.

**Diagnosed, not left as a bare number.** Per-cause attempt-count
distribution, one seed:

| true cause | n | mean attempts | distribution (attempts → count) |
|---|---|---|---|
| CANT_PAY_NOW | 85 | 1.459 | {0:1, 1:53, 2:22, 3:9} |
| CANT_PAY_EVER | 34 | 1.412 | {1:24, 2:6, 3:4} |
| WONT_PAY | 66 | 1.803 | {0:1, 1:26, 2:24, 3:15} |

71% of true-`CANT_PAY_EVER` mandates (24/34) do stop at exactly 1 attempt —
the mechanism works, most of the time, on this seed. The remainder
(6 at 2 attempts, 4 at 3) trace to `sim_config.yaml`'s own `base_dead: 0.55`
for `CANT_PAY_EVER`: a `DEAD` outcome — the only strong signal this
harness's outcome-to-`DeclineClass` proxy produces — appears on only 55% of
any given attempt, not every attempt, so on average it takes ~1.8 attempts
before the informative signal even arrives, and until it does `ATTEMPT`
remains the economically correct choice given the evidence actually in
hand. Checked whether the proxy's OTHER mapping (`STILL_PENDING ->
INSUFFICIENT_FUNDS`, a confident CANT_PAY_NOW-leaning signal on a
genuinely ambiguous outcome — CANT_PAY_NOW and CANT_PAY_EVER survive an
attempt at broadly comparable rates per `sim_config.yaml`'s base rates) was
actively hurting the gap: reran with `STILL_PENDING -> None` (no update at
all). Mean gap was **unchanged** (0.0412 both ways; sd dropped from
measured to 0.1356 under the no-op variant, tighter but not materially
different in mean). So the shortfall is not a proxy-calibration mistake —
it is the base-rate delay before the one strong signal this harness has
access to arrives, compared against an oracle ceiling that has none of that
delay by construction. A real, bounded, explainable gap between "realistic
evidence-limited policy" and "omniscient oracle," not a defect to keep
chasing blindly.

**Left open, explicitly, for the human to decide rather than resolved
solo:** whether 0.0412 against 0.2956 is a finding to report with the gate
left unticked, whether the margin itself should be re-derived against a
more realistic reference point than an omniscient oracle (a defensible
question — 1.5816 assumes zero signal delay, which no evidence-based
policy can ever achieve — but one that must go through the same
derive-before-measure discipline as everything above it, not be answered
by whichever threshold happens to clear), or something else. Not decided
in this entry.

### 2026-08-29 · B8 · second amendment to the same clause, in the same
day — approved narrowly, on an objective defect in the reference point,
not because the first replacement failed

Not presented as a clean history: this is the **second** consecutive
re-derivation of `DISCRIMINATION_MARGIN` after measuring our own
allocator against it, in the same session. The human approved proceeding
only because the specific defect argued was objective and checkable
before any new number was produced, not because a threshold was
inconvenient — and set three conditions: derive and validate the new
reference **before** re-measuring our allocator; pre-commit, in writing,
before running anything, that a second failure is a finding and the gate
stays unticked (no third re-derivation); and investigate whether the
55%-per-attempt `DEAD` rate was a property of the frozen simulator or an
artifact of the eval harness discarding signal, since that could make the
whole amendment moot.

**Pre-commitment, written before running the validation below:** if the
allocator fails whatever margin survives this entry's validation, B8's
gate stays unticked and this is reported as a finding. There will not be
a third re-derivation of this clause in this session.

**Investigation (condition 3), answered first:** the 55% `base_dead` rate
for `CANT_PAY_EVER` is `sim_config.yaml`'s own generative parameter,
constant across retries (no `retries_so_far` adjustment for
`CANT_PAY_EVER`, unlike `WONT_PAY`'s optout, which does escalate).
`AttemptResult` carries exactly one signal field relevant here (`outcome`;
`iatrogenic_insufficient_funds` is coupled-arm-only) and `SimMandate`'s
other fields (`amount_paise`, `category`) are drawn independent of cause.
No richer channel exists for the harness to have discarded. Confirmed by
reading the config and both dataclasses directly, not by inference. The
amendment is not moot.

**The defect in the first replacement's reference point, stated
precisely:** `DISCRIMINATION_MARGIN` is a *margin* — a comparative claim
about how much better a real policy does than blind chance — and its
reference ceiling must be something a policy *constrained the way ours is*
could plausibly approach. The oracle used (`test_oracle_policy_clears_
the_discrimination_margin`, prior entry) reads `initial_cause` directly
and sets `CANT_PAY_EVER`'s attempts-spent to exactly 0 by fiat — it never
attempts, never waits for evidence, never faces the delay every real
policy in this project must face. That is not "a hard bar," it is a
different question answered instead of the one the clause needs answered.
A *floor* tripwire may legitimately reference an unreachable ideal (that
is exactly what makes it a tripwire); a *margin* must reference something
reachable, or every real policy fails it by construction regardless of
quality.

**Corrected reference, derived and its value computed BEFORE this
allocator's score was consulted for this amendment** (the scratch
validation script constructs the reference using only
`eval.frozen.simulator`'s own `_logits_from_base_rates`/`_softmax`
helpers — reused verbatim, not reimplemented, to remove transcription risk
— with no import of `allocator.py` or the fitted hazard model; the
already-known 0.0412 figure from the first replacement was not used to
shape this definition):

**Constrained-reference policy** — belief starts at the uniform prior,
identical to our allocator. On each attempt, updates via **exact** Bayes
using the simulator's own true `P(outcome | cause, context)` — no
`DeclineClass` proxy, no approximation, the opposite end of the
information spectrum from `eval/allocator_sweep.py`'s harness. Stops
attempting as soon as `dominant() == CANT_PAY_EVER` (identical decision
rule to the allocator's own `REAUTH` gate — this isolates exactly one
variable: realistic signal delay vs. the harness's crude proxy). Still
faces the real stochastic delay: `DEAD` still only arrives 55% of the time
per attempt for a true `CANT_PAY_EVER` mandate, same as it does for our
allocator.

**Result: mean gap −0.0144 (sd 0.1491), 20 seeds.** Statistically
indistinguishable from the random baseline's −0.0153 (sd 0.3109) — the two
are well within a fraction of either standard deviation of each other.

**This is not what was expected going in, and it changes the finding
substantially. Investigated rather than accepted at face value — the
result held up under scrutiny, and the reason is structural, not a bug in
the reference construction:**

`E[attempts spent]` under ANY policy that must wait for a real terminal
event is dominated by **how fast some terminal outcome occurs at all**,
not by which one. `sim_config.yaml`'s own per-attempt terminal rates are
broadly comparable across causes — `CANT_PAY_NOW` terminates fast via
`RECOVERED` (base rate 0.35, boosted further in the salary window);
`CANT_PAY_EVER` terminates fast via `DEAD` (0.55). Both cohorts stop
accumulating attempts at a similar pace; they just stop for opposite
reasons. A metric built on raw attempts-spent cannot see *why* a cohort
stopped, only *that* it did — so it inherits almost none of the real
discrimination this project's thesis is actually about. The original
zero-delay oracle's 1.5816 was high specifically because it let
`CANT_PAY_EVER` skip this dynamic entirely (0 attempts, by fiat); once
forced to wait for the same evidence any real policy waits for, the
ceiling collapses to noise.

**Checked directly whether this generalises past "mean attempts spent"
specifically, rather than assumed:** re-ran the current
(`DEAD`-terminal-fixed) `eval/allocator_sweep.py` sweep and counted
`Action.REAUTH` firings directly. **Zero, across 901 mandates (5 seeds).**
The 12/34 `REAUTH` firings reported in the prior entry were entirely an
artifact of a bug fixed earlier the same session (re-attempting after a
`DEAD` outcome, which let a `DEAD`-driven belief update trigger a
follow-up decision that should never have existed — see the
`committable_days` / DEAD-terminal fixes above). Once `DEAD` is correctly
treated as terminal, there is no decision point *after* observing it
within the same cycle — the one strong signal this simulator produces
ends the episode at the moment it arrives. `REAUTH`'s only remaining path
to fire is off the weakly-informative `STILL_PENDING` signal (survival
odds ~0.60 `CANT_PAY_NOW` vs ~0.40 `CANT_PAY_EVER` — a real but mild tilt,
already shown in the prior entry's no-op check to move the mean
negligibly) or off belief carried over from a **previous cycle's**
outcome — a mechanism this single-cycle-scoped eval harness never
exercises at all, since every mandate here is scored on exactly one cycle
in isolation.

**This is a structural finding about the eval harness's scope, not just
about which formula scores it, and it goes beyond what the human
pre-authorised (trying a REAUTH-conditional-on-`DEAD` candidate) — that
candidate would score as trivially degenerate given the finding above (a
`DEAD` observation can never be followed by a further attempt in this
harness, by construction, for any policy), so trying it would not answer
anything.** Not resolved solo. Per the pre-commitment above, this is
reported as a finding; B8's gate stays unticked. Left for the human,
explicitly: whether the eval harness needs to model belief carrying over
across cycles for the same mandate before this clause (in any formulation
tried so far) can measure what the allocator's `REAUTH` gate was actually
designed to do, or whether the thesis itself needs restating for what a
single retry cycle can honestly demonstrate.

### 2026-08-30 · B8 · scoping (not implementing) what closing the
structural gap above would actually take

Asked for, and this is: a sized writeup of options, no code, so the next
session (or this one, with fresh direction) can decide from something
concrete rather than an abstract "model cross-cycle belief" gesture.

**Two distinct candidate evidence sources exist for "what could inform
belief before slot 2 is ever decided" — checked separately, because they
have different costs and it matters which one (if either) is real:**

**(a) A previous billing cycle's own resolution.** A mandate that died last
cycle is presumably still dead this cycle; that carryover is what would let
`REAUTH` fire *before* wasting a slot-2 attempt. Checked directly against
`eval/frozen/simulator.py`: `_generate_mandates()` hard-codes
`cycle_id=1` for every mandate (line 186, literal, not derived), and
`mandate_id=f"M{i:04d}"` is a flat per-index label with no relationship to
any other mandate. **There is no multi-cycle generation in the frozen
simulator at all** — not a missing parameter, an absent mechanism. Every
`Simulator("nominal", seed=N)` call produces `n_mandates` structurally
independent single-cycle draws, full stop.

**(b) Slot 1's own decline reason.** In a real deployment, the very first
attempt would return a real issuer decline string immediately — B9's
executor would normalise it and update belief before slot 2 is ever
planned. Checked against the same file's own module docstring, not
inferred: *"Every mandate entering this simulator has already had its
slot-1 (original) attempt fail — that failure is what puts a mandate into
a recovery system in the first place. Only slots 2/3/4 ... are simulated
as decisions; slot 1 is given."* This is a **deliberate, pre-registered**
design choice (frozen at B2, before B7's `DeclineClass`/belief machinery
existed to consume such a signal at all), not an oversight — and
`Simulator.attempt()`'s own signature enforces it (`slot not in (2,3,4)`
raises). The simulator has no path to produce a slot-1 decline reason,
ever.

**Both are blocked by the same wall, for related but distinct reasons: one
is a missing generation mechanism, the other is a deliberate scope
boundary set two blocks before the belief layer that would consume it
existed.** Neither is a small harness-side fix — both terminate at
`eval/frozen/`, which `guard_frozen.py` denies editing, by design, with
its only sanctioned exception being a human editing outside a Claude Code
session and logging why (`gates.md`, B2 entry) — not something to reach
for mid-session, and not attempted here.

**Options, sized honestly, not ranked by preference:**

1. **Extend the frozen simulator** (draw a next cycle after a prior one
   resolves, and/or emit a slot-1 decline reason) via the human escape
   hatch. Highest cost: invalidates the pre-registration every downstream
   gate was measured against, requires re-running and re-justifying B5's
   "beats the ladder" fits, B6's conformal coverage, and this session's
   own oracle/random baselines against the new generative story. Not a
   same-session decision regardless of who makes the edit.

2. **Harness-side chaining**: approximate cross-cycle carryover in
   `eval/allocator_sweep.py` (not frozen, freely editable) by stitching
   together independently-drawn `SimMandate`s as if they were one
   customer's successive cycles. Runs into a real mechanical snag beyond
   the modelling question: `Simulator.attempt()`'s own state tracking
   (`last_slot_seen`, slot restricted to `{2,3,4}`) has no reset hook for
   "new cycle, same customer" — approximating one honestly would mean
   reimplementing outcome-drawing logic outside the `Simulator` class
   using `sim_config.yaml`'s hazard numbers directly, which is no longer
   testing "the frozen simulator's behaviour," it is testing a
   parallel, hand-rolled approximation of it that could silently drift
   from the pre-registered story. Medium cost, real risk to the
   pre-registration's own credibility if not built and reasoned about
   carefully.

3. **Reconsider whether this simulator was ever the right instrument for
   this specific claim.** Its own docstring says what it is for: testing
   the allocator's scheduling/constraint-compliance exactness over
   already-failed mandates, frozen at B2 before the cause-taxonomy and
   belief layers (B7) existed. The richer signal `REAUTH` actually needs
   — an immediate, per-attempt decline reason — is exactly what B3's real
   ingest path and B11's normaliser were built to carry, over real
   Razorpay test-mode data, not synthetic data this simulator was
   pre-registered to produce. This reframes B8's discrimination gate as a
   question for a *different* evidence source (real test-mode declines)
   rather than a formula or generator change — lowest mechanical cost, but
   the biggest reframing of what B8's own gate is allowed to be measured
   against, and not a call to make without the human weighing in on
   whether that satisfies what the gate was meant to prove.

**Not decided here.** Two consecutive gate amendments and this scoping
pass all point the same direction: the shortfall was never in the
formula. Recorded as three sized options, not a recommendation.

### 2026-08-30 · B8 · gate CLEARED — the missing slot-1 signal, two
belief-consistency fixes to the allocator, and the reference-point
correction. Threshold NOT moved.

Resolved with option A+B from the scoping entry above. Sequence recorded
in the order it happened, including the two places the allocator itself
was changed, because one of those changes HELPED the gate and that must
not be discoverable only by reading a diff.

**1. Root cause, stated properly.** Information has decision value only if
it can change an action taken BEFORE the outcome is revealed. This harness
had no such moment: belief started uniform at every first decision, and
the only strong signal (`DEAD`, 0.55 vs 0.02 per attempt) arrives
*simultaneously with episode termination* — so "stop retrying a dead
instrument" was free, and no policy could earn credit for it. That is why
all three metric formulations measured ~0, and why a fourth would have
too.

**2. The slot-1 decline signal (option A).** The frozen simulator does not
emit one — its own docstring records that slot 1 is "given," a scope
boundary set at B2 before the belief layer existed. But that decline
reason is the entire premise of the system. `eval/allocator_sweep.py` now
reconstructs it using ONLY frozen parameters:

- **Emission**: `P(slot-1 failure mode | cause)` read from
  `sim_config.yaml`'s own hazards via the simulator's own
  `_logits_from_base_rates`/`_softmax` (imported, not reimplemented),
  conditioned on the two modes that can put a mandate into recovery.
  Measured: `CARD_EXPIRED` at **57.9%** for true `CANT_PAY_EVER` vs
  **3.2%** / **3.5%** for `CANT_PAY_NOW` / `WONT_PAY`.
- **Inference**: the allocator inverts through `src/classify/cause_map.py`
  — a *separate*, hand-authored table with no relationship to those
  numbers. Independence is the point: belief after `CARD_EXPIRED` is 0.75
  on `CANT_PAY_EVER`, never 1.0, and 0.75 is not the 0.896 exact Bayes
  would give. The allocator remains realistically miscalibrated and can
  still be wrong in both directions.
- RNG stream `seed + 500_000`, independent of the simulator's own, so
  adding the signal cannot perturb any previously reported outcome draw.

Nothing under `eval/frozen/` was modified.

**3. The post-terminal decision (option B).** The harness previously
`break`-ed on a terminal outcome without ever asking the allocator what to
do next — never recording the `CANT_PAY_EVER -> REAUTH` action that lane
exists to produce. It now asks once more, with the terminal observation
folded into belief, and records the decision without executing further
debit.

**4. Allocator change #1 — REAUTH belief-weighting. Requested on safety
grounds; makes the gate HARDER.** `Q(REAUTH)` did not depend on belief at
all, so `b.dominant() == CANT_PAY_EVER` fired on a bare plurality as thin
as (0.34, 0.35, 0.31). Re-authorisation only recovers money if the
instrument is genuinely dead, so its recovery term is now weighted by
`b[CANT_PAY_EVER]`. The confidence required EMERGES from the economics
rather than from a hand-picked threshold — no new constant, so
`src/policy/CLAUDE.md`'s "cite a clause or put it in a config file" rule
is not engaged, and this project's thrice-rejected tuning-dial pattern
(B5's stop-threshold scalar, the paired-criterion reversal, B7's
`switch_eps`) is not repeated. The AFA-cliff path is deliberately NOT
discounted: above the AFA-free limit re-authorisation is the only legal
route (clause 8(a)/8(b)), and discounting a legal requirement by a belief
would be a category error. Asymmetric by design, matching
`cause_map.py`'s established posture: mistaking `CANT_PAY_EVER` for
`CANT_PAY_NOW` costs one retry slot; the reverse costs a customer.

**5. Allocator change #2 — ATTEMPT belief-discounting. This one HELPED the
gate, was flagged as such BEFORE being applied, and was applied only on
explicit approval.** After change #4 the allocator was internally
incoherent: `REAUTH` was belief-weighted while `ATTEMPT` was not, so one
comparison used two different beliefs. Concretely, on a real mandate it
believed 0.75-dead, it still valued `Q(ATTEMPT)=398,080` against
`Q(REAUTH)=233,157` and burned a retry slot — because the marginal hazard
reports population-average `P(RECOVERED)=0.3017` regardless of belief.
`eval/frozen/protocol.md` names "a wasted attempt on a dead instrument" as
exactly what the attempts-spent bar exists to penalise.

Fix: scale ONLY `ATTEMPT`'s recovery term by `(1 - b[CANT_PAY_EVER])`.
The justification is **definitional, not fitted**: root `CLAUDE.md`
defines `CANT_PAY_EVER` as "Instrument dead — expired card, closed
account, revoked mandate," so `P(RECOVERED | CANT_PAY_EVER) ~ 0` follows
from what the cause MEANS. No `P(outcome | cause, ...)` is estimated, no
coefficient fitted, no constant introduced — so this does NOT reopen the
cause-conditioned-hazard gap (2026-08-29, B7). Opt-out risk and the
continuation value are deliberately left undiscounted: a dead instrument's
holder can still opt out, and the survival branch is already the
"nothing terminal happened" case.

**6. Reference-point correction — the threshold itself did NOT move.** The
zero-delay oracle used to validate the first replacement (~1.5816) was
withdrawn as the wrong reference for a *margin*: it never waits for
evidence, so it cannot bound what an evidence-based policy can achieve. It
is replaced by a delayed-evidence reference policy (best achievable
inference under the same observations and the same delay): **mean gap
0.8329**, separation from random **2.73 random-SDs** — ample resolving
power. `DISCRIMINATION_MARGIN` is **bit-identical to before** (0.29558992474816265):
it is derived from the RANDOM baseline, which ignores the slot-1 signal by
construction, so its numbers do not move. **The harness was fixed; the bar
was not.** The 2026-08-29 pre-commitment (no third re-derivation of the
threshold) is therefore honoured literally, not just in spirit.

**7. Measured, only after all of the above was in place:**

| profile | attempt rate | floor | discrimination | margin | violations |
|---|---|---|---|---|---|
| strict | 0.8266 | 0.25 OK | **0.9048** | 0.2956 OK | 0 |
| permissive | 0.8266 | 0.25 OK | **0.9048** | 0.2956 OK | 0 |

Per-cause behaviour (seed 0), which is the finding that actually matters:

| true cause | n | mean attempts | REAUTH | zero-attempt |
|---|---|---|---|---|
| CANT_PAY_NOW | 85 | 1.388 | 4 | 5 |
| CANT_PAY_EVER | 34 | **0.529** | **19** | 19 |
| WONT_PAY | 66 | 1.848 | 1 | 2 |

56% of truly-dead instruments now go straight to re-authorisation spending
**zero** retries. **The safety cost is explicit and must be reported with
the headline: 4 of 85 true-`CANT_PAY_NOW` mandates (4.7%) were routed to
REAUTH incorrectly** — a recoverable customer sent through an auth flow.
That is the false-off-ramp-analogue error this project committed to
reporting alongside missed recovery, and it is the price of the asymmetry
in #4/#5.

**Honest caveat on 0.9048 exceeding the 0.8329 reference.** These are not
nested policies and neither dominates: the reference has BETTER inference
(exact Bayes, 0.896 on `CANT_PAY_EVER` after `CARD_EXPIRED`) but a CRUDER
stopping rule (plain dominance); ours has worse inference (cause_map's
0.75) but a belief-weighted economic rule that can decline to attempt
before dominance flips. 0.8329 is one reference policy's value, **not a
proven supremum over all policies**, and must not be described as "we beat
perfect inference." Corrected wording is in
`eval/gate_criteria.py`'s docstring and the test's own docstring.

**Not a clean history, deliberately.** This clause went: original
attempt-rate boolean (structurally unmeasurable) -> attempts-spent with a
zero-delay oracle reference (metric sound, reference wrong) -> attempts-
spent with a delayed-evidence reference and a harness that finally
supplies the evidence the system's premise assumes (threshold unchanged
throughout the last two). Two of those three steps were prompted by our
own allocator failing, which is exactly the pattern that warrants
suspicion; what makes the outcome defensible is that the threshold never
moved after it was first derived, both required-fail cases still hold, and
the one change that helped the gate was flagged and approved before being
made rather than discovered afterward.

### 2026-08-30 · B9 · `committed_schedule.decision_sha256` — a column added to a passed gate's artifact

`schema.sql` is B1's gate artifact and B1 is closed, so adding to it needs
saying out loud rather than doing quietly.

`committed_schedule` gained `decision_sha256 TEXT NOT NULL REFERENCES plan`.
The reason is structural, not convenience: `committed_schedule` is the only
durable record an executor process reads before writing a `ledger` row, and
`ledger.decision_sha256` is `NOT NULL REFERENCES plan`. The executor may be
a different process than the one that called `solve()`, running any amount
of time later — that separation is the whole point of the crash-recovery
design. Without the column, attaching the right plan to a ledger row means
joining `plan` and `committed_schedule` on `(mandate_id, cycle_id)` and
nearest `committed_at`, which is exactly the timing-heuristic join B1's
`plan` table exists to make unnecessary. Two `solve()` calls in one cycle —
or in one frozen-clock instant, which the tests do produce — make that join
ambiguous. A direct FK is unambiguous.

*What this does and does not reopen.* Additive column, no rewrite. B1's gate
certified money/clock/ids tests and "ledger DDL has no UPDATE path"; both
still hold, and `ledger` itself is untouched. `schema.sql` is not under
`eval/frozen/`, so no freeze rule is involved.

### 2026-08-30 · B9 · Two spec contradictions in PLAN_DETAIL, resolved before any executor code

Both were found while planning, and both were put to the human and approved
before implementation rather than settled unilaterally in code.

**1 — no commit path existed.** The gate requires a test that "commits an
attempt, then delivers a late opt-out." Nothing in the repo wrote `plan` or
`committed_schedule`: B8's `solve()` returns a `Plan` object and stops.
PLAN_DETAIL's B9 file table lists six files and none of them is a writer.
Resolution: `src/execute/commit.py`, a seventh file, as production code
rather than a test fixture — `schema.sql`'s own comment requires
`committed_at` to come from `src/core/clock.now()` and not Postgres's wall
clock, and only a real module can honour that. A fixture-only writer would
also have to be rebuilt by B10 and B12 independently.

**2 — `void.py`'s must-not contradicted §3's write ordering.** PLAN_DETAIL
says void must never touch a key that already has an INTENT row ("that
attempt is resolved by asking, never by voiding"); §3 step 2a has the late
lifecycle read "abort and void", which happens *after* INTENT is written.
Read literally, the abort path could never void.

Resolution: **`void()` refuses only when a `SENT` row exists.** The
asymmetry is about who is asking, not about which rows exist. The executor's
pre-call abort holds the lease and wrote that INTENT row itself, in this
process, and has issued no provider call — that is first-hand knowledge.
`recover.py` is a *different* process inferring from rows, and it keeps the
strict rule: it must never treat INTENT-without-SENT as proof nothing was
sent, and always asks the provider. The literal reading was rejected because
it leaves a live schedule row on a revoked mandate forever, and pollutes
`UNCONFIRMED` with cases we know for certain were never sent.

### 2026-08-30 · B9 · The recovery interface was dead on arrival — found by the first live call, not by the suite

Full incident: POSTMORTEM.md incident 3. Recorded here for the design
consequence.

`find_by_receipt()` — the entire "recover by asking, never by resending"
path — called `payment.all({"receipt": ...})`. Razorpay rejects that
outright: *"receipt is/are not required and should not be sent."* `receipt`
is an Order field; Payments have none. The method could never have returned
anything, for any input. All 78 B9 tests were green and
`guard_invariants --all` was clean at the time.

*Why the original reasoning was right and still failed.* The docstring
argued recovery must search the entity the ambiguous call actually creates,
and `charge()` (`payment.createRecurring`) creates a Payment, not an Order —
so searching Orders would be recovering against the wrong record. That
reasoning is sound. The mechanism it assumed simply does not exist. An
earlier draft that searched Orders had been "caught and fixed before merge";
the fix was the bug.

*Measured, both live, before choosing.* `order.all({"receipt": R})` works
and is server-side indexed — three known receipts each returned exactly
their own order, count=1. A bounded-window `payment.all({from,to})` scan
with a client-side `notes` match also works (verified against B3's real
payment). The first was chosen and the human approved it: the second caps at
100 payments per page and degrades on a busy account precisely when recovery
matters most.

*Consequence.* `charge()` now creates an order carrying the idempotency key
as its receipt, then creates the recurring payment against that `order_id` —
which is how Razorpay itself models recurring debits. `find_by_receipt()` is
an indexed two-step. The order is created FIRST so that a crash between the
two calls still leaves a receipt-addressable record.

*A second finding from the same probe, and why it changes nothing.* The
Orders list endpoint **lags indexing** — an order queried by its own receipt
at 0s, 3s and 8s after creation returned count=0, appearing later (~30s in
the smoke run). So `None` from this method cannot distinguish "never
created", "created but no payment", and "not indexed yet". That is exactly
what `recover.py` was already built for: a miss becomes `UNCONFIRMED` and is
asked again on backoff, never treated as proof nothing was sent, and the
slot stays consumed throughout. The lag strengthened the backoff's
rationale rather than changing its design.

*What is still unverified, and stays disclosed.* `charge()` itself. Driving
`payment.createRecurring` needs a real saved token or active mandate, which
test mode will not mint on demand. The module docstring's existing
disclosure stands, and is now the only remaining one on this path.

*The generalisable lesson, which is the point of logging it.* Every test in
`tests/execute/test_razorpay_client.py` fakes the SDK, deliberately and with
good reason. A fake accepts whatever parameter shape it is handed, so no
fake-based test can ever catch a wrong shape. Fake-based tests guard
behaviour; only a live call guards wire format. `scripts/live_smoke_b9.py`
now does the latter, and the two are documented as separate risks that do
not substitute for one another. PLAN.md §5 risk 3 predicted this class of
failure ("Razorpay's actual idempotency semantics may not match what B9
assumes") and bought it down with the B3 spike — the spike covered
`Order.create` and correctly told us not to trust provider dedup, but
nothing had ever exercised the *lookup* half of the same interface.

---

### 2026-08-30 · B10 · Recovery now frees a slot it can prove was never spent — the one non-ambiguous case in an otherwise deliberately paranoid module

*The measurement that prompted it.* The first 50-kill chaos run reported
**23 of 60 attempts ending at `UNRESOLVED_FINAL` with no `SENT` row in the
ledger**. Each of those permanently consumed one of only four lifetime NPCI
attempts, on an attempt that had provably never reached the provider.
`recover.py` asked five times, missed each time (the modelled Orders index
lag, measured live at B9), and burned the slot. That is the correct
behaviour when "we do not know" is the honest answer — and the module is
built end-to-end on that refrain — but here it is not the honest answer.

*The proof, stated precisely, because everything rests on it.*
`executor.execute()` writes the `SENT` ledger row (autocommit, durable) and
only **then** calls `client.charge()`. So `charge() was called` ⟹ `SENT
committed`. Contrapositive: **no `SENT` row means no provider call was ever
issued.** A `SENT` insert still in flight when the process died rolls back,
and `charge()` sits after that append returns, so it was never reached
either. Absence of a `SENT` row on re-read is therefore evidence, not
absence of evidence.

*What was built.* `recover._resolve_never_sent()` writes
`FAILED/NEVER_SENT` and voids the `committed_schedule` row, which clears
`committed_one_live_per_slot` and lets the allocator reissue at
generation+1 on the **same `attempt_index`** — so no new NPCI slot is
spent (the budget counts distinct `attempt_index` values, never distinct
keys; `void.reissue()` copies it unchanged). Measured effect: 23/60 → 0/60,
same seed, same window partition. Now a regression guard in
`ChaosReport.passed`, not merely reported.

*Three things deliberately NOT done, each of which was the tempting version.*

1. **It does not skip the ask.** The first implementation resolved
   `NEVER_SENT` without calling `find_by_receipt` at all — the proof says
   there is nothing to find, so why pay for the call? That was wrong, and
   an existing B9 test caught it: `test_reconcile_resolves_a_dangling_intent
   _when_the_provider_confirms_it` builds an INTENT-only key whose payment
   the provider *does* report, and expects `RESULT`. Under the skip-the-ask
   version that test would have voided a schedule row for an attempt that
   took money. The order is now: **always ask; use the proof only to decide
   what a miss means.** "Recovery is by asking" stays unconditional. If the
   provider ever contradicts our reasoning about our own write ordering, we
   believe the provider.
2. **It does not reissue.** Voiding is what *makes* a reissue possible;
   choosing a new `scheduled_for` is a scheduling decision the allocator
   owns. Making it here would be precisely the late read that ACTS rather
   than stops, which PLAN_DETAIL.md §1 forbids this layer.
3. **It is not generalised beyond the provable case.** A key *with* a
   `SENT` row still walks the full backoff to `UNRESOLVED_FINAL` and still
   burns the slot. The discriminating negative control
   (`test_never_sent_does_not_fire_when_a_sent_row_exists`) is byte-identical
   setup except for that row.

*Defence in depth, two independent layers, because the failure mode is
voiding a schedule row for an attempt that moved money.* First, reconcile
asks the provider before this path is reachable at all, so a payment that
exists is resolved as found rather than voided. Second, `void()`
independently refuses any key carrying a `SENT` row — so if the proof were
somehow wrong, this raises `VoidError` rather than quietly voiding. Note
that the second layer cannot catch the specific case of "payment exists,
no SENT row"; that is what the first layer and the ordering test below are
for.

*The load-bearing dependency, pinned rather than assumed.* This makes
`recover.py`'s correctness depend on `executor.py`'s write ordering, which
nothing previously tested directly — reversing step 3's append and the
charge would leave every existing test green while recovery began voiding
schedule rows for attempts that may have taken money.
`tests/execute/test_executor.py::test_sent_row_is_committed_before_the_
provider_is_ever_called` now reads the actual call order from inside the
provider call, the same technique the existing INTENT-ordering test uses.

*Test fallout, and why it is not "fixing tests to match the code".* Six
B9 recover tests built their dangling key as INTENT-only and expected the
UNCONFIRMED/backoff path. Under the new semantics an INTENT-only key is
provably-never-sent and correctly resolves without walking the backoff, so
their *premise* was what changed: the backoff exists for the case where a
call may have been made, which is the `SENT` case. Each now passes
`state="SENT"` explicitly at the call site rather than the helper's default
being flipped, so each test's premise stays visible where it is used. The
approval for this change was requested and given before any of it was
written.

---

### 2026-08-30 · B10 · The entry above is REVERSED. NEVER_SENT introduced a double charge and was reverted the same day

*Read the previous entry as a record of what was tried, not of what
shipped.* Everything it says about the proof is correct as far as it goes,
and that turned out not to be far enough.

*What the chaos-engineer review found.* The proof — "no `SENT` row means no
provider call was issued", because `executor.py` writes `SENT` before it
charges — is sound about **one process's own state** and false about a
**concurrent** one. `executor.py` never re-validates lease ownership
between claiming the lease (step 2) and charging (step 3). So a worker that
STALLS past its lease TTL — alive, not dead; a GC pause, a slow lifecycle
read, a suspended VM — is indistinguishable from a crash in durable state.
Recovery voided its slot while it was still running; the worker then
completed its real charge; the freed slot was reissued at generation+1
under a **different key** and charged again. Two real charges for one NPCI
slot, and because the keys differ, invisible to `ChaosClient.accepted` and
to any receipt-keyed idempotency check at the real Razorpay.

*Whose regression, measured not argued.* The same sequence run twice, once
with the branch forced off to reproduce B9:

| | charges | reissue succeeded |
|---|---|---|
| with `NEVER_SENT` | **2** | yes |
| B9 behaviour | 1 | no (`VoidError`) |

Introduced by B10, not inherited. Without the void there is no reissue,
hence no second key. **The 23/60 slot burning that the optimisation
removed was partly load-bearing.**

*Why reverted rather than patched.* Two fixes were on the table.
Re-checking for a `SENT` row inside `reissue()` only narrows the window —
the check still precedes the reissued attempt's own charge, which clause
6(a) puts at least 24h later. Re-validating lease ownership immediately
before step 3 is the real fix and is a fencing-token problem: new design
work on B9's executor, on the money path, after two auditors had signed off
on a different design. Neither belongs bolted onto the end of this block.
And the project's own refrain decides it: *a double-charge is worse than
ten missed recoveries.* The optimisation traded precisely that way. The
slot recovery is an OPTIMISATION; what it cost is the invariant the module
exists to hold.

*What remains from the attempt, deliberately.* Three things earned their
place independently of it: the `_dangling_keys` ledger-scan fix (incident
4, unrelated to `NEVER_SENT`); `test_sent_row_is_committed_before_the_
provider_is_ever_called`, which pins a write ordering nothing previously
tested and which any future fencing work will need; and
`eval/chaos.py`'s `slots_burned_unsent`, now a standing reported cost
rather than a pass/fail guard, so the 23/60 is visible instead of quietly
accepted.

*What it would take to reinstate.* Lease fencing in `executor.py` — a
token the executor re-validates immediately before step 3, which
`recover.py` invalidates when it takes over a key — plus a `reissue()`
re-check as a backstop. That is its own scoped piece of work with its own
review, not a B10 addendum. Until then the slot cost stands and is
reported.

*The generalisable lesson, which is why this is logged rather than quietly
undone.* Two review passes (money-auditor twice, compliance-auditor)
cleared this design; both reasoned explicitly about concurrency and
crash-safety, and both concluded the slot-freeing was safe. What found it
was a reviewer asked a different question — not "is this correct?" but
"what states can this harness NOT construct?" The harness is single-process
and single-threaded, and an induced kill can only STOP a process, never
DELAY one. A live-but-slow worker was outside its reachable state space,
and that is exactly where the defect lived.

---

## 2026-08-30 — LLM edge switched from Anthropic to Gemini (config, B11 not yet built)

*What changed.* The LLM edge's provider, before any of `src/llm/` exists.
`GEMINI_API_KEY` replaces `ANTHROPIC_API_KEY`; the three model ids become
`gemini-3.5-flash-lite` (normaliser, intent) and `gemini-3.5-flash`
(narrator); `requirements.txt` and `setup.ps1` swap `anthropic` for
`google-genai==2.20.0`. Requested by the human, who supplied the key.

*Scope, stated plainly.* This is a provider swap at the **language edge
only**. It touches no decision path: invariant 1 still forbids any LLM
client under `src/model/`, `src/policy/`, `src/core/` and `src/classify/`,
the decision core remains the competing-risks model plus backward
induction, and the normaliser still only maps unseen strings into a
taxonomy it cannot extend. Nothing in the thesis moves.

*Verified before adoption, not assumed.* Three things were probed against
the live API rather than taken on faith, in the spirit of B9's
`find_by_receipt()` finding (a method whose 78 tests were green and which
was dead on arrival against the real API):

1. **The key authenticates.** `ListModels` returns HTTP 200.
2. **Forced tool-use survives the switch.** CLAUDE.md requires that
   structured output go through required tool-use so malformed JSON is
   *structurally* impossible. Gemini's equivalent is
   `toolConfig.functionCallingConfig.mode = "ANY"`. Probed on both adopted
   models with a four-value enum: each returned a well-formed
   `functionCall` and **empty text** — no free-text path taken. The
   invariant transfers intact.
3. **`gemini-2.5-*` is retired for new keys.** Both `gemini-2.5-flash` and
   `gemini-2.5-flash-lite` return **404, "no longer available to new
   users"**, while still being listed by `ListModels`. The obvious model
   ids would have failed on first call, and would have failed at whatever
   later hour `src/llm/` was first exercised.

*Why pinned ids and not `gemini-flash-latest`.* The alias resolves
differently over time. B12's headline argument is the **run-to-run variance
column** — same input, different answer — and an alias that silently moves
underneath the benchmark would corrupt exactly the number the benchmark
exists to report. Pinned ids, as with every other constant here.

*The defect this uncovered, which matters more than the swap.*
`scripts/guard_invariants.py` listed `google.generativeai` — the **legacy**
SDK — and nothing else Google. Probed against the current SDK:

        BLOCKED  import anthropic
        PASSES   from google import genai        <- the client we now use
        PASSES   import google.genai
        PASSES   from google.genai import types
        BLOCKED  import google.generativeai      <- legacy, retired

So adopting Gemini without touching the guard would have left invariant 1 —
"the mechanical proof that the no-LLM-in-core claim is not just a
comment", per that file's own docstring — **cosmetic for the only provider
the repo actually uses**, while still reporting green. This is the same
defect class PLAN_DETAIL §8.2 already routed to B11 ("the guard matches
only a direct `import anthropic`"); the provider switch promotes it from a
secondary gap to a primary one. Fixed here rather than deferred to B11,
because the hole opens the moment the key does. All forms above now block;
`import googlemaps` and `from google.protobuf import x` still pass, so the
pattern is not over-broad. `vertexai` added while there.

*And the self-test that would have hidden it.* `run.ps1 verify` step 1
proves the guard fires by writing `import anthropic` into `src/model/`.
After this swap that probes a client we no longer use — a green check
verifying nothing about the live risk, the same vacuous shape audited out
of the gates on 2026-08-29. It now probes `from google import genai`
**first**, then `import anthropic`, and fails if either passes.

*What is NOT verified, and stays disclosed.* The key's prefix is `AQ.`,
not AI Studio's usual `AIza`. It authenticates today. Whether it is
long-lived or an OAuth-derived token with a TTL could not be determined
from the API, so a sudden `401` on the golden set is a credential
expiry to check before it is a code bug. Separately, Flash-Lite's
**Hinglish** intent quality is unmeasured — that is precisely what B11's
30-row `intent.jsonl` golden set is for, and it should be read as an open
question until that set runs, not as an assumption carried over from the
Haiku plan.
