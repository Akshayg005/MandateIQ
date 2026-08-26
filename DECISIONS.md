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
