# Block gates

Tick only when the stated condition is actually verified, not when the code
exists. `show_state.py` reads this at every session start, and `checkpoint.py` summarises it into STATE.md at every session end.

Blocks are ordered by dependency, not by calendar. A block is sized by what
it can prove, so two may close in one sitting and one may take three.
**Progress is gates passed, never blocks started** — a block with every file
written and its gate unmet counts as zero.

`★` = critical path · `∥` = parallel, can run any time after its entry block.

- [x] **B0** environment: `.\run.ps1 verify` passes all five, including a real test-mode order
- [x] **B1** ★ core + ledger: money/clock/ids tests pass; `ids.py` imports no `time`/`uuid`/`os`/`random`; ledger DDL has no UPDATE path
- [x] **B2** ★ **the freeze**: baseline ladder produces a number; FREEZE_HASH recorded; `guard_frozen.py` denies an edit under `eval/frozen/`
- [x] **B3** ingest + taxonomy: a real test-mode `payment.failed` lands in the ledger's ingest table (`ingested_event`) with a classified cause; mandate lifecycle table exists; provider idempotency spike result written to DECISIONS.md
      <!-- changed from "lands in the ledger" to "lands in ingested_event",
           2026-08-27, before any webhook had been received and before this
           block's code existed beyond the plan file: ledger.decision_sha256
           is NOT NULL REFERENCES plan, and no plan row can exist before B8's
           allocator -- a bare observed decline has no decision to attach to.
           Logged in DECISIONS.md, does not touch eval/frozen/. -->
- [x] **B4** ★ person-period frame: `validate()` rejects every malformed shape; a censored episode round-trips with all four rows intact; split is mandate-level; no feature encodes a future slot
- [x] **B5** ★ competing risks + CIF: held-out multinomial log-loss and per-cause Brier on the `test` split beat an intercept-only MNLogit null; transfer degradation of that same fit reported on `misspecified` and `coupled` frames; calibration-in-the-large reported per `slot × in_salary_window` cell; `Σ_c CIF_c(4) + S(4) == 1`; stats-reviewer returns clean
      <!-- changed from "both" at PLAN_DETAIL v2 (B1), before the freeze commit
           and before any file under src/policy/ existed: §8.1 decision 1 added
           a third frozen arm (coupled). Logged in DECISIONS.md.
           2026-08-27: changed from "all three, one bar" to per-metric,
           per-arm binding -- oracle_policy.py's timing oracle found coupled
           does not discriminate on recovered money (9/11/0, mean/SE=-0.22,
           20 seeds); a cause-aware oracle then found it DOES discriminate on
           iatrogenic count (mean/SE~5.6). coupled was never built to test
           recovering more, only wasting less under contention. Logged in
           DECISIONS.md, does not touch eval/frozen/.
           2026-08-28: changed from "beats the ladder" (four policy-comparison
           clauses) to model-fit evidence (policy-free), before any coefficient
           existed. A null policy (attempt slot 2 once, stop) was measured to
           clear mandates-preserved on all three arms (+5.90/+5.41/+6.32
           pooled SD) and coupled's attempts-spent/iatrogenic clauses
           (32.07/7.72 pooled SD) with no model at all -- all three metrics
           are monotonically decreasing in attempt count by construction
           (NOT_PRESERVED = {DEAD, OPTED_OUT}, both reachable only via
           attempt()). The one clause that was informative (misspecified/
           recovered-money) was separately found structurally unwinnable from
           nominal-only training: its mechanism (replenishment_exponent) only
           enters via _cloglog_probs, and nominal's link is logit, so the
           relevant coefficient is identically zero in the training arm.
           "Beats the ladder" moves to B8, where an allocator exists to
           honestly bear it. Full reasoning, the measured table, and the
           reversal of a paired-criterion amendment approved earlier the same
           session logged in DECISIONS.md, 2026-08-28 B5 entry. Does not
           touch eval/frozen/. -->

- [x] **B6** ★ calibration + conformal: reliability diagram roughly diagonal; empirical coverage matches nominal on held-out data
      <!-- 2026-08-28: stats-reviewer found 2 blocking issues before this
           was ticked -- an outcome-dependent imputation leak in
           hazard_tensor()'s schedule=None fallback (fixed by threading
           eval/corpus.py's real committed schedule through, closing a gap
           this block had deferred and disclosed but not yet fixed), and a
           conformal p-value formula bug under-smoothing test-point ties
           (one-line fix). Both reverified via eval-runner on the real
           40-seed corpus: corrected marginal coverage 0.9517 mean (min
           0.9327), all four per-class means within ~1.5pp of nominal.
           Full findings and both before/after number tables in
           DECISIONS.md, 2026-08-28, the two B6 entries. Does not touch
           eval/frozen/. -->

- [x] **B7** ★ policy foundation: every constant cites its clause; both profiles instantiate; compliance-auditor all-VERIFIED
      <!-- 2026-08-29: compliance-auditor, fresh context, returned all 8
           checklist clauses VERIFIED (constants cited, both AFA limits,
           4(c) ceiling, NPCI cap, no-cancellation invariant, both profiles
           reachable and neither hard-coded) plus 3 NOT COVERED items
           (24h-lag enforcement, attempt-cap enforcement, contact-frequency/
           quiet-hours) correctly attributed to B8/B9's scope, not this
           block's -- B7 defines constants and profile dispatch, it does not
           enforce them at runtime. Zero VIOLATED. Full report and the three
           B7 design decisions (likelihood inversion, overconfidence
           disclosed not damped, cause-conditioned-hazard Protocol) in
           DECISIONS.md, 2026-08-29. Does not touch eval/frozen/. -->
- [x] **B8** ★ allocator + stopping + off-ramp: 2-slot brute-force equivalence test passes; zero constraint violations across the eval, with an attempt rate on AFA-eligible mandates ≥ `eval.gate_criteria.ATTEMPT_RATE_FLOOR` (0.25) AND a mean discrimination gap (true-CANT_PAY_NOW attempt rate minus true-CANT_PAY_EVER attempt rate, seeds 0-19) exceeding `DISCRIMINATION_MARGIN` (~0.0808) -- the null policy must fail the first, a uniform-random policy at the floor rate must fail the second, both proven by test (`tests/eval/test_gate_criteria.py`); both profiles produce numbers
      <!-- flagged at B4, 2026-08-28, from stats-reviewer's B4 finding 4: the
           allocator MUST apply src.policy.constraints.afa_free_limit_paise()
           before ever consulting the hazard model, routing any above-cliff
           mandate straight to Action.REAUTH -- eval/corpus.py already
           excludes these from B5's training data on that assumption (they
           are 9% of the frozen batch, e.g. subscription mandates above
           Rs 15,000), and the model has zero support for them. Scoring one
           through the CIF/backward-induction path anyway is out-of-support
           extrapolation, not a compliance nuance. Logged in DECISIONS.md,
           does not touch eval/frozen/.
           2026-08-29, from B7: PLAN_DETAIL.md section 4's Q(b, ATTEMPT)
           assumes a CAUSE-CONDITIONED hazard, h(outcome | cause, slot, day,
           amount) -- B5 shipped hazards MARGINAL over cause instead, and
           Cause has no production label, ever, so no fit can close that gap
           (see DECISIONS.md, 2026-08-28, B6). B7 adds
           src/policy/hazards.py's CauseConditionedHazard Protocol -- a type
           declaration only, not an implementation -- so the allocator must
           name its hazard source in the type system rather than silently
           defaulting to a cause-marginal substitution. OPEN QUESTION FOR
           THIS GATE, not settled at B7: whether the allocator instead
           avoids the gap by design -- Cause entering only through action
           gating (REAUTH when CANT_PAY_EVER dominant, OFFER on a singleton
           conformal set) rather than the hazard arithmetic itself. Decide
           and record the reasoning here when B8 answers it. Full writeup:
           DECISIONS.md, 2026-08-29, B7. Does not touch eval/frozen/.
           2026-08-29, vacuous-checks audit: this gate's original text --
           "zero constraint violations across the eval" -- is trivially
           satisfiable by an allocator that never attempts anything (B5's
           null-policy finding, DECISIONS.md 2026-08-28, recurring here).
           Amended, before any allocator code exists, to add an attempt-
           rate floor and a discrimination-margin clause, both derived
           from the frozen simulator's own generative parameters (seeds
           0-19) rather than chosen by hand, and both proven to reject a
           real failing case by test, not asserted. Original headline
           text: "★ allocator + stopping + off-ramp: 2-slot brute-force
           equivalence test passes; zero constraint violations across the
           eval; both profiles produce numbers." New text is the gate
           line above. Full derivation, including 48.86% (the true
           CANT_PAY_NOW fraction) measured and DELIBERATELY REJECTED as
           the floor value in favour of 25%, and the uniform-random
           baseline simulation the discrimination margin is derived from:
           DECISIONS.md, 2026-08-29, two entries, "B8 gate amended" and
           "B8 gate floor lowered." Constants live in
           `eval/gate_criteria.py`, not restated here. Does not touch
           eval/frozen/ -- gate_criteria.py is a new file under `eval/`,
           outside the frozen directory.
           2026-08-30, CLOSED. Measured: attempt rate 0.8266 (floor
           0.25), discrimination gap 0.9048 (margin 0.2956), zero
           constraint violations, both profiles, 20 seeds; 2-slot
           brute-force equivalence passes against an independent
           unmemoised reimplementation across 7 scenarios. NOT a clean
           history and must not be read as one -- the discrimination
           CLAUSE was replaced twice before it measured anything (the
           original attempt-rate boolean was structurally saturated at
           the first, necessarily uninformed decision; its zero-delay
           oracle reference was then withdrawn as the wrong reference
           for a margin). The THRESHOLD (0.29558992474816265) was derived
           once, on 2026-08-29, and never moved after: it depends only on
           the cause-blind random baseline, which the later fixes do not
           touch. What changed was the eval harness (it now supplies the
           slot-1 decline signal the system's own premise assumes, built
           from frozen sim_config parameters only) and two belief-
           consistency fixes in the allocator -- one requested on safety
           grounds that made the gate HARDER (REAUTH weighted by
           b[CANT_PAY_EVER]), one that made it EASIER (ATTEMPT's recovery
           term discounted by the same belief), the latter flagged as
           gate-helping and approved BEFORE being applied. Reported cost
           of the asymmetry, not hidden: 4/85 true-CANT_PAY_NOW mandates
           (4.7%) routed to REAUTH incorrectly. Full sequence, both
           allocator diffs, the per-cause table, and the caveat that
           0.9048 exceeding the 0.8329 reference does NOT mean beating
           perfect inference (the policies are not nested):
           DECISIONS.md, 2026-08-29 and 2026-08-30. Does not touch
           eval/frozen/. -->
- [x] **B9** ★ executor + idempotency: keys test passes (no clock/uuid/pid); **an opt-out arriving inside the 24h window is honoured — proven by a test that actively constructs the race** (commits an attempt, then delivers a late opt-out event inside that window, and asserts the attempt is aborted — not merely a test that happens never to generate one); `UNCONFIRMED` has a resolution path that is actually reachable
      <!-- 2026-08-29, vacuous-checks audit: strengthened from "an opt-out
           arriving inside the 24h window is honoured" -- a test that
           simply never generates a late opt-out would satisfy that text
           by construction, without ever exercising the actual race. No
           executor code exists yet; this amendment is made before B9
           starts, the same standard as the other gate amendments this
           block. Full reasoning: DECISIONS.md, 2026-08-29.
           2026-08-30, CLOSED. All three clauses met, 719 tests green,
           guard_invariants exit 0 (--all AND explicitly by path for the
           new untracked files, per the known B3 issue), eval/frozen/
           untouched. money-auditor: zero findings. compliance-auditor:
           10/10 VERIFIED, zero VIOLATED, zero NOT COVERED -- including
           the three items B7 had to leave NOT COVERED and attribute
           forward (24h-lag enforcement, attempt-cap enforcement,
           contact-frequency/quiet-hours), all of which land here.
           Clause 1 (keys): key_for() delegates to core/ids and a
           source-level test asserts keys.py imports no time/uuid/os/
           random. Clause 2 (the 6(c) race): shipped as a DISCRIMINATING
           PAIR -- a positive case (commit, advance the frozen clock to
           scheduled_for-2h, deliver a REVOKED lifecycle event with
           effective_at inside the window, assert INTENT->FAILED with no
           SENT row, provider never called, schedule row voided) and a
           byte-identical negative control WITHOUT the opt-out that must
           REACH the provider. The pair was additionally mutation-tested
           before ticking: emptying _TERMINAL_LIFECYCLE_STATES made the
           identical positive setup reach the provider, proving the abort
           is caused by the lifecycle read specifically and not by the
           test's own setup. That probe was temporary and deleted; it is
           described here rather than kept, since a permanent test that
           monkeypatches the module under test would itself be the kind
           of check this project distrusts. Clause 3 (UNCONFIRMED):
           driven all the way to UNRESOLVED_FINAL across four reconcile
           passes by test, with UNRESOLVED_FINAL asserted terminal-and-
           reported, never silently dropped.
           NOT a clean history, and must not be read as one. The FIRST
           LIVE test-mode call ever made through razorpay_client.py --
           the last verification step, run after all 78 B9 tests were
           green and guards were clean -- found find_by_receipt() DEAD ON
           ARRIVAL: it filtered PAYMENTS by `receipt`, which the API
           rejects outright ("receipt is/are not required and should not
           be sent") because receipt is an ORDER field. Since B3 proved
           `receipt` does not dedupe Order.create, that method is the
           ENTIRE recover-by-asking path, so clause 3 was false against
           the real API while every test passed. Fixed by anchoring the
           ATTEMPT path to an Order (charge() creates the order carrying
           the key as receipt FIRST, then the recurring payment against
           that order_id; find_by_receipt() became an indexed two-step),
           chosen over a verified-but-pagination-fragile alternative and
           approved before being applied. A second measured finding: the
           Orders list endpoint LAGS INDEXING (count=0 at 0s/3s/8s,
           resolved ~30s later), so None means "not found YET", never
           "never sent" -- which is what recover.py's backoff already
           assumed, so the lag strengthened that design rather than
           changing it. charge()'s own field shape remains UNVERIFIED
           against live traffic (test mode will not mint a recurring
           token on demand) and stays disclosed in the module docstring,
           not assumed correct. Guard added: scripts/live_smoke_b9.py --
           fake-based tests guard behaviour, only a live call guards wire
           format, and a fake accepts whatever shape it is handed. Full
           incident: POSTMORTEM.md incident 3. Full reasoning, both
           measured alternatives, and the two spec contradictions
           resolved before coding (the missing commit path -> commit.py
           as a seventh file; void() refusing only on a SENT row rather
           than on any INTENT row): DECISIONS.md, 2026-08-30, three B9
           entries. Does not touch eval/frozen/. -->

- [x] **B10** chaos: 50 induced kills; zero double-charges; zero lost jobs; ledger complete; **the denominator is reported** — how many kills landed inside the unsafe window
      <!-- 2026-08-30, CLOSED. Measured, seed 0, reproducible by
           `.\run.ps1 chaos -Kills 50`; full output in reports/chaos.md.
           50 uniform kills partitioned by the window each ACTUALLY landed
           in (judged from durable state after the fact, never from where
           the kill was aimed): pre-INTENT 5, INTENT->lease 5,
           lease->SENT 18, SENT->ack-not-accepted 9, SENT->RESULT-ACCEPTED
           4 (the unsafe window), post-RESULT 9. Plus 10 fault-seam runs,
           reported SEPARATELY and never folded into the 50. Zero
           double-charges, zero lost jobs, zero ledger violations.
           TWO LIMITS ON WHAT THE DENOMINATOR MEANS, stated in
           reports/chaos.md rather than left for a reader to assume.
           (1) The partition is over DATABASE STATEMENTS plus the provider
           call -- a reproducible, exhaustively coverable space -- NOT over
           wall-clock time, which is dominated by the network call. It is
           the denominator for this sampler and says nothing about
           production frequency. (2) A kill signal to our own process
           CANNOT reach the most dangerous state at all: it can only
           destroy work not yet done, never make the provider accept money
           and then lose the answer. That is why the FaultSpec seam exists
           (src/execute/razorpay_client.py, construction guarded outside
           eval/chaos.py + tests/ by scripts/guard_invariants.py) and why
           ChaosReport.passed REQUIRES unsafe_window_covered > 0 -- with
           seed 0 at 24 kills the uniform block drew the unsafe window
           ZERO times, so the headline would have been an artifact of
           sampling exactly as PLAN_DETAIL.md section 8.2 finding 2
           predicts. The gate is met at 50; it was not at 24, by luck.
           NOT a clean history and must not be read as one. The harness
           found TWO defects on its FIRST run, both in code whose 82 tests
           were green:
           (a) A PERMANENTLY LOST JOB -- POSTMORTEM.md incident 4.
           recover._dangling_keys() discovered abandoned work by iterating
           lease.expired(), so it could not see an INTENT row written
           before the lease was claimed (executor.py writes INTENT at step
           1 and claims the lease at step 2, so a crash between them
           leaves an INTENT row and NO lease row at all). Such a key was
           invisible to both scans forever while step 1's ON CONFLICT DO
           NOTHING blocked any re-execute: slot consumed, customer never
           debited, nothing reported. src/execute/lease.py's own docstring
           names the rule that was broken -- the lease is "an OPTIMISATION
           over ledger_intent_once, not the concurrency control" -- and
           recovery had been keyed solely on it. Fixed by scanning the
           LEDGER and using the lease only to exclude keys a live worker
           still holds. The regression test was PROVEN discriminating
           against a verbatim reimplementation of the old algorithm on the
           same state (old returns [], new returns the key), not merely
           asserted to be.
           (b) 23 of 60 NPCI SLOTS BURNED FOR NOTHING -- attempts ending
           at UNRESOLVED_FINAL with no SENT row, i.e. never sent, each
           permanently consuming 1 of only 4 lifetime attempts. A fix
           (recover._resolve_never_sent) was approved, built, cleared by
           money-auditor AND compliance-auditor, and then REVERTED THE
           SAME DAY when the chaos-engineer review found it introduced a
           CROSS-GENERATION DOUBLE CHARGE -- POSTMORTEM.md incident 5.
           Its proof ("no SENT row means no call was issued") is sound
           about one process's own state and false about a concurrent
           one: executor.py never re-validates lease ownership between
           claiming the lease and charging, so a worker STALLED past its
           lease TTL (alive, not dead) is indistinguishable from a crash
           in durable state. Recovery voided a live worker's slot, the
           worker completed its real charge, and the freed slot was
           reissued at generation+1 -- a DIFFERENT key -- and charged
           again. Measured directly, same sequence run both ways: 2
           charges with the fix, 1 without, so B10 INTRODUCED it rather
           than inheriting it. Invisible to every oracle in the harness,
           since both are keyed per receipt and the charges land on
           different keys. Reverted rather than patched because the slot
           recovery is an OPTIMISATION and this project's refrain --
           a double-charge is worse than ten missed recoveries -- decides
           it; reinstating it needs real lease fencing in executor.py,
           which is its own scoped work. THE SLOT COST THEREFORE STANDS
           AT 23/60 and is reported by eval/chaos.py as a standing
           measurement, not silently accepted.
           Kept from that attempt because each earned its place
           independently: the _dangling_keys fix in (a); tests/execute/
           test_executor.py::test_sent_row_is_committed_before_the_
           provider_is_ever_called, which pins a write ordering nothing
           previously tested; and a green regression guard,
           tests/eval/test_chaos.py::test_a_stalled_worker_cannot_have_
           its_slot_voided_and_reissued.
           THE REVIEW LESSON, worth more than the code: money-auditor
           (twice) and compliance-auditor all cleared the reverted design,
           both reasoning explicitly about concurrency. What found it was
           a reviewer asked not "is this correct?" but "what states can
           this harness NOT construct?" -- the harness is single-threaded
           and an induced kill can only STOP a process, never DELAY one,
           so a live-but-slow worker was outside its reachable state space
           entirely. That limit is now stated in reports/chaos.md as one
           of three things the report does not license concluding.
           A third finding, NOT fixed and NOT a defect: UNRESOLVED_FINAL
           is a permanent dead end (recover._stuck_keys matches
           reason=UNCONFIRMED only), so a charge that becomes findable
           after the backoff ends stays misfiled. That is B9's stated
           design ("terminal and reported") and the slot correctly stays
           consumed; recorded as POSTMORTEM.md incident 6 and pinned by a
           test asserting the CURRENT behaviour, so changing it has to be
           deliberate. Correcting such keys needs a path outside this
           module (the B3 webhook, or an operator tool) -- B13's.
           Full reasoning: DECISIONS.md, 2026-08-30, BOTH B10 entries (the
           second reverses the first); POSTMORTEM.md incidents 4, 5, 6;
           reports/chaos.md. Does not touch eval/frozen/. -->
- [x] **B11** ∥ LLM edge + golden set: golden set passes; no LLM import in core; normaliser output is versioned in the ledger before it can touch a belief
      <!-- 2026-08-31, CLOSED. All three clauses measured, not asserted.
           Golden set: aggregate 47/50 (94.0%, floor 90%); ESCALATION-ONLY
           subset (the 12 rows decline_taxonomy.classify() itself leaves
           UNKNOWN -- the only rows normalize() actually sees in
           production, since 38/50 are answered confidently upstream and
           never reach the LLM) 12/12 (100%), gated independently of the
           aggregate so the component cannot look fine on average while
           failing at its actual job. Both zero-tolerance checks clean
           (0 INSUFFICIENT_FUNDS<->MANDATE_REVOKED confusions, 0 any-label
           false MANDATE_REVOKED). Intent 29/30 (96.7%, floor 85%), 0 false
           HIGH on LOW-labeled (false off-ramp risk, the safety-critical
           direction). guard_invariants.py denies src.llm from all four
           PROTECTED_DIRS, transitively (both `import src.llm.x` and
           `from src import llm` forms). belief.update()'s source_version
           is a required keyword-only parameter, no default -- a belief
           cannot be constructed without naming which classifier version
           produced its observation.
           NOT a clean history. money-auditor cleared the ledger/belief
           changes outright (zero findings). payments-domain -- this
           block's specified review, PLAN_DETAIL.md -- found NINE real
           problems on the classify/llm boundary, each independently
           reverified (a probe script or a live measurement) before being
           acted on: two golden-set labels were wrong, not model errors
           (grounded against decline_taxonomy.py's own keyword rules and
           standard ISO 8583 banking codes); three intent labels were
           wrong, the worst literally reading "can you pause it for a
           month or two" and labeled LOW (stay in the retry lane) when it
           is a direct request for the off-ramp's own first stage; the
           narrator's claims guard was ANTI-CORRELATED WITH TRUTH -- tested
           against 8 legitimate off-ramp sentences and 9 real false-agency
           claims, it blocked 4/8 of the first group and missed 8/9 of the
           second -- rebuilt as a SAFE-then-DANGER design, 0/0 on both sets
           after; a guard bypass (`from src import llm`) reproduced a hole
           the same file had already closed for `from google import genai`
           three lines above it; the zero-tolerance decline check only
           caught one symmetric swap and missed exactly what the live run
           produced (payment_cancelled -> MANDATE_REVOKED, the adversarial
           case lifted from decline_taxonomy.py's own documented finding),
           widened to any false MANDATE_REVOKED; confidence was computed,
           used, then discarded -- now persisted to
           normalized_decline.confidence, its missing-key default flipped
           from the most permissive value (1.0) to the safest (0.0); the
           cache-busting version hash covered the prompt but not the model
           id or the confidence floor, so an env override could report
           PASSED on a different model's stale answers -- both folded into
           the hash.
           Two follow-on findings were investigated and confirmed
           genuinely blocked, not forced: belief.update()'s honesty gap
           (source_version proves something was supplied, not that it was
           honest) needs a fix living in src/execute/, not src/policy/ --
           PLAN_DETAIL.md section 6's own dependency graph draws ledger/
           and policy/ as siblings that never point at each other,
           converging only at the executor, which does not exist for this
           purpose yet. intent_score()'s float has no defined path into
           src/policy/gate.py's ConformalGate anywhere in this codebase --
           the real gate takes a Belief built from payment data and the
           fitted predictor consumes nonconformity scores, neither of
           which intent_score() produces -- inventing a threshold to close
           the item would have meant fabricating unreviewed statistical
           machinery in the exact path CLAUDE.md's safety design section
           requires split-conformal rigor for. Both left as real design
           work for whichever block does the relevant wiring, not guessed
           at here.
           Full reasoning, every fixed/disagreed/deferred item, and the
           exact repository citations for both blocked findings:
           DECISIONS.md, 2026-08-31, two entries. Does not touch
           eval/frozen/. -->
- [x] **B12** ∥ benchmark + shadow: benchmark table in DECISIONS.md including the variance column; shadow mode produces a delta log over the full batch
      <!-- 2026-08-31, PARTIALLY MET, then TICKED ON A HUMAN SCOPE DECISION
           (see the end of this note). Two clauses; one is met and
           one is blocked on an external quota, so the box stays empty.
           SHADOW MODE: MET. src/execute/shadow.py produces a delta log over
           all 200 frozen mandates (reports/shadow_delta.md + .jsonl).
           Measured: the fixed ladder commits 600 attempts (3 per mandate,
           unconditionally), this system commits 141; 56 REAUTH of which 18
           are bound by the AFA cliff and 38 by belief alone, 3 STOP, 141
           agreeing exactly. The no-live-write invariant was proven against a
           REAL Postgres, not only against test doubles: 6 rows written to
           shadow_ledger and 0 rows in ledger, committed_schedule,
           attempt_lease and plan. Test coverage is a positive/negative
           control PAIR -- a conn permitting only INSERT INTO shadow_ledger
           must let run_shadow finish, AND one also rejecting shadow_ledger
           must make it raise, so "no forbidden statement issued" cannot be
           satisfied by a function that never touched the database.
           BENCHMARK TABLE: NOT MET. The stats side is complete (intercept-
           only null log loss 1.2795 / AUC 0.5000; competing-risks 1.2509 /
           0.5990 on the n=140 seed-0 sample) but the LLM arm is missing, and
           a table without it has no variance column -- which is the entire
           argument the block exists to make. Gemini's free tier caps
           requests per day PER MODEL and the caps differ: flash-lite 500/day,
           flash 20/day, both measured from real 429 bodies and both
           confirmed exhausted by direct probe. Three runs were attempted;
           POSTMORTEM.md incidents 7 and 8 record what each cost, including
           400 completed calls discarded because results were written only at
           the end of a run.
           NOT a clean history. The originally pinned `--n 200 --repeats 5`
           plans 600 calls per model and could never have completed on this
           tier -- it was sized against a limit that had not been read. The
           first fix encoded DAILY_QUOTA_PER_MODEL = 500, which was the same
           error one level up and would have waved a 440-call flash run past
           a cap of 20; it is now a measured per-model table defaulting an
           unknown model to the SMALLEST observed cap.
           Also NOT clean: stats-reviewer found the benchmark was measuring
           the wrong thing. Per-class AUC is 0.534/0.569/0.487/0.714, i.e.
           DEAD is BELOW CHANCE by construction, so macro AUC had no power to
           decide "the LLM must lose" either way; the headline moved to
           multiclass log loss with per-class Brier, an intercept-only null
           arm, and a mandate-cluster bootstrap CI. Worse, the prompt defined
           p_still_pending as surviving "to a further slot" while 74 of 146
           slot-4 rows carry exactly that label -- instructing the model to
           zero out the correct answer on half of them, biasing the result in
           THIS PROJECT'S OWN FAVOUR. Every finding was reverified by
           measurement before being acted on. Full account: DECISIONS.md,
           2026-08-31, five B12 entries.
           To close: run the two commands in DECISIONS.md's "To finish it"
           block after the quota resets. The call cache added here banks
           partial work (21 flash answers already on disk), so the resume
           does not re-bill. Whether to buy paid quota, ship flash-lite
           alone, or shrink the flash arm is a decision for the human.
           ENVIRONMENT, fixed the same day and affecting how much of this
           block is actually covered: 128 Postgres tests had been SKIPPING
           because Windows reserved the dynamic TCP range 5341-5440, which
           contains 5432, so the mrdb container could not bind. Re-created on
           port 15432 with the same volume (B3's 2 ingested_event and 2
           webhook_event rows verified intact afterwards). The suite now runs
           872 tests with ZERO skips, so this block's shadow_ledger DDL and
           store.append_shadow() are covered by the real suite rather than
           only by the throwaway containers used while building it, and
           .
un.ps1 verify passes 5/5 including the postgres check that had
           been failing. DECISIONS.md, 2026-08-31.
           TICKED 2026-08-31 BY EXPLICIT HUMAN DECISION, on the stated
           criterion "tick B12 if the only work left is due to the Gemini
           quota being reached." That criterion was verified true before
           ticking: shadow mode complete, benchmark module complete and
           tested (34 offline tests), stats and null arms measured, and the
           ONLY outstanding item blocked solely by the daily free-tier cap.
           READ THIS BEFORE CITING THE GATE. The tick records a scope
           decision, NOT a measurement. Clause 2 (shadow) is verified in the
           ordinary way. Clause 1 is PARTIALLY verified: the benchmark table
           is in DECISIONS.md, but the variance column that clause names
           explicitly is EMPTY, because no LLM arm has been run. Nobody may
           later read this [x] as evidence that run-to-run variance was
           measured -- it was not, and DECISIONS.md's "The benchmark"
           section still carries a Status: incomplete banner saying so. The
           two files agree deliberately; if they ever disagree, DECISIONS.md
           is the truth.
           What would make clause 1 fully true, unchanged: run the two
           commands in DECISIONS.md's "To finish it" block once quota
           allows, then replace the pending rows. The call cache banks 21
           flash answers so the resume does not re-bill. B16 in particular
           needs this number -- the variance column is the AI-judgment
           evidence the submission rests on, and it is the one figure this
           block did not produce.
           Does not touch eval/frozen/. -->
- [x] **B13** ★ stress regimes + report: every number reproducible by one command; at least one regime where we lose, explained
      <!-- 2026-08-31, MET, both clauses, verified rather than asserted.

           CLAUSE 1 -- reproducible by one command. `.\run.ps1 eval` runs
           eval/run.py over 6 regimes x 3 arms x 2 profiles x 2 policies
           (64 cells) and then eval/report.py, which computes NOTHING: every
           number in reports/regimes.md is read out of reports/regimes.json,
           so the report cannot drift from the run. Verified by DELETING
           regimes.json, regimes.md and all three figures and re-running from
           empty -- identical numbers. Checked, not assumed: eval/frozen/ is
           untouched (git status clean; FREEZE_HASH still 4daf9ec56db2), the
           regimes are config OVERLAYS on the frozen sim_config.yaml and an
           overlay key the base config lacks is refused rather than ignored.

           CORRECTION, 2026-09-03 (B16). Clause 1 as ticked said "identical
           numbers", and that was true of every number. It was NOT true of
           the artifact bytes: each cell carried a `seconds` wall-clock
           timing, so a reader checking the claim by hashing regimes.json --
           rather than by reading its numbers -- got a mismatch and no way
           to tell jitter from a real divergence. Nothing read the field.
           It is still measured in memory and is no longer serialised
           (UNSERIALISED_CELL_FIELDS, eval/run.py), pinned by three tests in
           tests/eval/test_run_regimes.py including two full runs of the
           same seed compared as strings. Re-verified by re-running
           `.\run.ps1 eval` end to end: all 1024 cells equal the previous
           artifact field-for-field once `seconds` is dropped, and
           reports/regimes.md is byte-identical. The gate holds; the
           evidence for it is now checkable the cheap way.

           CLAUSE 2 -- at least one regime where we lose, explained. Three,
           found mechanically by report.py rather than chosen by hand:
             * festival_season -- the largest money loss: -57.6% (nominal)
               and -62.2% (misspecified) against the ladder, 38 and 48
               mandates that would have paid, Rs 20.2 lakh and Rs 23.8 lakh.
               The AFA cliff (clause 8a) pushes high-value mandates onto the
               re-auth path; the ladder has no concept of the cliff and
               collects money it is not entitled to attempt for.
             * stacking_spike -- a PRE-REGISTERED HYPOTHESIS THAT FAILED. The
               spec predicted the engine would cause fewer iatrogenic
               failures by attempting less. It caused MORE: 137 -> 142. Same
               inversion under baseline/coupled (115 -> 125) and
               issuer_outage/coupled (39 -> 44). Fewer attempts do not imply
               less household contention, and the engine models households
               not at all.
             * delayed_salary/nominal -- -21.7% money, the regime written in
               advance as the one we expected to lose.

           THREE FINDINGS THAT ARE LOSSES FOR THE THESIS, NOT FOR THE GATE.
           Recorded here because a future session must not rediscover them as
           good news. Full detail in DECISIONS.md, six B13 entries.
             1. OFFER = 0 in all 32 engine cells, with the REAL conformal gate
                live (new ConformalCauseGate; B8's FullSetGate stub is no
                longer what is being measured). Coverage 0.960-0.985 vs a 0.95
                target, but achieved by returning 2.82-2.90 labels out of 3 --
                B6's own trivially-satisfiable failure mode. After one slot-1
                decline a true WONT_PAY mandate's belief points at
                CANT_PAY_NOW. So every preserved-mandate win in the report is
                won by RESTRAINT, and none of it by identifying exit intent.
                The off-ramp lane -- one of the three the project is built on
                -- contributes nothing measurable here.
             2. No timing discrimination, ANYWHERE. Every committed attempt
                lands on day 2, the earliest legal day, in all 5 regimes, all
                3 arms, both profiles. This SETTLES B12's open question:
                SAME_ACTION_DIFFERENT_DAY = 0 was not an artifact of looking
                only at the first decision point. The hazard model's only
                temporal feature is in_salary_window, so backward induction
                cannot prefer day 4 to day 2.
             3. strict and permissive are byte-identical in all 32 cell pairs.
                The profiles are read by allocator.solve(), but the
                constraint never binds at an optimum that is always "earliest
                legal day" -- a consequence of finding 2, not independent of
                it. The two RBI interpretations are indistinguishable on this
                evidence, which is not the same as the ambiguity not
                mattering.

           REVIEWED AFTER THE FIRST TICK, and the numbers above are the
           POST-REVIEW ones. payments-domain and stats-reviewer were run
           against this block; between them they invalidated two published
           numbers and found three checks that could not fail. The gate still
           holds -- both clauses re-verified after the fixes -- but several
           conclusions in the first draft were WRONG. Full detail in
           DECISIONS.md, three further B13 entries. The corrections that
           change what this block claims:
             * Finding 1 above understated it. OFFER = 0 is ARITHMETIC, not
               measurement: the proxy decline alphabet gives WONT_PAY an
               identical likelihood under both symbols it can emit, so
               P(WONT_PAY) is pinned at 0.10 and the singleton is unreachable
               for any alpha, seed or regime. The off-ramp lane is UNTESTED,
               not tested-and-negative, and retry_storm's hypothesis about it
               is vacuous rather than falsified.
             * The gate's reported coverage was an artifact. Its smoothing key
               was derived from the belief, which collapsed the WONT_PAY
               p-value to a hash of a constant; and coverage was scored over
               only the 200 slot-1 beliefs rather than the ~4,700 queries the
               gate actually receives. Both fixed. Real marginal coverage is
               0.876-0.925 against a 0.95 target -- the gate UNDER-covers.
               The earlier 0.980 should not be cited.
             * The headline was unfalsifiable and is now bounded. `null`
               (never attempt) preserves 200/200 in EVERY cell; `one_shot`
               (one attempt, no model, no belief, no gate) preserves more
               than the engine in 14 of 16 cells while spending fewer
               attempts. Both are now first-class arms in the table, the
               figures and the README. The test asserting the engine always
               spends fewer attempts than the ladder was DELETED: it pinned
               the confound by test.
             * issuer_outage's own pre-registered falsification criterion
               (false-REAUTH) was never computed. Now measured: 810 of 1,494
               REAUTHs go to mandates whose true cause is not CANT_PAY_EVER.
             * Finding 3 above was right about the fact and wrong about the
               reason: strict and permissive are provably the SAME FUNCTION
               (the lead term is absorbed by the monotonicity clamp at every
               reachable context), not merely equal at this policy's optimum.
             * Every rupee in the report was float-divided outside money.py,
               and guard_invariants scanned money only in PROTECTED_DIRS --
               i.e. exactly where the rule already held. Both sides fixed;
               the guard now scans MONEY_DIRS.

           TWO CHECKS THAT COULD NOT FAIL, fixed:
             * `.\run.ps1 test` returned 0 on a RED suite -- bare `& $Py` in a
               switch branch does not set the script exit code -- which made
               CLAUDE.md's definition-of-done step 3 unfalsifiable. Proven
               with a repro, then fixed via the already-existing Invoke-Step.
             * The golden-set freshness advisory tested file existence, so a
               quota-killed run leaving 1 of 30 rows cached reported "current".
               Now compares against the golden set's rows. Currently, and
               honestly, PARTIAL.

           STILL NOT DONE, not claimed: the belief layer cannot conclude
           CANT_PAY_EVER from an observed dead instrument (502 post-terminal
           re-solves returned ATTEMPT); the 24h pre-notification lead is not
           modelled as a real lead; the five regimes never perturb
           CANT_PAY_EVER.base_dead, never model external debit-order
           competition, and never stress the decline-signal channel. The
           regimes are deliberately UNCHANGED -- they are pre-registered, and
           choosing a sixth now, knowing what breaks the engine, is what
           pre-registration exists to prevent.

           SEED SWEEP, 2026-09-01. Everything above was originally seed 0
           with no error bar. `.\run.ps1 eval` now runs 8 seeds (1024 cells,
           ~15 min) and the report's headline is a per-seed SIGN TEST over
           256 paired comparisons rather than the averaged table, because a
           mean can hide a comparison that flips between seeds. Result:
             * vs the LADDER the thesis holds and is stable -- the engine
               preserves more in 256/256 and spends fewer attempts in
               256/256, recovering more money in only 36/256. Deliberately
               recovering less to protect lifetime value is the thesis.
             * vs ONE_SHOT it does not. A single attempt on day 2 with no
               model, no belief and no gate preserves more mandates than the
               engine in 214/256, and the engine spends MORE attempts in
               226/256. The engine's only edge is money, 146/256 (57%). On
               two of the three bars a policy with no model in it beats this
               one -- and more seeds made that finding STRONGER, which is the
               opposite of what a noise explanation would predict.
           The defensible claim is therefore against the incumbent ladder,
           not against every trivial baseline, and both the report and the
           README now lead with that rather than with the ladder comparison
           alone. The sweep also produced worse figures than seed 0 had:
           per-class conformal coverage down to 0.741 (CANT_PAY_EVER), and
           4,032 post-terminal ATTEMPTs.

           915 tests pass, zero skips; guards clean on 122 files; run.ps1
           verify 5/5. Does not touch eval/frozen/. -->
- [x] **B14** ∥ dashboard: merchant + acquirer views; per-mandate drill-down shows belief, chosen slot, binding constraint, conformal set, ledger trail
      <!-- 2026-09-01, CLOSED. All six sub-conditions verified against a
           running instance (Vite dev server, port 4317, real data).
           MERCHANT VIEW: honesty callout listing six disclosed limitations
           (vs one_shot, OFFER=0, false re-auth, post-terminal, no timing
           discrimination, profiles tie); three-bar chart for four policies
           (engine, ladder, one_shot, null) over 8 seeds / 256 paired
           comparisons; batch table of 200 mandates with outcome/category
           filters; clickable rows opening the drill-down.
           ACQUIRER VIEW (clause 10c): regime/profile/seed selectors over
           1024 cells; attempt budget and AFA cliff table per arm x policy;
           error table (false re-auth, attempt after terminal, missed
           recovery, false off-ramp) for engine cells; conformal gate table
           (marginal coverage, target, mean set size, singleton rate,
           {WONT_PAY} rate, worst class); invariant violation section.
           PER-MANDATE DRILL-DOWN, all five fields:
             (1) BELIEF: three-cause posterior as horizontal bars with
                 numeric probability, dimmed when outside the conformal set.
             (2) CHOSEN SLOT: "slot N · day D" in decision header.
             (3) BINDING CONSTRAINT: kv field, null → "none — an unforced
                 value comparison". 17 of 316 decisions have one.
             (4) CONFORMAL SET: brace notation with excluded causes
                 struck-through; all-three → "excluded nothing — the set
                 is uninformative here".
             (5) LEDGER TRAIL: full table (state, action, amount, outcome,
                 decline_class, reason, idempotency key prefix). 159 of
                 200 mandates have rows; 41 never spent a slot.
           Post-terminal re-solves (272 ATTEMPTs on dead instruments) are
           annotated in the drill-down rather than hidden.
           DATA PIPELINE: scripts/dashboard_data.py copies reports/ to
           dashboard/public/data/ (manifest.json, results.json,
           regimes.json, mandates.json); zero computation in the
           dashboard. TypeScript compiles clean. Vite production build
           succeeds (320ms). guard_invariants --all exit 0 on 122 files.
           Does not touch eval/frozen/. -->
- [x] **B15** ∥ landing page: 60fps on a mid laptop; reduced-motion fallback; canvas-failure fallback; counters wired to real report output, not hard-coded
      <!-- 2026-09-03. Three of four criteria verified in a real browser
           (headless Chromium 151/152 over CDP, this machine's GPU via ANGLE
           d3d11), screenshots in the session scratchpad:
             * reduced motion  -- Emulation.setEmulatedMedia forced; the
               static three-frame storyboard renders, 200 dots per frame, and
               its captions carry the same figures as the scene.
             * canvas failure  -- launched with --disable-gpu. This FAILED
               first time: react-three-fiber does not throw during React's
               render pass when context creation fails, so
               CanvasErrorBoundary never fired and the reader got a black
               rectangle with captions floating over nothing. Fixed by
               probing WebGL before mounting the canvas
               (site/src/hooks/useWebGLSupport.ts); re-verified.
             * counters        -- site/ssr-check.tsx asserts the real figures
               reach the output AND that PLAN.md's placeholders do not;
               verified to fail by reintroducing "14 mandates lost".
           The fourth, measured over CDP while scrolling the scene:
             * before: 320 samples, median 16.7ms (59.9fps), p95 16.9ms,
               WORST FRAME 283ms at scene entry -- a visible freeze, and
               exactly what the user reported ("gets stuck when scrolling,
               then smooth").
             * after:  358 samples, median 16.7ms (59.9fps), p95 16.9ms,
               worst frame 33.4ms. One dropped frame instead of a freeze,
               and 38 more frames delivered in the same 6s window.
           Three causes, all fixed in Scene.tsx: `phase` state sat in the
           component rendering <Canvas>, so the first phase change
           reconciled the whole R3F tree mid-scroll (Canvas now lives in a
           memoized child); the scroll handler called
           getBoundingClientRect() per event against a 520vh container
           (Motion's useScroll replaces it); and shaders compiled on the
           first visible frame (<Warmup> calls gl.compile at mount).
           2026-09-03, later session: the real-display run that was owed.
           HEADED Chrome 152 driven over CDP against the production build
           (vite preview), window visible, throttling of occluded windows
           disabled so rAF reflects presented frames. GPU as reported by the
           page itself: ANGLE (Intel Iris Xe Graphics, D3D11) -- integrated
           graphics, which is what "a mid laptop" means. Frame deltas sampled
           in-page with requestAnimationFrame while scrolling the narrative
           section end to end over 8s:
             run 1: 480 samples, median 16.7ms (59.9fps), p95 17.0ms,
                    worst frame 17.8ms, frames >20ms: 0, dropped (>32ms): 0
             run 2: 479 samples, median 16.7ms (59.9fps), p95 17.0ms,
                    worst frame 17.7ms, frames >20ms: 0, dropped (>32ms): 0
           Better than the headless figure it replaces (worst frame 17.8ms
           vs 33.4ms), so the B15 fixes in Scene.tsx hold on real hardware.
           The other three criteria were re-verified first-hand in that same
           real browser rather than carried over on trust:
             * reduced motion -- Emulation.setEmulatedMedia forced: canvas
               NOT mounted, 3 storyboard frames present, 142/110/29.0%/45.3%
               all on the page, PLAN.md placeholders absent.
             * canvas failure -- second instance launched --disable-gpu
               --disable-software-rasterizer: canvas NOT mounted, the HTML
               fallback rendered (not the old black rectangle), same real
               figures present, placeholders absent.
             * counters -- npm run render-check green against the staged
               results.json, which is byte-identical to reports/results.json.
           Also removed in this session: site/src/assets/{hero.png,react.svg,
           vite.svg}, three unreferenced leftovers, closing PLAN_DETAIL's
           "Vite demo assets replaced wholesale".
           Verified after those deletions: build 677ms, render-check green,
           guard_invariants --all exit 0 on 125 files, 785 passed / 141
           skipped. Does not touch eval/frozen/.

           CORRECTION, same day, after the tick. The four criteria above do
           hold and were re-verified on the shipped build. But the evidence
           for them came from browsers launched with a throwaway
           --user-data-dir, i.e. a CLEAN PROFILE WITH DEFAULT FLAGS, and that
           is not the browser a reader has. On the human's own Chrome, same
           laptop, same build, the page served the no-WebGL fallback instead
           of the scene: useWebGLSupport asked for a context with
           failIfMajorPerformanceCaveat: true and read the refusal as "no
           GPU", when that flag only means the browser is being conservative.
           A clean automation profile is never in that state, so no run here
           could have caught it -- the strict-pass and no-context cases were
           both tested, the case between them was not. Found by the human in
           one screenshot after four green automated runs.
           Fixed in fe10ad2: the probe is tiered (strict, then looser terms
           with a renderer-string check, and only a software rasteriser
           reaches the fallback), and Scene.tsx's Canvas gl config follows
           the tier rather than re-failing independently. The degraded path
           is now verified by overriding getContext over CDP to refuse every
           strict request and asserting the canvas still mounts. Logged as
           POSTMORTEM Incident 9. The tick stands; this note records that it
           was taken on a non-representative browser. Relevant to B16: the
           video will be captured in the same kind of throwaway browser. -->

- [ ] **B16** ship: README has "What this can't do" with ≥4 items; video under 5:00; three takes max
      <!-- 2026-09-03, PARTIAL. Two of the three README deliverables are
           closed; the video is not, so this gate stays OPEN.

           DONE -- "What this can't do", 8 items, README.md:153. Every one
           is reproducible from reports/ and the first is the one that hurts:
           the off-ramp never fires, which makes the lane the project exists
           for untested rather than tested-and-negative. The clause asked for
           4; withholding the other 4 to look better would defeat the point
           of the clause.

           DONE -- the two remaining README placeholders, which were the real
           open items behind this gate:
             * "## The problem" was a parenthetical Day-12 stub. Now written:
               the ladder's defect is one rule applied to two customers with
               nothing in common, and the three India-specific constraints
               (RBI 6(a)'s 24h wall, 6(c)'s per-notification opt-out, NPCI's
               budget of four ever) that stop a Stripe-shaped answer from
               transferring. The 20-40% involuntary-churn figure is marked
               as an industry range, NOT as something measured here.
             * "## Architecture" promised an Excalidraw diagram that did not
               exist. Shipped as docs/architecture.svg -- hand-authored SVG,
               so it renders inline on GitHub, diffs as text, and needs no
               account to open. The colour split IS the argument: blue core
               moves money and may not import an LLM; amber edge reads
               language and returns one symbol. Rendered and inspected at
               1200x812 rather than assumed correct.

           NOT DONE -- video under 5:00, three takes max. Nothing recorded.
           Incident 9's lesson applies directly and is written down there:
           the capture will run in a throwaway clean-profile browser, which
           is exactly the configuration that produced a green verification of
           a page a real reader could not see. Whatever the video shows, one
           pass must be watched on the human's own everyday browser before
           this gate is ticked.

           ALSO DONE, same block, at the human's direction.

           * The LLM benchmark can run at all now. `.
un.ps1 bench` had been
             refusing to start since the per-model quotas were measured: the
             flash arm plans 440 calls against a 20/day cap, which is 22 days.
             fit_plan_to_quota shrinks a plan instead of refusing it, so flash
             runs as a 20-call variance-only probe with no accuracy pass and
             flash-lite keeps its full 440/500. README limitation 9 records
             that the accuracy numbers are budget-bound. Verified offline; no
             live calls spent. Quota rolls over 2026-09-04 12:30 IST.
           * Both apps now explain themselves. The human's call, and it
             REVERSES this repo's earlier decision to keep caveats out of the
             landing page -- see DECISIONS.md, 2026-09-03, "Explain, do not
             hide". Findings stay on the page; every assumed term carries a
             definition that appears on hover, focus or tap; the landing page
             gained a "How it decides" section whose fifth card is the
             off-ramp finding. ssr-check.tsx's guard was rewritten to assert
             the opposite of what it used to, rather than deleted -- a guard
             encoding a superseded decision looks like a reason. -->

---

# Post-B16 remediation gates

Written **before** any of this work started (R0, 2026-09-04), for the same
reason the Day-1 freeze exists: a pass condition invented after seeing the
result is not a pass condition. These close findings the B13 seed sweep and
the B14 dashboard surfaced and disclosed rather than fixed.

Each gate names a number or an artifact that can be checked. "The code
exists" is not a gate here either.

- [x] **R1a** design matrix carries amount and category: `_design_matrix()`
      builds an amount-band term and a category term, selectable via a new
      `WIDENED_FEATURE_COLUMNS` alongside the unchanged default
      `FEATURE_COLUMNS` — the model is not required to adopt a widened
      design as its production default, only to fit it and report what
      happens; held-out multiclass log-loss for the widened design vs the
      narrow one is reported via a properly-powered inferential test
      (pooled out-of-fold per-row differences clustered by mandate_id — the
      20-seed stability sweep alone is NOT independent-sample evidence, see
      `eval/model_fit_report.py`'s own docstring), **with the result stated
      whichever way it goes** — a widened design that does not beat the
      narrow one is a finding, not a failure, and must not be re-framed as
      one
      <!-- 2026-09-04, CLOSED. src/model/competing_risks.py:
           WIDENED_FEATURE_COLUMNS (amount_band_2/3/4,
           category_insurance_premium/mutual_fund/credit_card_bill) fit via
           fit(feature_columns=...); FEATURE_COLUMNS stays the unchanged
           default. eval/design_matrix_comparison.py -> reports/
           model_defensibility.md, Phase A: PRIMARY test (pooled
           out-of-fold log-loss, clustered by mandate_id, 12316 rows/7154
           mandates) mean(widened-narrow)=+0.00103, clustered SE=0.00036,
           t=+2.88, df=7153, p=0.0040 -- WIDENED IS MEASURABLY WORSE at 95%
           confidence, not merely no better. 0/18 new coefficients (6
           columns x 3 outcomes) have a 95% CI excluding zero. Matches the
           DGP directly: eval/frozen/simulator.py's _draw_outcome never
           reads category in any arm and reads amount_paise only inside
           coupled's household-balance side mechanic, never in the base
           hazard logits fit() trains against (nominal). Two real bugs
           found and fixed during this work, both disclosed in
           DECISIONS.md and reports/model_defensibility.md rather than
           quietly corrected: (1) the FIRST version of this comparison used
           a normal-approximation |t|>2 threshold on a 5-fold-mean t-stat
           with only 4 degrees of freedom (correct 95% critical value
           2.776, not 2.0) and published a "does not beat" verdict that
           was not actually significant (p=0.076) -- replaced with the
           pooled/clustered test as primary; (2) the amount-band cut
           points were documented as quartiles of "the only range fit()
           trains on", which is false -- elevated categories'
           higher AFA limit (clause 8(b)) lets 4.42% of the training
           sample exceed that stated range, confounding amount_band_4 with
           category. Neither bug changes the null finding; both are
           disclosed in the report and the module docstring. stats-reviewer
           reviewed the full change (design, tests, script) and confirmed:
           no leakage (amount/category are static per-mandate, present at
           slot 1), censoring discipline intact, CV split genuinely
           mandate-disjoint for both models, no data-snooping in the cut
           points (fixed arithmetic on the config's own documented range,
           chosen before any fit ran). Also found and fixed: a test-fixture
           bug (idx % 4 was a perfect bijection with the outcome within
           every cell, given the fixture's own 25-row inner loop --
           fabricated a large false amount effect purely from fixture
           construction; fixed to idx // 5, verified no bijection remains)
           and a silent-wrong-answer gap in production code (an
           unrecognized/null category scored as the reference level with
           no error; now raises). Does not touch eval/frozen/. -->

- [x] **R1b** `reports/model_defensibility.md` exists and reports per-cause
      coefficients with confidence intervals for issuer, instrument type and
      mandate age, fit on a simulator that actually generates them; a guard
      asserts `eval/run.py` never imports that simulator
      <!-- 2026-09-04, CLOSED. `eval/sim2.py` -- a second, non-frozen
           simulator whose DGP actually varies dead-hazard by issuer_id/
           instrument_type and CANT_PAY_NOW's recovery hazard by
           mandate_age_days -- built, tested (59 new tests), and run:
           40-seed corpus, 8,000 mandates, 12,242 estimable rows, 7/18
           issuer/instrument/age coefficients significant at 95% (the
           mirror-image of Phase A's honest null). `SIM2_IMPORT` guard in
           scripts/guard_invariants.py denies eval/run.py importing
           eval.sim2, scoped and tested both directions.

           NOT A CLEAN HISTORY. stats-reviewer found the report's own FIRST
           interpretation was wrong, not just incomplete, on all three of
           its severity-flagged findings: (1) the paragraph explaining why
           mandate_age_years came out significant on DEAD/OPTED_OUT (despite
           being coded as a direct CANT_PAY_NOW-recovery-only effect) argued
           a mechanism that predicts a NEGATIVE coefficient, while the table
           three lines above it reports positive numbers -- re-derived
           independently here (MNLogit's reference category is
           STILL_PENDING; raising CANT_PAY_NOW's recovery drains the
           survivor denominator faster than the dead numerator, so pooled
           dead-vs-survive log-odds mechanically RISE) and confirmed by
           direct analytic marginalisation (+0.13/year vs fitted
           +0.175/+0.170). (2) A second, structurally identical artifact
           (issuer_gamma -- coded as dead-only -- also significant on
           RECOVERED, +0.1775) was present in the original table but never
           flagged, and counted as an unqualified "signal" win instead.
           (3) The fitted CIs do not cover the DGP's own coded additive
           logits for any of the three directly-coded effects (attenuated
           1.7-3.2x by cause-marginal pooling) -- true and expected (the
           same cause-specific-vs-cause-marginal distinction
           src/model/competing_risks.py's own docstring already names for
           slot3_x_in_salary_window), but undisclosed in the first version.

           ALL THREE FIXED, verified independently before being acted on
           (not taken on the reviewer's word): eval/sim2.py's report writer
           gained a "Direct effects vs cause-marginal artifacts" section
           (structural classification, not hand-counted per run), a
           "fitted CIs do not cover the DGP's own coded values" table
           (computed live from the same constants the DGP uses, so it
           cannot go stale on a rerun), an in-sample-only disclosure, and a
           closing paragraph naming the least-comfortable assumption:
           initial_cause is independent of issuer/instrument/age in this
           corpus, so every artifact and every attenuation measured here
           has zero confounding -- real correlated covariate data could
           flip these signs, not just shrink them. Re-ran after the fix:
           byte-identical 18-row coefficient table (the fit was never
           wrong, only its explanation); Phase A's section re-confirmed
           byte-identical.

           TWO MORE FINDINGS, also fixed. A dangling citation: three sites
           (module docstring, two test docstrings) cited a "DECISIONS.md
           R1b entry" margin derivation that did not exist yet -- now
           written (this file's 2026-09-04 R1b entry), with the aggregate
           dead-rate gap re-derived independently here by analytic
           marginalisation (+6.75pp issuer, +6.42pp instrument, consistent
           with stats-reviewer's own +7.38pp/+6.70pp). And a real flake
           risk: the two DGP hazard-difference tests' original 20-seed
           window measured only +5.51pp/+5.69pp -- ~0.26 SD above the 5pp
           floor against a measured ~1.9pp/1.4pp window-SD, a confirmed
           ~7-10% flake rate despite passing today. Widened to 150 seeds
           (empirically verified: 1.57s runtime, measures +7.83pp/+7.18pp,
           a real ~2.5-2.9 SE margin this time).

           Disclosed, not fixed (LOW/LATENT, none blocking): the SIM2_IMPORT
           guard is direct-textual-import matching only (same pre-existing
           limitation SRC_LLM_IMPORT already has, now documented in-line);
           Sim2Episode omits eval.corpus.Episode's `schedule` field (latent
           -- nothing calls hazard_tensor() on a sim2 episode today, and the
           docstring now says one must be added first if that changes); a
           guard-coverage note (not a live bug) on eval/sim2.py's amount/
           ceiling locals evading FLOAT_MONEY's identifier-prefix matching
           the same way the frozen simulator's identically-shaped code
           already does.

           Full account, all six findings, all independently re-verified:
           DECISIONS.md, 2026-09-04, "R1b" entry. 1065 tests pass, 1
           skipped; guard_invariants --all clean on 127 files. Does not
           touch eval/frozen/. -->

- [x] **R2a** `n_attempt_after_terminal == 0` across all 256 engine cells,
      **proven by a test that constructs the sequence and fails against the
      pre-R2 code** — a counter that reads zero because the path was deleted
      is not evidence
      <!-- 2026-09-04, CLOSED, after a review-driven redesign mid-block.
           n_attempt_after_terminal == 0 confirmed across all 256 engine
           cells on the fresh 8-seed sweep. tests/eval/test_run_regimes.py:
           three scripted-sequence regression tests (_ScriptedSimulator,
           forcing an exact STILL_PENDING->DEAD / OPTED_OUT / RECOVERED
           sequence) verified to FAIL against the pre-fix code before the
           fix landed -- not merely written to pass.

           THE FIX, AS SHIPPED: src/policy/stopping_rules.py's
           AllocationContext gained `instrument_dead: bool = False`;
           permitted() denies ATTEMPT when set (same shape as the existing
           REVOKED rule); with_terminal(outcome) sets it for DEAD or the
           EXISTING opted_out for OPTED_OUT. eval/run.py's terminal branch
           now calls this for both outcomes (previously OPTED_OUT's
           re-solve was skipped ENTIRELY -- _proxy_decline_class() returns
           None for it -- so no decision was ever recorded).

           A FIRST VERSION OF THIS FIX WAS WRONG, found by stats-reviewer
           and payments-domain BEFORE this gate was ticked, not after:
           belief.observe_terminal() originally collapsed belief to an
           exact DEGENERATE (1.0/0/0) posterior, on the reasoning "DEAD
           means CANT_PAY_EVER -- that is what the cause label MEANS, not
           a hypothesis about it." Checked against eval/frozen/
           sim_config.yaml's own generative process (200-seed direct
           simulation, cross-validated on a disjoint 300-seed range) and
           found FALSE: P(CANT_PAY_EVER|DEAD) = 0.899, P(WONT_PAY|
           OPTED_OUT) = 0.904 -- roughly 10% of each terminal outcome has
           a DIFFERENT true cause. A degenerate 1.0 was additionally
           IRREVERSIBLE: cause_map._PRIORS has no zeros, so update() on an
           exact (0,1,0) belief returns it unchanged forever (0 * anything
           = 0) -- an absorbing state dormant only because no belief
           survives past a mandate's own terminal outcome in this eval
           harness, a real hazard for R4's future multi-cycle persistence.
           FIXED: observe_terminal()'s signature changed to
           observe_terminal(cause_probs: Mapping[Cause, float], *,
           source_version) -- no prior-belief parameter at all (matching
           init()'s own shape) -- and eval/run.py now passes the MEASURED
           distributions above via a new, fully-derived-and-cited
           _TERMINAL_OBSERVED_CAUSE_PROBS constant. Re-ran the full sweep
           after the fix: EVERY action count (n_reauth, n_stop,
           false_reauth_count and all R2b splits) is BYTE-IDENTICAL to the
           degenerate-collapse version -- REAUTH's economics dominate STOP
           at 90% confidence exactly as they did at 100%, for every
           realistic amount in this corpus -- so the correction cost
           nothing in policy quality while removing a false certainty
           claim and an irreversible state.

           TWO MORE BUGS FOUND BY THE SAME REVIEW PASS, both fixed:
           (1) src/policy/allocator.py's _binding_constraint() did not
           check ctx.instrument_dead at all -- a REAUTH forced ONLY by
           this rule wrote binding_constraint=None to the Plan, which
           src/execute/shadow.py renders as "(none -- decided on belief
           and expected value)": the ledger stating a hard-forced decision
           was a free economic choice. Fixed (checked first, matching
           AFA_CLIFF's precedence); regression test added
           (test_instrument_dead_denies_attempt_and_names_itself_as_the_
           binding_constraint). (2) Fixing OPTED_OUT's re-solve meant the
           conformal gate was queried, for the first time, on a
           RETROSPECTIVE belief already collapsed by observe_terminal() --
           not exchangeable with calib_conf's live-inference calibration
           pool, so mixing it into coverage/singleton-rate MEASUREMENT
           would contaminate exactly the diagnostic the off-ramp's safety
           claim depends on (payments-domain: "the entire nonzero
           singleton rate is arithmetically one query per opted-out
           mandate... the gate has still never produced a {WONT_PAY}
           singleton from a belief it actually reasoned about"). Fixed:
           _RecordingGate now tags each query live/retrospective (via the
           `;observed=terminal` provenance marker) and
           _score_recorded_queries() computes coverage_marginal/
           singleton_rate/singleton_wont_pay_rate/mean_set_size/
           coverage_per_class over LIVE queries only -- confirmed on the
           fresh sweep: singleton_wont_pay_rate is back to exactly 0.000
           in every one of 256 engine cells, with 20,592 retrospective
           queries correctly excluded and disclosed via the new
           coverage_n_retrospective field (matches payments-domain's own
           independently-cited 1,162,576 total query count exactly: 1,141,984
           live + 20,592 retrospective). The Plan object's own
           conformal_set audit field is UNCHANGED -- a specific mandate's
           drill-down CAN still show a retrospective {WONT_PAY} singleton
           (46/200 in the headline cell) next to binding_constraint=
           OPTED_OUT; dashboard/src/Drilldown.tsx's comment corrected to
           explain this rather than claim it never happens.

           DISCLOSED, not fixed (out of R2's scope): observe_terminal()/
           with_terminal() have zero callers outside eval/ and tests/ --
           src/execute/shadow.py builds one AllocationContext per mandate
           with opted_out hard-coded False and calls solve() exactly once,
           no multi-attempt cycle exists in src/ to wire this into. This
           fix is proven true of the evaluation harness; R4 (the cycle
           orchestrator, not yet built) is where it must reach the money
           path. README's "What this can't do" item 5 rewritten to
           disclose this rather than the now-fixed 4,032 count.

           Reviewed by stats-reviewer, compliance-auditor (all 6 checked
           clauses VERIFIED, singleton-on-departed-customer concern
           confirmed inert), and payments-domain (found all of the above;
           confirmed the permitted()/instrument_dead/signature() core of
           the fix is correct and was "the actual fix"). Does not touch
           eval/frozen/. -->
- [x] **R2b** four re-auth numbers published side by side (pre-registered
      `false_reauth_count`, compliance-path count, inference-path false count,
      and the same against `effective_cause`); `false_reauth_count` keeps its
      Day-1 meaning byte-for-byte
      <!-- 2026-09-04, CLOSED. eval/run.py's CellResult: compliance_reauth_
           count (REAUTH via requires_afa() -- clause 8(a)/8(b), legally
           mandatory), false_reauth_inference_count (false_reauth_count
           restricted to the inference route only), and both scored again
           against sim.effective_cause() (false_reauth_count_effective /
           false_reauth_inference_count_effective) for the misspecified
           arm's cause-switching. false_reauth_count/false_reauth_paise
           themselves UNCHANGED -- confirmed via test
           (test_reauth_via_compliance_path_is_not_counted_as_inference_
           false: cell.false_reauth_count == 1, "still true against the
           pre-registered, unredefined criterion"). Fresh 8-seed sweep:
           17,554 REAUTH total; 8,832 false_reauth_count (pre-registered,
           unchanged meaning); 6,784 compliance_reauth_count; **3,022**
           false_reauth_inference_count -- the genuinely-interesting
           number, about a third of the pre-registered count. Effective-
           cause-scored versions slightly lower (8,542 / 2,732), as
           expected. payments-domain confirmed the route attribution is
           correct (requires_afa() at counting time cannot disagree with
           _best_action's own branch: ctx is object-identical between
           decision and count, and amount_paise/category are never
           mutated by with_attempt/with_contact/with_terminal) but flagged
           that after the observe_terminal() redesign, false_reauth_
           inference_count on a post-DEAD re-solve is no longer a pure
           "belief was wrong" signal -- b.dominant()==CANT_PAY_EVER is now
           TRUE on every DEAD-terminated mandate by construction (P=0.899,
           still dominant), so some fraction of this count reflects the
           simulator's own ~10% off-diagonal draw rather than a model
           error. Disclosed in DECISIONS.md rather than re-defined again:
           the number is real and reproducible, its INTERPRETATION as
           "belief was wrong" is approximate, not exact. Also flagged,
           disclosed not fixed here (R5's territory): false_offramp_count
           is structurally 0 (n_offer=0 everywhere), so root CLAUDE.md's
           "report both error costs" pairing is currently one real number
           and one zero by construction, not two measurements. Reviewed by
           compliance-auditor (clause 8(a)/8(b) reasoning: VERIFIED, purely
           a scoring change, no effect on which action the allocator
           chose) and payments-domain. Does not touch eval/frozen/. -->
- [x] **R3** `reports/regimes.md` names the LTV break-even ratio as a ratio to
      mean mandate amount, and every point on the sensitivity curve is
      reproducible by one command
      <!-- 2026-09-04, CLOSED. `python -m eval.ltv_sensitivity` (new,
           separate from eval/run.py -- reuses run_engine_cell()/
           run_ladder_cell() unmodified via dataclasses.replace(costs,
           mandate_ltv_paise=...), zero changes to CellResult or the main
           sweep loop) writes reports/ltv_sensitivity.json; `python -m
           eval.report` reads it (computes nothing) into a new "LTV
           sensitivity" section of reports/regimes.md.

           TWO SLICES, both fixed before the first sweep ran (see
           eval/ltv_sensitivity.py's own docstring for the exact,
           mechanically-applied, non-cherry-picked selection rule --
           restated in full in DECISIONS.md, 2026-09-04, "R3"):
             * HEADLINE (baseline/nominal/strict/seed=0, this project's own
               canonical comparison slice): ZERO crossings across a
               66-point grid (0 to 100,000,000 paise). engine.recovered_
               paise stays below ladder.recovered_paise at every point,
               worst at LTV=0 (-Rs 1,50,470.99) and only getting worse.
               interpolate_crossing() correctly refuses to compute a
               break-even (no sign change exists) rather than extrapolate
               one. THIS IS THE HEADLINE FINDING, not a gap: the gate asks
               to name a break-even ratio, and the honest answer for the
               project's own canonical slice is that none exists -- the
               engine's money deficit here is structural (AFA-cliff
               routing, hazard-informed stopping), not an LTV trade-off,
               and no achievable LTV buys it back.
             * WORKED EXAMPLE (issuer_outage/nominal/strict/seed=0 -- the
               first of the pre-existing 36/256 engine-beats-ladder-on-
               money cells reports/gates.md's own B13 entry already
               measured, found by mechanical search over orderings that
               already exist in this codebase): TWO crossings, a
               non-monotonic rise-then-fall (loses at LTV=0, wins by the
               configured default 180,000, loses again by 500,000).
               Crossings at ratio 0.090 and ratio 0.343 to mean mandate
               amount; the current default (ratio ~0.135) sits inside the
               winning window, consistent with this cell's already-known
               default-LTV win.

           `src.core.money.interpolate_crossing()` (added earlier this
           session, unused until now) got its first real caller here --
           and, checked before building on it, its claimed test coverage
           from a prior session handoff was FALSE (zero references under
           tests/ anywhere). 10 tests added to tests/core/test_money.py
           before any R3 code was written.

           REVIEWED by money-auditor before ticking. One real, fixed
           finding: eval/report.py's rendering read the JSON artifact's
           FLOAT convenience fields rather than the EXACT Fraction strings
           stored alongside them for exactly this purpose -- no rendered
           digit was actually wrong at the 2-3 decimals ever displayed,
           but it defeated interpolate_crossing()'s own reason for
           returning a Fraction. Fixed: the renderer now parses the exact
           strings. Everything else money-auditor checked (exact-arithmetic
           ordering, dataclasses.replace() safety on the frozen
           PolicyCosts, _signed_rupees()'s zero-sign handling, no ledger/
           execute import anywhere in this path, the worked example's
           non-monotonic curve checked against allocator.py's Q-function
           and found economically plausible rather than a bug) came back
           clean.

           Disclosed, not fixed: the worked-example cell search checks
           only seeds 0-2 per (regime, arm, profile), not the full 8-seed
           B13 grid -- a stated scope narrowing, and it raises rather than
           silently widening the search if empty (tested). The 50,000-
           paise grid step is coarse enough that a crossing bracket can
           straddle more than one true discrete allocator decision-flip --
           interpolate_crossing()'s own docstring already frames its
           result as an interpolation between two swept points, never a
           third measurement, and the report repeats that caveat beside
           every crossing table.

           Full test suite: 1086 passed (+21 new), 1 skipped.
           guard_invariants --all clean. Does not touch eval/frozen/. -->
- [x] **R4** one command plans a cycle and a second executes what is due; a
      test drives read → solve → commit → execute end to end against a live
      schema with the clock advanced 24h between the two phases
      <!-- 2026-09-04, CLOSED. `src/execute/cycle.py` (new): `plan_cycle()`
           reads durable state for every registered mandate (a new,
           additive `mandate` table -- no FK from ledger/plan/
           committed_schedule, deliberately, per R4_PLAN.md) and calls the
           already-gated `solve()`/`commit()`; `run_due()`, called >=24h
           later, scans `committed_schedule` for due rows and calls the
           already-gated `execute()`. Neither writes a ledger/
           committed_schedule row directly -- both delegate entirely, so
           R4 is orchestration wiring the B8/B9 core got its first
           production caller for, not new money logic.

           THE GATE'S OWN TEST, driven end to end against a real
           (throwaway) Postgres schema:
           `test_plan_cycle_then_run_due_end_to_end` -- seed a `mandate`
           row, freeze the clock at T, `plan_cycle()` -> assert a `plan`
           row and a `committed_schedule` row with `scheduled_for >= T +
           24h`, advance the frozen clock to that moment, `run_due()`
           against a fake Razorpay client -> assert `INTENT -> SENT ->
           RESULT` and that `charge()` was actually called.

           A SECOND FULL PASS proves `belief.observe_terminal()` and
           `AllocationContext.with_terminal()` their first PRODUCTION
           callers (both had zero outside `eval/`/`tests/` before this):
           `run_due()` resolves an attempt `DEAD` (a decline
           `decline_taxonomy.classify()` maps to `MANDATE_REVOKED`); a
           second `plan_cycle()` call is asserted to commit no new
           `committed_schedule` row and to have used the measured ~0.8991
           `CANT_PAY_EVER` posterior (`;observed=terminal` provenance, not
           a degenerate 1.0) to produce REAUTH -- found, since `plan` has
           no `chosen_action` column, by the absence of a matching
           `committed_schedule` row rather than by a column that does not
           exist (a real bug the test itself caught and the fix corrected
           before this gate was ticked).

           RELOCATED, NOT REINVENTED: `TERMINAL_OBSERVED_CAUSE_PROBS` /
           `TERMINAL_OBSERVATION_SOURCE_VERSION` moved from `eval/run.py`
           to `src/policy/belief.py` -- their natural home, since `src/`
           must never import `eval/` and `cycle.py` is the first `src/`
           caller that needs them. Values and the point-in-time-
           measurement caveat unchanged; `eval/run.py` now aliases
           `belief_mod`'s objects (pinned by an identity test, not just an
           equality one) rather than keeping a second copy that could
           drift.

           THREE SCOPE DECISIONS, each disclosed rather than silently
           made (full reasoning: DECISIONS.md, 2026-09-04, "R4"):
           `_is_eligible()` checks only in-flight-commitment and
           already-`RECOVERED` -- NOT "a prior terminal decision already
           made this cycle" -- because re-solving such a mandate is
           provably idempotent (same durable state -> same
           decision_sha256 -> `commit()`'s own `ON CONFLICT DO NOTHING`),
           and adding that check would need a race-free "latest plan row"
           ordering `plan.created_at` (DB-clock `DEFAULT now()`, no serial
           ordinal) does not safely provide; `_read_context()`'s
           `committed_days`/`plan_day` are exact only when `cycle_start`
           is supplied (`plan_cycle()`'s path) since `run_due()`'s own
           `execute()` call never reaches `solve()` and
           `stopping_rules.permitted()` reads neither field;
           `contacts_sent` tracks `ATTEMPT` contacts only, since
           `REAUTH`/`OFFER` never produce a ledger row to count.

           REVIEWED by both `money-auditor` and `compliance-auditor`
           before ticking (this is the first path that could move real
           money end to end via a real Postgres-backed cycle, and clause
           6(a) is the constraint this module exists to honour). Both
           clean: money-auditor confirmed the ledger-before-money-action
           ordering holds throughout, idempotency keys stay derived from
           deterministic fields only, and the NPCI attempt-cap
           reconstruction cannot be raced past `_is_eligible()`'s
           in-flight gate. compliance-auditor returned 10/10 VERIFIED (6a,
           6c, 8a, 8b, 4c, the NPCI cap, no-cancellation, both profiles
           reachable, contact-frequency/quiet-hours, constant citations),
           zero VIOLATED, zero NOT COVERED -- including confirming
           `plan_cycle()`'s no-try/except-around-`commit()` choice is a
           disclosed "let it crash loudly" design, not a silent gap.

           Full test suite: 1101 passed (+15 new), 1 skipped.
           `guard_invariants` clean on 127 tracked files (`--all`) plus
           the 6 new/changed files explicitly (the new files are
           untracked, so `--all` alone does not see them -- the same
           known limitation named at B9's gate entry). Does not touch
           `eval/frozen/`. -->
- [x] **R5** `n_offer > 0` in at least one regime, with **both** the recovery
      cost and the false-off-ramp rate reported at every point of a
      channel-quality sweep, and the synthetic channel's own ROC published
      beside them. The conformal singleton stays the only firing rule
      <!-- 2026-09-05, CLOSED. n_offer = 1292 across 256 engine cells on the
           republished 8-seed grid, and > 0 in ALL SIX regimes (baseline 284,
           issuer_outage 246, delayed_salary 264, stacking_spike 92,
           festival_season 94, retry_storm 312) -- against exactly 0 in every
           cell before. false_offramp_count = 200 of offramp_scored_count =
           1292, a 15.5% false-off-ramp rate (Wilson 95% CI 12.9-18.5% on the
           TRUE distinct sample -- see the CORRECTED note below; the naive CI
           on the number as first stated would have read 13.6-17.6%). Live
           singleton-{WONT_PAY} rate 0.0395 mean, against exactly 0 before.

           CORRECTED, R5 review pass, 2026-09-05 (stats-reviewer): "256
           engine cells" / "1292" DOUBLE-COUNTS. `strict` and `permissive`
           are byte-identical on every field across all 128 (regime, arm,
           seed) triples (verified: 0 differing cells out of 128) -- the
           pre-existing, already-disclosed consequence of this engine
           having no timing discrimination (regimes.md, "Compliance
           profiles"). The TRUE sample is 128 distinct engine cells, 646
           distinct OFFERs, 100 distinct false ones -- same rate either way
           (0.1548), but stating it on the doubled n=1292 understates the
           real uncertainty: the Wilson interval on the true n=646 (12.9-
           18.5%) is measurably wider than the one on the doubled n=1292
           (13.6-17.6%). The per-regime breakdown two lines up is likewise
           doubled (true per-regime n_offer is half of each figure shown).

           HOW IT WAS MADE REACHABLE, and what that costs in claim strength.
           Two channels, both pre-registered in DECISIONS.md 2026-09-04 (R0)
           and neither dropped: (A) DeclineClass.CUSTOMER_DECLINED, a new
           8th taxonomy class for Razorpay's `payment_cancelled` -- a real
           WONT_PAY-flavoured event that src/classify/decline_taxonomy.py's
           own docstring (finding 3, payments-domain's B3 review) had already
           identified as distinct from MANDATE_REVOKED and which fell through
           to UNKNOWN for want of a home. cause_map prior 0.70 WONT_PAY /
           0.20 CANT_PAY_NOW / 0.10 CANT_PAY_EVER, dominant but deliberately
           not near-degenerate; PRIOR_VERSION v2->v3, TAXONOMY_VERSION
           v1->v2. The anti-conflation guard was NARROWED, not deleted:
           payment_cancelled still cannot reach MANDATE_REVOKED, now by a
           positive classification ordered ahead of it rather than by a
           negative lookahead. (B) The intent channel:
           src/policy/belief.update_from_likelihood_ratio() (generic, no LLM
           and no Outcome knowledge) plus src/execute/intent_channel.py,
           which turns src/llm/intent.py's float into a ratio at a DECLARED
           operating point (threshold 0.70, tpr 0.65, fpr 0.20 -- declared,
           because Cause is latent and has no production label to fit
           against). src/policy/ still never imports src.llm; the score
           crosses as a plain float.

           WHY THE PRE-R5 STATE WAS ARITHMETIC, RE-DERIVED NOT QUOTED: the
           proxy alphabet's two symbols have IDENTICAL WONT_PAY likelihood
           components (0.30 each), so the WONT_PAY likelihood ratio is
           monotone non-increasing and exhaustive enumeration over every
           sequence reachable within the NPCI cap gives max P(WONT_PAY) =
           0.10. tests/eval/test_wontpay_channel.py re-derives that 0.10 by
           direct enumeration rather than citing it.

           THE CHANNEL IS SYNTHETIC AND READS PRIVILEGED GROUND TRUTH. It
           reads SimMandate.initial_cause -- which the policy must never see
           -- and feeds a fabricated observation into the DECISION path.
           That is materially stronger than the score-only privileged read
           false_reauth_count already makes, which is why the gate demanded
           the ROC beside every number. Own provenance stamps
           (eval-wontpay-channel-v1, eval-intent-channel-v1, never
           PROXY_SOURCE_VERSION and never a taxonomy version). Disclosed as
           synthetic in the module docstring, the artifact itself
           (`"synthetic": true` plus a disclosure string), reports/regimes.md
           finding 2 AND its own section, README's "What this can't do" item
           1, the dashboard's Merchant and Acquirer views, and the landing
           page's limit card.

           THE SWEEP (eval/offramp_channel.py -> reports/offramp_channel.json,
           rendered by eval/report.py, which computes nothing). Slice and
           grid and operating point all PRE-REGISTERED in that module's
           docstring before its first run: baseline/nominal/strict, 8 seeds,
           8 (tpr,fpr) points per channel from AUC 0.5 to 1.0, operating
           point tpr 0.60 / fpr 0.15 (AUC 0.725 -- chosen because it is
           unambiguously not an oracle). eval/run.py IMPORTS that operating
           point rather than restating it. 16/16 points reach n_offer > 0,
           and the false-off-ramp RATE degrades exactly as it should:
           decline channel 10.4% at the oracle -> 44.1% at realised AUC
           0.498; intent channel 9.3% -> 75.0% at AUC 0.507.

           CORRECTED, R5 review pass, 2026-09-05 (stats-reviewer): this
           paragraph previously said "the intent channel is measurably
           worse than the decline channel... the honest consequence of its
           adapter being misspecified." Independently re-derived via
           two-proportion z-tests on the artifact's own counts and found
           NOT SUPPORTED: significant at 1 of 8 grid points (z=+2.01, the
           AUC-0.5 row), REVERSES at the oracle (z=-0.26), and the pooled
           difference is negligible (51/244=0.209 intent vs 96/467=0.206
           decline, z=+0.11; sign test across the 7 non-tied points,
           p=0.070). The two channels are also confounded on two axes the
           "misspecification" attribution ignored: apply_intent_channel()
           never fires on a DEAD attempt (unlike the decline channel), so
           they observe different numbers of decisions per mandate; and
           their per-observation likelihood-ratio magnitudes differ by
           construction. Correct statement: the intent channel's rate is
           numerically higher at 7 of 8 points but not distinguishable from
           the decline channel's at this sample size, and this design
           cannot attribute a cause. Every row carries the
           REALISED ROC (measured from the draws that happened, never the
           configured parameter) with a mandate-level cluster-bootstrap CI,
           via bench/llm_vs_stats.py's own macro_ovr_auc and
           cluster_bootstrap_ci -- imported, not reimplemented, because that
           implementation is tie-aware and this predictor is all ties.

           RE-CALIBRATION. The gate is refit at every sweep point and for
           the published grid, on a calibration pool drawn under the SAME
           channel -- reusing a pool drawn under a different one would break
           the exchangeability the coverage guarantee rests on. The Mondrian
           floor (ceil(1/alpha)-1 = 19/class at alpha=0.05) still holds
           (calibrate() raises below it; the per-class counts are recorded
           anyway). Coverage moved and is re-reported: 0.899-0.986 across
           cells against the 0.95 target, still under-covering, with
           per-class coverage as low as 0.795 on CANT_PAY_EVER. Degradation
           is a result, not a bug.

           CRITICAL, R5 review pass, 2026-09-05 (stats-reviewer), NOT FIXED
           IN THIS PASS. "0.95 target" is a WEAKER claim than the code
           supports even that far: the calibration pool (200 slot-1
           beliefs) has only 2-3 DISTINCT nonconformity values per class --
           every slot-1 belief is exactly cause_map.prior(dc) for one of a
           handful of DeclineClasses, so the LAC score can only take a
           handful of values (verified exactly: CANT_PAY_NOW {0.20 x81,
           0.80 x22, 0.85 x2}; CANT_PAY_EVER {0.25 x27, 0.90 x21}; WONT_PAY
           {0.30 x28, 0.90 x19}). Probed directly: the fitted gate's own
           {WONT_PAY}-singleton boundary moves from p(WONT_PAY)>=0.90 at
           alpha=0.05 to >=0.80 at alpha=0.20/0.30/0.40 and then STAYS
           there -- an 8x range of nominal miscoverage tolerance (95% down
           to 60%) produces almost no change in what the gate actually
           does. Separately, the pool's own maximum score is 0.90, while a
           real query sequence exceeds that within the four-attempt NPCI
           budget (three CUSTOMER_DECLINED observations already push belief
           past 0.997) -- a SUPPORT MISMATCH, not only a small-sample one.
           The gate behaves like a near-fixed threshold around p(WONT_PAY)
           0.80-0.90, not a function of the stated confidence level.

           This is not new to R5 -- the under-coverage NUMBER above was
           already disclosed before R5 ran -- but R5 is what makes it
           consequential: before R5 the singleton was unreachable at all,
           so a coverage failure in that lane was inert; it is not inert
           now. NOT FIXED here: the real fix is to calibrate on the QUERY
           distribution across all four slots, from a held-out disjoint
           corpus, not one row per mandate at slot 1 -- that changes
           fit_gate() for every consumer in this project, not only R5's
           channel, and needs its own investigation rather than a patch
           inside a review response. CLAUDE.md's "at 95% coverage" line is
           corrected to remove the unsupported guarantee rather than left
           standing next to this finding.

           FIXED, R8, 2026-09-05 (DECISIONS.md, "R8 · The conformal gate's
           CRITICAL calibration bug, fixed"). fit_gate() now grinds each
           calibration mandate through its own slot 2/3 too (n_calib
           200 -> 333), closing the support mismatch above -- the pool now
           spans the confidence range a real multi-decline trajectory
           reaches instead of stopping at slot 1. Measured effect on the
           published sweep: OFFER 1292 -> 300, false-off-ramp rate
           15.5% -> 1.3%, per-class coverage 0.795-0.986 -> 0.836-1.0
           (marginal 0.883-0.985) -- still short of the 0.95 target,
           CANT_PAY_NOW specifically, so this closes the SUPPORT MISMATCH,
           not the under-coverage finding in full. A follow-on stats-review
           pass found the fix's own "can only widen the pool" claim false
           (11.4% of the shipped pool is unreachable as a live query,
           biasing the singleton boundary toward MORE conservative -- the
           safe direction, but a real, measured bias) plus two disclosed-
           not-fixed sensitivities (the pool is now arm/regime-dependent
           though only ever fit once, on nominal; up to 3 rows share one
           mandate, a mild departure from i.i.d. calibration). Full
           numbers, both reviews, in DECISIONS.md.

           TWO GAPS THE GATE TEXT DID NOT NAME, BOTH CLOSED.
           (1) construct_offer() had NO CALLER anywhere in src/ -- a chosen
           OFFER had never produced an Offer object, while offramp.py's own
           docstring asserted the opposite. Now wired in _build_plan(), with
           the Offer on Plan and in the export's audit trail and rendered in
           the dashboard drill-down. Deliberately NOT in decision_sha256's
           payload (verified byte-identical to HEAD's payload block, and a
           literal digest pinned by test) so no already-persisted hash moves.
           (2) false_offramp_count had NO DENOMINATOR: it was computed inside
           the would_pay branch, so an OFFER to a mandate that would not have
           paid was counted nowhere. Its Day-1 meaning is UNCHANGED BYTE FOR
           BYTE (the R2b lesson); offramp_scored_count, true_offramp_count
           and true_offramp_paise are ADDED BESIDE it, and the rate is
           computed at render time from the exact denominator.

           THE FIRING RULE. tests/policy/test_allocator.py asserts, by AST
           walk rather than grep, that should_act() has EXACTLY ONE call
           site in src/ -- so a second way to fire the off-ramp fails the
           suite instead of appearing quietly in a diff.

           A TRIPWIRE FIRED AND WAS HONOURED, NOT SILENCED.
           tests/eval/test_export_mandates.py::
           test_the_wont_pay_singleton_is_unreachable_via_live_inference
           failed on this change, and its own message named the reason in
           advance ("which is R5's job, not a silent side effect of
           something else"). Rewritten to the claims that are still true and
           still load-bearing (a LIVE {WONT_PAY} singleton must be
           accompanied by an actual OFFER; an OFFER must carry a real Offer
           in PAUSE/DOWNGRADE/CANCEL order; an OFFER must never sit on an
           OPTED_OUT context, clause 6(c)) rather than deleted. Two more
           checks were found to be passing for the WRONG reason and made
           stricter: dashboard/ssr-check.tsx matched "slot " against the
           literal "no slot spent" rather than any chosen slot, and
           site/ssr-check.tsx required the words "never fired", which R5
           made false -- replaced with three phrases that hold ("made-up
           signal", "hidden answer", "coin flip").

           NOT DONE, DISCLOSED: the golden set gained 6 rows for the new
           class (56 total) and a THIRD zero-tolerance check (any false
           CUSTOMER_DECLINED, on the same footing as any false
           MANDATE_REVOKED, because it routes a paying customer to the
           exit), and src/llm/normalizer.py's prompt now names the class --
           which changes NORMALIZER_VERSION (a content hash) and so busts
           the on-disk golden cache by design. `run.ps1 golden` therefore
           needs ~56 live calls before it can be ticked again; it was NOT
           re-run here, and no gate is claimed from it. eval/frozen/
           untouched.

           STATS-REVIEW PASS, 2026-09-05, AFTER THE ABOVE WAS CLOSED. Full
           account in DECISIONS.md's own entry. Nine findings, one
           CRITICAL (the conformal-validity finding folded into
           RE-CALIBRATION above), three statistical overclaims corrected
           in this text (the intent-vs-decline comparison, the 256-vs-128-
           cell double count, the bare headline rate), two real low-
           severity bugs fixed (a cluster-bootstrap CI grouped by bare
           mandate_id, colliding across seeds; a stale comment claiming a
           draw happens on every branch, false for two of four outcomes),
           one vacuous test replaced with a real regression floor, one
           pre-existing (not R5's) coverage-table mislabelling fixed
           because the review found it live, and one HIGH-value gap closed
           with a new sweep dimension rather than only disclosed: within-
           mandate correlation in the false-firing channel, held at exactly
           zero by the main grid, now swept separately
           (`WontPayChannel.habitual_fraction`,
           `eval.offramp_channel.dependence_sweep()`) at the pre-registered
           operating point -- false-off-ramp rate 12.5% -> 32.6% while the
           realised fpr stays within 0.136-0.165, same discrimination, same
           published ROC point. Every fix independently re-derived before
           being acted on, not taken on the reviewer's word (this
           project's standing discipline). Full suite green after every
           change; guard_invariants clean. -->
- [x] **R6** `/plan/{mandate_id}`, `/ledger/{mandate_id}` and
      `/decision/{decision_sha256}` return real rows from a live schema,
      covered by tests
      <!-- 2026-09-05, CLOSED. New src/api/read.py (an APIRouter, mounted in
           src/ingest/app.py beside the webhook router; a separate package
           because ingest means events arriving and these are reads going
           out). New named store functions rather than raw SQL in the
           router, matching every other DB read in this repo: PlanRow,
           find_plan(), plans_for_mandate(), committed_for_decision(),
           ledger_for_decision(). src/ingest/app.py's docstring asserted no
           second router would ever be needed; REWRITTEN, not left standing
           next to the code disproving it.

           VERIFIED LIVE, not only by test: schema.sql applied into an
           `r6_demo` schema on the running Postgres, two mandates seeded and
           driven through the real R4 plan_cycle()/run_due() path, uvicorn
           started, all three routes curl'd. /ledger returned the real
           INTENT -> SENT -> RESULT trail with outcome RECOVERED; /plan
           returned the derived action, the committed slot with
           scheduled_for exactly 24h after committed_at (clause 6(a)), the
           conformal set as a LIST and the belief with its verbatim
           provenance (cause_map=v3;reference_prior=ref-v1); /decision
           returned the plan plus every ledger row citing it, all three
           carrying that hash. All three 404s confirmed, and the webhook
           router confirmed still mounted (400 = signature check reached).

           tests/api/test_read.py (16 tests) seeds EVERY populated case by
           running the real R4 cycle against pg_schema rather than
           hand-writing rows -- an endpoint proven against invented fixtures
           proves only that it can read a shape someone imagined.

           WHAT IS HONESTLY UNDECIDABLE, AND SAID SO. `plan` has no
           chosen_action column. Three SOUND rules derive what can be
           derived -- a committed_schedule row => ATTEMPT (commit()'s own
           gate); binding_constraint = OPTED_OUT => STOP (clause 6(c)
           denies every other action); a conformal set that is not the
           {WONT_PAY} singleton => not OFFER (should_act()'s requirement) --
           and anything left is reported as NOT_ATTEMPT with a
           chosen_action_candidates superset, never as a guessed label.
           R5_R6_R7_PLAN.md's test list said "derived as REAUTH/STOP"; that
           is not derivable from this schema, and naming one would be the
           same failure R2's _binding_constraint() bug was (a hard-forced
           decision recorded as a free choice). The real fix is a
           chosen_action column, which is a schema change with no migration
           path in this repo -- named in src/api/read.py's docstring rather
           than left as a surprise.

           Serialization reuses existing conventions rather than inventing
           any: amount_paise AND money.fmt() (invariant 2, and src/api/ was
           added to guard_invariants.py's MONEY_DIRS at the same time the
           package was created, with a test pinning that coverage); the
           parsed belief AND the verbatim belief_json, because the parsed
           form drops provenance; conformal_set split to a list with "" ->
           [] (never [""]); Outcome by .name; datetimes isoformat.

           A REAL FINDING, DISCLOSED NOT FIXED: this machine's dev database
           had drifted materially from schema.sql (no `mandate` table, no
           `normalized_decline`, `committed_schedule` missing
           `decision_sha256`) because schema.sql is only ever applied into
           throwaway test schemas and there is NO MIGRATION PATH. The three
           missing tables were created, the demo used its own schema rather
           than destroying `public`, and the gap itself is R7's territory
           and is named there. -->
- [x] **R7** a reviewer on Linux or macOS can install, test, run the eval and
      read the report using commands printed in the README, without
      translating anything
      <!-- 2026-09-05, CLOSED, RE-READ AGAINST THE FIRST CI RESULT AS
           PROMISED. The first run (commit 4771298) FAILED, exactly as the
           plan pre-registered ("expect the first runs to fail; each
           failure is the finding"), and the failure was real: step 5
           (`pip install -r requirements.txt`) failed on ubuntu-latest,
           python 3.13.

           THE FINDING, diagnosed without access to the raw CI log (the
           unauthenticated jobs/logs endpoint 403s -- "must have admin
           rights"; the run's own status/conclusion is public via the
           unauthenticated runs API, its log text is not). Reproduced
           locally instead, in a real `python:3.13` Docker container (not
           guessed at from reading requirements.txt): `pip install -r
           requirements.txt` fails with `ResolutionImpossible` --
           `websockets==17.0.1` (the pinned line) directly conflicts with
           `google-genai==2.20.0`'s own declared requirement,
           `websockets<17.0,>=13.0.0`. `requirements.txt` was internally
           inconsistent -- a file that could not have installed cleanly
           into ANY fresh environment, Windows included, had anyone run
           `pip install -r` there either. This is precisely the gap R7's
           own plan named before writing a line of code: "`pip install -r
           requirements.txt` is an install path nothing in this repo
           documented or exercised" -- CI is the first thing that ever
           exercised it, and it found a real bug on the first try.

           WHY THE LOCAL VENV NEVER HIT THIS: `pip show websockets` on the
           actual working `.venv` reports `16.1.1`, not the `17.0.1`
           `requirements.txt` claimed -- the file had drifted from the
           real environment at some point after being frozen, and nothing
           had re-frozen it since. A second, related discrepancy in the
           same diff: the actual venv also had `anthropic==1.0.0`
           installed with NOTHING depending on it (`pip show anthropic`:
           no `Required-by`) and NOTHING in the codebase importing it
           (grepped, confirmed) -- a leftover from before this project's
           own 2026-08-30 "LLM edge switched from Anthropic to Gemini"
           change, never uninstalled. A bare `pip freeze` over the
           existing venv would have "fixed" the conflict while silently
           RESURRECTING the old SDK and its transitive dependency chain
           (cryptography/cffi/pycparser/distro/pyasn1) into the committed
           manifest -- the wrong fix, caught by diffing the fresh freeze
           against the committed file before writing it, not by running
           freeze and trusting it. `anthropic` uninstalled from the venv
           first; the six genuinely-needed transitive deps it shared with
           `google-auth`/`google-genai` (google-auth, tenacity, distro,
           pyasn1, pyasn1_modules -- all confirmed via `Required-by`, not
           assumed) stay, because they were always real requirements this
           project's OLD requirements.txt was simply missing.

           FIXED AND RE-VERIFIED, not just reasoned about: `requirements.txt`
           regenerated from the now-corrected venv (86 pins, was 78 --
           websockets 17.0.1->16.1.1, anthropic gone, six real google-genai
           transitive deps added that were previously missing), and the
           EXACT install command re-run in a fresh `python:3.13` container
           against the corrected file: exit 0. Full local suite re-run
           after the `anthropic` uninstall to confirm nothing depended on
           it: green.

           SECOND CI RUN, CONFIRMED GREEN. Commit `ac1cee3`:
           https://github.com/Akshayg005/MandateIQ/actions/runs/33940629559
           -- `conclusion: success`, all steps completed (install, lint,
           the full suite against the postgres service, eval-quick,
           report, the byte-identity check). This is R7's own
           pre-registered evidence, now real rather than pending: a
           reviewer on Linux can install, test, run the eval and read the
           report using the commands this README prints, and a machine
           re-proves it on every push.

           SHIPPED. `run.sh`, mirroring run.ps1 (scope decision 1: a README
           telling a reviewer to translate six PowerShell lines IS the
           translation the gate forbids). It carries run.ps1's own hard-won
           exit-code discipline: `step()` propagates non-zero, and
           tests/scripts/test_run_sh.py proves `./run.sh test` exits
           non-zero on a red suite by driving it with a stub interpreter,
           plus the control case (zero on a green one) so the first
           assertion cannot pass because everything fails. Four actions are
           DECLINED rather than approximated -- up/down (Start-Pane's
           per-server consoles and Win32_Process teardown have no honest
           POSIX equivalent), verify (a live Razorpay desktop pre-flight),
           freeze (block B2 only, already executed) -- and a test fails if
           either runner grows an action the other neither implements nor
           declines. `setup.sh`, which installs FROM requirements.txt: the
           pinned path setup.ps1 never exercised, since it pip-installs a
           hand-listed unpinned set and then OVERWRITES requirements.txt
           from pip freeze.

           TWO REAL BUGS FOUND WHILE BUILDING THIS, both by tests rather
           than by reading. (1) `step()`'s first version put a bare `"$@"`
           before its `rc` check: under `set -e` the script aborts at that
           line, so the exit code was right but the FAILED banner never
           printed -- a mirror that silently dropped run.ps1's own
           diagnostic. Fixed with `|| rc=$?`. (2) run.sh was written with
           CRLF line endings on this Windows checkout, and bash then dies
           with "set: pipefail<CR>: invalid option name" before running
           anything. Both are now pinned by test, the second by a byte-level
           check on the working-tree file (.gitattributes pins only the
           stored form, and it is the working-tree form that executes).

           BYTE-IDENTITY, which was FALSE ACROSS PLATFORMS. README claimed
           two runs of the same seeds produce byte-identical output,
           "precisely so that the claim can be checked by hashing". Every
           artifact writer used write_text() with no `newline=`, so the same
           run emitted CRLF on Windows and LF on Linux and no two platforms
           could ever agree. `newline="\n"` added to all 16 writers across
           eval/ and scripts/ and src/execute/shadow.py, plus a
           .gitattributes; CI re-runs eval-quick and checks the sha256
           against the first run.

           FALSE CLAIM CORRECTED, NOT MADE TRUE. CLAUDE.md and README both
           said the invariants are enforced by GIT hooks. `.git/hooks/` holds
           only the stock samples and core.hooksPath is unset; they are
           CLAUDE CODE hooks in .claude/settings.json, which do not gate a
           commit made outside Claude Code. Both sentences rewritten to say
           what is true (the guard runs as a PostToolUse hook, as
           `run.sh lint` / `.\run.ps1 lint`, and in CI, which is what
           actually gates a push). Installing git hooks nobody asked for
           would change how contributors work; that is not a documentation
           fix.

           GENERATED STRINGS FIXED AT THE SOURCE, since hand-editing the
           .md would be overwritten on the next eval: eval/report.py's
           regimes.md command block (both forms now), its README caption and
           its not-found error; scripts/dashboard_data.py's error;
           tests/conftest.py's require_pg message -- and
           tests/test_pg_guard.py, which PINS that message, now asserts BOTH
           runners so the POSIX half cannot be dropped later without the
           suite noticing.

           README rewritten: prerequisites (Python 3.13, Docker, optional
           Node, optional API key), side-by-side install/run blocks for both
           platforms, the Postgres section in both forms, a CI section, an
           HTTP API section covering R6's three endpoints, and an explicit
           statement that a fresh clone has NO reports/*.json (.gitignore
           excludes them) so report/dashboard/site have nothing to read
           until eval has run. requirements.txt stripped of its UTF-8 BOM
           and CRLF, and setup.ps1 changed to stop reintroducing them
           (PS 5.1's `-Encoding utf8` always writes a BOM).

           OUT OF SCOPE, DISCLOSED: .claude/settings.json's Stop hook is
           `powershell -NoProfile -File scripts/stop_hook_ci.ps1` and its
           other three hooks call a bare `python` (often absent on Linux,
           and never the project venv). Those are Claude Code dev-loop
           concerns, not commands a reviewer executes, and they are left
           alone. Also disclosed: schema.sql has no migration path (found
           concretely at R6 -- this machine's own dev database had drifted
           several tables and one column behind it), and `run.sh test`
           depends on a Postgres the reviewer must start first. -->

## What these gates deliberately do NOT say

- Nothing here promises the widened design matrix improves the model. R1a is
  a gate on **measuring and reporting**, not on winning.
- Nothing here promises the false re-auth count falls. R2b is a gate on
  **attributing** it correctly. If the inference path turns out to be as bad
  as the conflated number implied, that is the answer and it gets published.
- R5 does not promise the off-ramp is *correct*, only that it is **reachable
  and measured**. Untested-and-central is a weaker position than
  tested-and-imperfect; this gate buys the second one, not a good result.
