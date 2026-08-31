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

           910 tests pass, zero skips; guards clean on 122 files; run.ps1
           verify 5/5. Does not touch eval/frozen/. -->
- [ ] **B14** ∥ dashboard: merchant + acquirer views; per-mandate drill-down shows belief, chosen slot, binding constraint, conformal set, ledger trail
- [ ] **B15** ∥ landing page: 60fps on a mid laptop; reduced-motion fallback; canvas-failure fallback; counters wired to real report output, not hard-coded
- [ ] **B16** ship: README has "What this can't do" with ≥4 items; video under 5:00; three takes max
