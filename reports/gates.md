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
- [ ] **B8** ★ allocator + stopping + off-ramp: 2-slot brute-force equivalence test passes; zero constraint violations across the eval, with an attempt rate on AFA-eligible mandates ≥ `eval.gate_criteria.ATTEMPT_RATE_FLOOR` (0.25) AND a mean discrimination gap (true-CANT_PAY_NOW attempt rate minus true-CANT_PAY_EVER attempt rate, seeds 0-19) exceeding `DISCRIMINATION_MARGIN` (~0.0808) -- the null policy must fail the first, a uniform-random policy at the floor rate must fail the second, both proven by test (`tests/eval/test_gate_criteria.py`); both profiles produce numbers
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
           outside the frozen directory. -->
- [ ] **B9** ★ executor + idempotency: keys test passes (no clock/uuid/pid); **an opt-out arriving inside the 24h window is honoured — proven by a test that actively constructs the race** (commits an attempt, then delivers a late opt-out event inside that window, and asserts the attempt is aborted — not merely a test that happens never to generate one); `UNCONFIRMED` has a resolution path that is actually reachable
      <!-- 2026-08-29, vacuous-checks audit: strengthened from "an opt-out
           arriving inside the 24h window is honoured" -- a test that
           simply never generates a late opt-out would satisfy that text
           by construction, without ever exercising the actual race. No
           executor code exists yet; this amendment is made before B9
           starts, the same standard as the other gate amendments this
           block. Full reasoning: DECISIONS.md, 2026-08-29. -->

- [ ] **B10** chaos: 50 induced kills; zero double-charges; zero lost jobs; ledger complete; **the denominator is reported** — how many kills landed inside the unsafe window
- [ ] **B11** ∥ LLM edge + golden set: golden set passes; no LLM import in core; normaliser output is versioned in the ledger before it can touch a belief
- [ ] **B12** ∥ benchmark + shadow: benchmark table in DECISIONS.md including the variance column; shadow mode produces a delta log over the full batch
- [ ] **B13** ★ stress regimes + report: every number reproducible by one command; at least one regime where we lose, explained
- [ ] **B14** ∥ dashboard: merchant + acquirer views; per-mandate drill-down shows belief, chosen slot, binding constraint, conformal set, ledger trail
- [ ] **B15** ∥ landing page: 60fps on a mid laptop; reduced-motion fallback; canvas-failure fallback; counters wired to real report output, not hard-coded
- [ ] **B16** ship: README has "What this can't do" with ≥4 items; video under 5:00; three takes max
