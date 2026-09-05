# Mandate Recovery Engine

## What this is

A decision engine for failed recurring debits (subscriptions / UPI AutoPay
mandates) in India, built on Razorpay test-mode APIs.

Every retry system on the market asks one question: *"will a retry succeed?"*
It then retries on a fixed schedule and halts after three failures.

We ask a different question: **which of three things went wrong?**

| Latent cause | Meaning | Correct action |
|---|---|---|
| `CANT_PAY_NOW` | Transient liquidity gap | Spend a slot, timed to their replenishment rhythm |
| `CANT_PAY_EVER` | Instrument dead — expired card, closed account, revoked mandate | Stop retrying. Request re-authorisation |
| `WONT_PAY` | Wants out, passively resisting | **Offer** an exit: pause, then downgrade, then cancel |

The system will sometimes deliberately recover less money this cycle to
protect lifetime value. That is the thesis, not a bug.

---

## Non-negotiable invariants

These are enforced by `scripts/guard_invariants.py`, not by trust -- it runs
over the whole tree via `.un.ps1 lint` / `./run.sh lint`, and on every push
in CI, exiting non-zero on a violation. It is also wired as a local editor
write-guard during development. It is NOT a git hook: `.git/hooks/` holds
only the stock `*.sample` files and `core.hooksPath` is unset, so nothing
blocks a commit made locally. (This paragraph claimed git hooks until R7,
2026-09-05, which is a stronger claim than the repo can support. Corrected
rather than made true: installing git hooks nobody asked for is a change to
how contributors work, not a documentation fix. CI is what actually gates a
push.)
If you find yourself wanting to violate one, stop and raise it instead.

1. **`src/model/`, `src/policy/` and `src/core/` may NEVER import `google.genai`,
   `anthropic`, `openai`, or any LLM client.** The decision core is deterministic +
   statistical. Generative models do not make money decisions.
2. **All money is integer paise.** A float touching a money value is a bug.
3. **The ledger write happens BEFORE the money action, never after.** This is
   what makes crash recovery correct.
4. **`eval/frozen/` is immutable after the Day-1 freeze commit.** The hash
   lives in `reports/FREEZE_HASH`.
5. **`rzp_live_` must never appear anywhere in this repo.** Test mode only.
6. **The system NEVER cancels a mandate.** It only ever *offers* an off-ramp.
   The customer decides.

---

## Regulatory constants

Source: RBI **"Digital Payments – E-mandate Framework, 2026"**,
circular `RBI/DPSS/2026-27/396`, dated 21 April 2026, effective immediately.

| Clause | Rule | Consequence for this codebase |
|---|---|---|
| 6(a) | Issuer sends pre-transaction notification ≥24h before every debit | Attempts must be **committed ≥24h ahead**. Reactive retry is structurally impossible. We forecast; we do not react. This is why Stripe's approach (reacting to signals in the last N hours) does not transfer to India |
| 6(c) | Notification carries an AFA-validated opt-out for that transaction **or the whole mandate** | Every attempt risks losing the customer permanently. `OPTED_OUT` is a **distinct outcome**, never folded into "declined" |
| 6(d) | Pre-notification exempt only for FASTag / NCMC auto-replenishment | Out of scope; assert we never hit this path |
| 8(a) | AFA-free up to ₹15,000 per transaction | `AFA_FREE_LIMIT_PAISE = 1_500_000`. Above → re-auth path, not silent retry |
| 8(b) | ₹1,00,000 for insurance premiums, mutual fund subscriptions, credit card bills | `AFA_FREE_LIMIT_ELEVATED_PAISE = 10_000_000`, category-gated |
| 4(c) | Variable e-mandates carry a customer-set maximum per transaction | Never attempt above the mandate ceiling |
| 10(c) | Acquirers must ensure merchant compliance | Why a payment aggregator — not just a merchant — wants this. Drives the acquirer dashboard view |

**NPCI attempt limit: 1 original + 3 retries = 4 attempts total, ever.**
This is a scarce-budget allocation problem, not a scheduling problem.

**Open ambiguity — handled, not assumed away.** The circular never uses the
word "retry". Whether a reattempt needs its own fresh notification is
genuinely unresolved. We therefore ship **two compliance profiles**
(`strict` / `permissive`, see `src/policy/profiles.py`) and evaluate under
both. Never hard-code one interpretation.

**Consumer-protection context.** The CCPA *Guidelines for Prevention and
Regulation of Dark Patterns, 2023* list "subscription trap" among 13
prohibited practices, with active enforcement through 2026. Grinding an
exit-intent customer is a legal exposure, not just unkind. The off-ramp is a
compliance artifact.

---

## Where AI lives, and where it deliberately does not

**Decision core — no LLM.** Discrete-time competing-risks survival model
(multinomial logit over person-period data) → cumulative incidence per cause
→ exact backward induction over the four attempt slots.

**LLM edge — language only:**
- `src/llm/normalizer.py` — Gemini 3.5 Flash-Lite. Normalises decline strings that differ
  across issuers into the taxonomy.
- `src/llm/intent.py` — Gemini 3.5 Flash-Lite. Extracts cancellation intent from support
  ticket text, including Hinglish.
- `src/llm/narrator.py` — Gemini 3.5 Flash. Merchant-facing root-cause narrative.
  Runs **once per batch**, never per transaction.

All structured LLM output goes through forced function calling
(`tool_config.function_calling_config.mode = "ANY"`, declared in
`src/llm/tools.py`) so malformed JSON is structurally impossible rather than
caught downstream. Verified against the live API before adoption: the model
returns a function call and no stray text.

`bench/llm_vs_stats.py` benchmarks an LLM-as-classifier baseline against the
statistical model on the same held-out split, reporting AUC, p95 latency,
cost per 1k decisions, and **run-to-run variance on identical input**. Ship
the table even though — especially because — the LLM loses. The variance
column is the argument: same input, different retry time, is disqualifying
in a payments path regardless of accuracy.

---

## Safety design

Exit intent is **weakly identified** from payment data alone. A false
positive cancels a paying customer — the exact harm we exist to prevent.
Therefore:

- Split conformal prediction gates the off-ramp. Offer only when the
  conformal prediction set is the **singleton `{WONT_PAY}`**. Everything
  else stays in the retry lane. (This line said "at 95% coverage" until a
  2026-09-05 review — the calibration pool had only 2-3 distinct
  nonconformity values per class, so the fitted threshold was nearly
  insensitive to `alpha` across an 8x range, and its own maximum score sat
  below beliefs the gate was actually queried at after two or more
  observations — a SUPPORT MISMATCH, corrected at R8 the same day:
  `fit_gate()` now calibrates across each mandate's own slot 2/3 trajectory,
  not slot 1 alone, which is why measured class-conditional coverage moved
  from 0.795-0.986 to **0.836-1.0** (marginal 0.883-0.985). Still not a
  validated 0.95 — `CANT_PAY_NOW` specifically remains under-covered — so
  this line still does not claim one. See `reports/gates.md`'s R5 entry for
  the original finding and DECISIONS.md's "R8" entry for the fix, its
  measured effect on the published report, OFFER 1292 -> 300 and the
  false-off-ramp rate 15.5% -> 1.3% chief among them, and a follow-on
  review's own disclosed caveats on the fix itself.)
- The system offers; the customer decides.
- Report **both** error costs: missed recovery, and false off-ramp.

---

## Conventions

- Python 3.13 · FastAPI · Postgres · statsmodels · lifelines · scikit-learn
- Every constant in `src/policy/constraints.py` carries its clause reference
  in a docstring. **No unattributed magic numbers.**
- Type hints everywhere. `.\run.ps1 test` passes before any commit.
- **Windows-native project.** PowerShell, not bash. Task runner is
  `run.ps1`, not a Makefile. Guards resolve their own file paths
  (argv -> env -> stdin -> git) so hooks work regardless of shell.
- Money helpers live in `src/core/money.py`. Nothing else formats currency.
- Time helpers live in `src/core/clock.py`. Nothing else calls `datetime.now()` —
  tests must be able to freeze the clock.

## Working discipline

The goal is narrow and easy to drift away from. Two habits keep it honest.

**Decisions live in files, not in memory.** Anything that changes what the
system does -- a threshold, a clause reading, a scope cut -- gets written
down at the moment it is made, with its reasoning, not reconstructed later.

**Do not reconstruct the goal from the source.** The code will happily
support a generic dunning tool; the thesis above is what makes it something
else. Re-read this file before adding a feature.

| File | Role | Read when |
|---|---|---|
| `DECISIONS.md` | Where a model was used and where it wasn't | When making or revisiting a design choice |
| `POSTMORTEM.md` | What broke during the build | When something breaks |
| `WHAT_BROKE.md` | The short version of the same | For the highlights |
| `reports/gates.md` | Every acceptance gate and its evidence | When checking a claim |

**Keeping the evidence trail usable:**

- Batch output belongs in a file, never pasted into a review thread --
  write a script that prints a summary instead.
- Never paste a CSV or a full test log into a discussion.
- If you find yourself re-explaining a constraint, it belongs in a
  `DESIGN.md` -- the root one, or the directory-local one.

## Definition of done for any module

1. Tests written before implementation
2. `python scripts\guard_invariants.py --all` exits 0
3. `.un.ps1 test` passes
4. If it touches money or the ledger: a dedicated review pass over the diff,
   covering amounts, idempotency and ledger ordering
5. If it touches a regulatory constraint: a review pass against the clause
   table above, clause by clause
