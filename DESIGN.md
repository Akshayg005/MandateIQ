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

These are enforced by git hooks in `.claude/settings.json`, not by trust.
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
  conformal prediction set is the **singleton `{WONT_PAY}`** at 95% coverage.
  Everything else stays in the retry lane.
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

## Context discipline across sessions

Claude Code starts every session with no memory of the last one. The window
does not carry over, and a bigger window would not help — the reset is the
problem, not the size. State therefore lives in files, not in conversation.

**Every session begins with `/orient`** — read `STATE.md`, answer its six
drift-check questions, restate today's gate. Do not reconstruct the goal
from source files; that is how this becomes a generic dunning tool by day
eight.

**Every session ends with `/checkpoint`** — regenerate `STATE.md`, tick
gates honestly, write the handoff.

| File | Role | Read when |
|---|---|---|
| `STATE.md` | Current position + drift check. Under 145 lines | Every session, first |
| `SESSIONS.md` | Append-only history, one line per session | Only when tracing how something happened |
| `PLAN.md` | The 12-day schedule and gates | Every session, today's day only |
| `PLAN_DETAIL.md` | File-by-file plan from the Opus planning session | Today's day only |
| `DECISIONS.md` | Where a model was used and where it wasn't | When making or revisiting a design choice |
| `POSTMORTEM.md` | What broke during the build | When something breaks |

**Keeping the window usable within a session:**

- Delegate anything log-heavy to the `eval-runner` subagent. Batch output
  must never enter the main context — that is the single largest lever.
- Never paste a CSV or a full test log. Write a script that prints a summary.
- Use read-only review subagents (`stats-reviewer`, `money-auditor`,
  `payments-domain`, `compliance-auditor`) so exploration burns their context,
  not yours.
- `/clear` between days. **Never `/compact` mid-task** — compaction silently
  drops invariants, and you will not notice which ones.
- If you find yourself re-explaining a constraint, it belongs in a
  `CLAUDE.md` — the root one, or the directory-local one.
- **Never spawn the general-purpose subagent.** It has no scoped
  instructions, so it reads broadly to work out what it is doing, and that
  reading is billed. Use a named subagent from `.claude/agents/`, or do the
  work in the main session.

## Definition of done for any module

1. Tests written before implementation (use the `test-writer` subagent)
2. `python scripts\guard_invariants.py --all` exits 0
3. `.\run.ps1 test` passes
4. If it touches money or the ledger: reviewed by the `money-auditor` subagent
5. If it touches a regulatory constraint: reviewed by `compliance-auditor`
