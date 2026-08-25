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
