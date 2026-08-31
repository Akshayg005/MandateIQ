# Shadow-mode delta log

run_id `shadow-20260831T085344-83b66a95` · arm `nominal` · profile `strict` · 200 mandates

Decisions only. Nothing was executed, no provider was called, and no row
was written to `ledger`, `committed_schedule`, `attempt_lease` or `plan`.

## Committed attempts, at the first decision point

| | attempts committed |
|---|---|
| fixed ladder (T+1/T+2/T+3, every mandate) | 600 |
| this system | 141 |

The ladder commits three attempts per mandate up front regardless of cause;
this system commits at most one and re-decides after each observation.

## Where the two policies disagree

| divergence | mandates |
|---|---|
| `LADDER_ATTEMPTS_WE_OFFER` | 0 |
| `LADDER_ATTEMPTS_WE_REAUTH` | 56 |
| `LADDER_ATTEMPTS_WE_STOP` | 3 |
| `SAME_ACTION_DIFFERENT_DAY` | 0 |
| `SAME_ACTION_SAME_DAY` | 141 |

Agree: 141 · diverge: 59

## What bound each decision

| binding constraint | mandates |
|---|---|
| (none -- decided on belief and expected value) | 182 |
| AFA_CLIFF | 18 |

This decomposition matters: a hard constraint and a belief are not the
same kind of reason. `AFA_CLIFF` rows are routed by regulation (clause
8(a)) and would be routed identically by any compliant system. Every
other REAUTH is this system's own inference that the instrument is dead
-- which is the part that can be wrong, in both directions.

## What this report does NOT say

- **No money delta.** "We would have recovered X more" requires executing
  both policies against outcomes; shadow mode by definition executes neither.
  That comparison is the frozen eval's (B13), not this file's.
- **The slot-1 decline signal is simulated here.** Against the frozen batch
  it is drawn from the simulator's own generative parameters, not observed
  from an issuer. `source_version` on every row says so. Against live
  traffic the same function reads a real normalised decline.
- **One decision point per mandate**, not a full retry cycle.
- **Nothing about the off-ramp.** `LADDER_ATTEMPTS_WE_OFFER` is 0 because
  no `ConformalGate` is passed here, so `solve()` falls back to
  `FullSetGate` and the prediction set is never the singleton
  `{WONT_PAY}` an OFFER requires. That is the safe default working as
  specified, not a defect -- but this log is therefore silent on the
  off-ramp and must not be read as evidence about it either way.
- **Nothing about retry timing.** `SAME_ACTION_DIFFERENT_DAY` is 0:
  every attempt this system commits lands on the same slot and day the
  ladder would have picked. The timing discrimination the thesis claims
  is not visible at the first decision point, and this report does not
  demonstrate it.
