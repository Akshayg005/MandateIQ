"""Pre-registered scoring: turns a sequence of simulated attempts into the
three-bar metric (recovered, attempts spent, mandates preserved). Frozen
alongside simulator.py and sim_config.yaml -- what counts as a win or a loss
must not be redefinable after a result has been seen.

"preserved" follows directly from the person-period schema (PLAN_DETAIL.md
section 2): a mandate that exhausts its attempt budget without resolving is
right-censored, not lost -- it lives to the next cycle. Only DEAD (instrument
confirmed dead) and OPTED_OUT (customer walked away) count as not preserved.
RECOVERED and STILL_PENDING-at-budget-exhaustion both count as preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import Outcome
from eval.frozen.simulator import AttemptResult, SimMandate

NOT_PRESERVED = frozenset({Outcome.DEAD, Outcome.OPTED_OUT})


@dataclass(frozen=True)
class MandateResult:
    mandate_id: str
    attempts: tuple[AttemptResult, ...]
    final_outcome: Outcome
    amount_recovered_paise: int
    preserved: bool
    iatrogenic_failures: int


@dataclass(frozen=True)
class BatchResult:
    arm: str
    profile: str
    n_mandates: int
    total_recovered_paise: int
    total_attempts_spent: int
    mandates_recovered: int
    mandates_dead: int
    mandates_opted_out: int
    mandates_censored: int
    mandates_preserved: int
    total_iatrogenic_failures: int
    per_mandate: tuple[MandateResult, ...] = field(repr=False)

    def summary(self) -> str:
        return (
            f"[{self.arm}/{self.profile}] n={self.n_mandates} "
            f"recovered_paise={self.total_recovered_paise} "
            f"attempts_spent={self.total_attempts_spent} "
            f"preserved={self.mandates_preserved}/{self.n_mandates} "
            f"(recovered={self.mandates_recovered} dead={self.mandates_dead} "
            f"opted_out={self.mandates_opted_out} censored={self.mandates_censored}) "
            f"iatrogenic_failures={self.total_iatrogenic_failures}"
        )


def score_mandate(mandate: SimMandate, attempts: list[AttemptResult]) -> MandateResult:
    """Pure aggregation -- no randomness, no I/O. `attempts` must be the
    ordered sequence of attempts actually made for this mandate (a policy
    may stop early; a censored mandate simply has fewer terminal-free rows,
    per the person-period schema)."""
    if not attempts:
        raise ValueError(f"{mandate.mandate_id}: scored with zero attempts")

    terminal = [a for a in attempts if a.outcome != Outcome.STILL_PENDING]
    if terminal:
        final = terminal[-1].outcome
        if len(terminal) > 1 or terminal[-1] is not attempts[-1]:
            raise ValueError(
                f"{mandate.mandate_id}: a terminal outcome was followed by "
                "another attempt -- the episode continued after resolving"
            )
    else:
        final = Outcome.STILL_PENDING  # budget exhausted, still censored

    recovered_paise = mandate.amount_paise if final == Outcome.RECOVERED else 0
    preserved = final not in NOT_PRESERVED
    iatrogenic = sum(1 for a in attempts if a.iatrogenic_insufficient_funds)

    return MandateResult(
        mandate_id=mandate.mandate_id,
        attempts=tuple(attempts),
        final_outcome=final,
        amount_recovered_paise=recovered_paise,
        preserved=preserved,
        iatrogenic_failures=iatrogenic,
    )


def aggregate(results: list[MandateResult], *, arm: str, profile: str) -> BatchResult:
    if not results:
        raise ValueError("aggregate() called with zero mandate results")

    return BatchResult(
        arm=arm,
        profile=profile,
        n_mandates=len(results),
        total_recovered_paise=sum(r.amount_recovered_paise for r in results),
        total_attempts_spent=sum(len(r.attempts) for r in results),
        mandates_recovered=sum(1 for r in results if r.final_outcome == Outcome.RECOVERED),
        mandates_dead=sum(1 for r in results if r.final_outcome == Outcome.DEAD),
        mandates_opted_out=sum(1 for r in results if r.final_outcome == Outcome.OPTED_OUT),
        mandates_censored=sum(1 for r in results if r.final_outcome == Outcome.STILL_PENDING),
        mandates_preserved=sum(1 for r in results if r.preserved),
        total_iatrogenic_failures=sum(r.iatrogenic_failures for r in results),
        per_mandate=tuple(results),
    )
