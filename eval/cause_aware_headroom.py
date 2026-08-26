"""Diagnostic, not frozen, not part of any block's file list -- same status
as `oracle_policy.py`, which it depends on. Answers the question B2's
handoff left open: does perfect cause-targeting (as opposed to `oracle_
policy.run()`'s perfect *timing*) show a headroom gap over the fixed ladder
on `attempts_spent` and `iatrogenic_failures`, on all three frozen arms --
including `coupled`, which the timing-only oracle could not discriminate on
(9/11/0 wins, mean/SE=-0.22 over a 20-seed sweep; see STATE.md/DECISIONS.md).

Money recovered and mandates-preserved are deliberately NOT reported here:
a mandate whose true cause is offered an exit needs an offer-acceptance
model to score as preserved/not, and inventing one is exactly the kind of
after-the-fact assumption this diagnostic has no standing to make (see
protocol.md, Known limitations, last bullet). Attempts spent and iatrogenic
count need no such model -- they are observable the moment a policy decides
not to attempt.

Pre-registered comparison convention (protocol.md, "Comparisons are across
seeds, not one seed"): seeds 0-19, paired per seed (same seed -> same
underlying mandate population for both policies), reporting mean and SD for
each policy plus the paired mean/SE of (ladder - cause_aware) -- positive
means the cause-aware oracle spends fewer attempts / causes fewer
iatrogenic failures than the ladder.

Run: python -m eval.cause_aware_headroom
"""
from __future__ import annotations

import statistics

from src.core.types import Profile
from eval import baseline_ladder, oracle_policy
from eval.frozen.simulator import Simulator

ARMS = ("nominal", "misspecified", "coupled")
SEEDS = range(20)


def _sweep(arm: str) -> dict:
    ladder_attempts, oracle_attempts = [], []
    ladder_iatro, oracle_iatro = [], []
    n_reauth = n_offer = 0
    for seed in SEEDS:
        ladder_result = baseline_ladder.run(Simulator(arm, seed=seed), Profile.strict)
        oracle_result = oracle_policy.run_cause_aware(Simulator(arm, seed=seed), Profile.strict)
        ladder_attempts.append(ladder_result.total_attempts_spent)
        oracle_attempts.append(oracle_result.total_attempts_spent)
        ladder_iatro.append(ladder_result.total_iatrogenic_failures)
        oracle_iatro.append(oracle_result.total_iatrogenic_failures)
        n_reauth += oracle_result.n_stopped_reauth
        n_offer += oracle_result.n_offered_exit

    def _mean_sd(xs: list[int]) -> tuple[float, float]:
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    def _paired_mean_se(a: list[int], b: list[int]) -> tuple[float, float]:
        diffs = [x - y for x, y in zip(a, b)]
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        se = sd / (len(diffs) ** 0.5)
        return mean, se

    attempts_mean_se = _paired_mean_se(ladder_attempts, oracle_attempts)
    iatro_mean_se = _paired_mean_se(ladder_iatro, oracle_iatro)
    return {
        "arm": arm,
        "attempts_ladder": _mean_sd(ladder_attempts),
        "attempts_oracle": _mean_sd(oracle_attempts),
        "attempts_diff_mean_se": attempts_mean_se,
        "iatro_ladder": _mean_sd(ladder_iatro),
        "iatro_oracle": _mean_sd(oracle_iatro),
        "iatro_diff_mean_se": iatro_mean_se,
        "avg_reauth_per_seed": n_reauth / len(SEEDS),
        "avg_offer_per_seed": n_offer / len(SEEDS),
    }


def main() -> None:
    print(f"cause-aware oracle vs fixed ladder -- seeds {SEEDS.start}-{SEEDS.stop - 1}, n=20")
    print(
        f"{'arm':<13} {'attempts(ladder)':>18} {'attempts(oracle)':>18} "
        f"{'attempts diff mean/SE':>22} {'iatro(ladder)':>15} {'iatro(oracle)':>15} "
        f"{'iatro diff mean/SE':>20} {'avg_reauth':>10} {'avg_offer':>10}"
    )
    for arm in ARMS:
        r = _sweep(arm)
        al_m, al_sd = r["attempts_ladder"]
        ao_m, ao_sd = r["attempts_oracle"]
        ad_m, ad_se = r["attempts_diff_mean_se"]
        il_m, il_sd = r["iatro_ladder"]
        io_m, io_sd = r["iatro_oracle"]
        id_m, id_se = r["iatro_diff_mean_se"]
        print(
            f"{arm:<13} {al_m:>10.1f}±{al_sd:<6.1f} {ao_m:>10.1f}±{ao_sd:<6.1f} "
            f"{ad_m:>10.2f}/{ad_se:<9.2f} {il_m:>7.1f}±{il_sd:<6.1f} {io_m:>7.1f}±{io_sd:<6.1f} "
            f"{id_m:>9.2f}/{id_se:<8.2f} {r['avg_reauth_per_seed']:>10.1f} {r['avg_offer_per_seed']:>10.1f}"
        )


if __name__ == "__main__":
    main()
