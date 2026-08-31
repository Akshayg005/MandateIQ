"""B12: LLM-as-classifier against the competing-risks model, on the same
held-out split -- log loss and per-class Brier, macro AUC with a cluster
bootstrap CI, p95 latency, cost per 1k decisions, and run-to-run variance on
identical input.

Ship the table whichever way it falls. The variance columns are the
argument, not accuracy: same input, different retry time, is disqualifying
in a payments path regardless of how well the model scores, because the
decision cannot be reproduced in a dispute.

=== What is being compared, exactly ==========================================

TARGET. Both arms predict `Outcome` (STILL_PENDING / RECOVERED / DEAD /
OPTED_OUT), NOT `Cause`. The statistical model never predicts Cause -- root
CLAUDE.md's own design puts Cause behind action-gating, and no production
label for it exists, ever. Scoring the LLM on Cause would compare it against
nothing, so this file scores the thing the model actually emits.

SPLIT. src/model/splits.py's 4-way mandate-level split, `test` arm only,
estimable rows only (slot 1 is a structural zero). The stats model is fit on
`train` and never sees a test row. The LLM was fit on neither, which is the
easy half of PLAN_DETAIL.md's review clause; the load-bearing half is that
the STATS arm must not be scored on anything it was fit on, which the split
guarantees structurally rather than by assertion.

INFORMATION SET -- read this before trusting any number below. The LLM is
shown a DELIBERATE SUPERSET of what the stats model uses. FEATURE_COLUMNS in
competing_risks.py is only ("const", "slot_3", "slot_4", "in_salary_window");
PROMPT_FIELDS here adds amount, ceiling, category, prior failures, committed
day of month and days since the last attempt. Showing it only the four
covariates would reduce the whole exercise to two priors over a 6-cell
contingency table, which says nothing about language models.

Note the honest limit of this framing, which an earlier version of this
docstring overstated as "unfair in the LLM's favour": strictly MORE
INFORMATION is what is demonstrated, and more information is not
automatically an advantage to a zero-shot model -- six extra covariates can
mislead as easily as inform. Claim the former, not the latter.

What it must NEVER see is any label or latent:
PROMPT_FIELDS is asserted disjoint from src/model/features.py's FORBIDDEN by
tests/eval/test_bench.py, and render_prompt() applies the allowlist rather
than trusting its caller's dict.

WHY AUC IS NOT THE HEADLINE. PLAN_DETAIL.md names AUC, so it is reported --
but it cannot decide this block's claim, and saying so is the point of this
paragraph. Two independent problems, both measured rather than suspected:

  (1) Two of four classes are unrankable BY CONSTRUCTION. Measured per-class
      one-vs-rest AUC: STILL_PENDING 0.534, RECOVERED 0.569, DEAD **0.487 --
      below chance**, OPTED_OUT 0.714. The frozen simulator sets the DEAD
      hazard from latent cause alone, and a design matrix of (const, slot_3,
      slot_4, in_salary_window) cannot separate it. Macro-averaging spends
      half its weight there, leaving almost no room above chance for any arm
      to lose in.
  (2) Ties. The stats arm emits at most 6 distinct probability vectors, so
      its ROC is a 6-point step function compared against a free-form
      model's continuous scores. macro_ovr_auc() uses sklearn's tie-aware
      implementation, which mitigates but does not remove the mismatch.

The headline is therefore multiclass_log_loss(), with brier_per_class()
beside it and an intercept-only null arm so both real arms are measured
against a shared reference. Log loss is a proper scoring rule: it rewards
CALIBRATION, not merely ranking, and calibration is what the allocator's
backward induction actually consumes -- a miscalibrated model that ranks
well is useless downstream and would still beat the stats arm on AUC. Every
AUC printed carries a mandate-level cluster bootstrap CI, because rows are
clustered (one mandate contributes up to three slot rows) and overlapping
intervals mean a tie however many decimals the point estimates differ by.

Found by stats-reviewer, 2026-08-31; see DECISIONS.md.

TEMPERATURE. Both 0.0 and 1.0 are run. PLAN_DETAIL.md explicitly forbids
running the LLM at temperature 0 only, and the sharpest finding available is
that variance at t=0.0 is not zero -- a model pinned to its most
deterministic setting still moves between identical calls.

COST. Free-tier keys make the marginal price zero, which is not a number
worth reporting. The cost column prices this exact workload at PAID-TIER
rates from config/llm_pricing.yaml, which carries its source URL and the date
it was read. No price and no exchange rate is written into this file.

=== Budget ===================================================================

TWO limits, and the second is the one that bites. Per MINUTE: ~15 requests
per model, handled by _pace(). Per DAY: capped per model, and the caps are
NOT equal -- gemini-3.5-flash-lite 500/day, gemini-3.5-flash 20/day, both
measured from real 429 bodies (DAILY_QUOTA_BY_MODEL).

The accuracy pass runs once over n rows per arm; the variance pass runs
`repeats` times over a smaller `variance_n` subsample at each of two
temperatures, and the subsample size is printed in the table rather than
left implicit. plan_budget()/assert_within_budget() compute the total from
the actual arguments and REFUSE to start a run that cannot fit, because the
alternative -- discovering a cap partway through -- cost this block 400
completed calls (POSTMORTEM.md incident 8).

CallCache flushes every answer to disk as it arrives, so an interrupted run
resumes instead of re-billing. End-of-run persistence is no protection
against the failure that actually happens, which is the run not reaching its
end.

A full two-model table is NOT reachable on the free tier: a flash variance
pass alone is 5 x 30 x 2 = 300 calls against a 20/day cap.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

# Bare-script invocation (`.\run.ps1 bench` runs `python bench\llm_vs_stats.py`),
# so the repo root is not on sys.path the way `-m` would put it there. Same
# fix-up as eval/golden_check.py, for the same reason.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import corpus  # noqa: E402
from src.core.types import Outcome  # noqa: E402
from src.model import competing_risks, features, person_period, splits  # noqa: E402

PRICING_PATH = _REPO_ROOT / "config" / "llm_pricing.yaml"
REPORT_PATH = _REPO_ROOT / "reports" / "bench.json"

# Outcome int order. hazards() returns columns in exactly this order and a
# silent transposition would invert the AUC, so it is named once here and
# pinned by test rather than assumed at each use site.
OUTCOME_ORDER: tuple[Outcome, ...] = (
    Outcome.STILL_PENDING,
    Outcome.RECOVERED,
    Outcome.DEAD,
    Outcome.OPTED_OUT,
)

# The exact allowlist of frame columns the LLM is shown. See the module
# docstring's INFORMATION SET note. Deliberately excludes every identifier
# (mandate_id, cycle_id, row_id -- nothing to generalise from), `profile`
# (a policy setting, not a fact about the customer), `estimable` (a function
# of slot) and, load-bearingly, `event_code` (the label).
PROMPT_FIELDS: tuple[str, ...] = (
    "slot",
    "in_salary_window",
    "amount_paise",
    "ceiling_paise",
    "category",
    "prior_failures_this_cycle",
    "committed_day_of_month",
    "days_since_last_attempt",
)


BENCH_TOOL: dict[str, Any] = {
    "name": "emit_slot_probabilities",
    "description": (
        "Emit a calibrated probability for each of the four terminal states "
        "of this retry slot. The four values must sum to 1.0."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "p_still_pending": {
                "type": "number",
                # NOT "survives to a further slot" -- see BENCH_SYSTEM_PROMPT's
                # note. At slot 4 there is no further slot, yet 50.7% of slot-4
                # rows carry this label.
                "description": (
                    "Probability the attempt fails and the mandate is neither recovered, "
                    "dead, nor opted out by the end of this slot (right-censored if this "
                    "is the last slot)."
                ),
            },
            "p_recovered": {
                "type": "number",
                "description": "Probability the debit succeeds and the money is collected.",
            },
            "p_dead": {
                "type": "number",
                "description": "Probability the instrument is confirmed dead (expired card, closed account).",
            },
            "p_opted_out": {
                "type": "number",
                "description": "Probability the customer opts out of the mandate entirely.",
            },
        },
        "required": ["p_still_pending", "p_recovered", "p_dead", "p_opted_out"],
    },
}

BENCH_SYSTEM_PROMPT = """You are scoring a single retry slot of a failed recurring debit (UPI AutoPay / card mandate) in India.

The original debit has already failed. You are given the facts known before this retry is attempted. Estimate how this retry slot resolves, as four probabilities summing to 1.0:

- p_still_pending: the retry fails, and at the end of this slot the mandate is neither recovered, nor dead, nor opted out -- it is simply unresolved
- p_recovered: the retry succeeds and the money is collected
- p_dead: the instrument is confirmed dead -- expired card, closed account, revoked mandate
- p_opted_out: the customer actively opts out of the mandate

IMPORTANT about p_still_pending at the LAST slot: it does NOT mean "a further slot remains". It means no resolving event happened during this slot. If this is slot 4 and the debit simply fails again without the instrument dying and without the customer opting out, the correct answer is still p_still_pending -- the episode ends unresolved (right-censored), which is a real and common result, not an impossible one.

Context that matters: only 4 attempts exist per mandate ever (1 original + 3 retries, NPCI). Salary in India typically lands in the first week of the month, so a retry inside that window meets a replenished balance. Amounts above Rs 15,000 sit above the AFA-free limit and behave differently.

Return only the function call."""


# --- pricing -----------------------------------------------------------------


def load_pricing(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Read config/llm_pricing.yaml. Kept in a file with its source URL and
    retrieval date rather than as constants here -- see that file's header
    and PLAN.md's "never fabricate a number"."""
    with open(path or PRICING_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cost_per_1k_paise(
    *,
    prompt_tokens: int,
    output_tokens: int,
    n_calls: int,
    model: str,
    usd_inr_paise: int,
    pricing: Mapping[str, Any] | None = None,
) -> int:
    """Cost of 1,000 decisions at paid-tier rates, in integer paise.

    `prompt_tokens` and `output_tokens` are TOTALS observed across `n_calls`,
    not per-call figures, so doubling the calls at a fixed token total halves
    the amortised per-1k cost.

    `usd_inr_paise` is REQUIRED and has no default. Gemini is priced in USD
    and this repo reports paise; defaulting the rate would mean an exchange
    rate invented inside the benchmark and silently baked into a published
    table. The caller passes it so the table can cite it.

    Integer arithmetic throughout, with a single floor at the end. Prices are
    stored as integer micro-USD per million tokens precisely so that root
    CLAUDE.md invariant 2 ("a float touching a money value is a bug") holds
    through the whole computation and not merely at the final rounding.

    Raises KeyError on a model with no price entry -- a model swapped in via
    an env override must not silently produce a free-looking row.
    """
    if n_calls <= 0:
        raise ValueError(f"n_calls must be positive, got {n_calls}")
    table = (pricing or load_pricing())["models"]
    if model not in table:
        raise KeyError(
            f"no price entry for model {model!r} in {PRICING_PATH.name}; "
            f"priced models are {sorted(table)}"
        )
    rates = table[model]
    # micro-USD * 1e6, kept unfloored so a small token count cannot round to
    # zero before the final division.
    numerator = (
        prompt_tokens * int(rates["input_micro_usd_per_1m"])
        + output_tokens * int(rates["output_micro_usd_per_1m"])
    )
    return (numerator * 1000 * usd_inr_paise) // (1_000_000 * n_calls * 1_000_000)


# --- metrics -----------------------------------------------------------------


def macro_ovr_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    """Macro one-vs-rest AUC over the four Outcome classes.

    sklearn's implementation is tie-aware (it integrates the ROC with ties
    handled as a diagonal segment rather than a step), which matters more
    here than usual: the stats arm emits at most 6 distinct probability
    vectors, so most pairs are tied. See the module docstring.

    Classes absent from `y_true` are dropped from the macro average rather
    than scored as 0.5, and a fold with fewer than 2 present classes raises
    -- an AUC over one class is not a number.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(y_true):
        raise ValueError(f"p must be ({len(y_true)}, n_classes); got {p.shape}")

    present = [c for c in range(p.shape[1]) if (y_true == c).any() and not (y_true == c).all()]
    if len(present) < 2:
        raise ValueError(
            f"macro_ovr_auc needs at least 2 classes both present and absent "
            f"in y_true; got {present}"
        )
    scores = [roc_auc_score((y_true == c).astype(int), p[:, c]) for c in present]
    return float(np.mean(scores))


def multiclass_log_loss(y_true: np.ndarray, p: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean negative log likelihood of the true class.

    THE HEADLINE METRIC, and the reason the AUC column is no longer it.
    stats-reviewer (2026-08-31) measured what AUC was actually doing on this
    problem: per-class OvR AUCs of 0.534 / 0.569 / 0.487 / 0.714, i.e. DEAD
    BELOW CHANCE and STILL_PENDING barely above it. That is structural, not a
    bad fit -- on the nominal arm the simulator sets the DEAD hazard from the
    latent cause alone, and nothing in a design matrix of (const, slot_3,
    slot_4, in_salary_window) can separate it. Macro-averaging then spends
    half its weight on classes the model provably cannot rank, and the 0.5759
    headline leaves almost no room above chance for an LLM to lose in.

    Log loss is a PROPER SCORING RULE: it rewards calibration, not just
    ranking. That matters here beyond statistical taste, because the allocator
    consumes probabilities, not ranks -- a badly calibrated model that ranks
    well is useless to backward induction and would nonetheless beat the stats
    arm on AUC. Reported against two references (an intercept-only null and
    uniform) so both arms are measured against a common baseline.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(len(y_true)), np.asarray(y_true, dtype=int)]).mean())


def brier_per_class(y_true: np.ndarray, p: np.ndarray) -> dict[int, float]:
    """One-vs-rest Brier score per Outcome class. Also a proper scoring rule,
    and it localises WHERE a model is wrong rather than averaging it away."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    return {
        c: float(np.mean(((y_true == c).astype(float) - p[:, c]) ** 2))
        for c in range(p.shape[1])
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray, p: np.ndarray, groups: Sequence[str], *,
    stat=None, n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling MANDATES, not rows.

    Rows are clustered -- one mandate contributes up to three slot rows -- so
    a naive row-level interval overstates precision. Without an interval the
    table reports a tie to four decimal places as though it were a result,
    which is exactly how "the LLM must lose" gets claimed on noise.
    """
    stat = stat or macro_ovr_auc
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in unique}
    out: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_of[g] for g in drawn])
        try:
            out.append(stat(y_true[idx], p[idx]))
        except ValueError:
            continue  # a resample missing a class entirely; skip, do not invent
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The flip rates are printed to three decimals off n=40 rows, where the
    binomial SE alone is ~0.06 -- three decimals of a number that is plus or
    minus six points. Wilson rather than normal-approximation because it
    stays inside [0, 1] and behaves at the boundaries, and 0.000 is exactly
    the value this block most wants to report honestly (stats-reviewer,
    2026-08-31). Still ignores WITHIN-mandate clustering in the variance
    subsample, so treat it as a floor on the true width.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def p95(latencies_s: Sequence[float]) -> float:
    """95th percentile, numpy's linear-interpolation convention."""
    if len(latencies_s) == 0:
        raise ValueError("p95() of an empty sequence is undefined")
    return float(np.percentile(np.asarray(latencies_s, dtype=float), 95))


@dataclass(frozen=True)
class VarianceReport:
    n_rows: int
    n_repeats: int
    argmax_flip_rate: float
    max_prob_stddev: float
    mean_prob_stddev: float
    cause_ordering_flip_rate: float
    max_optout_swing: float = 0.0
    argmax_flip_ci: tuple[float, float] = (float('nan'), float('nan'))

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} repeats={self.n_repeats} "
            f"argmax_flip={self.argmax_flip_rate:.3f} "
            f"[{self.argmax_flip_ci[0]:.3f},{self.argmax_flip_ci[1]:.3f}] "
            f"cause_order_flip={self.cause_ordering_flip_rate:.3f} "
            f"max_sd={self.max_prob_stddev:.4f} mean_sd={self.mean_prob_stddev:.4f} "
            f"max_optout_swing={self.max_optout_swing:.3f}"
        )


def variance_report(runs: Sequence[Sequence[np.ndarray]]) -> VarianceReport:
    """Run-to-run variance on byte-identical input.

    `runs[r]` is the (n_rows, 4) probability array from repeat r. Every
    repeat must cover the same rows in the same order.

    Three things are measured, because they answer three different
    objections. `argmax_flip_rate` is how often the model changed its mind
    about the most likely state. `max/mean_prob_stddev` is how far the
    numbers moved even when the argmax held -- a decision downstream of a
    threshold can flip without the argmax flipping. `decision_flip_rate` is
    the one that matters in a payments path: how often the implied retry
    slot changed, i.e. how often the same customer, same facts, would have
    been debited on a different day.

    Raises on fewer than 2 repeats: a variance computed from a single run is
    not a variance.
    """
    if len(runs) < 2:
        raise ValueError(
            f"variance_report needs at least 2 repeats to measure anything; got {len(runs)}"
        )
    stacked = np.asarray([np.asarray(r, dtype=float) for r in runs])  # (R, N, 4)
    if stacked.ndim != 3:
        raise ValueError(f"every repeat must be a 2-D (n_rows, n_classes) array; got {stacked.shape}")
    n_repeats, n_rows, _ = stacked.shape

    argmaxes = stacked.argmax(axis=2)  # (R, N)
    flipped = (argmaxes != argmaxes[0]).any(axis=0)  # (N,)
    # ddof=1: the unbiased estimator of run-to-run SD. ddof=0 is ~12% smaller
    # at R=5, in the direction that UNDERSTATES this block's own argument.
    sds = stacked.std(axis=0, ddof=1)  # (N, 4)

    # Whether RECOVERED outranks DEAD -- the contrast that separates "spend a
    # slot" from "this instrument is gone", and the one argmax_flip_rate misses
    # when both orderings sit under a dominant STILL_PENDING.
    #
    # An earlier version named this `decision_flip_rate` and the table called it
    # "retry-slot flip", claiming it measured how often the customer "would have
    # been debited on a different day". That was an OVERCLAIM on this block's
    # headline number and stats-reviewer was right to attack it: no slot and no
    # day is computed anywhere in this module, and the real allocator does exact
    # backward induction over CIFs, which a swap in this ordering neither implies
    # nor is implied by. Renamed to say only what it measures.
    #
    # It also ignores OPTED_OUT entirely -- the cause gating the off-ramp -- so a
    # model whose P(OPTED_OUT) swung 0.1 -> 0.6 across repeats would score 0.000
    # here. It is therefore a LOWER BOUND on decision instability, which is the
    # safe direction for an argument but must not be read as the full measure.
    # max_optout_swing below covers the gap it leaves.
    implied = stacked[:, :, int(Outcome.RECOVERED)] > stacked[:, :, int(Outcome.DEAD)]
    ordering_flipped = (implied != implied[0]).any(axis=0)
    optout = stacked[:, :, int(Outcome.OPTED_OUT)]
    max_optout_swing = float((optout.max(axis=0) - optout.min(axis=0)).max())

    return VarianceReport(
        n_rows=int(n_rows),
        n_repeats=int(n_repeats),
        argmax_flip_rate=float(flipped.mean()),
        max_prob_stddev=float(sds.max()),
        mean_prob_stddev=float(sds.mean()),
        cause_ordering_flip_rate=float(ordering_flipped.mean()),
        argmax_flip_ci=wilson_ci(int(flipped.sum()), int(n_rows)),
        max_optout_swing=max_optout_swing,
    )


# --- prompt ------------------------------------------------------------------


def render_prompt(row: Mapping[str, object]) -> str:
    """Render one frame row for the LLM, applying PROMPT_FIELDS as an
    ALLOWLIST rather than a denylist.

    The allowlist is applied here, not assumed of the caller: a row dict that
    happens to carry `event_code` or `initial_cause` must produce a prompt
    containing neither the value nor the key. A denylist would leak every
    column nobody thought to name, and the columns that matter are exactly
    the ones a future frame change would add silently.
    """
    missing = [name for name in PROMPT_FIELDS if name not in row]
    if missing:
        # Raise rather than skip. "Strictly more information than the stats
        # model" is this module's central fairness claim, and nothing else
        # enforced it at runtime: a future frame rename would quietly shrink
        # the prompt BELOW parity with FEATURE_COLUMNS while the table's
        # footnote went on printing the superset claim, biasing the result
        # against the LLM for a reason having nothing to do with the model.
        # Found by stats-reviewer, 2026-08-31.
        raise ValueError(
            f"render_prompt() is missing PROMPT_FIELDS {missing} -- the prompt "
            f"would silently drop below the information set the table claims. "
            f"Row carried: {sorted(row)}"
        )
    return "\n".join(f"{name}: {row[name]}" for name in PROMPT_FIELDS)


# --- the instrumented client -------------------------------------------------


# Same values as src/llm/client.py's, for the same reason (~15 req/min per
# model on the free tier). Restated rather than imported: bench/ must be able
# to measure the raw API without inheriting the production client's shape.
_MAX_RETRIES = 6
_DEFAULT_RETRY_DELAY_S = 15.0
# Free tier is 15 requests/minute per model (measured: the 429 body says
# `quotaValue: '15'`). 60/15 plus a cushion for clock skew against the
# server's own window.
_MIN_INTERVAL_S = 4.3
# Free tier is ALSO capped PER DAY, PER MODEL -- and the cap is NOT the same
# for every model, which is the part that cost this block two runs. Both
# figures below are measured from real 429 bodies
# (quotaId 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'), not from
# documentation and not from memory:
#
#   gemini-3.5-flash-lite   quotaValue '500'   (2026-08-31)
#   gemini-3.5-flash        quotaValue '20'    (2026-08-31)
#
# A single DAILY_QUOTA_PER_MODEL = 500 constant was the first version of this
# fix, and it was still wrong: it would have waved through a 440-call flash
# run against a cap of 20. POSTMORTEM.md incident 8.
DAILY_QUOTA_BY_MODEL: Mapping[str, int] = {
    "gemini-3.5-flash-lite": 500,
    "gemini-3.5-flash": 20,
}
# Used only for a model with no measured entry. Deliberately the SMALLEST
# observed cap, not the largest: guessing high is how a run discovers a
# limit at call 400 with nothing persisted.
DAILY_QUOTA_UNKNOWN_MODEL = 20


def daily_quota(model: str) -> int:
    return DAILY_QUOTA_BY_MODEL.get(model, DAILY_QUOTA_UNKNOWN_MODEL)


def _retry_delay_s(exc: Exception) -> float:
    """Honour the server's own RetryInfo when it sends one -- a 429 body
    carries `retryDelay: '14s'` -- and fall back to the production client's
    fixed 15s otherwise. Sleeping the server's figure rather than a guess is
    what keeps a 1,200-call run close to the quota's actual throughput."""
    details = getattr(exc, "details", None) or {}
    for detail in (details.get("error", {}) or {}).get("details", []) or []:
        raw = str(detail.get("retryDelay", "")).rstrip("s")
        try:
            return max(float(raw), 0.0) + 0.5  # small cushion past the window
        except ValueError:
            continue
    return _DEFAULT_RETRY_DELAY_S


@dataclass
class CallRecord:
    latency_s: float
    prompt_tokens: int
    output_tokens: int


class InstrumentedGemini:
    """A forced-call Gemini client that also records latency and token usage.

    Deliberately NOT src/llm/client.py's GeminiClient. That class returns
    only the parsed function-call args and discards response.usage_metadata,
    so exact token counts -- which the cost column needs -- are unrecoverable
    through it, and widening its return type to serve a benchmark would
    change a module the B11 golden gate depends on. The forced-calling config
    below is kept byte-identical to GeminiClient.forced_call's on purpose;
    if that one changes, this must follow.

    No caching, by construction. A cached answer would make the variance
    column read 0.000 for the wrong reason -- the single most important
    number in this benchmark is also the easiest to fake.
    """

    def __init__(self, api_key: str | None = None, *, min_interval_s: float = _MIN_INTERVAL_S) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client: Any = None
        self.records: list[CallRecord] = []
        self._min_interval_s = min_interval_s
        self._last_started: float | None = None
        # Counted, not silent: a degenerate answer that quietly becomes
        # uniform would degrade both AUC and variance with no trace in the
        # table or bench.json (stats-reviewer, 2026-08-31).
        self.degenerate_answers = 0

    def _pace(self) -> None:
        """Space call starts to the known quota instead of discovering it by
        failing. Firing flat-out and retrying on 429 thrashes: the first ~15
        calls land in a second, then every subsequent call burns a full
        retry-delay to buy one request, which measured far BELOW the quota's
        actual throughput. Pacing to ~1 call / 4.1s converts that into steady
        15/min. The 429 backoff stays as the safety net -- pacing is an
        optimisation over it, never a replacement, since the quota window is
        server-side and this clock is not."""
        if self._last_started is not None:
            wait = self._min_interval_s - (time.perf_counter() - self._last_started)
            if wait > 0:
                time.sleep(wait)
        self._last_started = time.perf_counter()

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # imported lazily; bench/ may do this

            if not self._api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _call_with_backoff(self, client: Any, *, model: str, user: str, config: Any) -> tuple[Any, float]:
        """One generate_content call, retrying ONLY on 429.

        Mirrors src/llm/client.py's _call_with_backoff, whose absence here
        killed the first full run 200 calls in (POSTMORTEM.md incident 7):
        the free tier allows ~15 requests/minute per model and this is the
        most rate-limit-exposed caller in the repo.

        Returns (response, elapsed) where `elapsed` times ONLY the attempt
        that succeeded. The backoff sleep is deliberately outside the timed
        region -- including it would add tens of seconds to throttled calls
        and turn the p95-latency column into a measurement of this project's
        quota tier rather than the model's response time.
        """
        from google.genai import errors as genai_errors

        for attempt in range(_MAX_RETRIES):
            self._pace()
            started = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=model, contents=user, config=config
                )
                return response, time.perf_counter() - started
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) != 429 or attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(_retry_delay_s(exc))
        raise RuntimeError("unreachable: retry loop exited without returning or raising")

    def probabilities(self, *, model: str, user: str, temperature: float) -> np.ndarray:
        """One forced call; returns the 4-vector in OUTCOME_ORDER, normalised."""
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=BENCH_SYSTEM_PROMPT,
            temperature=temperature,
            tools=[types.Tool(function_declarations=[BENCH_TOOL])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[BENCH_TOOL["name"]]
                )
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response, elapsed = self._call_with_backoff(client, model=model, user=user, config=config)

        usage = getattr(response, "usage_metadata", None)
        self.records.append(
            CallRecord(
                latency_s=elapsed,
                prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            )
        )

        args = _extract_call_args(response)
        raw = np.array(
            [
                float(args.get("p_still_pending", 0.0)),
                float(args.get("p_recovered", 0.0)),
                float(args.get("p_dead", 0.0)),
                float(args.get("p_opted_out", 0.0)),
            ],
            dtype=float,
        )
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total <= 0:
            # An all-zero answer is not a distribution. Fall back to uniform
            # and let it score as uninformative rather than crashing a
            # multi-hour run or, worse, silently propagating zeros into AUC.
            # Counted so the rate reaches the table instead of vanishing.
            self.degenerate_answers += 1
            return np.full(4, 0.25)
        return raw / total


def _extract_call_args(response: Any) -> dict[str, Any]:
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "args", None) is not None:
                return dict(fc.args)
    raise RuntimeError("forced call returned no function call")


# --- the run -----------------------------------------------------------------


CACHE_DIR = _REPO_ROOT / "reports" / ".bench_cache"

# Hash of everything that changes what a call MEANS. A prompt or tool-schema
# edit must invalidate the cache; a change to n/repeats must not.
PROMPT_VERSION = __import__("hashlib").sha256(
    (BENCH_SYSTEM_PROMPT + str(BENCH_TOOL) + str(PROMPT_FIELDS)).encode()
).hexdigest()[:12]


def plan_budget(*, n: int, repeats: int, variance_n: int, temperatures: Sequence[float],
                models: Sequence[str]) -> dict[str, int]:
    """Calls this configuration will make, per model. Checked BEFORE any client
    is built, because the alternative -- discovering the daily cap at call 400
    of 600 -- costs the whole run (POSTMORTEM.md incident 8)."""
    per_model = n + repeats * variance_n * len(temperatures)
    return {model: per_model for model in models}


def assert_within_budget(budget: Mapping[str, int], *, already_spent: int = 0) -> None:
    over = {
        m: (c, daily_quota(m)) for m, c in budget.items()
        if c + already_spent > daily_quota(m)
    }
    if over:
        detail = "; ".join(
            f"{m}: {c} planned vs {q}/day" for m, (c, q) in sorted(over.items())
        )
        raise ValueError(
            f"this run exceeds the daily free-tier quota -- {detail} "
            f"(already spent today: {already_spent}). Lower --n, --repeats or "
            f"--variance-n, or pass fewer --model values. Quotas differ PER MODEL "
            f"and are checked here rather than discovered mid-run -- see "
            f"POSTMORTEM.md incident 8."
        )


class CallCache:
    """Append-only JSONL cache, flushed after EVERY live call.

    Mirrors eval/golden_check.py's _persisting(): an interrupted multi-hundred
    call run must resume, never re-bill. End-of-run persistence is no
    protection against the failure that actually happens, which is the run not
    reaching its end.
    """

    def __init__(self, model: str, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = CACHE_DIR / f"{model.replace('/', '_')}__{PROMPT_VERSION}.jsonl"
        self.hits = 0
        self.misses = 0
        self._data: dict[str, list[float]] = {}
        if enabled and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._data[rec["k"]] = rec["p"]

    @staticmethod
    def key(*, temperature: float, repeat: int, row_index: int) -> str:
        return f"t{temperature}|r{repeat}|i{row_index}"

    def get(self, k: str) -> list[float] | None:
        if not self.enabled:
            return None
        hit = self._data.get(k)
        if hit is not None:
            self.hits += 1
        return hit

    def put(self, k: str, probs: Sequence[float]) -> None:
        self.misses += 1
        if not self.enabled:
            return
        self._data[k] = list(probs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "p": list(probs)}) + "\n")


@dataclass
class ArmResult:
    name: str
    auc: float
    auc_ci: tuple[float, float]
    log_loss: float
    brier: dict[int, float]
    p95_latency_s: float
    latency_kind: str
    cost_per_1k_paise: int
    n_scored: int
    degenerate_answers: int = 0
    variance: dict[str, VarianceReport] = field(default_factory=dict)
    note: str = ""
    # Kept so any metric can be recomputed later WITHOUT re-calling the API.
    # A 1200-call run costs ~80 minutes of quota; discarding the raw
    # probabilities would mean paying that again to add one column.
    probs: list[list[float]] = field(default_factory=list)
    # Every repeat of the variance pass, keyed by temperature. Without
    # these, adding a single variance metric later would cost another full
    # ~80-minute quota run.
    variance_runs: dict[str, list[list[list[float]]]] = field(default_factory=dict)
    variance_row_index: list[int] = field(default_factory=list)


def build_test_split(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `(train, test)`, where `test` is the held-out arm of the
    mandate-level 4-way split, estimable rows only.

    Both are returned because the caller needs them separately and must not
    be able to confuse them: the stats arm fits on `train` and scores `test`,
    and handing back only one frame would leave the fit/score boundary to a
    convention rather than to the type. Built from eval.corpus's TRAIN_SEEDS
    draw -- the SAME recipe, corpus and split seed as B8's fit. (An earlier
    draft said "the model this project actually ships"; stats-reviewer noted
    that is literally false -- this is a refit inside the benchmark, and the
    sentence is in any case conditional on seed 0, which `--seed` can change.
    Corrected rather than defended.)

    The split is GROUPED ON mandate_id (splits.py's default), so no mandate
    contributes rows to both sides. A row-level split would let the same
    mandate's slot-2 row train the model that scores its slot-3 row, which
    is the leakage src/model/CLAUDE.md's rule 2 exists to prevent.
    """
    episodes = corpus.generate(arm="nominal")
    pp = person_period.build(episodes)
    feat = features.featurize(pp)
    assembled = competing_risks.assemble(pp, feat)
    train, _calib_iso, _calib_conf, test = splits.split(assembled, seed=seed)
    return train, test[test["estimable"]].reset_index(drop=True)


def stats_arm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[ArmResult, np.ndarray]:
    """Fit on `train`, score `test`. The fit never sees a test row -- that is
    the clause PLAN_DETAIL.md's stats-reviewer note actually turns on."""
    started = time.perf_counter()
    model = competing_risks.fit(train)
    fit_s = time.perf_counter() - started

    # Per-ROW timing, matching the LLM arm's estimand. The previous version
    # timed three whole-batch hazards() calls and divided by row count, then
    # took a percentile over those three numbers -- which is the max of three
    # measurements of the same amortised throughput, not a tail statistic, and
    # not comparable to a per-call wall clock printed beside it under one
    # header (stats-reviewer, 2026-08-31).
    per_row: list[float] = []
    for i in range(min(len(test), 200)):
        one = test.iloc[[i]]
        t0 = time.perf_counter()
        competing_risks.hazards(model, one)
        per_row.append(time.perf_counter() - t0)
    probs = competing_risks.hazards(model, test)

    y_true = test["event_code"].to_numpy(dtype=int)
    groups = test["mandate_id"].to_numpy()
    return (
        ArmResult(
            name="competing-risks model",
            auc=macro_ovr_auc(y_true, probs),
            auc_ci=cluster_bootstrap_ci(y_true, probs, groups),
            log_loss=multiclass_log_loss(y_true, probs),
            brier=brier_per_class(y_true, probs),
            p95_latency_s=p95(per_row),
            latency_kind="per-row, local",
            cost_per_1k_paise=0,
            n_scored=len(test),
            note=(
                f"deterministic; fit {fit_s:.2f}s; "
                f"{len(np.unique(probs.round(12), axis=0))} distinct probability vectors"
            ),
            probs=probs.tolist(),
        ),
        probs,
    )


def null_arm(train: pd.DataFrame, test: pd.DataFrame) -> ArmResult:
    """An intercept-only fit: the base rates, and nothing else. Both real arms
    are measured against it so "better than the LLM" cannot quietly mean
    "better than nothing". stats-reviewer's point: without a shared null, two
    scores near chance are indistinguishable from two scores that are simply
    both uninformative."""
    model = competing_risks.fit(train, intercept_only=True)
    probs = competing_risks.hazards(model, test)
    y_true = test["event_code"].to_numpy(dtype=int)
    groups = test["mandate_id"].to_numpy()
    return ArmResult(
        name="intercept-only null (base rates)",
        auc=macro_ovr_auc(y_true, probs),
        auc_ci=cluster_bootstrap_ci(y_true, probs, groups),
        log_loss=multiclass_log_loss(y_true, probs),
        brier=brier_per_class(y_true, probs),
        p95_latency_s=0.0,
        latency_kind="n/a",
        cost_per_1k_paise=0,
        n_scored=len(test),
        note="reference, not a competitor",
        probs=probs.tolist(),
    )


def llm_arm(
    *,
    model: str,
    test: pd.DataFrame,
    repeats: int,
    variance_n: int,
    temperatures: Sequence[float],
    usd_inr_paise: int,
    pricing: Mapping[str, Any],
    use_cache: bool = True,
) -> ArmResult:
    """Accuracy over the full sample once at t=0.0, then `repeats` passes
    over a `variance_n` subsample at each temperature."""
    client = InstrumentedGemini()
    cache = CallCache(model, enabled=use_cache)
    y_true = test["event_code"].to_numpy(dtype=int)
    rows = [test.iloc[i].to_dict() for i in range(len(test))]

    def _score(
        rs: Sequence[Mapping[str, object]], temp: float, label: str, *,
        repeat: int, row_ids: Sequence[int],
    ) -> np.ndarray:
        out = []
        for i, (r, row_id) in enumerate(zip(rs, row_ids), 1):
            key = CallCache.key(temperature=temp, repeat=repeat, row_index=row_id)
            hit = cache.get(key)
            if hit is not None:
                out.append(np.asarray(hit, dtype=float))
            else:
                probs = client.probabilities(
                    model=model, user=render_prompt(r), temperature=temp
                )
                # Flushed to disk immediately, before the next call is made.
                # An interrupted run must resume, never re-bill -- 400 calls
                # were lost to end-of-run persistence (POSTMORTEM.md 8).
                cache.put(key, probs)
                out.append(probs)
            if i % 25 == 0 or i == len(rs):
                print(
                    f"    [{model} {label}] {i}/{len(rs)} "
                    f"(cache {cache.hits} hit / {cache.misses} live)",
                    flush=True,
                )
        return np.array(out)

    accuracy_probs = _score(
        rows, 0.0, "accuracy", repeat=0, row_ids=list(range(len(rows)))
    )

    variance: dict[str, VarianceReport] = {}
    raw_runs: dict[str, list] = {}
    # A draw, not rows[:variance_n]. At --n 200 the frame is already randomly
    # sampled so a head was harmless, but at --n 1212 it would silently become
    # the first ~26 mandates in mandate_id order (stats-reviewer, 2026-08-31).
    sub_idx = list(test.sample(n=min(variance_n, len(rows)), random_state=0).index)
    sub = [rows[i] for i in sub_idx]
    for temp in temperatures:
        runs = [
            _score(
                sub, temp, f"variance t={temp} rep{k + 1}",
                repeat=k + 1, row_ids=[int(i) for i in sub_idx],
            )
            for k in range(repeats)
        ]
        variance[f"t={temp}"] = variance_report(runs)
        raw_runs[f"t={temp}"] = [r.tolist() for r in runs]

    prompt_tokens = sum(rec.prompt_tokens for rec in client.records)
    output_tokens = sum(rec.output_tokens for rec in client.records)
    groups = test["mandate_id"].to_numpy()
    return ArmResult(
        name=f"{model} as classifier",
        auc=macro_ovr_auc(y_true, accuracy_probs),
        auc_ci=cluster_bootstrap_ci(y_true, accuracy_probs, groups),
        log_loss=multiclass_log_loss(y_true, accuracy_probs),
        brier=brier_per_class(y_true, accuracy_probs),
        p95_latency_s=p95([rec.latency_s for rec in client.records]),
        latency_kind="per-call, network",
        cost_per_1k_paise=cost_per_1k_paise(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            n_calls=len(client.records),
            model=model,
            usd_inr_paise=usd_inr_paise,
            pricing=pricing,
        ),
        n_scored=len(test),
        degenerate_answers=client.degenerate_answers,
        variance=variance,
        note=(
            f"{cache.misses} live calls, {cache.hits} cached; variance subsample n={len(sub)}; "
            f"{client.degenerate_answers} degenerate answer(s) coerced to uniform"
        ),
        probs=accuracy_probs.tolist(),
        variance_runs=raw_runs,
        variance_row_index=[int(i) for i in sub_idx],
    )


def _fmt_latency(seconds: float) -> str:
    """Pick a unit that keeps significant digits. The two arms are ~7 orders
    of magnitude apart -- a local matrix multiply against a network round
    trip -- so a single fixed unit prints one of them as 0.0."""
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.1f} us"


def render_table(arms: Sequence[ArmResult], *, pricing: Mapping[str, Any], seed: int) -> str:
    header = (
        "| arm | log loss (lower=better) | macro OvR AUC [95% CI] | p95 latency | "
        "cost / 1k decisions | argmax flip t=0.0 | argmax flip t=1.0 | "
        "REC-vs-DEAD order flip t=0.0 |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for a in arms:
        v0 = a.variance.get("t=0.0")
        v1 = a.variance.get("t=1.0")
        det = "0.000 (deterministic)"
        cost = "0 (local)" if a.cost_per_1k_paise == 0 else f"{a.cost_per_1k_paise} paise"
        lat = "n/a" if a.latency_kind == "n/a" else f"{_fmt_latency(a.p95_latency_s)} ({a.latency_kind})"
        lines.append(
            f"| {a.name} | {a.log_loss:.4f} | "
            f"{a.auc:.4f} [{a.auc_ci[0]:.3f}, {a.auc_ci[1]:.3f}] | {lat} | {cost} | "
            f"{det if v0 is None else f'{v0.argmax_flip_rate:.3f} [{v0.argmax_flip_ci[0]:.2f},{v0.argmax_flip_ci[1]:.2f}]'} | "
            f"{det if v1 is None else f'{v1.argmax_flip_rate:.3f} [{v1.argmax_flip_ci[0]:.2f},{v1.argmax_flip_ci[1]:.2f}]'} | "
            f"{det if v0 is None else f'{v0.cause_ordering_flip_rate:.3f}'} |"
        )

    brier_rows = ["", "Per-class Brier (lower is better):", "",
                  "| arm | STILL_PENDING | RECOVERED | DEAD | OPTED_OUT |", "|---|---|---|---|---|"]
    for a in arms:
        brier_rows.append(
            f"| {a.name} | " + " | ".join(f"{a.brier.get(c, float('nan')):.4f}" for c in range(4)) + " |"
        )

    swings = [
        f"- {a.name}, {k}: max |delta P(OPTED_OUT)| across repeats = {v.max_optout_swing:.3f}"
        for a in arms for k, v in sorted(a.variance.items())
    ]
    swing_block = (["", "Opt-out probability swing on identical input:", ""] + swings) if swings else []

    degen = [f"- {a.name}: {a.degenerate_answers}" for a in arms if a.degenerate_answers]
    degen_block = (["", "Degenerate answers coerced to uniform (excluded from no metric, "
                    "so they DEGRADE the arm that produced them):", ""] + degen) if degen else []

    footnotes = f"""

**Headline is log loss, not AUC.** On this problem the per-class AUCs are
0.534 / 0.569 / 0.487 / 0.714 for STILL_PENDING / RECOVERED / DEAD /
OPTED_OUT -- DEAD is *below chance* and STILL_PENDING barely above it. That
is structural, not a bad fit: the frozen simulator sets the DEAD hazard from
the latent cause alone, and a design matrix of (const, slot_3, slot_4,
in_salary_window) cannot separate it. Macro-averaging spends half its weight
on classes the model provably cannot rank, leaving almost no room above
chance for either arm to lose in. Log loss is a proper scoring rule, rewards
calibration rather than ranking alone, and is the quantity the allocator's
backward induction actually consumes -- a miscalibrated model that ranks
well is useless downstream and would still win on AUC.

**Read the CI before reading the AUC.** It is a mandate-level cluster
bootstrap (2000 resamples of mandates, not rows). Overlapping intervals mean
a tie, however many decimal places the point estimates differ by.

Split: `src/model/splits.py` 4-way mandate-level split, seed {seed}, `test`
arm, estimable rows only. The stats model is fit on `train` and never sees a
scored row; the LLM was fit on neither. Rows are clustered -- one mandate
contributes up to three slot rows -- which is why every interval here
resamples mandates.

**Information set.** The LLM is shown strictly more of the frame than the
stats model uses (`PROMPT_FIELDS`: eight columns against the design matrix's
`slot` and `in_salary_window`). It never sees `event_code` or any latent
cause, and `render_prompt()` raises rather than silently dropping a field.
Note the honest limit of this framing: strictly more information is not
automatically an advantage for a zero-shot model -- six additional covariates
can mislead as easily as inform -- so "shown more" is demonstrated, while
"handicapped in its favour" is a weaker claim than it first sounds.

**"REC-vs-DEAD order flip"** is exactly what its name says: how often
P(RECOVERED) > P(DEAD) changed across identical repeats. It is NOT a count of
changed retry slots -- no slot or day is computed in this module -- and it
ignores OPTED_OUT entirely, so it is a LOWER BOUND on decision instability.
The opt-out swing figures above cover the gap it leaves.

Cost is priced at PAID-TIER rates ({pricing['source']}, read
{pricing['retrieved_on']}); the run used free-tier keys, whose marginal price
is zero. Converted at USD 1 = INR {pricing['usd_inr_paise'] / 100:.2f}
({pricing['usd_inr_retrieved_on']}).
"""
    return (
        header
        + "\n".join(lines)
        + "\n"
        + "\n".join(brier_rows + swing_block + degen_block)
        + footnotes
    )


def main(n: int = 200, repeats: int = 5, *, variance_n: int = 40, seed: int = 0,
         models: Sequence[str] | None = None, dry_run: bool = False,
         use_cache: bool = True, already_spent: int = 0) -> int:
    pricing = load_pricing()
    usd_inr_paise = int(pricing["usd_inr_paise"])
    models = tuple(models or ("gemini-3.5-flash-lite", "gemini-3.5-flash"))

    if not dry_run:
        budget = plan_budget(n=n, repeats=repeats, variance_n=variance_n,
                             temperatures=(0.0, 1.0), models=models)
        assert_within_budget(budget, already_spent=already_spent)
        print("planned live calls: " + ", ".join(
            f"{m}={c} (quota {daily_quota(m)}/day)" for m, c in sorted(budget.items())
        ))

    train, test = build_test_split(seed=seed)
    if n < len(test):
        test = test.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"test split: {len(test)} estimable rows, {test['mandate_id'].nunique()} mandates")

    arms: list[ArmResult] = []
    null = null_arm(train, test)
    arms.append(null)
    print(f"  {null.name}: log loss {null.log_loss:.4f} AUC {null.auc:.4f}")

    stats, _ = stats_arm(train, test)
    arms.append(stats)
    print(f"  {stats.name}: log loss {stats.log_loss:.4f} AUC {stats.auc:.4f} ({stats.note})")

    if dry_run:
        print("dry run -- no live calls made, LLM arms skipped")
    else:
        for model in models:
            arm = llm_arm(
                model=model, test=test, repeats=repeats, variance_n=variance_n,
                temperatures=(0.0, 1.0), usd_inr_paise=usd_inr_paise, pricing=pricing,
                use_cache=use_cache,
            )
            arms.append(arm)
            print(f"  {arm.name}: log loss {arm.log_loss:.4f} AUC {arm.auc:.4f} ({arm.note})")
            for label, v in arm.variance.items():
                print(f"    variance {label}: {v.summary()}")

    table = render_table(arms, pricing=pricing, seed=seed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "seed": seed, "n": len(test), "repeats": repeats, "variance_n": variance_n,
                # str(): PyYAML parses an unquoted ISO date into datetime.date,
                # which json.dumps refuses.
                "pricing_source": pricing["source"],
                "pricing_retrieved_on": str(pricing["retrieved_on"]),
                "usd_inr_paise": usd_inr_paise,
                # y_true and per-arm `probs` are persisted so ANY metric can be
                # recomputed offline. A full run costs ~80 minutes of free-tier
                # quota; throwing the raw probabilities away would mean paying
                # that again to add a single column, which is how a benchmark
                # quietly stops being re-analysable.
                "y_true": test["event_code"].astype(int).tolist(),
                "mandate_id": test["mandate_id"].tolist(),
                "slot": test["slot"].astype(int).tolist(),
                "arms": [
                    {
                        "name": a.name, "auc": a.auc, "auc_ci": list(a.auc_ci),
                        "log_loss": a.log_loss,
                        "brier": {str(k): v for k, v in a.brier.items()},
                        "p95_latency_s": a.p95_latency_s, "latency_kind": a.latency_kind,
                        "cost_per_1k_paise": a.cost_per_1k_paise, "n_scored": a.n_scored,
                        "degenerate_answers": a.degenerate_answers,
                        "note": a.note,
                        "variance": {k: vars(v) for k, v in a.variance.items()},
                        "probs": a.probs,
                        "variance_runs": a.variance_runs,
                        "variance_row_index": a.variance_row_index,
                    }
                    for a in arms
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {REPORT_PATH.relative_to(_REPO_ROOT)}")
    print("\n" + table)
    print("Paste the table above into DECISIONS.md under 'The benchmark'.")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=200, help="rows in the accuracy pass")
    ap.add_argument("--repeats", type=int, default=5, help="identical-input repeats for the variance pass")
    ap.add_argument("--variance-n", type=int, default=40, help="subsample size for the variance pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", action="append", dest="models", default=None)
    ap.add_argument("--dry-run", action="store_true", help="stats arm only; no live API calls")
    ap.add_argument("--no-cache", action="store_true",
                    help="force a live call for every row (ignores reports/.bench_cache/)")
    ap.add_argument("--already-spent", type=int, default=0,
                    help="calls already made against today's per-model quota, for the budget check")
    return ap.parse_args(argv)


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    _a = _parse_args()
    raise SystemExit(
        main(_a.n, _a.repeats, variance_n=_a.variance_n, seed=_a.seed,
             models=_a.models, dry_run=_a.dry_run, use_cache=not _a.no_cache,
             already_spent=_a.already_spent)
    )
