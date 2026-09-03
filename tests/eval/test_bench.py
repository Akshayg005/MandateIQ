"""tests/eval/test_bench.py -- benchmarking LLM-as-classifier against the
statistical model.

bench/llm_vs_stats.py is a bare script, not a package -- tests import it via
importlib from the repo root. All tests are offline (no API calls), use
deterministic synthetic data, and verify contract invariants (money is integer
paise, outcome order matches the model's enum, prompt leakage is blocked).

The most load-bearing test is the leakage guard: even though PROMPT_FIELDS
is declared as a fixed tuple, render_prompt() must be tested to prove it
actually applies that allowlist, not merely asserts it exists. The contract
depends on this being enforced in the code, not hope.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pytest

# Import the bare script via importlib.
_BENCH_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "bench" / "llm_vs_stats.py"
_IMPORT_ERROR: str | None = None
_llm_vs_stats = None
try:
    _spec = importlib.util.spec_from_file_location("llm_vs_stats", _BENCH_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"could not build an import spec for {_BENCH_PATH}")
    _llm_vs_stats = importlib.util.module_from_spec(_spec)
    sys.modules["llm_vs_stats"] = _llm_vs_stats
    _spec.loader.exec_module(_llm_vs_stats)
except Exception as exc:  # noqa: BLE001 -- reported as a test failure below
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Import types from the codebase.
from src.core.types import Outcome
from src.model.features import FORBIDDEN


# --- Fixture: safely import bench module or skip if not yet written ---


@pytest.fixture
def bench_module():
    """Import bench/llm_vs_stats.py via importlib -- as a bare script from
    the repo root, which is how .\\run.ps1 bench actually invokes it.

    A missing or unimportable module FAILS here; it must never skip. An
    earlier draft of this file skipped instead, which meant a renamed or
    deleted bench script would turn every test in this file green-by-absence
    and the B12 gate would read as passing having checked nothing. That is
    the same vacuous shape the 2026-08-29 audit removed from the gates and
    from run.ps1's live-key scan (DECISIONS.md); it does not get to
    reappear here."""
    if _IMPORT_ERROR is not None:
        pytest.fail(f"cannot import {_BENCH_PATH}: {_IMPORT_ERROR}")
    return _llm_vs_stats


# --- Tests ---


def test_prompt_fields_excludes_forbidden_columns(bench_module):
    """PROMPT_FIELDS (the LLM allowlist) must not include any column from
    src.model.features.FORBIDDEN. This enforces the information barrier:
    the LLM sees only features safe to show, never the model's own internal
    columns."""
    prompt_fields_set = set(bench_module.PROMPT_FIELDS)
    forbidden_set = set(FORBIDDEN)

    overlap = prompt_fields_set & forbidden_set
    assert overlap == set(), (
        f"PROMPT_FIELDS overlaps FORBIDDEN: {overlap}"
    )


def test_prompt_fields_excludes_event_code_outcome_and_cause_keywords(bench_module):
    """PROMPT_FIELDS must not contain "event_code", "outcome", or any name
    containing "cause" -- these are the labels and latent causes the LLM
    must never see, to prevent data leakage."""
    prompt_fields = bench_module.PROMPT_FIELDS

    forbidden_exact = {"event_code", "outcome"}
    for name in prompt_fields:
        assert name not in forbidden_exact, (
            f"{name!r} in PROMPT_FIELDS is forbidden"
        )
        assert "cause" not in name.lower(), (
            f"{name!r} in PROMPT_FIELDS contains 'cause' -- forbidden"
        )


def test_render_prompt_does_not_leak_forbidden_columns(bench_module):
    """render_prompt() receives a row dict that includes both allowed and
    forbidden columns. The returned prompt string must contain NEITHER the
    forbidden column values nor the words 'event_code', 'outcome',
    'initial_cause', etc. This proves the allowlist is actually applied,
    not merely declared."""
    row = {
        # Every allowed column -- render_prompt now RAISES on a missing
        # PROMPT_FIELD (stats-reviewer, 2026-08-31), so a partial row would
        # test incompleteness rather than leakage.
        "slot": 3,
        "in_salary_window": 1,
        "amount_paise": 50_000,
        "ceiling_paise": 200_000,
        "category": "subscription",
        "prior_failures_this_cycle": 2,
        "committed_day_of_month": 7,
        "days_since_last_attempt": 4,
        # Forbidden columns that should NOT appear in the prompt.
        "event_code": "ATTEMPT_FAILED_INSUFFICIENT_FUNDS",
        "initial_cause": "CANT_PAY_NOW",
        "outcome": "STILL_PENDING",
        "ledger_id": 12345,
    }

    prompt = bench_module.render_prompt(row)

    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # The prompt must NOT contain the forbidden values.
    assert "ATTEMPT_FAILED_INSUFFICIENT_FUNDS" not in prompt
    assert "CANT_PAY_NOW" not in prompt
    assert "STILL_PENDING" not in prompt
    assert str(12345) not in prompt

    # The prompt must NOT contain the forbidden column names themselves.
    assert "event_code" not in prompt.lower()
    assert "initial_cause" not in prompt.lower()
    assert "outcome" not in prompt
    assert "ledger_id" not in prompt.lower()


def test_macro_ovr_auc_perfect_separation_returns_1_0(bench_module):
    """macro_ovr_auc on a perfectly separable 4-class problem (each class
    has probability 1 on its own true label) returns 1.0."""
    # y_true: 0, 1, 2, 3, 0, 1, 2, 3, ... one of each class.
    n_samples = 100
    y_true = np.array([i % 4 for i in range(n_samples)])

    # p: perfectly confident on each row's true class.
    p = np.zeros((n_samples, 4))
    for i in range(n_samples):
        p[i, y_true[i]] = 1.0

    auc = bench_module.macro_ovr_auc(y_true, p)
    assert isinstance(auc, (float, np.floating))
    assert auc == pytest.approx(1.0)


def test_macro_ovr_auc_uniform_probabilities_returns_approx_0_5(bench_module):
    """macro_ovr_auc on uniform probabilities (each class gets 0.25) for a
    balanced 4-class problem returns approximately 0.5."""
    n_samples = 100
    y_true = np.array([i % 4 for i in range(n_samples)])

    # p: uniform 0.25 on all classes for every sample.
    p = np.full((n_samples, 4), 0.25)

    auc = bench_module.macro_ovr_auc(y_true, p)
    assert isinstance(auc, (float, np.floating))
    assert 0.4 < auc < 0.6, f"uniform probabilities should yield ~0.5, got {auc}"


def test_macro_ovr_auc_tie_heavy_case_is_finite_and_in_range(bench_module):
    """macro_ovr_auc on a tie-heavy distribution (only 6 distinct probability
    vectors across many rows, simulating real model output where many rows
    have similar predictions) returns a finite value in [0, 1]. This guard
    proves ties do not cause NaN or infinity."""
    n_samples = 1000
    n_distinct = 6

    # Create only 6 distinct probability vectors, repeated across 1000 rows.
    y_true = np.tile(np.arange(4), n_samples // 4)[:n_samples]

    # 6 distinct probability vectors (some tie, most favoring class 0).
    base_vectors = [
        [0.7, 0.1, 0.1, 0.1],  # Favors 0
        [0.6, 0.2, 0.1, 0.1],  # Favors 0, weaker
        [0.5, 0.3, 0.1, 0.1],  # Favors 0, weaker still
        [0.3, 0.3, 0.2, 0.2],  # Tie between 0 and 1
        [0.25, 0.25, 0.25, 0.25],  # Uniform
        [0.1, 0.1, 0.4, 0.4],  # Favors 2 and 3
    ]
    p = np.array([base_vectors[i % n_distinct] for i in range(n_samples)])

    auc = bench_module.macro_ovr_auc(y_true, p)
    assert isinstance(auc, (float, np.floating))
    assert np.isfinite(auc), f"macro_ovr_auc returned non-finite: {auc}"
    assert 0.0 <= auc <= 1.0, f"macro_ovr_auc out of range: {auc}"


def test_p95_on_known_list_matches_order_statistic(bench_module):
    """p95(latencies) on a known list returns the 95th percentile order
    statistic."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # 10 items
    # 95th percentile is between indices 8 and 9 (0-indexed): between 40 and 50.
    # numpy.percentile uses linear interpolation: 0.95 * 9 = 8.55, so 40 + 0.55*(50-40) = 45.5

    p95_value = bench_module.p95(latencies)
    expected = np.percentile(latencies, 95)
    assert p95_value == pytest.approx(expected)


def test_p95_on_single_element_list_returns_that_element(bench_module):
    """p95 on a single-element list returns that element (edge case)."""
    latencies = [5.0]
    p95_value = bench_module.p95(latencies)
    assert p95_value == 5.0


def test_p95_raises_on_empty_list(bench_module):
    """p95 on an empty list raises an error."""
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        bench_module.p95([])


def test_cost_per_1k_paise_returns_integer(bench_module):
    """cost_per_1k_paise returns an int (root CLAUDE.md invariant 2: all
    money is integer paise -- a float touching a money value is a bug)."""
    result = bench_module.cost_per_1k_paise(
        prompt_tokens=100,
        output_tokens=50,
        n_calls=10,
        model="gemini-3.5-flash-lite",
        usd_inr_paise=9569,
    )

    assert isinstance(result, int), (
        f"cost_per_1k_paise must return int, got {type(result)}"
    )


def test_cost_per_1k_paise_requires_an_explicit_fx_rate(bench_module):
    """usd_inr_paise is required, not defaulted. Gemini is priced in USD and
    this repo reports paise; a default would mean an exchange rate invented
    inside the benchmark and silently baked into a published table. PLAN.md:
    "Never fabricate a number." The caller must state the rate it used, so
    the table can cite it."""
    with pytest.raises(TypeError):
        bench_module.cost_per_1k_paise(
            prompt_tokens=100,
            output_tokens=50,
            n_calls=10,
            model="gemini-3.5-flash-lite",
        )


def test_cost_per_1k_paise_rejects_an_unpriced_model(bench_module):
    """An unknown model id raises rather than silently costing zero. A model
    swapped via MODEL_NORMALIZER with no price entry must not produce a
    free-looking row in the benchmark table."""
    with pytest.raises(KeyError):
        bench_module.cost_per_1k_paise(
            prompt_tokens=100,
            output_tokens=50,
            n_calls=10,
            model="gemini-9.9-imaginary",
            usd_inr_paise=9569,
        )


def test_cost_per_1k_paise_doubling_halves_per_1k(bench_module):
    """cost_per_1k_paise(n_calls=X) divided by cost_per_1k_paise(n_calls=2*X)
    should be approximately 2.0 (doubling calls halves the per-1k amortized
    cost). Verifies the function respects the name's "per_1k" semantics."""
    result_x = bench_module.cost_per_1k_paise(
        prompt_tokens=100,
        output_tokens=50,
        n_calls=10,
        model="gemini-3.5-flash-lite",
        usd_inr_paise=9569,
    )
    result_2x = bench_module.cost_per_1k_paise(
        prompt_tokens=100,
        output_tokens=50,
        n_calls=20,
        model="gemini-3.5-flash-lite",
        usd_inr_paise=9569,
    )

    # result_x / result_2x should be approximately 2.0, within rounding.
    ratio = result_x / result_2x if result_2x != 0 else 0
    assert 1.8 < ratio < 2.2, (
        f"doubling n_calls should roughly halve the per-1k cost, got "
        f"{result_x} / {result_2x} = {ratio}"
    )


def test_variance_report_identical_repeats_have_zero_variance(bench_module):
    """variance_report over runs where every repeat is byte-identical returns
    argmax_flip_rate == 0.0 and max_prob_stddev == 0.0. This proves the
    variance column can detect when there is NO variance."""
    n_rows = 10
    n_repeats = 3

    # All repeats are identical.
    p_fixed = np.array([
        [0.7, 0.1, 0.1, 0.1],  # Argmax = 0
        [0.1, 0.7, 0.1, 0.1],  # Argmax = 1
        [0.2, 0.2, 0.5, 0.1],  # Argmax = 2
        [0.2, 0.2, 0.2, 0.4],  # Argmax = 3
        [0.5, 0.3, 0.1, 0.1],  # Argmax = 0
        [0.1, 0.1, 0.1, 0.7],  # Argmax = 3
        [0.4, 0.3, 0.2, 0.1],  # Argmax = 0
        [0.2, 0.3, 0.2, 0.3],  # Argmax = 1
        [0.1, 0.1, 0.7, 0.1],  # Argmax = 2
        [0.25, 0.25, 0.25, 0.25],  # Argmax = 0 (first)
    ])

    runs = [p_fixed for _ in range(n_repeats)]

    report = bench_module.variance_report(runs)

    assert report.argmax_flip_rate == 0.0, (
        f"identical repeats should have no argmax flips, got "
        f"{report.argmax_flip_rate}"
    )
    # Tolerance, not exact equality: np.std over identical float64 vectors
    # returns ~1e-16, not 0.0, because the mean-subtraction step is not exact
    # in IEEE arithmetic. The tolerance lives HERE rather than as a snap-to-
    # zero inside variance_report(), deliberately -- a threshold in the
    # implementation would report genuine sub-threshold movement as "no
    # variance", and this is the column the entire B12 argument rests on.
    # 1e-12 is ten thousand times larger than the observed noise and still
    # far below any probability difference that could change a decision.
    assert report.max_prob_stddev == pytest.approx(0.0, abs=1e-12), (
        f"identical repeats should have zero probability stddev, got "
        f"{report.max_prob_stddev}"
    )


def test_variance_report_diverging_repeats_detect_variance(bench_module):
    """variance_report over runs where at least one row's argmax differs
    across repeats returns argmax_flip_rate > 0. This is the positive case:
    proves the variance column can detect ACTUAL variance. Without both this
    and the identical-repeats test, the variance reporting could be
    structurally incapable of detecting variance and every test would still
    pass."""
    n_rows = 10

    # Repeat 1: argmax = 0 for most rows.
    run1 = np.array([
        [0.7, 0.1, 0.1, 0.1],  # Argmax = 0
        [0.1, 0.7, 0.1, 0.1],  # Argmax = 1
        [0.2, 0.2, 0.5, 0.1],  # Argmax = 2
        [0.2, 0.2, 0.2, 0.4],  # Argmax = 3
        [0.5, 0.3, 0.1, 0.1],  # Argmax = 0
        [0.1, 0.1, 0.1, 0.7],  # Argmax = 3
        [0.4, 0.3, 0.2, 0.1],  # Argmax = 0
        [0.2, 0.3, 0.2, 0.3],  # Argmax = 1
        [0.1, 0.1, 0.7, 0.1],  # Argmax = 2
        [0.25, 0.25, 0.25, 0.25],  # Argmax = 0
    ])

    # Repeat 2: same as repeat 1, but row 0 has argmax = 1 instead of 0.
    run2 = np.array([
        [0.1, 0.7, 0.1, 0.1],  # Argmax = 1 (different from run1)
        [0.1, 0.7, 0.1, 0.1],  # Argmax = 1 (same)
        [0.2, 0.2, 0.5, 0.1],  # Argmax = 2 (same)
        [0.2, 0.2, 0.2, 0.4],  # Argmax = 3 (same)
        [0.5, 0.3, 0.1, 0.1],  # Argmax = 0 (same)
        [0.1, 0.1, 0.1, 0.7],  # Argmax = 3 (same)
        [0.4, 0.3, 0.2, 0.1],  # Argmax = 0 (same)
        [0.2, 0.3, 0.2, 0.3],  # Argmax = 1 (same)
        [0.1, 0.1, 0.7, 0.1],  # Argmax = 2 (same)
        [0.25, 0.25, 0.25, 0.25],  # Argmax = 0 (same)
    ])

    runs = [run1, run2]

    report = bench_module.variance_report(runs)

    assert report.argmax_flip_rate > 0.0, (
        f"diverging repeats should detect argmax flips, got "
        f"{report.argmax_flip_rate}"
    )
    assert report.argmax_flip_rate <= 1.0


def test_variance_report_raises_on_fewer_than_two_repeats(bench_module):
    """variance_report raises on fewer than 2 repeats. A variance report
    computed from a single run is meaningless."""
    single_run = [np.array([[0.7, 0.1, 0.1, 0.1]])]

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        bench_module.variance_report(single_run)


def test_outcome_order_matches_core_types(bench_module):
    """OUTCOME_ORDER matches src.core.types.Outcome's integer order exactly.
    The stats model returns hazards in this order, and a silent transposition
    would invert the AUC."""
    expected_order = (
        Outcome.STILL_PENDING,
        Outcome.RECOVERED,
        Outcome.DEAD,
        Outcome.OPTED_OUT,
    )

    assert bench_module.OUTCOME_ORDER == expected_order, (
        f"OUTCOME_ORDER {bench_module.OUTCOME_ORDER} != "
        f"expected {expected_order}"
    )

    # Also verify the int values.
    assert Outcome.STILL_PENDING == 0
    assert Outcome.RECOVERED == 1
    assert Outcome.DEAD == 2
    assert Outcome.OPTED_OUT == 3


def test_variance_report_has_required_fields(bench_module):
    """VarianceReport has all required fields: n_rows, n_repeats,
    argmax_flip_rate, max_prob_stddev, mean_prob_stddev, decision_flip_rate."""
    # Create a minimal valid input.
    p1 = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    p2 = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    runs = [p1, p2]

    report = bench_module.variance_report(runs)

    # All required fields must be present and have the right types.
    assert hasattr(report, "n_rows")
    assert isinstance(report.n_rows, int)

    assert hasattr(report, "n_repeats")
    assert isinstance(report.n_repeats, int)

    assert hasattr(report, "argmax_flip_rate")
    assert isinstance(report.argmax_flip_rate, (float, np.floating))

    assert hasattr(report, "max_prob_stddev")
    assert isinstance(report.max_prob_stddev, (float, np.floating))

    assert hasattr(report, "mean_prob_stddev")
    assert isinstance(report.mean_prob_stddev, (float, np.floating))

    assert hasattr(report, "cause_ordering_flip_rate")
    assert isinstance(report.cause_ordering_flip_rate, (float, np.floating))


# --- The 429 retry path (POSTMORTEM.md incident 7) ---------------------------


class _Fake429(Exception):
    """Stands in for google.genai.errors.ClientError with code 429. The real
    class is imported lazily inside the client, so a structural double with
    the two attributes the handler reads is enough and keeps this test
    offline."""

    def __init__(self, retry_delay: str = "0s") -> None:
        super().__init__("429 RESOURCE_EXHAUSTED")
        self.code = 429
        self.details = {
            "error": {"details": [{"retryDelay": retry_delay}]}
        }


def test_retry_delay_prefers_the_servers_own_figure(bench_module):
    """A 429 body carries RetryInfo (`retryDelay: '14s'`). Honouring it keeps
    a 1200-call run close to the quota's real throughput instead of sleeping
    a guessed constant; the cushion keeps us past the window's edge."""
    assert bench_module._retry_delay_s(_Fake429("14s")) == pytest.approx(14.5)


def test_retry_delay_falls_back_when_the_server_sends_none(bench_module):
    """No RetryInfo -- fall back to the production client's fixed 15s rather
    than retrying immediately into another 429."""
    bare = _Fake429()
    bare.details = {}
    assert bench_module._retry_delay_s(bare) == bench_module._DEFAULT_RETRY_DELAY_S


def test_backoff_retries_a_429_and_excludes_the_wait_from_latency(bench_module, monkeypatch):
    """The guard for POSTMORTEM.md incident 7, in two halves.

    First half: a 429 is retried rather than raised -- the defect that killed
    the first full run 200 calls in.

    Second half, and the load-bearing one: the recorded latency must NOT
    include the backoff sleep. A retry that silently folded a 15-second wait
    into the timing would still pass a retry-only test, while inflating the
    p95-latency column by tens of seconds per throttled call -- a wrong
    number that looks entirely plausible, in one of the four columns this
    block exists to produce.
    """
    import google.genai.errors as genai_errors

    monkeypatch.setattr(genai_errors, "ClientError", _Fake429, raising=False)

    slept: list[float] = []
    monkeypatch.setattr(bench_module.time, "sleep", lambda s: slept.append(s))

    calls = {"n": 0}

    class _Models:
        def generate_content(self, *, model, contents, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Fake429("3s")
            return "ok"

    class _Client:
        models = _Models()

    # min_interval_s=0 isolates the BACKOFF sleep from the PACING sleep --
    # both go through time.sleep, and this test is about the former. Pacing
    # has its own test below.
    client = bench_module.InstrumentedGemini(api_key="unused", min_interval_s=0.0)
    response, elapsed = client._call_with_backoff(
        _Client(), model="m", user="u", config=None
    )

    assert response == "ok"
    assert calls["n"] == 2, "the 429 should have been retried exactly once"
    assert slept == [pytest.approx(3.5)], f"expected one 3.5s sleep, got {slept}"
    # The sleep is faked to take no real time, so any leakage of the WAIT into
    # the timing would have to come from timing the failed attempt too. Assert
    # the elapsed figure is a single fast call, not an accumulation.
    assert elapsed < 1.0, (
        f"latency {elapsed}s includes more than the successful attempt -- the "
        f"backoff wait must sit outside the timed region"
    )


# --- Guards for the stats-reviewer findings (2026-08-31) ---------------------


def test_render_prompt_raises_on_a_missing_allowlist_field(bench_module):
    """The "shown strictly more information than the stats model" claim is the
    table's central fairness argument, and nothing else enforces it at runtime.
    A future frame rename that dropped a PROMPT_FIELD would silently shrink the
    prompt below parity with FEATURE_COLUMNS while the footnote kept printing
    the superset claim -- biasing the result against the LLM for a reason with
    nothing to do with the model."""
    complete = {name: 1 for name in bench_module.PROMPT_FIELDS}
    bench_module.render_prompt(complete)  # must not raise

    partial = dict(complete)
    partial.pop(bench_module.PROMPT_FIELDS[0])
    with pytest.raises(ValueError, match="missing PROMPT_FIELDS"):
        bench_module.render_prompt(partial)


def test_prompt_fields_cover_every_stats_model_covariate(bench_module):
    """The superset claim, asserted against the real design matrix rather than
    against a hand-copied list. `_design_matrix()` is built from `slot` and
    `in_salary_window`; if the model ever gains a covariate the prompt does not
    show, the comparison stops being a handicap in the LLM's favour."""
    for covariate in ("slot", "in_salary_window"):
        assert covariate in bench_module.PROMPT_FIELDS, (
            f"the stats model uses {covariate!r} but the LLM is not shown it"
        )


def test_multiclass_log_loss_rewards_calibration_not_just_ranking(bench_module):
    """The reason log loss replaced AUC as the headline. Two predictors with
    IDENTICAL ranking but different calibration must score differently -- AUC
    cannot tell them apart, and the allocator consumes probabilities, so the
    difference is exactly what matters downstream."""
    y = np.array([0, 1, 2, 3])
    confident = np.array([
        [0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85],
    ])
    timid = np.array([
        [0.28, 0.24, 0.24, 0.24], [0.24, 0.28, 0.24, 0.24],
        [0.24, 0.24, 0.28, 0.24], [0.24, 0.24, 0.24, 0.28],
    ])
    assert bench_module.macro_ovr_auc(y, confident) == pytest.approx(
        bench_module.macro_ovr_auc(y, timid)
    ), "these two must be indistinguishable to AUC, or the test proves nothing"
    assert bench_module.multiclass_log_loss(y, confident) < bench_module.multiclass_log_loss(y, timid)


def test_cluster_bootstrap_ci_brackets_the_point_estimate(bench_module):
    """A CI that does not contain its own point estimate is a broken CI, and
    the table asks readers to judge ties by interval overlap."""
    rng = np.random.default_rng(0)
    y = np.array([i % 4 for i in range(120)])
    p = rng.dirichlet(np.ones(4), size=120)
    groups = [f"M{i // 3}" for i in range(120)]  # 3 rows per mandate, as in the real frame
    point = bench_module.macro_ovr_auc(y, p)
    lo, hi = bench_module.cluster_bootstrap_ci(y, p, groups, n_boot=200, seed=0)
    assert lo <= point <= hi, f"CI [{lo}, {hi}] excludes point estimate {point}"
    assert lo < hi, "a degenerate interval would hide all sampling uncertainty"


def test_variance_report_max_optout_swing_catches_what_ordering_flip_misses(bench_module):
    """cause_ordering_flip_rate ignores OPTED_OUT entirely, so a model whose
    P(OPTED_OUT) swung wildly across repeats would score 0.000 on it. That is
    the cause gating the off-ramp, so the gap needs its own number."""
    run1 = np.array([[0.60, 0.30, 0.05, 0.05]])
    run2 = np.array([[0.20, 0.30, 0.05, 0.45]])
    report = bench_module.variance_report([run1, run2])
    assert report.cause_ordering_flip_rate == 0.0, (
        "RECOVERED still outranks DEAD in both repeats -- this metric is blind here"
    )
    assert report.max_optout_swing == pytest.approx(0.40), (
        "the opt-out swing is what makes the instability visible at all"
    )


# --- Budget and cache guards (POSTMORTEM.md incident 8) ----------------------


def test_plan_budget_counts_every_live_call(bench_module):
    """The budget must count the variance pass, not just the accuracy pass.
    Undercounting is how 600 calls got planned against a 500/day cap."""
    budget = bench_module.plan_budget(
        n=140, repeats=5, variance_n=30, temperatures=(0.0, 1.0), models=("m1", "m2"),
    )
    assert budget == {"m1": 140 + 5 * 30 * 2, "m2": 140 + 5 * 30 * 2}


def test_assert_within_budget_refuses_the_configuration_that_actually_failed(bench_module):
    """--n 200 --repeats 5 --variance-n 40 plans 600 calls per model against
    flash-lite's 500/day cap. It ran for 400 calls and lost all of them."""
    over = bench_module.plan_budget(
        n=200, repeats=5, variance_n=40, temperatures=(0.0, 1.0),
        models=("gemini-3.5-flash-lite",),
    )
    with pytest.raises(ValueError, match="daily free-tier quota"):
        bench_module.assert_within_budget(over)

    ok = bench_module.plan_budget(
        n=140, repeats=5, variance_n=30, temperatures=(0.0, 1.0),
        models=("gemini-3.5-flash-lite",),
    )
    bench_module.assert_within_budget(ok)  # must not raise


def test_daily_quota_is_per_model_not_one_global_number(bench_module):
    """The SECOND failure, and the one a single constant would have missed.
    gemini-3.5-flash allows 20 requests/day, not 500 -- both figures measured
    from real 429 bodies. A 440-call run that is legal for flash-lite is 22x
    over budget for flash, and the first version of this guard would have
    waved it straight through."""
    assert bench_module.daily_quota("gemini-3.5-flash-lite") == 500
    assert bench_module.daily_quota("gemini-3.5-flash") == 20

    budget = bench_module.plan_budget(
        n=140, repeats=5, variance_n=30, temperatures=(0.0, 1.0),
        models=("gemini-3.5-flash",),
    )
    with pytest.raises(ValueError, match="gemini-3.5-flash: 440 planned vs 20/day"):
        bench_module.assert_within_budget(budget)


def test_unknown_model_assumes_the_smallest_observed_quota(bench_module):
    """An unmeasured model must default to the SMALLEST known cap, not the
    largest. Guessing high is exactly how a run discovers its limit at call
    400 with nothing persisted."""
    assert bench_module.daily_quota("gemini-9.9-unmeasured") == 20
    assert bench_module.daily_quota("gemini-9.9-unmeasured") == min(
        bench_module.DAILY_QUOTA_BY_MODEL.values()
    )


def test_assert_within_budget_accounts_for_calls_already_spent(bench_module):
    """A quota is per DAY, not per run. A second run that fits on its own can
    still exceed what is left."""
    # A named model, not a placeholder: an unknown model now defaults to the
    # smallest observed quota (20), which would fail this at already_spent=0
    # for the wrong reason.
    budget = bench_module.plan_budget(
        n=140, repeats=5, variance_n=30, temperatures=(0.0, 1.0),
        models=("gemini-3.5-flash-lite",),
    )
    bench_module.assert_within_budget(budget, already_spent=0)
    with pytest.raises(ValueError, match="already spent today"):
        bench_module.assert_within_budget(budget, already_spent=100)


def test_call_cache_round_trips_and_is_flushed_per_call(bench_module, tmp_path, monkeypatch):
    """The load-bearing half: a second pass over identical inputs must issue
    ZERO live calls. A cache that silently missed would be indistinguishable
    from no cache at all, right up until the next 500-call bill.

    Also asserts the entry is on DISK before the next call would be made --
    end-of-run persistence is what lost 400 calls, so writing at close is not
    good enough.
    """
    monkeypatch.setattr(bench_module, "CACHE_DIR", tmp_path)

    first = bench_module.CallCache("test-model")
    key = bench_module.CallCache.key(temperature=0.0, repeat=0, row_index=7)
    assert first.get(key) is None, "empty cache must miss"
    first.put(key, [0.1, 0.2, 0.3, 0.4])

    # On disk immediately, not at close.
    assert first.path.exists(), "cache entry was not flushed to disk on write"

    second = bench_module.CallCache("test-model")
    got = second.get(key)
    assert got == [0.1, 0.2, 0.3, 0.4], f"cache did not round-trip, got {got}"
    assert second.hits == 1


def test_call_cache_is_invalidated_by_a_prompt_change(bench_module, tmp_path, monkeypatch):
    """A reworded prompt must not be served last version's answers. The
    slot-4 prompt fix is exactly this case: reusing pre-fix answers would have
    silently reported the corrected prompt's numbers as the old prompt's."""
    monkeypatch.setattr(bench_module, "CACHE_DIR", tmp_path)
    path_now = bench_module.CallCache("m").path
    monkeypatch.setattr(bench_module, "PROMPT_VERSION", "deadbeef1234")
    path_after = bench_module.CallCache("m").path
    assert path_now != path_after, "cache path must change when the prompt changes"


def test_call_cache_can_be_disabled(bench_module, tmp_path, monkeypatch):
    """--no-cache must force a live call for every row, mirroring
    eval/golden_check.py's own flag. A gate ticked against a cached run is
    weaker evidence than one ticked against a live run."""
    monkeypatch.setattr(bench_module, "CACHE_DIR", tmp_path)
    c = bench_module.CallCache("m", enabled=False)
    key = bench_module.CallCache.key(temperature=0.0, repeat=0, row_index=1)
    c.put(key, [0.25, 0.25, 0.25, 0.25])
    assert c.get(key) is None, "a disabled cache must never serve a hit"


# --- Fitting a plan to a quota, instead of refusing (B16) --------------------
#
# The flash arm plans 440 calls against a hard 20/day cap. assert_within_budget
# correctly refuses it, which means the arm could never run at all -- 440 at
# 20/day is 22 days, and the deadline is not 22 days away. Refusing is right;
# refusing and stopping there is what left reports/bench.json with no LLM row
# in it. fit_plan_to_quota shrinks the plan instead, and says what it gave up.


def test_a_plan_that_fits_is_left_exactly_alone(bench_module):
    plan = bench_module.fit_plan_to_quota(
        "gemini-3.5-flash-lite", n=140, repeats=5, variance_n=30,
        temperatures=(0.0, 1.0),
    )
    assert plan.n == 140
    assert plan.repeats == 5
    assert plan.variance_n == 30
    assert plan.calls == 440
    assert plan.shrunk is False
    assert plan.reason == ""


def test_the_flash_arm_is_shrunk_to_a_variance_only_probe(bench_module):
    """20 calls buys an EXISTENCE claim, never a rate. So the accuracy pass is
    dropped whole (it would produce an AUC over 4 rows sitting next to a
    140-row AUC in the same table) and every call goes to repeats."""
    plan = bench_module.fit_plan_to_quota(
        "gemini-3.5-flash", n=140, repeats=5, variance_n=30,
        temperatures=(0.0, 1.0),
    )
    assert plan.shrunk is True
    assert plan.n == 0, "the accuracy pass is dropped, not merely reduced"
    assert plan.variance_n == 1
    assert plan.calls <= bench_module.daily_quota("gemini-3.5-flash")


def test_a_shrunk_plan_spends_its_whole_budget_on_repeats(bench_module):
    """With variance_n pinned to 1, repeats are the only thing carrying the
    measurement, so the fit maximises them -- even ABOVE the requested 5.
    Leaving quota unspent here would weaken the one claim the arm can make."""
    quota = bench_module.daily_quota("gemini-3.5-flash")
    plan = bench_module.fit_plan_to_quota(
        "gemini-3.5-flash", n=140, repeats=5, variance_n=30,
        temperatures=(0.0, 1.0),
    )
    assert plan.repeats == quota // 2
    assert plan.repeats > 5, "the fit must not silently keep the smaller ask"
    assert plan.calls == plan.repeats * 2


def test_shrinking_keeps_both_temperatures(bench_module):
    """PLAN_DETAIL.md forbids running the LLM at temperature 0 only, and the
    sharpest finding available is that variance at t=0.0 is NOT zero. Buying
    repeats by dropping a temperature would spend the arm's whole point."""
    temps = (0.0, 1.0)
    plan = bench_module.fit_plan_to_quota(
        "gemini-3.5-flash", n=140, repeats=5, variance_n=30, temperatures=temps,
    )
    assert plan.calls == plan.repeats * plan.variance_n * len(temps)


def test_a_quota_too_small_to_measure_variance_raises(bench_module):
    """variance_report needs 2 repeats. A plan that cannot afford 2 per
    temperature must fail loudly, not emit a one-repeat 'variance' of zero --
    which would read as evidence of stability and is the exact false comfort
    this whole arm exists to avoid."""
    with pytest.raises(ValueError, match="too small to measure"):
        bench_module.fit_plan_to_quota(
            "gemini-3.5-flash", n=140, repeats=5, variance_n=30,
            temperatures=(0.0, 1.0), quota=3,
        )


def test_fitted_plans_always_clear_the_budget_guard(bench_module):
    """The fit and the guard must agree. If a fitted plan could still trip
    assert_within_budget, the fit would be decorative."""
    plans = bench_module.plans_for(
        models=("gemini-3.5-flash-lite", "gemini-3.5-flash"),
        n=140, repeats=5, variance_n=30, temperatures=(0.0, 1.0),
    )
    budget = {m: p.calls for m, p in plans.items()}
    bench_module.assert_within_budget(budget)  # must not raise
    assert budget["gemini-3.5-flash-lite"] == 440
    assert budget["gemini-3.5-flash"] == 20


def test_the_reason_names_the_quota_and_what_was_given_up(bench_module):
    """A shrunk arm that does not say what it lost is worse than no arm: the
    reader compares a 20-call number against a 440-call number as if they
    measured the same thing."""
    plan = bench_module.fit_plan_to_quota(
        "gemini-3.5-flash", n=140, repeats=5, variance_n=30,
        temperatures=(0.0, 1.0),
    )
    assert "20/day" in plan.reason
    assert "440" in plan.reason
    assert "accuracy" in plan.reason.lower()


def test_arm_result_tolerates_a_missing_accuracy_pass(bench_module):
    """A variance-only arm has no AUC and no log loss, and the table must
    render a dash rather than crash or, worse, print a zero."""
    arm = bench_module.ArmResult(
        name="gemini-3.5-flash as classifier (variance probe)",
        auc=None, auc_ci=None, log_loss=None, brier={},
        p95_latency_s=0.9, latency_kind="per-call, network",
        cost_per_1k_paise=12, n_scored=0,
        note="variance-only: 20-call quota",
    )
    table = bench_module.render_table([arm], pricing=bench_module.load_pricing(), seed=0)
    assert "—" in table or "--" in table
    assert "0.0000" not in table.split("\n")[0]
