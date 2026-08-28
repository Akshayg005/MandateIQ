"""Split conformal prediction sets -- the safety gate for the off-ramp.

The off-ramp (offering a customer an exit: pause, then downgrade, then
cancel) may only fire when a prediction set is the SINGLETON {label} at
95% coverage. A set that is falsely narrow -- excludes the true label --
risks a false off-ramp: cancelling a customer who would have paid, the
exact harm this system exists to prevent. Every design choice here is
in service of that one asymmetry: too SMALL is the harm; too LARGE just
declines to act.

Four choices, all defaults, none optional:

1. LAC (least-ambiguous-set) nonconformity scoring, `1 - p`, not APS.
   Measured on this project's own hazard model: APS gives smaller sets
   AND worse coverage on the safety-relevant class than LAC -- both
   directions wrong under the asymmetry above. APS's usual conditional-
   coverage advantage is asymptotic in score diversity, and a hazard model
   with only a handful of distinct predicted vectors has none.
2. Mondrian (class-conditional) conformal. Marginal coverage does not
   bound the false-off-ramp rate for any one class -- a rare or
   safety-relevant class can have terrible coverage while the marginal
   average still looks fine.
3. Smoothed (randomized) conformal, via Vovk's smoothed p-value. This
   project's hazard model produces very few distinct predicted vectors,
   so calibration scores are heavily tied. Unsmoothed conformal under
   ties is a step function of the calibration draw -- coverage can swing
   by tens of points between two otherwise-identical splits (see
   tests/model/test_conformal.py's heavy-ties regression guard). The
   smoothing draw is derived DETERMINISTICALLY from a caller-supplied
   stable key (mandate_id) via a hash, never a mutable RNG -- so the same
   (predictor, scores, key) always yields the same set, regardless of call
   order, batch composition, or process.
4. Label-space genericity. This class never sees a probability, only
   nonconformity SCORES -- it has no idea what a label "means", only that
   labels are hashable and there are K of them. That is what makes it
   usable for Outcome now and Cause later (once a cause posterior exists,
   at B7/B11) with no change to this file, and what makes it structurally
   impossible for src/model/ to acquire a Cause-shaped or LLM-shaped
   dependency through this path.

pred_set() returns frozenset[LabelT], never a point prediction and never
None -- an intentionally empty frozenset() is the abstain signal, and
counts as miscoverage in coverage_report(), never as "no data" or a
silent fallback to some default label.

No LLM client import here, and no import of this project's latent-cause
enum either -- see src/model/CLAUDE.md and this module's own test suite,
which greps the source for both.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Generic, Hashable, Sequence, TypeVar

import numpy as np
import pandas as pd

LabelT = TypeVar("LabelT", bound=Hashable)

_ACCEPTED_CALIBRATE_PROVENANCE = "calib_conf"

# 95% two-sided z for the Wilson score interval coverage_report() reports.
_WILSON_Z = 1.959963984540054


class ConformalUnderpowered(RuntimeError):
    """Raised when calibrate(..., mondrian=True) finds a class with fewer
    than the finite-sample floor ceil(1/alpha) - 1 calibration examples --
    that class's Mondrian quantile would be undefined/infinite. Names the
    offending class and both the required and actual counts, rather than
    silently returning the full label set for that class forever."""


class ConformalLeakError(RuntimeError):
    """Raised by assert_disjoint() when the group ids used to fit a
    predictor overlap with the group ids a coverage report is computed
    over."""


def lac_scores(p: np.ndarray) -> np.ndarray:
    """Least-ambiguous-set nonconformity score: (n, K) predicted
    probabilities -> (n, K) scores, elementwise 1 - p."""
    p = np.asarray(p, dtype=float)
    return 1.0 - p


def aps_scores(p: np.ndarray, *, keys: Sequence[str], seed: int) -> np.ndarray:
    """Adaptive prediction set scores: for each row, sort classes by
    descending probability and score each class by the cumulative
    probability mass up to and including it, with a deterministic
    randomized tie-break (derived from `keys`/`seed`, never a mutable
    RNG -- same determinism requirement as pred_set()) so the boundary
    class's inclusion is not always resolved the same way. Provided for
    comparison against LAC; not the default (see module docstring)."""
    p = np.asarray(p, dtype=float)
    n, k = p.shape
    scores = np.empty((n, k), dtype=float)
    for i in range(n):
        order = np.argsort(-p[i])
        cum = np.cumsum(p[i][order])
        u = _derive_u(seed, str(keys[i]))
        row_scores = np.empty(k, dtype=float)
        for rank, class_idx in enumerate(order):
            prev_cum = cum[rank - 1] if rank > 0 else 0.0
            row_scores[class_idx] = prev_cum + u * p[i][class_idx]
        scores[i] = row_scores
    return scores


def _derive_u(seed: int, key: str) -> float:
    """Deterministic pseudo-uniform draw in [0, 1), a pure function of
    (seed, key) via a cryptographic hash -- never a mutable RNG. This is
    what makes pred_set() bit-reproducible regardless of call order, batch
    composition, or process."""
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    as_int = int.from_bytes(digest, "big")
    return (as_int % (1 << 53)) / float(1 << 53)


def _p_value(s: float, pool: np.ndarray, *, smoothed: bool, u: float) -> float:
    """Vovk's (smoothed) conformal p-value of candidate score `s` against
    calibration pool `pool`: (#{pool > s} + u*(#{pool == s} + 1)) / (n+1).
    The `+1` inside the weighted term is the test point's own tie with
    itself (under exchangeability, the candidate is hypothetically the
    (n+1)-th point, tied with itself) -- it must be smoothed by `u` too,
    not added as a flat constant. Getting this wrong (weighting only the
    calibration ties, adding a bare +1 for the test point) is a real,
    caught bug: it under-smooths, giving systematic OVER-coverage that
    grows as the pool shrinks and ties get heavier -- exactly this
    project's regime (Mondrian pools as small as the ceil(1/alpha)-1=19
    floor, and this model's ~6-atom hazard vocabulary), where it measured
    up to +4.35 points of excess coverage (stats-reviewer, DECISIONS.md,
    2026-08-28 B6). Unsmoothed is the special case u=1, which reduces to
    the standard conservative non-randomized rule and is unaffected by
    this fix (weighting doesn't matter when it's always 1)."""
    n = len(pool)
    if n == 0:
        return 1.0
    greater = int(np.sum(pool > s))
    equal = int(np.sum(pool == s))
    weight = u if smoothed else 1.0
    return (greater + weight * (equal + 1)) / (n + 1)


def _wilson_interval(k: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def should_act(s: frozenset, target: object) -> bool:
    """The off-ramp firing rule: True iff `s` is the singleton {target}.
    An empty set, a multi-label set, or a singleton of the wrong label are
    all False -- there is exactly one condition under which this system
    acts on a conformal set, and it is this one."""
    return len(s) == 1 and target in s


def assert_disjoint(fit_ids: frozenset[str], report_ids: frozenset[str]) -> None:
    """Raise ConformalLeakError if `fit_ids` and `report_ids` share any
    element -- the machine-checked half of "never calibrate and report
    coverage on the same split"."""
    overlap = fit_ids & report_ids
    if overlap:
        raise ConformalLeakError(
            f"{len(overlap)} id(s) used to calibrate a predictor also "
            f"appear in the report set: {sorted(overlap)[:10]}"
        )


@dataclass(frozen=True)
class SplitConformal(Generic[LabelT]):
    """A fitted split-conformal predictor over label space `labels`. Sees
    only nonconformity scores, never probabilities -- see module docstring.

    calib_scores: (n_calib,) the TRUE label's own nonconformity score for
    each calibration row. calib_labels: (n_calib,) the integer index (into
    `labels`) of each calibration row's true label. fit_group_ids: the
    group (mandate) ids calibration was fit on, for leak-checking via
    assert_disjoint(). provenance: always "calib_conf" -- calibrate()
    rejects any other value."""

    labels: tuple[LabelT, ...]
    alpha: float
    mondrian: bool
    smoothed: bool
    smoothing_seed: int
    calib_scores: np.ndarray
    calib_labels: np.ndarray
    fit_group_ids: frozenset[str]
    provenance: str

    def pred_set(self, scores: Sequence[float], *, key: str) -> frozenset[LabelT]:
        """scores: length-K nonconformity scores for one row, in the same
        order as self.labels. key: a stable per-row id (e.g. mandate_id)
        deriving the smoothing draw deterministically -- one draw per row,
        shared across every candidate label's inclusion test for that row.
        Returns the subset of self.labels whose smoothed p-value exceeds
        alpha; frozenset() signals abstain, never a point prediction."""
        scores = np.asarray(scores, dtype=float)
        u = _derive_u(self.smoothing_seed, str(key)) if self.smoothed else 1.0

        included = []
        for c, label in enumerate(self.labels):
            pool = (
                self.calib_scores[self.calib_labels == c]
                if self.mondrian else self.calib_scores
            )
            p = _p_value(float(scores[c]), pool, smoothed=self.smoothed, u=u)
            if p > self.alpha:
                included.append(label)
        return frozenset(included)

    def empirical_coverage(
        self, score_rows: np.ndarray, y: np.ndarray, *, keys: Sequence[str]
    ) -> float:
        """Marginal coverage: fraction of rows where the true label is in
        pred_set(...). An empty set counts as miscoverage automatically
        (a label is never `in` an empty frozenset)."""
        score_rows = np.asarray(score_rows, dtype=float)
        y = np.asarray(y)
        keys = list(keys)
        n = len(score_rows)
        if n == 0:
            return float("nan")
        covered = 0
        for i in range(n):
            s = self.pred_set(score_rows[i], key=str(keys[i]))
            if self.labels[int(y[i])] in s:
                covered += 1
        return covered / n

    def coverage_report(
        self, score_rows: np.ndarray, y: np.ndarray, *, keys: Sequence[str]
    ) -> pd.DataFrame:
        """One row per label class plus one MARGINAL row: label, n,
        coverage, wilson_lo, wilson_hi, mean_set_size, singleton_count,
        singleton_error_count, empty_count. Coverage is never meaningful
        without set size and singleton rate alongside it -- a predictor
        that always returns every label clears any coverage target
        trivially and says nothing."""
        score_rows = np.asarray(score_rows, dtype=float)
        y = np.asarray(y).astype(int)
        keys = list(keys)
        n = len(score_rows)
        k = len(self.labels)

        pred_sets = [self.pred_set(score_rows[i], key=str(keys[i])) for i in range(n)]
        set_sizes = np.array([len(s) for s in pred_sets])
        is_singleton = set_sizes == 1
        is_empty = set_sizes == 0
        true_labels = [self.labels[y[i]] for i in range(n)]
        covered = np.array([true_labels[i] in pred_sets[i] for i in range(n)])
        singleton_error = is_singleton & (~covered)

        def _row(label, mask: np.ndarray) -> dict:
            n_mask = int(mask.sum())
            cov = float(covered[mask].mean()) if n_mask else float("nan")
            wilson_lo, wilson_hi = _wilson_interval(int(covered[mask].sum()), n_mask)
            return {
                "label": label,
                "n": n_mask,
                "coverage": cov,
                "wilson_lo": wilson_lo,
                "wilson_hi": wilson_hi,
                "mean_set_size": float(set_sizes[mask].mean()) if n_mask else float("nan"),
                "singleton_count": int(is_singleton[mask].sum()),
                "singleton_error_count": int(singleton_error[mask].sum()),
                "empty_count": int(is_empty[mask].sum()),
            }

        rows = [_row(self.labels[c], y == c) for c in range(k)]
        rows.append(_row("MARGINAL", np.ones(n, dtype=bool)))
        report = pd.DataFrame(rows)
        # object dtype for the integer count columns -- pandas' default
        # numeric dtype (int64) boxes each cell as numpy.int64 on access,
        # and numpy.int64 is not `isinstance(x, int)` (unlike numpy.float64,
        # which does subclass Python's float). Genuine Python int identity
        # is part of this report's contract.
        for col in ("n", "singleton_count", "singleton_error_count", "empty_count"):
            report[col] = report[col].astype(object)
        return report


def calibrate(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    labels: tuple[LabelT, ...],
    row_group_ids: Sequence[str],
    provenance: str,
    alpha: float = 0.05,
    mondrian: bool = True,
    smoothed: bool = True,
    smoothing_seed: int = 0,
) -> SplitConformal[LabelT]:
    """scores: (n, K) nonconformity scores (e.g. from lac_scores()). y:
    (n,) integer index into `labels` of each row's true label.
    row_group_ids: (n,) group (mandate) ids, stored for leak-checking.
    `provenance` must be "calib_conf" -- calibrate() refuses to fit on any
    other split, most importantly "test"."""
    if provenance != _ACCEPTED_CALIBRATE_PROVENANCE:
        raise ValueError(
            f"conformal.calibrate() may only be called with "
            f"provenance={_ACCEPTED_CALIBRATE_PROVENANCE!r}; got {provenance!r} "
            "-- the conformal quantile must never be calibrated on the "
            "split its coverage is reported on"
        )

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    row_group_ids = list(row_group_ids)
    k = len(labels)

    if len(scores) != len(y):
        raise ValueError(f"calibrate() scores/y length mismatch: {len(scores)} vs {len(y)}")
    if len(row_group_ids) != len(scores):
        raise ValueError(
            f"calibrate() row_group_ids length {len(row_group_ids)} does not "
            f"match scores length {len(scores)}"
        )

    calib_scores_true = scores[np.arange(len(scores)), y]

    if mondrian:
        min_required = math.ceil(1.0 / alpha) - 1
        for c in range(k):
            n_c = int((y == c).sum())
            if n_c < min_required:
                raise ConformalUnderpowered(
                    f"class {labels[c]!r} (index {c}) has {n_c} calibration "
                    f"example(s); Mondrian conformal at alpha={alpha} "
                    f"requires at least required={min_required} "
                    f"(= ceil(1/alpha) - 1), actual={n_c} -- refusing to "
                    "silently return the full label set for this class"
                )

    return SplitConformal(
        labels=tuple(labels),
        alpha=alpha,
        mondrian=mondrian,
        smoothed=smoothed,
        smoothing_seed=smoothing_seed,
        calib_scores=calib_scores_true,
        calib_labels=y,
        fit_group_ids=frozenset(row_group_ids),
        provenance=provenance,
    )
