"""Golden evaluation: score src/llm/normalizer.py and src/llm/intent.py
against hand-labeled data. Run via `.\\run.ps1 golden` -- NOT wired into the
Stop hook, despite PLAN_DETAIL.md B11's literal wording ("golden set ...
wired into the Stop hook"): even cached, the FIRST run after any prompt edit
still costs ~5.5 minutes of live calls (the cache only helps on UNCHANGED
prompts), which is a bad fit for something that has to finish before a
session can end. Deliberate deviation, not an oversight -- flagged for
confirmation, not silently decided either way (payments-domain review,
2026-08-31, caught this docstring claiming the PLAN_DETAIL wording as if it
were what was actually built).

Cached, by design (DECISIONS.md, 2026-08-30): the live edge is rate-limited
to ~15 requests/minute, and the golden set is 50 + 30 = 80 rows, so an
uncached run costs ~5.5 minutes -- unacceptable at every Stop-hook checkpoint.
The cache is namespaced by (normalizer_version | intent_version), which are
themselves content hashes of the system prompt + tool schema (src/llm/
normalizer.py, intent.py) -- so a prompt edit changes the version, which
changes the cache filename, which forces a full live re-run automatically.
A hand-maintained version string could be forgotten on a prompt edit and
let this check pass on stale answers -- the exact "vacuous check" failure
mode this project has amended gates over before (reports/gates.md, B8).

The cache stores MODEL OUTPUT only, never labels -- eval/golden/declines.jsonl
and intent.jsonl are hand-authored and must never be regenerated from model
output (PLAN_DETAIL.md's explicit "Must NOT" for those files). golden_check
always reports cache hits/misses/live-calls-made, so a green run can never be
silently misread as "the model was consulted" when it mostly wasn't.

Two zero-tolerance checks gate the exit code independently of aggregate
accuracy -- a golden set that hits its accuracy floor while still producing
a false MANDATE_REVOKED verdict, or false-off-ramping a paying customer,
must still fail. This is what "must NOT pass on a tie with a lowered
threshold" (PLAN_DETAIL.md B11) means in practice: the threshold cannot be
gamed by averaging away the one confusion this system exists to prevent.

The decline check gates on ANY false MANDATE_REVOKED, not only the
INSUFFICIENT_FUNDS<->MANDATE_REVOKED swap it originally checked -- widened
2026-08-31 after payments-domain review found the narrow version blind to
e.g. UNKNOWN->MANDATE_REVOKED, which is the actual failure the escalation
path produced on eval/golden/declines.jsonl's payment_cancelled row (a
string decline_taxonomy.py deliberately leaves UNKNOWN rather than guesses
at) once the taxonomy escalated it. Any false MANDATE_REVOKED stops
retrying a mandate that may still be alive -- the specific label it was
confused FROM doesn't change that cost. The reverse direction (a real
MANDATE_REVOKED missed, i.e. still attempting a dead mandate) is a real,
different cost too, but is intentionally NOT zero-tolerance here, matching
how the intent check gates only the false-off-ramp direction: this project
reports both error costs but gates only the one a false positive cannot
walk back (money-auditor's framing, same principle applied here).

Only 12 of these 50 rows would actually reach the LLM in production
(decline_taxonomy.py's classify() answers the other 38 confidently) -- a
THIRD gate, separate from the aggregate floor, requires escalation-only
accuracy to clear DECLINE_ACCURACY_FLOOR on its own: the aggregate can be
made to look fine by a component doing badly on exactly the rows that are
its actual job, since 76% of the set never exercises it at all. Reuses the
existing floor rather than inventing a second number for a 12-row subset
this project has no principled way to derive one for yet. DECISIONS.md,
2026-08-31, has the full reasoning for why the golden set keeps both
populations rather than being cut to 12 rows.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
CACHE_DIR = GOLDEN_DIR / ".cache"
DECLINES_PATH = GOLDEN_DIR / "declines.jsonl"
INTENT_PATH = GOLDEN_DIR / "intent.jsonl"

DECLINE_ACCURACY_FLOOR = 0.90
INTENT_BAND_ACCURACY_FLOOR = 0.85


@dataclass(frozen=True)
class DeclineResult:
    """Result of scoring decline classifications against golden data."""

    accuracy: float
    total: int
    correct: int
    cache_hits: int
    cache_misses: int
    insufficient_funds_revoked_confusions: int  # zero-tolerance: must be 0
    # Broader than the field above: ANY label falsely predicted
    # MANDATE_REVOKED, not only from INSUFFICIENT_FUNDS. Zero-tolerance for
    # the same reason -- a false MANDATE_REVOKED stops retrying a mandate
    # that may still be alive, regardless of which class it was confused
    # from (payments-domain review, 2026-08-31; see module docstring).
    any_to_mandate_revoked_confusions: int


@dataclass(frozen=True)
class IntentResult:
    """Result of scoring intent bands against golden data."""

    band_accuracy: float
    total: int
    correct: int
    cache_hits: int
    cache_misses: int
    false_high_on_low_labeled: int  # zero-tolerance: false positive toward off-ramp


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cache(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: pathlib.Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _persisting(cache: dict, cache_path: pathlib.Path, fn: Callable):
    """Wrap a classify/score function so every fresh (non-cached) live call
    is flushed to disk immediately after it returns, not just once at the
    end of main(). At ~15 requests/minute, a 50-row run legitimately takes
    several minutes and can be interrupted or hit a mid-run error (measured
    on this block's first live run); without incremental persistence,
    every already-PAID-FOR call before the interruption would be silently
    thrown away and re-billed on the next attempt. score_declines /
    score_intent stay pure (no I/O, easy to test with a plain dict) --
    this wrapper is the only place that touches the filesystem, and it is
    main()-only, never exercised by the fake-scorer unit tests."""
    def wrapped(x):
        result = fn(x)
        cache[x] = result
        _save_cache(cache_path, cache)
        return result
    return wrapped


def score_declines(
    rows: list[dict],
    classify_fn: Callable[[str], str],
    cache: dict,
    no_cache: bool = False,
) -> DeclineResult:
    """Score decline classifications against golden labels.

    rows: [{"raw": <issuer decline string>, "label": <DeclineClass.value>}]
    classify_fn: raw string -> predicted DeclineClass.value string
    cache: dict, keyed by the raw string, modified in place with fresh
        answers. Not the persistent on-disk cache directly -- main() decides
        WHICH on-disk cache file (i.e. which model version) this dict came
        from before calling in.
    """
    total = correct = cache_hits = cache_misses = confusions = any_revoked = 0
    _SWAP = {"INSUFFICIENT_FUNDS", "MANDATE_REVOKED"}

    for row in rows:
        raw, label = row["raw"], row["label"]
        if not no_cache and raw in cache:
            predicted = cache[raw]
            cache_hits += 1
        else:
            predicted = classify_fn(raw)
            cache[raw] = predicted
            cache_misses += 1

        total += 1
        if predicted == label:
            correct += 1
        if {predicted, label} == _SWAP:
            confusions += 1
        if label != "MANDATE_REVOKED" and predicted == "MANDATE_REVOKED":
            any_revoked += 1

    accuracy = correct / total if total else 0.0
    return DeclineResult(
        accuracy=accuracy, total=total, correct=correct,
        cache_hits=cache_hits, cache_misses=cache_misses,
        any_to_mandate_revoked_confusions=any_revoked,
        insufficient_funds_revoked_confusions=confusions,
    )


def score_intent(
    rows: list[dict],
    score_fn: Callable[[str], float],
    cache: dict,
    no_cache: bool = False,
) -> IntentResult:
    """Score exit-intent bands against golden labels.

    rows: [{"text": <support message>, "band": "HIGH"|"LOW"}]
    score_fn: text -> score in [0.0, 1.0]. Predicted band is HIGH if
        score >= 0.5 else LOW.
    """
    total = correct = cache_hits = cache_misses = false_high = 0

    for row in rows:
        text, label_band = row["text"], row["band"]
        if not no_cache and text in cache:
            score = cache[text]
            cache_hits += 1
        else:
            score = score_fn(text)
            cache[text] = score
            cache_misses += 1

        predicted_band = "HIGH" if score >= 0.5 else "LOW"
        total += 1
        if predicted_band == label_band:
            correct += 1
        # Zero-tolerance direction only: a paying customer (LOW) scored as
        # exit intent (HIGH) risks a false off-ramp -- the harm this system
        # exists to prevent. The reverse (HIGH scored LOW) just costs a
        # later wasted retry attempt, reported in aggregate accuracy but not
        # gated -- see this module's docstring and DECISIONS.md.
        if label_band == "LOW" and predicted_band == "HIGH":
            false_high += 1

    band_accuracy = correct / total if total else 0.0
    return IntentResult(
        band_accuracy=band_accuracy, total=total, correct=correct,
        cache_hits=cache_hits, cache_misses=cache_misses,
        false_high_on_low_labeled=false_high,
    )


def _report(
    decline: DeclineResult, intent: IntentResult, escalation: DeclineResult | None = None,
) -> None:
    print("=" * 70)
    print(
        f"declines  {decline.correct}/{decline.total} = {decline.accuracy:.1%}  "
        f"(floor {DECLINE_ACCURACY_FLOOR:.0%})   "
        f"cache: {decline.cache_hits} hit / {decline.cache_misses} live-called"
    )
    if escalation is not None:
        print(
            f"  of which, ESCALATION-ONLY (decline_taxonomy.classify() itself "
            f"leaves UNKNOWN -- the subset that actually reaches this component "
            f"in production): {escalation.correct}/{escalation.total} = "
            f"{escalation.accuracy:.1%}. This is the number that matters more "
            f"than the headline one above."
        )
    print(
        f"  INSUFFICIENT_FUNDS <-> MANDATE_REVOKED confusions: "
        f"{decline.insufficient_funds_revoked_confusions}  (zero-tolerance)"
    )
    print(
        f"  any label falsely predicted MANDATE_REVOKED: "
        f"{decline.any_to_mandate_revoked_confusions}  (zero-tolerance)"
    )
    print(
        f"intent    {intent.correct}/{intent.total} = {intent.band_accuracy:.1%}  "
        f"(floor {INTENT_BAND_ACCURACY_FLOOR:.0%})   "
        f"cache: {intent.cache_hits} hit / {intent.cache_misses} live-called"
    )
    print(
        f"  false HIGH on LOW-labeled (false off-ramp risk): "
        f"{intent.false_high_on_low_labeled}  (zero-tolerance)"
    )
    print("=" * 70)


def check_freshness() -> bool:
    """Advisory, no live calls: does a cache file exist for the CURRENT
    NORMALIZER_VERSION / INTENT_VERSION? If not, the prompts changed (or
    this is a fresh clone) since the golden set was last actually run
    against the live models, and `.\\run.ps1 golden` should be run before
    shipping. Never fatal -- see main()'s --check-freshness handling and
    the module docstring on why this is not in the Stop hook's blocking
    path (payments-domain review, 2026-08-31)."""
    from src.llm.intent import INTENT_VERSION
    from src.llm.normalizer import NORMALIZER_VERSION

    decline_fresh = (CACHE_DIR / f"declines__{NORMALIZER_VERSION}.json").exists()
    intent_fresh = (CACHE_DIR / f"intent__{INTENT_VERSION}.json").exists()
    return decline_fresh and intent_fresh


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    no_cache = "--no-cache" in argv

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    if "--check-freshness" in argv:
        # Advisory only -- see check_freshness()'s docstring. Runs no live
        # calls, so it's cheap enough for the Stop hook / `ci` to call on
        # every session end without adding the multi-minute cost caching
        # was built to avoid. Always exits 0: warns, never blocks.
        if check_freshness():
            print("golden-set cache is current for the active prompts.")
        else:
            print(
                "WARNING: golden-set cache is missing or stale for the current "
                "normalizer/intent prompt version -- run `.\\run.ps1 golden` "
                "before shipping. (Advisory only, not blocking session end.)",
                file=sys.stderr,
            )
        return 0

    from src.classify.decline_taxonomy import classify as taxonomy_classify
    from src.core.types import DeclineClass
    from src.llm.intent import INTENT_VERSION, intent_score
    from src.llm.normalizer import NORMALIZER_VERSION, normalize

    decline_cache_path = CACHE_DIR / f"declines__{NORMALIZER_VERSION}.json"
    intent_cache_path = CACHE_DIR / f"intent__{INTENT_VERSION}.json"

    decline_cache = {} if no_cache else _load_cache(decline_cache_path)
    intent_cache = {} if no_cache else _load_cache(intent_cache_path)

    decline_rows = _load_jsonl(DECLINES_PATH)
    intent_rows = _load_jsonl(INTENT_PATH)

    classify_fn = _persisting(
        decline_cache, decline_cache_path, lambda raw: normalize(raw).value.value,
    )
    decline_result = score_declines(decline_rows, classify_fn, decline_cache, no_cache)
    intent_result = score_intent(
        intent_rows,
        _persisting(intent_cache, intent_cache_path, intent_score),
        intent_cache, no_cache,
    )

    # Escalation-only subset: rows the deterministic taxonomy itself leaves
    # UNKNOWN, i.e. the only rows normalize() actually sees in production
    # (payments-domain review, 2026-08-31 -- 38/50 rows are answered
    # confidently by classify() and never reach the LLM at all). Reuses
    # decline_cache, already fully populated by the pass above -- zero extra
    # live calls, no_cache=False regardless of the outer flag since these
    # answers were already freshly computed this run.
    escalation_rows = [
        r for r in decline_rows
        if taxonomy_classify(None, r["raw"]) == DeclineClass.UNKNOWN
    ]
    escalation_result = score_declines(escalation_rows, classify_fn, decline_cache, False)

    _report(decline_result, intent_result, escalation_result)

    failures = []
    if decline_result.accuracy < DECLINE_ACCURACY_FLOOR:
        failures.append(
            f"decline accuracy {decline_result.accuracy:.1%} < floor "
            f"{DECLINE_ACCURACY_FLOOR:.0%}"
        )
    # Gated separately from the aggregate, not folded into it: 38/50 rows
    # never reach normalize() in production (decline_taxonomy.classify()
    # answers them first), so an aggregate-only gate can clear its floor
    # while doing badly on the 12 rows that are the component's actual job
    # (payments-domain review, 2026-08-31; DECISIONS.md same date). Reuses
    # DECLINE_ACCURACY_FLOOR rather than a second invented number -- this
    # project derives thresholds where it can (B8's gate_criteria) and
    # otherwise reuses an existing, already-disclosed judgment call rather
    # than adding a new unjustified one.
    if escalation_result.total > 0 and escalation_result.accuracy < DECLINE_ACCURACY_FLOOR:
        failures.append(
            f"escalation-only accuracy {escalation_result.accuracy:.1%} < floor "
            f"{DECLINE_ACCURACY_FLOOR:.0%} ({escalation_result.correct}/"
            f"{escalation_result.total}) -- the subset that actually reaches "
            "this component in production"
        )
    if decline_result.insufficient_funds_revoked_confusions > 0:
        failures.append(
            f"{decline_result.insufficient_funds_revoked_confusions} "
            "INSUFFICIENT_FUNDS<->MANDATE_REVOKED confusion(s) -- zero-tolerance"
        )
    if decline_result.any_to_mandate_revoked_confusions > 0:
        failures.append(
            f"{decline_result.any_to_mandate_revoked_confusions} label(s) falsely "
            "predicted MANDATE_REVOKED -- zero-tolerance (stops retrying a mandate "
            "that may still be alive)"
        )
    if intent_result.band_accuracy < INTENT_BAND_ACCURACY_FLOOR:
        failures.append(
            f"intent band accuracy {intent_result.band_accuracy:.1%} < floor "
            f"{INTENT_BAND_ACCURACY_FLOOR:.0%}"
        )
    if intent_result.false_high_on_low_labeled > 0:
        failures.append(
            f"{intent_result.false_high_on_low_labeled} false-HIGH-on-LOW "
            "(false off-ramp risk) -- zero-tolerance"
        )

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
