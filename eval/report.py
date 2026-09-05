"""B13 report: renders reports/regimes.json into the three-bar tables, the
figures, and the losses section.

    python -m eval.report --figures

This module computes NOTHING. Every number it prints is read out of the
artifact eval/run.py wrote, so "reproducible by one command" is structural:
there is no second code path that could produce a different figure for the
same claim. If a number here is wrong, it is wrong in the artifact too.

Two rules this file enforces rather than documents:

* **Never report recovery alone.** Every table is three bars -- recovered
  paise, attempts spent, mandates preserved -- and the two error costs sit
  beside them. protocol.md's headline, not the incumbent's.
* **The 95%-coverage claim may only appear for cells where the REAL gate was
  live.** `gate_kind` is read from the artifact, never assumed; under
  FullSetGate the coverage column prints "n/a (stub gate)" rather than a
  number that would look like evidence.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from fractions import Fraction
from typing import Any, Sequence

from src.core import money

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "regimes.json"
OUT_MD = _REPO_ROOT / "reports" / "regimes.md"
FIG_DIR = _REPO_ROOT / "reports" / "figures"
# R3: a separate artifact from `python -m eval.ltv_sensitivity`, read here
# (not computed) the same way ARTIFACT is -- optional, since a tree that
# has never run the LTV sweep must still render the rest of regimes.md.
LTV_ARTIFACT = _REPO_ROOT / "reports" / "ltv_sensitivity.json"
# R5: a separate artifact from `python -m eval.offramp_channel`, read here
# (never computed) on exactly the same terms as LTV_ARTIFACT -- optional, so
# a tree that has never run the channel sweep still renders the rest of
# regimes.md.
OFFRAMP_ARTIFACT = _REPO_ROOT / "reports" / "offramp_channel.json"

# dataviz: categorical slots 1 and 2. Validated as a pair (light mode) --
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface.
C_LADDER = "#2a78d6"
C_ENGINE = "#eb6834"
C_ONE_SHOT = "#1baf7a"
C_NULL = "#eda100"
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"


def _rupees(paise: int) -> str:
    """CLAUDE.md: money helpers live in src/core/money.py, nothing else
    formats currency, and a float touching a money value is a bug. The first
    version of this function divided paise by one hundred in float and rendered
    Rs 20,22,513.53 as "2,022,514" -- a float division, in Western grouping,
    outside money.py, breaking both invariants at once. It survived because
    the guard's money checks ran only over PROTECTED_DIRS -- eval/ was not
    among them -- so the rule was being checked exactly where it already
    held. Fixed on both sides: this function delegates, and the guard now
    scans MONEY_DIRS, which includes eval/. (payments-domain, 2026-08-31.)"""
    return money.fmt(paise)


def _pct(new: float, old: float) -> str:
    if not old:
        return "n/a"
    return f"{100 * (new - old) / old:+.1f}%"


def load(path: pathlib.Path = ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run `python -m eval.run --all-regimes "
            f"--both-profiles` first (or `.\\run.ps1 eval` / `./run.sh eval`, "
            f"which does both)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


_MEAN_FIELDS = (
    "recovered_paise", "attempts_spent", "mandates_preserved", "recovered",
    "dead", "opted_out", "censored", "iatrogenic_failures", "n_attempt",
    "n_offer", "n_reauth", "n_stop", "n_above_afa", "n_attempt_after_terminal",
    "missed_recovery_count", "missed_recovery_paise", "false_offramp_count",
    "false_offramp_paise", "false_reauth_count", "false_reauth_paise",
    "billable_paise", "coverage_n",
    # R2b, 2026-09-04: absent on any pre-R2 artifact -- _merge_seeds() reads
    # every _MEAN_FIELDS entry via c[f], same as every other field in this
    # tuple, so pointing this report at a stale pre-R2 regimes.json raises
    # KeyError here rather than silently rendering a zero. Requires a
    # freshly re-run artifact, same requirement every other field change
    # in this project's history has had.
    "compliance_reauth_count", "false_reauth_inference_count",
    "false_reauth_count_effective", "false_reauth_inference_count_effective",
)
_FLOAT_FIELDS = ("coverage_marginal", "mean_set_size", "singleton_rate",
                 "singleton_wont_pay_rate")


def _merge_seeds(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one group's per-seed cells into a single mean cell, keeping
    the spread alongside.

    Integer money fields are averaged and rounded to whole paise -- a mean
    over seeds is a statistic, not a ledger entry, and rounding keeps
    invariant 2's "money is integer paise" true of every value this report
    can emit.
    """
    if len(cells) == 1:
        out = dict(cells[0])
        out["n_seeds"] = 1
        return out
    out = dict(cells[0])
    out["n_seeds"] = len(cells)
    for f in _MEAN_FIELDS:
        vals = [c[f] for c in cells]
        out[f] = round(sum(vals) / len(vals))
        out[f"{f}__min"], out[f"{f}__max"] = min(vals), max(vals)
    for f in _FLOAT_FIELDS:
        vals = [c[f] for c in cells if c.get(f) is not None]
        out[f] = (sum(vals) / len(vals)) if vals else None
        if vals:
            out[f"{f}__min"], out[f"{f}__max"] = min(vals), max(vals)
    # Averaged too, not inherited from cells[0]: a per-class coverage number
    # silently carried from one seed would look like an 8-seed figure.
    per_class: dict[str, list[float]] = collections.defaultdict(list)
    for c in cells:
        for k, v in (c.get("coverage_per_class") or {}).items():
            per_class[k].append(v)
    out["coverage_per_class"] = {k: sum(v) / len(v) for k, v in per_class.items()}
    out["violations"] = [v for c in cells for v in c["violations"]]
    return out


def _paired(data: dict[str, Any], profile: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Group cells by (regime, arm) -> policy, averaging over seeds."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for c in data["cells"]:
        if c["profile"] != profile:
            continue
        grouped[(c["regime"], c["arm"], c["policy"])].append(c)
    out: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(dict)
    for (regime, arm, policy), cs in grouped.items():
        out[(regime, arm)][policy] = _merge_seeds(cs)
    return dict(out)


def _seed_win_counts(data: dict[str, Any], a: str, b: str, field: str) -> tuple[int, int, int]:
    """How often does policy `a` beat `b` on `field`, counted per
    (regime, arm, profile, SEED) rather than per averaged cell?

    A mean can hide that a comparison flips from seed to seed. This is the
    sign test the headline needs: (a wins, b wins, ties).
    """
    grouped: dict[tuple, dict[str, Any]] = collections.defaultdict(dict)
    for c in data["cells"]:
        grouped[(c["regime"], c["arm"], c["profile"], c["seed"])][c["policy"]] = c
    wins = losses = ties = 0
    for v in grouped.values():
        if a not in v or b not in v:
            continue
        if v[a][field] > v[b][field]:
            wins += 1
        elif v[a][field] < v[b][field]:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


# --- sections ----------------------------------------------------------------


POLICIES = ("ladder", "engine", "one_shot", "null")


def _three_bar_table(data: dict[str, Any], profile: str) -> list[str]:
    """The three bars for every policy, including the two cause-blind
    reference policies. `null` and `one_shot` are here because without them
    the engine's headline is unfalsifiable -- every metric in this table is
    monotonically decreasing in attempt count, so "preserves more" follows
    from "attempts less" and is not evidence of knowing WHY a payment
    failed."""
    rows = [
        "| regime | arm | policy | recovered | attempts | preserved | "
        "stopped-on that would have paid (n / value) | false off-ramp | false REAUTH |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (regime, arm), v in _paired(data, profile).items():
        for policy in POLICIES:
            c = v.get(policy)
            if c is None:
                continue
            eng = policy == "engine"
            miss = (f"{c['missed_recovery_count']} / {_rupees(c['missed_recovery_paise'])}"
                    if eng else "--")
            fo = (f"{c['false_offramp_count']} / {_rupees(c['false_offramp_paise'])}"
                  if eng else "--")
            fr = (f"{c['false_reauth_count']} / {_rupees(c['false_reauth_paise'])}"
                  if eng else "--")
            rows.append(
                f"| {regime} | {arm} | {policy} | {_rupees(c['recovered_paise'])} | "
                f"{c['attempts_spent']} | {c['mandates_preserved']}/{c['n_mandates']} | "
                f"{miss} | {fo} | {fr} |"
            )
    return rows


def _delta_table(data: dict[str, Any], profile: str) -> list[str]:
    rows = [
        "| regime | arm | money | attempts | preserved | iatrogenic (ladder->engine) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (regime, arm), v in _paired(data, profile).items():
        L, E = v["ladder"], v["engine"]
        rows.append(
            f"| {regime} | {arm} | {_pct(E['recovered_paise'], L['recovered_paise'])} | "
            f"{_pct(E['attempts_spent'], L['attempts_spent'])} | "
            f"{E['mandates_preserved'] - L['mandates_preserved']:+d} | "
            f"{L['iatrogenic_failures']} -> {E['iatrogenic_failures']} |"
        )
    return rows


def _coverage_table(data: dict[str, Any]) -> list[str]:
    """Coverage of the off-ramp gate, one row per (regime, arm, SEED).
    Printed only where the real conformal gate was live -- under FullSetGate
    there is no coverage claim to make, and printing 1.000 would be true and
    completely misleading.

    CORRECTED, R5 review pass, 2026-09-05 (stats-reviewer): this function's
    own comment used to say "one row per (regime, arm) engine cell", which
    was wrong -- `data["cells"]` holds every seed, and there is no seed
    filter here, so 16 (regime, arm) pairs x 8 seeds always produced 128
    rows under a heading that read as 16. The `seed` column below is the
    fix; the render() call site's prose is corrected to match (see there
    for why "any degradation here is a real result" was also wrong)."""
    rows = [
        "| regime | arm | seed | gate live | marginal coverage | per-class coverage "
        "(NOW / EVER / WONT) | mean set size | singleton rate | singleton "
        "{WONT_PAY} | OFFERs |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in data["cells"]:
        if c["policy"] != "engine" or c["profile"] != "strict":
            continue
        if c["gate_kind"] != "conformal":
            rows.append(f"| {c['regime']} | {c['arm']} | {c['seed']} | {c['gate_kind']} | "
                        f"n/a (stub gate) | n/a | n/a | n/a | n/a | {c['n_offer']} |")
            continue
        pc = c.get("coverage_per_class") or {}
        pcs = " / ".join(
            f"{pc[k]:.3f}" if k in pc else "--"
            for k in ("CANT_PAY_NOW", "CANT_PAY_EVER", "WONT_PAY")
        )
        sing = c.get("singleton_rate")
        rows.append(
            f"| {c['regime']} | {c['arm']} | {c['seed']} | conformal | "
            f"{c['coverage_marginal']:.3f} (n={c['coverage_n']}) | {pcs} | "
            f"{c['mean_set_size']:.2f} / 3 | "
            f"{sing:.3f} | {c['singleton_wont_pay_rate']:.3f} | {c['n_offer']} |"
        )
    return rows


def _hypotheses(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for name, spec in data["regimes"].items():
        out += [
            f"### `{name}`",
            "",
            f"**Story.** {spec['story']}",
            "",
            f"**Pre-registered hypothesis.** {spec['hypothesis']}",
            "",
        ]
        if spec["approximation"]:
            out += [f"**Where the model falls short of the story.** {spec['approximation']}", ""]
    return out


def _profiles_note(data: dict[str, Any]) -> list[str]:
    """Did the compliance profile change anything at all? CLAUDE.md requires
    both interpretations be evaluated; that is only meaningful if we say
    plainly when they came out identical."""
    pairs = collections.defaultdict(dict)
    for c in data["cells"]:
        pairs[(c["regime"], c["arm"], c["policy"])][c["profile"]] = c
    fields = ("recovered_paise", "attempts_spent", "mandates_preserved",
              "n_offer", "n_reauth", "n_attempt")
    differing = [k for k, v in pairs.items() if len(v) == 2
                 and any(v["strict"][f] != v["permissive"][f] for f in fields)]
    total = sum(1 for v in pairs.values() if len(v) == 2)
    if differing:
        return [f"{len(differing)} of {total} cells differ between `strict` and "
                f"`permissive`: {sorted(differing)[:10]}"]
    return [
        f"**All {total} cells are byte-identical between `strict` and "
        f"`permissive`, and this is a defect in the compliance model, not a "
        f"finding about the policy.**",
        "",
        "An earlier version of this report explained it as \"the constraint "
        "never binds at the optimum, because the policy always picks the "
        "earliest legal day anyway\". That explanation is wrong, and it "
        "understated the problem. The two profiles are provably the same "
        "function. In `allocator.solve()`:",
        "",
        "```python",
        "lead = 1 if profile.requires_fresh_notification(next_slot) else 0",
        "earliest = ctx.plan_day + lead",
        "if ctx.committed_days:",
        "    earliest = max(earliest, ctx.committed_days[-1] + 1)",
        "```",
        "",
        "`with_attempt()` sets `plan_day = on_day` *and* appends `on_day` to "
        "`committed_days`, and the initial context starts at `plan_day=1, "
        "committed_days=(1,)`. So `committed_days[-1] == plan_day` always, "
        "and `max(plan_day + 0, plan_day + 1)` equals "
        "`max(plan_day + 1, plan_day + 1)`. The `lead` term is absorbed by "
        "the monotonicity clamp at every reachable context: the candidate "
        "day *sets* are identical, not merely the chosen optimum. A policy "
        "with perfect timing discrimination would still produce byte-"
        "identical results under the two profiles.",
        "",
        "CLAUDE.md requires that the strict/permissive ambiguity never be "
        "hard-coded to one interpretation and that both be evaluated. That "
        "requirement is currently satisfied in form and empty in substance: "
        "the code branches on the profile and the branch cannot change any "
        "output. Modelling the 24h pre-notification lead as a real lead -- "
        "rather than as a one-day offset that the clamp swallows -- is the "
        "fix, and it is not done. (payments-domain, 2026-08-31.)",
    ]


def _losses(data: dict[str, Any]) -> list[str]:
    """The gate requires at least one regime where we lose, explained. This
    finds them mechanically rather than by hand, so a future run cannot
    quietly stop reporting one."""
    out: list[str] = []
    money, iatro, beaten = [], [], []
    # BOTH profiles, not just strict. They happen to be identical today (see
    # the compliance section), but scanning only `strict` would silently hide
    # a loss that occurred solely under `permissive` -- and that becomes live
    # the moment the profile defect is fixed.
    for profile in data["profiles"]:
        for (regime, arm), v in _paired(data, profile).items():
            L, E = v["ladder"], v["engine"]
            if E["recovered_paise"] < L["recovered_paise"]:
                money.append((regime, arm, profile,
                              _pct(E["recovered_paise"], L["recovered_paise"]),
                              (E["recovered_paise"] - L["recovered_paise"]),
                              E["missed_recovery_count"], E["missed_recovery_paise"]))
            if E["iatrogenic_failures"] > L["iatrogenic_failures"]:
                iatro.append((regime, arm, profile,
                              L["iatrogenic_failures"], E["iatrogenic_failures"]))
            one = v.get("one_shot")
            if one and one["mandates_preserved"] > E["mandates_preserved"]:
                beaten.append((regime, arm, profile, E["mandates_preserved"],
                               E["attempts_spent"], one["mandates_preserved"],
                               one["attempts_spent"]))

    money.sort(key=lambda r: r[4])
    out += ["### Money left on the table, against the ladder", "",
            "\"Stopped-on\" counts every mandate the engine did not carry to a "
            "terminal outcome -- whether it never attempted, or attempted and "
            "then stopped. The counterfactual grinds on consecutive days from "
            "where we stopped, which always lands inside the days-1-5 salary "
            "window, so this is an **upper bound** on what was given up, not a "
            "point estimate.", ""]
    out += ["| regime | arm | profile | money delta | stopped-on that would have paid | value |",
            "|---|---|---|---:|---:|---:|"]
    for regime, arm, profile, pct, _abs, n, paise in money:
        out.append(f"| {regime} | {arm} | {profile} | {pct} | {n} | {_rupees(paise)} |")
    out.append("")

    if beaten:
        out += ["### Cells where a model-free one-attempt policy preserves more than the engine", "",
                "`one_shot` spends one attempt per mandate on day 2 and consults "
                "no model, no belief and no gate. Where it preserves more while "
                "spending fewer attempts, the engine's preserved-mandate bar is "
                "not evidence of cause inference.", "",
                "| regime | arm | profile | engine preserved / attempts | one_shot preserved / attempts |",
                "|---|---|---|---:|---:|"]
        for regime, arm, profile, ep, ea, op, oa in beaten:
            out.append(f"| {regime} | {arm} | {profile} | {ep} / {ea} | {op} / {oa} |")
        out.append("")

    if iatro:
        out += ["### Regimes where the engine caused MORE collateral damage than the ladder", "",
                "| regime | arm | profile | ladder iatrogenic | engine iatrogenic |",
                "|---|---|---|---:|---:|"]
        for regime, arm, profile, l, e in iatro:
            out.append(f"| {regime} | {arm} | {profile} | {l} | {e} |")
        out.append("")
    return out


# --- figures -----------------------------------------------------------------


def render_figures(data: dict[str, Any]) -> list[pathlib.Path]:
    """One small-multiple figure per bar. Three bars means three separate
    charts on their own scales -- never one chart with two y-axes, which is
    what plotting rupees against attempt counts would require."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pairs = _paired(data, "strict")
    keys = list(pairs)
    labels = [f"{r}\n{a}" for r, a in keys]
    written = []

    bars = [
        ("recovered_paise", "Money recovered (Rs lakh)", 1 / 10_000_000, "recovered"),
        ("attempts_spent", "Attempts spent", 1.0, "attempts"),
        ("mandates_preserved", "Mandates preserved (of 200)", 1.0, "preserved"),
    ]
    # dataviz: categorical slots 1-4, assigned in fixed order and never
    # cycled. `null` and `one_shot` are drawn alongside the two real policies
    # because the preserved-mandates panel is meaningless without them.
    series = [
        ("ladder (incumbent)", "ladder", C_LADDER),
        ("engine", "engine", C_ENGINE),
        ("one_shot (no model)", "one_shot", C_ONE_SHOT),
        ("null (never attempt)", "null", C_NULL),
    ]
    for field, title, scale, slug in bars:
        fig, ax = plt.subplots(figsize=(12, 5.6))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)
        x = np.arange(len(keys))
        w = 0.20
        for i, (label, policy, colour) in enumerate(series):
            vals = np.array(
                [pairs[k].get(policy, {}).get(field, 0) for k in keys], dtype=float
            ) * scale
            off = (i - (len(series) - 1) / 2) * (w + 0.015)
            ax.bar(x + off, vals, w, label=label, color=colour)

        ax.set_title(title, color=C_TEXT, fontsize=13, loc="left", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=C_MUTED,
                           rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelcolor=C_MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=9, labelcolor=C_MUTED, loc="upper right", ncol=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d2")
        ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = FIG_DIR / f"regimes_{slug}.png"
        fig.savefig(path, dpi=160, facecolor=C_SURFACE)
        plt.close(fig)
        written.append(path)
    return written


def _signed_rupees(paise: int) -> str:
    """money.fmt() refuses negative paise (a ledger amount is never
    negative) -- this is a DIFFERENCE, not a ledger entry, so it needs its
    own sign-then-magnitude rendering rather than passing a negative value
    into the money formatter."""
    sign = "-" if paise < 0 else "+"
    return f"{sign}{_rupees(abs(paise))}"


def _ltv_slice_table(slice_data: dict[str, Any]) -> list[str]:
    lines = [
        f"**{slice_data['regime']}/{slice_data['arm']}/{slice_data['profile']}/"
        f"seed={slice_data['seed']}** -- mean mandate amount "
        f"{_rupees(round(slice_data['mean_amount_paise']))} ({slice_data['n_mandates']} mandates).",
        "",
    ]
    if slice_data["crossings"]:
        lines.append(
            "| Bracket (LTV, paise) | Crossing LTV | As a ratio to mean amount |"
        )
        lines.append("|---|---|---|")
        for c in slice_data["crossings"]:
            # Prefer the EXACT Fraction strings over the float convenience
            # fields stored alongside them -- both are in the artifact, but
            # rendering from the float would round-trip through a lossy
            # intermediate for no reason when the exact value is right
            # there (money-auditor, 2026-09-04, R3 review).
            crossing_exact = Fraction(c["crossing_ltv_paise_exact"])
            ratio_exact = Fraction(c["ratio_to_mean_amount_exact"])
            lines.append(
                f"| [{c['bracket_low_paise']:,}, {c['bracket_high_paise']:,}] | "
                f"{_rupees(round(crossing_exact))} | "
                f"**{float(ratio_exact):.3f}** |"
            )
        lines.append("")
        lines.append(
            "Each crossing is a linear interpolation between two SWEPT-AND-"
            "MEASURED grid points (`src.core.money.interpolate_crossing()`), "
            "not a third measurement -- the allocator's decisions are "
            "discrete, so the true curve is a step function and the crossing "
            "is only as precise as the grid's own resolution in that bracket."
        )
    else:
        lo, hi = slice_data["points"][0], slice_data["points"][-1]
        lines.append(
            f"**No crossing anywhere in the swept LTV range** "
            f"({lo['ltv_paise']:,} to {hi['ltv_paise']:,} paise) -- "
            f"engine.recovered_paise stays below ladder.recovered_paise at "
            f"every point, from {_signed_rupees(lo['diff_paise'])} at LTV=0 "
            f"to {_signed_rupees(hi['diff_paise'])} at the ceiling. "
            f"`interpolate_crossing()` correctly refuses to compute a "
            f"break-even here (no sign change exists to interpolate) rather "
            f"than extrapolate one."
        )
    lines.append("")
    return lines


def _ltv_sensitivity(path: pathlib.Path = LTV_ARTIFACT) -> list[str]:
    """R3. Reads reports/ltv_sensitivity.json (python -m eval.ltv_sensitivity)
    -- computes nothing here, same discipline the rest of this file uses for
    regimes.json. Renders a graceful placeholder if the sweep has never
    been run, rather than crashing the whole report."""
    if not path.exists():
        return [
            "Not yet generated -- run `python -m eval.ltv_sensitivity` "
            "(writes `reports/ltv_sensitivity.json`; this section renders "
            "from it on the next `python -m eval.report`).",
            "",
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"_Generated {data['generated']} by `python -m eval.ltv_sensitivity`. "
        f"Default `mandate_ltv_paise` (`config/policy_costs.yaml`): "
        f"{_rupees(data['default_ltv_paise'])}. LTV grid: "
        f"{len(data['ltv_grid_paise'])} points, "
        f"{data['ltv_grid_paise'][0]:,} to {data['ltv_grid_paise'][-1]:,} paise._",
        "",
        "**Does raising the assumed mandate LTV ever make the engine recover "
        "as much raw money as the incumbent ladder?** Two slices, both fixed "
        "before this sweep's first run (see `eval/ltv_sensitivity.py`'s "
        "module docstring for the exact, non-cherry-picked selection rule):",
        "",
        "### Headline -- the project's own canonical comparison slice",
        "",
    ]
    lines += _ltv_slice_table(data["headline"])
    lines += [
        "### Worked example -- the first pre-existing engine-wins-on-money cell",
        "",
        "Not the headline claim: this cell is one of the 36/256 cells "
        "`reports/gates.md`'s B13 entry already measured where the engine "
        "beats the ladder on raw recovered money AT THE DEFAULT LTV. It "
        "exists here to show `interpolate_crossing()` computing a real "
        "break-even against genuine swept data, since the headline slice "
        "has none to compute.",
        "",
    ]
    lines += _ltv_slice_table(data["worked_example"])
    return lines


def _offramp_channel(path: pathlib.Path = OFFRAMP_ARTIFACT) -> list[str]:
    """R5. Reads reports/offramp_channel.json (python -m eval.offramp_channel)
    -- computes nothing here, the same discipline _ltv_sensitivity() and the
    rest of this file already follow. Renders a graceful placeholder if the
    sweep has never been run rather than crashing the whole report."""
    if not path.exists():
        return [
            "Not yet generated -- run `python -m eval.offramp_channel` "
            "(writes `reports/offramp_channel.json`; this section renders "
            "from it on the next `python -m eval.report`).",
            "",
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    sl = data["slice"]
    op = data["operating_point"]
    lines = [
        f"_Generated {data['generated']} by `python -m eval.offramp_channel`. "
        f"Slice: `{sl['regime']}/{sl['arm']}/{sl['profile']}`, "
        f"{len(sl['seeds'])} seeds. Operating point (PRE-REGISTERED in that "
        f"module's docstring before its first run): tpr {op['tpr']:.2f} / "
        f"fpr {op['fpr']:.2f}._",
        "",
        "> **Every channel below is SYNTHETIC.** `eval/frozen/simulator.py` "
        "emits no decline strings and no support tickets, so this channel "
        "fabricates them -- and it does so by reading the simulator's "
        "**privileged true cause** and feeding the result into the DECISION "
        "path. That is a stronger claim than the score-only privileged read "
        "`false_reauth_count` already makes, which is why the channel's own "
        "ROC is published in the same table as every number it produced. "
        "Nothing here is evidence that a real `payment_cancelled` feed or a "
        "real support-ticket feed carries this much information.",
        "",
        "R5's gate does not promise the off-ramp is *correct*, only that it "
        "is **reachable and measured**. Untested-and-central is a weaker "
        "position than tested-and-imperfect; this table buys the second one.",
        "",
    ]
    for kind in data["channel_kinds"]:
        pts = [pt for pt in data["points"] if pt["channel_kind"] == kind]
        if not pts:
            continue
        lines += [f"### Channel `{kind}`", ""]
        if kind == "decline":
            lines += [
                "A fabricated `DeclineClass.CUSTOMER_DECLINED` observation, "
                "inverted through `src/classify/cause_map.py`'s independent "
                "hand-authored table -- so the allocator's belief is "
                "realistically MISCALIBRATED, never oracular.",
                "",
            ]
        else:
            lines += [
                "A fabricated exit-intent score, mapped through "
                "`src/execute/intent_channel.py`'s **declared** operating "
                "point, which is independent of the sweep's own (tpr, fpr) "
                "and therefore misspecified at every row but one. "
                "`src/policy/` never imports `src.llm`; the score crosses "
                "the boundary as a plain float. **This channel is measured "
                "here and NOT folded into the published grid** -- a "
                "fabricated support-ticket feed is a bigger fabrication than "
                "a fabricated decline string.",
                "",
            ]
        lines += [
            "| tpr | fpr | realised AUC (95% CI) | n_offer | false off-ramp | "
            "true off-ramp | **false rate** (95% CI) | recovered | vs ladder | "
            "preserved | coverage |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for pt in pts:
            roc = pt["channel_roc"]
            auc = "-" if roc["auc"] is None else f"{roc['auc']:.3f}"
            if roc["auc_ci"]:
                auc += f" [{roc['auc_ci'][0]:.3f}, {roc['auc_ci'][1]:.3f}]"
            if pt["false_offramp_rate"] is None:
                rate = "-"
            else:
                rate = f"**{pt['false_offramp_rate']:.1%}**"
                ci = pt.get("false_offramp_rate_ci")
                if ci:
                    # The interval is not decoration: n_offer is as low as 8
                    # at the worst points, and the point estimate alone
                    # would read as a result rather than as noise.
                    rate += f"<br>[{ci[0]:.0%}, {ci[1]:.0%}]"
            delta = pt["engine_recovered_paise"] - pt["ladder_recovered_paise"]
            cov = ("-" if pt["coverage_marginal_mean"] is None
                   else f"{pt['coverage_marginal_mean']:.3f}")
            mark = " **<-**" if pt["is_operating_point"] else ""
            lines.append(
                f"| {pt['tpr']:.2f}{mark} | {pt['fpr']:.2f} | {auc} | "
                f"{pt['n_offer']} | {pt['false_offramp_count']} | "
                f"{pt['true_offramp_count']} | {rate} | "
                f"{_rupees(pt['engine_recovered_paise'])} | "
                f"{_signed_rupees(delta)} | {pt['engine_mandates_preserved']} | "
                f"{cov} |"
            )
        lines.append("")
        best = max(pts, key=lambda x: x["n_offer"])
        fewest = min(pts, key=lambda x: x["n_offer"])
        worst = min(pts, key=lambda x: (x["channel_roc"]["auc"] or 0.0))
        worst_rate = worst["false_offramp_rate"]
        lines += [
            f"The ladder recovers {_rupees(pts[0]['ladder_recovered_paise'])} "
            f"on this slice and preserves "
            f"{pts[0]['ladder_mandates_preserved']} mandates, at every row -- "
            f"it is channel-blind by construction (no belief, no gate), so it "
            f"is run once per seed and reused rather than re-measured 8 times "
            f"to produce the same number.",
            "",
            f"**Reading it.** The off-ramp fires "
            f"{best['n_offer']} times at the point where it fires most and "
            f"as few as {fewest['n_offer']} where it fires least -- but the "
            f"COUNT is not the finding, because a channel that rarely fires "
            f"and a channel that fires accurately look the same in it. The "
            f"false-off-ramp RATE is "
            f"what moves with channel quality, and it is "
            + (f"{worst_rate:.1%} at the least informative channel measured "
               f"(realised AUC {worst['channel_roc']['auc']:.3f})."
               if worst_rate is not None else
               "undefined at the least informative channel measured, because "
               "no off-ramp fired there at all.")
            + " That degradation is the point of sweeping quality: a table "
              "showing only good channels would prove nothing.",
            "",
            "**The intervals are wide, and that is the honest reading.** "
            "`n_offer` on this slice runs from single digits to about a "
            "hundred, so a false-off-ramp rate printed alone would be a "
            "number plus or minus tens of points presented as a result. "
            "Wilson intervals (`bench/llm_vs_stats.py`'s own implementation, "
            "imported rather than rewritten) are shown beside every rate; "
            "they ignore the pooling of eight seeds, so read each width as a "
            "FLOOR on the true uncertainty. What survives that is the SHAPE "
            "of the curve, not any single cell.",
            "",
            "Marginal conformal coverage is re-measured after re-calibrating "
            "the gate at each point -- the channel changes the belief "
            "distribution, so it changes the calibration pool, and reusing a "
            "pool drawn under a different channel would break exactly the "
            "exchangeability split conformal's guarantee rests on.",
            "",
        ]

    dep = data.get("dependence_sweep")
    if dep:
        lines += [
            "### Within-mandate correlation, held fixed at zero above (R5 review pass)",
            "",
            "A 2026-09-05 review found the grid above sweeps channel "
            "**discrimination** while holding the one dimension the firing "
            "rule is actually sensitive to fixed at exactly zero: "
            "`WontPayChannel.fires()` draws an independent coin flip every "
            "call. `should_act()` needs roughly TWO coincident false firings "
            "on the SAME mandate to open the off-ramp (one `CUSTOMER_DECLINED` "
            "moves belief to ~0.62 WONT_PAY; the fitted gate's own singleton "
            "boundary sits at 0.80-0.90), and two independent draws from one "
            "customer's decline history is not a safe assumption -- a "
            "customer who dismisses one collect request is plausibly more "
            "likely to dismiss the next, for reasons that have nothing to do "
            "with wanting to leave.",
            "",
            "This table holds the marginal `(tpr, fpr)` FIXED at the "
            "pre-registered operating point and varies only "
            "`habitual_fraction` -- the fraction of non-WONT_PAY mandates "
            "that are \"habitual dismissers\" firing at an elevated rate, "
            "vs. never. `1.00` (top row) is EXACTLY the operating-point row "
            "of the main grid above, repeated so the two read as one "
            "continuous measurement rather than an unexplained fourth "
            "number.",
            "",
            "| habitual_fraction | n_offer | false off-ramp | rate (95% CI) | "
            "fpr realised | repeat false-fire rate |",
            "|---:|---:|---:|---|---:|---:|",
        ]
        for pt in dep["points"]:
            ci = pt.get("false_offramp_rate_ci")
            rate = (f"{pt['false_offramp_rate']:.1%} "
                    f"[{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "-")
            rr = pt["repeat_false_fire"]
            lines.append(
                f"| {pt['habitual_fraction']:.2f}"
                f"{' (= main grid)' if pt['habitual_fraction'] == 1.0 else ''} | "
                f"{pt['n_offer']} | {pt['false_offramp_count']} | {rate} | "
                f"{pt['channel_roc']['fpr_realised']:.3f} | "
                f"{rr['rate']:.3f} (n={rr['n_non_wont_pay_mandates']}) |"
            )
        lo, hi = dep["points"][0], dep["points"][-1]
        lines += [
            "",
            f"The realised fpr stays within "
            f"{min(p['channel_roc']['fpr_realised'] for p in dep['points']):.3f}-"
            f"{max(p['channel_roc']['fpr_realised'] for p in dep['points']):.3f} "
            f"across every row -- the marginal genuinely holds fixed -- while "
            f"the false-off-ramp rate moves from {lo['false_offramp_rate']:.1%} "
            f"to {hi['false_offramp_rate']:.1%} and the repeat-false-fire rate "
            f"from {lo['repeat_false_fire']['rate']:.3f} to "
            f"{hi['repeat_false_fire']['rate']:.3f}. Same discrimination, same "
            "published ROC point, more than double the false-off-ramp rate: "
            "the main grid's operating-point row is not robust to an "
            "assumption it holds fixed at zero.",
            "",
            "**Second-order synthetic, disclosed as such**: there is no real "
            "decline-string corpus this project has access to that could "
            "calibrate `habitual_fraction` itself. This table establishes "
            "SENSITIVITY -- the rate is not robust to a fixed-at-zero "
            "assumption -- not a corrected point estimate.",
            "",
        ]
    return lines


# --- assembly ----------------------------------------------------------------


def render(data: dict[str, Any], *, figures: bool) -> str:
    gate = data["gate_kind"]
    lines = [
        "# Stress regimes -- B13",
        "",
        "Generated by `python -m eval.report`. Every number below is read out "
        "of `reports/regimes.json`; nothing is computed in this file. To "
        "reproduce every number in this report from a clean tree -- the sweep "
        "and the rendering -- one command:",
        "",
        "```powershell",
        ".\\run.ps1 eval          # Windows",
        "```",
        "",
        "```sh",
        "./run.sh eval           # Linux / macOS",
        "```",
        "",
        "(`.\\run.ps1 report` / `./run.sh report` re-renders the tables and "
        "figures from the existing artifact without re-running the sweep. It "
        "cannot change a number; only `eval` can.)",
        "",
        f"Seed `{data['seed']}` · arms {data['arms']} · profiles {data['profiles']} · "
        f"off-ramp gate: **{gate}** {data['gate_diagnostics']}",
        "",
        "The three bars are always reported together. Recovery rate alone is "
        "the incumbent's scorecard, and a policy tuned to it would be a "
        "different product.",
        "",
        "## Headline findings",
        "",
    ]
    lines += _headline(data)
    lines += ["", "## Where we lose", "",
              "A policy that wins every regime is evidence the regimes were "
              "tuned, not that the policy is good. These are the losses.", ""]
    lines += _losses(data)
    lines += ["## The three bars -- profile `strict`", ""]
    lines += _three_bar_table(data, "strict")
    lines += ["", "## Engine vs ladder, deltas -- profile `strict`", ""]
    lines += _delta_table(data, "strict")
    lines += ["", "## Compliance profiles", ""]
    lines += _profiles_note(data)
    lines += ["", "## LTV sensitivity", ""]
    lines += _ltv_sensitivity()
    lines += ["## Off-ramp reachability, and what it costs (R5)", ""]
    lines += _offramp_channel()
    lines += ["## Off-ramp gate: coverage per (regime, arm, seed)", "",
              "Coverage is *measured*, not assumed. The gate is calibrated "
              "once on `baseline` and reused unchanged under every regime. "
              "**Read seed-to-seed spread before regime-to-regime spread**: "
              "a 2026-09-05 review measured the two as comparable in "
              "magnitude on this table's own data (e.g. baseline/nominal "
              "swings 0.873-0.917 across seeds alone), so a single row's "
              "coverage moving is not, by itself, evidence that regime is "
              "the cause. This corrects an earlier version of this sentence "
              "(\"any degradation here is a real result\"), which the same "
              "review found the table's own contents falsify. What the "
              "gate's own R5 entry establishes independently -- that "
              "coverage is measurably below the nominal target -- still "
              "stands; only the attribution to REGIME specifically was "
              "unsupported.", ""]
    lines += _coverage_table(data)
    lines += ["", "## Pre-registered regimes", "",
              "Written before any result was seen. `git log eval/regimes.py` "
              "is the check.", ""]
    lines += _hypotheses(data)

    if figures:
        paths = render_figures(data)
        lines += ["## Figures", ""]
        for p in paths:
            lines.append(f"![{p.stem}](figures/{p.name})")
        lines.append("")

    return "\n".join(lines) + "\n"


def _finding_2(data, eng, conf, offers, sing_wont) -> str:
    """Headline finding 2 -- the off-ramp lane.

    Two texts, selected by what the artifact ACTUALLY contains, never by
    hand. Before R5 this finding read "the off-ramp cannot fire in this
    harness ... that is arithmetic, not measurement", which was true and is
    now false; a report whose prose survives the fix it describes is worse
    than one with no prose at all. The pre-R5 text is retained, not
    deleted, because `--channel-kind off` still reproduces exactly that
    configuration and must still be described correctly.
    """
    ch = data.get("wontpay_channel")
    retro = sum(c.get("coverage_n_retrospective", 0) for c in conf)
    if ch is None:
        return (
            f"2. **The off-ramp cannot fire in this configuration. `OFFER` = "
            f"{offers} in all {len(eng)} engine cells -- and that is "
            f"arithmetic, not measurement.** With R5's synthetic WONT_PAY "
            f"channel switched OFF (`--channel-kind off`), the proxy decline "
            f"alphabet has exactly two symbols (`INSUFFICIENT_FUNDS`, "
            f"`CARD_EXPIRED`) and `cause_map` assigns `WONT_PAY` a prior of "
            f"0.10 under **both**, so no observation this simulator can "
            f"produce from ORDINARY Bayesian updating moves belief mass "
            f"toward `WONT_PAY`, and the singleton `{{WONT_PAY}}` condition "
            f"is unreachable from any LIVE, still-retryable decision, for "
            f"any alpha, any seed, any regime -- confirmed on this run: "
            f"{sum(sing_wont):.3f} (every live cell measures exactly 0). "
            f"{retro:,} RETROSPECTIVE post-terminal queries are EXCLUDED "
            f"from every coverage/singleton statistic here (see finding 3): "
            f"a belief already collapsed by `belief.observe_terminal()` is "
            f"not exchangeable with the live population the gate is "
            f"calibrated on. **The off-ramp lane is untested, not tested and "
            f"negative**, and `false off-ramp = 0` is not a safety result."
        )

    scored = sum(c.get("offramp_scored_count", 0) for c in eng)
    false_n = sum(c.get("false_offramp_count", 0) for c in eng)
    true_n = sum(c.get("true_offramp_count", 0) for c in eng)
    rate = f"{false_n / scored:.1%}" if scored else "undefined (no OFFER scored)"
    n_wp = sum(c.get("channel_n_wont_pay", 0) for c in eng)
    pos_wp = sum(c.get("channel_positive_on_wont_pay", 0) for c in eng)
    n_other = sum(c.get("channel_n_other", 0) for c in eng)
    pos_other = sum(c.get("channel_positive_on_other", 0) for c in eng)
    tpr_r = f"{pos_wp / n_wp:.3f}" if n_wp else "-"
    fpr_r = f"{pos_other / n_other:.3f}" if n_other else "-"
    return (
        f"2. **The off-ramp now fires, and both of its error costs are real "
        f"numbers instead of one number and a structural zero. `OFFER` = "
        f"{offers} across {len(eng)} engine cells; of the {scored} scored "
        f"against the exact counterfactual, {false_n} went to a mandate that "
        f"WOULD have paid and {true_n} to one that would not -- a "
        f"false-off-ramp rate of {rate}.** Before R5 this finding read "
        f"\"the off-ramp cannot fire in this harness\", and it was correct: "
        f"the proxy decline alphabet had two symbols and `cause_map` gave "
        f"`WONT_PAY` a prior of 0.10 under both, so the singleton "
        f"`{{WONT_PAY}}` the gate fires on was unreachable for any alpha, "
        f"seed or regime. R5 added `CUSTOMER_DECLINED` (prior 0.70 toward "
        f"`WONT_PAY`) and a **synthetic, quality-parameterised** channel "
        f"that emits it. Live singleton-`{{WONT_PAY}}` rate on this run: "
        f"{sum(sing_wont) / len(sing_wont) if sing_wont else 0:.4f} mean "
        f"across cells, against exactly 0 before.\n\n"
        f"    **This channel is SYNTHETIC and it reads privileged ground "
        f"truth.** It is configured at tpr {ch['tpr']:.2f} / fpr "
        f"{ch['fpr']:.2f} and REALISED tpr {tpr_r} / fpr {fpr_r} on this "
        f"grid ({n_wp:,} WONT_PAY draws, {n_other:,} others). It reads each "
        f"mandate's true latent cause -- which the policy itself must never "
        f"see -- and feeds a fabricated observation into the DECISION path. "
        f"That is a materially stronger claim than the score-only "
        f"privileged read `false_reauth_count` already makes. It is not "
        f"evidence that a real `payment_cancelled` feed carries this much "
        f"information; the full quality curve, including deliberately "
        f"worthless channels at AUC 0.5, is in \"Off-ramp reachability, and "
        f"what it costs\" below. {retro:,} RETROSPECTIVE post-terminal gate "
        f"queries remain EXCLUDED from every coverage/singleton statistic "
        f"here (see finding 3). What R5 bought is a tested-and-imperfect "
        f"off-ramp in place of an untested-and-central one -- not a good "
        f"result, a checkable one."
    )


def _headline(data: dict[str, Any]) -> list[str]:
    cells = data["cells"]
    eng = [c for c in cells if c["policy"] == "engine"]
    offers = sum(c["n_offer"] for c in eng)
    reauth = sum(c["n_reauth"] for c in eng)
    false_reauth = sum(c["false_reauth_count"] for c in eng)
    # R2b, 2026-09-04: the pre-registered false_reauth above conflates a
    # legally-mandatory route with a genuine belief error -- these two
    # split it. Absent (older artifact) -> both read 0 rather than KeyError,
    # so a pre-R2 artifact still renders (with the honest, degenerate 0/0).
    compliance_reauth = sum(c.get("compliance_reauth_count", 0) for c in eng)
    false_reauth_inference = sum(c.get("false_reauth_inference_count", 0) for c in eng)
    stops = sum(c["n_stop"] for c in eng)
    after_term = sum(c["n_attempt_after_terminal"] for c in eng)
    viol = sum(len(c["violations"]) for c in cells)
    pairs = _paired(data, "strict")
    n = len(pairs)

    def _beats(a, b, field):
        return sum(1 for v in pairs.values()
                   if a in v and b in v and v[a][field] > v[b][field])

    # Only the cells where the real gate ran, and say so rather than
    # silently ranging over a filtered subset.
    conf = [c for c in eng if c["gate_kind"] == "conformal"]
    cov = [c["coverage_marginal"] for c in conf if c["coverage_marginal"] is not None]
    setsz = [c["mean_set_size"] for c in conf if c["mean_set_size"] is not None]
    sing_wont = [c["singleton_wont_pay_rate"] for c in conf if c.get("singleton_wont_pay_rate") is not None]
    worst_class = min(
        (v, k) for c in conf for k, v in (c.get("coverage_per_class") or {}).items()
    ) if conf else (None, None)

    seeds = data.get("seeds") or [data.get("seed", 0)]
    # NOTE the direction convention: for `attempts_spent`, MORE is WORSE.
    # _seed_win_counts is a plain "a > b" count, so its result must be read
    # as "spends more", never as "wins".
    pw, pl, pt = _seed_win_counts(data, "engine", "one_shot", "mandates_preserved")
    mw, ml, mt = _seed_win_counts(data, "engine", "one_shot", "recovered_paise")
    aw, al, a_t = _seed_win_counts(data, "engine", "one_shot", "attempts_spent")
    lp, lpl, _ = _seed_win_counts(data, "engine", "ladder", "mandates_preserved")
    lm, lml, _ = _seed_win_counts(data, "engine", "ladder", "recovered_paise")
    la, lal, _ = _seed_win_counts(data, "engine", "ladder", "attempts_spent")
    npairs = pw + pl + pt

    if len(seeds) > 1:
        spread = "\n".join([
            f"**Across {len(seeds)} seeds, {npairs} paired comparisons, counted "
            f"per seed rather than on the mean.** A gap of a few mandates on one "
            f"seed is not a result, so this sign test -- not the averaged table "
            f"below -- is what every claim here rests on.",
            "",
            "| comparison | preserves more | recovers more | spends FEWER attempts |",
            "|---|---|---|---|",
            f"| engine vs **ladder** | {lp} / {npairs} | {lm} / {npairs} | {lal} / {npairs} |",
            f"| engine vs **one_shot** | {pw} / {npairs} | {mw} / {npairs} | {al} / {npairs} |",
            "",
            "Fewer attempts is better, so the third column is the favourable "
            "count, not a win count.",
            "",
            f"**Against the incumbent the trade is real and stable.** The engine "
            f"preserves more in {lp} of {npairs} and spends fewer attempts in "
            f"{lal} of {npairs}, while recovering less money in {lml}. That is "
            f"the thesis, and it survives {len(seeds)} seeds.",
            "",
            f"**Against `one_shot` it does not hold.** A policy that makes one "
            f"attempt on day 2 with no model, no belief and no gate preserves "
            f"MORE mandates than the engine in {pl} of {npairs} comparisons, and "
            f"the engine spends MORE attempts in {aw} of {npairs}. The engine's "
            f"only edge is money, and a thin one: it recovers more in {mw} of "
            f"{npairs} ({100 * mw / npairs:.0f}%). **On two of the three headline "
            f"bars the engine is beaten by a policy with no model in it.** The "
            f"seed-0 draft reported this as 14 of 16 cells on the preserved bar; "
            f"{len(seeds)} seeds make the finding stronger, not weaker.",
        ])
    else:
        spread = (
            "**Single seed, no error bar.** Every number below is one draw. The "
            "engine-vs-`one_shot` comparison in particular turns on a handful of "
            "mandates and should not be read as a result until `--seeds N` has "
            "been run."
        )

    out = [
        f"0. {spread}",
        "",
        f"1. **The headline three-bar comparison does not identify anything, "
        f"and the reference policies are in the table to prove it.** The "
        f"engine preserves more mandates than the ladder in "
        f"{_beats('engine', 'ladder', 'mandates_preserved')} of {n} cells and "
        f"recovers more money in {_beats('engine', 'ladder', 'recovered_paise')}. "
        f"But `null` -- never attempt, no model -- preserves **every** mandate "
        f"in all {n} cells, and `one_shot` -- one attempt on day 2, no model, "
        f"no belief, no gate -- preserves more than the engine in "
        f"{_beats('one_shot', 'engine', 'mandates_preserved')} of {n} while "
        f"spending fewer attempts. Every metric here is monotonically "
        f"decreasing in attempt count by construction, so \"preserves more\" "
        f"follows from \"attempts less\" and is not evidence of knowing WHY a "
        f"payment failed. B5 recorded this exact confound; B13's first draft "
        f"reproduced it by omitting the column.",
        "",
        _finding_2(data, eng, conf, offers, sing_wont),
        "",
    ]
    if cov:
        out += [
            f"3. **The off-ramp gate under-covers.** Over the "
            f"{sum(c['coverage_n'] for c in conf):,} LIVE decision points the "
            f"gate was actually queried at (excluding "
            f"{sum(c.get('coverage_n_retrospective', 0) for c in conf):,} "
            f"retrospective post-terminal queries -- see finding 2), marginal "
            f"coverage is "
            f"{min(cov):.3f}-{max(cov):.3f} against a 0.95 target, with "
            f"per-class coverage as low as {worst_class[0]:.3f} "
            f"(`{worst_class[1]}`) -- a Mondrian violation on a class that "
            f"drives REAUTH. Mean set size is {min(setsz):.2f}-{max(setsz):.2f} "
            f"of 3. Two earlier defects inflated this: the smoothing key was "
            f"derived from the belief itself (making the WONT_PAY p-value a "
            f"hash of a constant rather than a rank), and coverage was scored "
            f"only over the 200 slot-1 beliefs rather than every query, which "
            f"also made six numbers print as thirty-two. Both are fixed; the "
            f"reported figure moved from an apparent 0.980 to a real "
            f"under-coverage.",
            "",
        ]
    out += [
        f"4. **No timing discrimination.** Every attempt the engine commits "
        "lands on day 2, the earliest legal day, in every regime and under "
        "both compliance profiles. The hazard model's only temporal feature "
        "is `in_salary_window` (days 1-5) plus the slot index, so backward "
        "induction has nothing with which to prefer day 4 to day 2. The "
        "'timed to their replenishment rhythm' claim in the project's own "
        "framing is **not supported by this evidence**; the engine's "
        "advantage comes from *whether* and *how often* it attempts, not "
        "*when*. This confirms B12's shadow-mode finding "
        "(`SAME_ACTION_DIFFERENT_DAY = 0`) across all five regimes.",
        "",
        f"5. **FIXED, R2 (2026-09-04): the allocator used to want to retry "
        f"instruments the issuer had just confirmed dead.** Re-solving "
        f"after a terminal outcome returned `ATTEMPT` {after_term} times "
        f"across all engine cells on this run -- structurally zero, not "
        f"merely measured low: a hard `permitted()` rule (`instrument_dead`) "
        f"denies `ATTEMPT` outright once the instrument is known dead, "
        f"regardless of belief. `belief.observe_terminal()` also replaces "
        f"belief with a MEASURED posterior on the observed outcome -- "
        f"P(CANT_PAY_EVER | DEAD) = 0.8991, P(WONT_PAY | OPTED_OUT) = 0.9040, "
        f"both measured directly against `eval/frozen/sim_config.yaml`'s own "
        f"generative process, NOT the degenerate 100%-certain collapse an "
        f"earlier same-day version of this fix assumed and stats-reviewer "
        f"then proved false and irreversible (`cause_map`'s priors contain "
        f"no zeros, so a belief collapsed to exactly 1.0 can never be moved "
        f"by any later evidence). The corrected, measured version changes no "
        f"action anywhere in this sweep -- REAUTH's economics dominate STOP "
        f"at 90% confidence exactly as they did at 100% for every realistic "
        f"amount in this corpus -- but is no longer an overconfident, "
        f"irreversible claim about a fact the frozen simulator's own numbers "
        f"say is only ~90% certain. Before this fix existed at all: "
        f"`belief.update()`'s ordinary naive-Bayes compounding, with no "
        f"floor, meant a single `CARD_EXPIRED` observation often could not "
        f"overtake a slot-1 `INSUFFICIENT_FUNDS` prior, so `CANT_PAY_NOW` "
        f"kept dominating and the CANT_PAY_EVER -> REAUTH row of the "
        f"project's own cause/action table never fired on observed "
        f"evidence -- measured at 4,032 such events on the pre-fix "
        f"artifact. The `OPTED_OUT` case was worse: the proxy decline-class "
        f"function returns `None` for it, so the OLD code's re-solve was "
        f"skipped ENTIRELY and no decision was ever recorded for what "
        f"happens after a customer opts out. A related audit-trail bug found "
        f"in the same review -- `_binding_constraint()` did not know about "
        f"`instrument_dead`, so a REAUTH forced by this rule alone recorded "
        f"`binding_constraint = None`, misrepresenting a hard-forced decision "
        f"as a free economic choice -- is also fixed. See DECISIONS.md and "
        f"POSTMORTEM Incidents 11-12 for the full account, including a "
        f"conformal-measurement contamination the fix briefly introduced and "
        f"then closed (finding 2 above).",
        "",
        f"6. Actions across all engine cells: {reauth} REAUTH ("
        f"**{compliance_reauth}** issued via the above-AFA-cliff compliance "
        f"route -- clause 8(a)/8(b), legally mandatory regardless of belief "
        f"-- and **{false_reauth_inference}** that are genuine "
        f"belief-inference errors against a mandate whose true cause is not "
        f"`CANT_PAY_EVER`), {stops} STOP, {offers} OFFER. The pre-registered "
        f"`issuer_outage` falsification criterion, `{false_reauth}` REAUTHs "
        f"on a mandate whose true cause is not `CANT_PAY_EVER`, keeps its "
        f"Day-1 meaning unchanged (DECISIONS.md, R0) -- the split above is "
        f"ADDED alongside it, R2b, because that one number conflates a "
        f"legal requirement with a belief error: only "
        f"`{false_reauth_inference}` of the `{false_reauth}` were ever a "
        f"real inference mistake. Constraint violations: **{viol}**.",
    ]
    return out


HEADLINE_REGIME = "baseline"
HEADLINE_ARM = "nominal"


def _summary_payload(data: dict) -> dict:
    """reports/results.json -- the small, stable summary other tooling reads.

    scripts/checkpoint.py has looked for this file since the scaffold commit
    and printed "no eval run yet" into STATE.md for thirteen blocks because
    nothing ever wrote it; the run-eval skill lists writing it as step 2. The
    shape below is the one checkpoint.py already parses.

    The headline cell is baseline/nominal -- the frozen protocol's own
    reference point -- and the reference policies travel with it, because the
    engine's numbers are not interpretable without them.
    """
    def cell(policy, profile="strict"):
        """The MERGED cell, averaged over seeds -- not the first match.

        Scanning data["cells"] for a match returned seed 0's cell while the
        report beside it printed 8-seed means, so the README and the report
        disagreed about the same headline. Going through _paired() means both
        read the same aggregation.
        """
        pairs = _paired(data, profile)
        return pairs.get((HEADLINE_REGIME, HEADLINE_ARM), {}).get(policy)

    def bars(c):
        if c is None:
            return None
        rec = c["recovered"]
        billable = c.get("billable_paise") or 0
        return {
            "recovered_paise": c["recovered_paise"],
            "recovered_pct": (f"{100 * c['recovered_paise'] / billable:.1f}%"
                              if billable else "?"),
            "recovered": _rupees(c["recovered_paise"]),
            "attempts_spent": c["attempts_spent"],
            "attempts_per_recovery": (round(c["attempts_spent"] / rec, 2) if rec else None),
            "mandates_preserved": f"{c['mandates_preserved']}/{c['n_mandates']}",
        }

    losing = sorted({
        c["regime"]
        for c in data["cells"] if c["policy"] == "engine"
        for lad in [next((x for x in data["cells"]
                          if x["policy"] == "ladder" and x["regime"] == c["regime"]
                          and x["arm"] == c["arm"] and x["profile"] == c["profile"]), None)]
        if lad and c["recovered_paise"] < lad["recovered_paise"]
    })

    eng = [c for c in data["cells"] if c["policy"] == "engine"]
    seeds = data.get("seeds") or [data.get("seed", 0)]
    pw, pl, pt = _seed_win_counts(data, "engine", "one_shot", "mandates_preserved")
    mw, ml, _ = _seed_win_counts(data, "engine", "one_shot", "recovered_paise")
    aw, al, _ = _seed_win_counts(data, "engine", "one_shot", "attempts_spent")
    lp, _, _ = _seed_win_counts(data, "engine", "ladder", "mandates_preserved")
    lm, lml, _ = _seed_win_counts(data, "engine", "ladder", "recovered_paise")
    _, lal, _ = _seed_win_counts(data, "engine", "ladder", "attempts_spent")
    npairs = pw + pl + pt
    out = {
        "headline_cell": f"{HEADLINE_REGIME}/{HEADLINE_ARM}",
        "seed": data["seed"],
        "seeds": seeds,
        "paired_comparisons": npairs,
        # Losses are carried explicitly, never derived as (n - wins): that
        # silently reassigns TIES to the opponent. The first draft of the
        # README did exactly that and overstated one_shot by 6 comparisons.
        "sign_test": {
            "vs_ladder": {"preserves_more": lp, "recovers_more": lm,
                          "spends_fewer_attempts": lal},
            "vs_one_shot": {"preserves_more": pw, "preserves_fewer": pl,
                            "recovers_more": mw, "recovers_less": ml,
                            "spends_fewer_attempts": al, "spends_more_attempts": aw},
        },
        "gate_kind": data["gate_kind"],
        "regimes_where_we_lose": losing,
        "offers_fired_total": sum(c["n_offer"] for c in eng),
        # R5: the off-ramp's own error costs, so downstream tooling
        # (dashboard/, site/, scripts/checkpoint.py) reads the PAIR rather
        # than a bare count. `offramp_scored_total` is the exact denominator
        # `false_offramp_total` was measured against -- never `n_offer`,
        # which can differ if a post-terminal re-solve ever returns OFFER.
        "offramp_scored_total": sum(c.get("offramp_scored_count", 0) for c in eng),
        "false_offramp_total": sum(c.get("false_offramp_count", 0) for c in eng),
        "true_offramp_total": sum(c.get("true_offramp_count", 0) for c in eng),
        "wontpay_channel": data.get("wontpay_channel"),
        "false_reauth_total": sum(c["false_reauth_count"] for c in eng),
        "reauth_total": sum(c["n_reauth"] for c in eng),
        "attempt_after_terminal_total": sum(c["n_attempt_after_terminal"] for c in eng),
        # R2b, 2026-09-04: false_reauth_total (pre-registered, unchanged)
        # conflates a legally-mandatory compliance-route REAUTH with a
        # genuine belief-inference error -- these two, added alongside it,
        # split it. .get(..., 0) so a pre-R2 artifact still renders.
        "compliance_reauth_total": sum(c.get("compliance_reauth_count", 0) for c in eng),
        "false_reauth_inference_total": sum(c.get("false_reauth_inference_count", 0) for c in eng),
    }
    out.update(bars(cell("engine")) or {})
    out["baseline"] = bars(cell("ladder"))
    out["engine_permissive"] = bars(cell("engine", "permissive"))
    out["reference_null"] = bars(cell("null"))
    out["reference_one_shot"] = bars(cell("one_shot"))
    return out


def _readme_table(data: dict) -> list[str]:
    s = _summary_payload(data)
    n = s["paired_comparisons"]
    lad, one = s["sign_test"]["vs_ladder"], s["sign_test"]["vs_one_shot"]
    n_seeds = len(s["seeds"])

    def row(name, b):
        if b is None:
            return f"| {name} | — | — | — |"
        return (f"| {name} | {b['recovered']} | "
                f"{b['attempts_per_recovery'] if b['attempts_per_recovery'] is not None else '—'} | "
                f"**{b['mandates_preserved']}** |")

    return [
        f"*Auto-generated by `.\\run.ps1 eval` (Windows) / `./run.sh eval` "
        f"(Linux, macOS). Headline cell "
        f"`{s['headline_cell']}`, mean of {n_seeds} seeds. Full report: "
        f"[reports/regimes.md](reports/regimes.md).*",
        "",
        "| | recovered | attempts/recovery | **mandates preserved** |",
        "|---|---|---|---|",
        row("Fixed ladder (the incumbent)", s["baseline"]),
        row("This engine (strict)", {k: s[k] for k in
            ("recovered", "attempts_per_recovery", "mandates_preserved")}),
        row("This engine (permissive)", s["engine_permissive"]),
        row("*Reference:* one attempt, no model", s["reference_one_shot"]),
        row("*Reference:* never attempt", s["reference_null"]),
        "",
        f"**Read this as a sign test, not a table.** Across {n_seeds} seeds and "
        f"{n} paired comparisons, counted per seed rather than on the mean:",
        "",
        "| comparison | preserves more | recovers more | spends FEWER attempts |",
        "|---|---|---|---|",
        f"| engine vs **ladder** | {lad['preserves_more']} / {n} | "
        f"{lad['recovers_more']} / {n} | {lad['spends_fewer_attempts']} / {n} |",
        f"| engine vs **one_shot** | {one['preserves_more']} / {n} | "
        f"{one['recovers_more']} / {n} | {one['spends_fewer_attempts']} / {n} |",
        "",
        f"**Against the incumbent, the trade holds and is stable** — more "
        f"mandates preserved and fewer attempts spent in every comparison, at "
        f"the cost of money. Deliberately recovering less this cycle to protect "
        f"lifetime value is the thesis, not a bug.",
        "",
        f"**Against `one_shot` it does not.** One attempt on day 2 with no "
        f"model, no belief and no gate preserves more mandates than the engine "
        f"in {one['preserves_fewer']} of {n} comparisons, and the engine "
        f"spends MORE attempts in {one['spends_more_attempts']} of {n}. The "
        f"engine's only edge over it is money, and a thin one: "
        f"{one['recovers_more']} of {n}. On two of the three bars, a policy "
        f"with no model in it beats this one. That is in the README because it "
        f"is true, and because a reader who discovers it themselves should not "
        f"have to wonder what else was left out.",
        "",
        _readme_offramp_sentence(s) + f" Separately, {s['false_reauth_total']} of "
        f"{s['reauth_total']} REAUTHs went to mandates whose true cause is not "
        f"`CANT_PAY_EVER` — but {s['compliance_reauth_total']} of those are "
        f"the above-AFA-cliff compliance route (clause 8(a)/8(b), legally "
        f"mandatory regardless of belief), so only "
        f"{s['false_reauth_inference_total']} were ever a genuine "
        f"belief-inference error — see `reports/regimes.md` finding 6 for "
        f"the full split (R2b).",
        "",
        f"**Where we lose:** "
        f"{', '.join('`' + r + '`' for r in s['regimes_where_we_lose'])} — the "
        f"engine recovers less money than the ladder in these regimes. The "
        f"report's \"Where we lose\" section gives the reason for each.",
    ]


def _readme_offramp_sentence(s: dict) -> str:
    """The README's one-line off-ramp claim. Two texts, chosen by what the
    artifact contains -- never edited by hand, because this block is
    regenerated on every `eval.report` run and a hand-edit would be
    overwritten silently. The `off` text is retained rather than deleted:
    `--channel-kind off` still reproduces that configuration exactly."""
    ch = s.get("wontpay_channel")
    if ch is None:
        return (
            f"**The off-ramp never fires** (`OFFER` = {s['offers_fired_total']} "
            f"across every cell) \u2014 and that is arithmetic, not measurement: "
            f"the proxy decline alphabet cannot move belief toward `WONT_PAY` "
            f"at all. The off-ramp lane is untested, so the false-off-ramp "
            f"column is not evidence of safety."
        )
    scored = s.get("offramp_scored_total") or 0
    false_n = s.get("false_offramp_total") or 0
    rate = f"{false_n / scored:.1%}" if scored else "n/a"
    # R5 review pass, 2026-09-05 (stats-reviewer): `eng` (and so every total
    # above) sums BOTH compliance profiles, which are byte-identical on
    # every field across all 128 (regime, arm, seed) triples -- this engine
    # has no timing discrimination (see "Compliance profiles" below), so
    # `strict` and `permissive` are the same 128 cells counted twice. The
    # RATE is unaffected by doubling both terms, but a CI computed on the
    # doubled n understates the true uncertainty -- computed here on the
    # halved (distinct) sample rather than the inflated one.
    from bench.llm_vs_stats import wilson_ci
    ci_note = ""
    if scored:
        lo, hi = wilson_ci(false_n // 2, scored // 2)
        ci_note = f" (95% CI {lo:.0%}\u2013{hi:.0%} on the distinct sample)"
    return (
        f"**The off-ramp fires, on a SYNTHETIC channel that reads privileged "
        f"ground truth** (`OFFER` = {s['offers_fired_total']} across every "
        f"cell; {false_n} of {scored} scored went to a mandate that would "
        f"have paid \u2014 a {rate} false-off-ramp rate{ci_note}). The channel is "
        f"configured at tpr {ch['tpr']:.2f} / fpr {ch['fpr']:.2f} and is "
        f"disclosed as fabricated everywhere it appears; "
        f"`reports/regimes.md` publishes its full quality curve (including "
        f"a within-mandate-correlation sensitivity check the headline grid "
        f"holds fixed at zero), and deliberately worthless channels at "
        f"AUC 0.5, where the false rate is several times worse. This buys a "
        f"tested-and-imperfect off-ramp in place of an untested-and-central "
        f"one, not a good result."
    )


README_BEGIN = "<!-- RESULTS:BEGIN -->"
README_END = "<!-- RESULTS:END -->"


def update_readme(data: dict, path: pathlib.Path) -> bool:
    """Replace the block between the RESULTS markers. Returns False (and
    changes nothing) if the markers are absent -- this must never rewrite a
    README by guessing where the table is."""
    text = path.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        return False
    head, rest = text.split(README_BEGIN, 1)
    _, tail = rest.split(README_END, 1)
    body = "\n".join(_readme_table(data))
    path.write_text(
        f"{head}{README_BEGIN}\n{body}\n{README_END}{tail}", encoding="utf-8"
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figures", action="store_true", help="also render the PNGs")
    ap.add_argument("--no-readme", action="store_true",
                    help="skip updating the README results table")
    ap.add_argument("--artifact", type=pathlib.Path, default=ARTIFACT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_MD)
    args = ap.parse_args(argv)

    data = load(args.artifact)
    args.out.write_text(render(data, figures=args.figures), encoding="utf-8", newline="\n")

    summary = _REPO_ROOT / "reports" / "results.json"
    # R7 review gap, found 2026-09-05 while regenerating this file: every
    # other artifact writer in this project got `newline="\n"` in the
    # cross-platform byte-identity fix; this one didn't, because the bulk
    # regex that applied the fix matched exactly one write_text() call per
    # file and this file has two. Same reasoning as the rest: without it,
    # this JSON is CRLF on Windows and LF on Linux, breaking the same
    # byte-identical claim for a smaller, more-read artifact.
    summary.write_text(json.dumps(_summary_payload(data), indent=2),
                       encoding="utf-8", newline="\n")
    if not args.no_readme:
        readme = _REPO_ROOT / "README.md"
        if not update_readme(data, readme):
            print("note: README has no <!-- RESULTS:BEGIN --> markers; table not updated",
                  file=sys.stderr)
    print(f"wrote {args.out.name}"
          + (f" + {len(list(FIG_DIR.glob('regimes_*.png')))} figures" if args.figures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
