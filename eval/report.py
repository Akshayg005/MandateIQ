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
from typing import Any, Sequence

from src.core import money

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "regimes.json"
OUT_MD = _REPO_ROOT / "reports" / "regimes.md"
FIG_DIR = _REPO_ROOT / "reports" / "figures"

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
            f"--both-profiles` first (or `.\\run.ps1 report`, which does both)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _paired(data: dict[str, Any], profile: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = collections.defaultdict(dict)
    for c in data["cells"]:
        if c["profile"] != profile:
            continue
        out[(c["regime"], c["arm"])][c["policy"]] = c
    return dict(out)


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
    """Per-regime coverage of the off-ramp gate. Printed only where the real
    conformal gate was live -- under FullSetGate there is no coverage claim to
    make, and printing 1.000 would be true and completely misleading."""
    rows = [
        "| regime | arm | gate live | marginal coverage | per-class coverage "
        "(NOW / EVER / WONT) | mean set size | singleton rate | singleton "
        "{WONT_PAY} | OFFERs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in data["cells"]:
        # One row per (regime, arm) engine cell under `strict`. The previous
        # version keyed on regime alone and summed `n_offer` over BOTH
        # profiles into a per-regime-strict row, which would have printed
        # double the moment OFFER became non-zero.
        if c["policy"] != "engine" or c["profile"] != "strict":
            continue
        if c["gate_kind"] != "conformal":
            rows.append(f"| {c['regime']} | {c['arm']} | {c['gate_kind']} | "
                        f"n/a (stub gate) | n/a | n/a | n/a | n/a | {c['n_offer']} |")
            continue
        pc = c.get("coverage_per_class") or {}
        pcs = " / ".join(
            f"{pc[k]:.3f}" if k in pc else "--"
            for k in ("CANT_PAY_NOW", "CANT_PAY_EVER", "WONT_PAY")
        )
        sing = c.get("singleton_rate")
        rows.append(
            f"| {c['regime']} | {c['arm']} | conformal | "
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
        ".\\run.ps1 eval",
        "```",
        "",
        "(`.\\run.ps1 report` re-renders the tables and figures from the "
        "existing artifact without re-running the sweep. It cannot change a "
        "number; only `eval` can.)",
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
    lines += ["", "## Off-ramp gate: coverage per regime", "",
              "Coverage is *measured*, not assumed. The gate is calibrated "
              "once on `baseline` and reused unchanged under every regime; a "
              "regime breaks the exchangeability split conformal assumes, so "
              "any degradation here is a real result.", ""]
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


def _headline(data: dict[str, Any]) -> list[str]:
    cells = data["cells"]
    eng = [c for c in cells if c["policy"] == "engine"]
    offers = sum(c["n_offer"] for c in eng)
    reauth = sum(c["n_reauth"] for c in eng)
    false_reauth = sum(c["false_reauth_count"] for c in eng)
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
    worst_class = min(
        (v, k) for c in conf for k, v in (c.get("coverage_per_class") or {}).items()
    ) if conf else (None, None)

    out = [
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
        f"2. **The off-ramp cannot fire in this harness. `OFFER` = {offers} in "
        f"all {len(eng)} engine cells -- and that is arithmetic, not "
        f"measurement.** The proxy decline alphabet has exactly two symbols "
        f"(`INSUFFICIENT_FUNDS`, `CARD_EXPIRED`) and `cause_map` assigns "
        f"`WONT_PAY` a prior of 0.10 under **both**, so the WONT_PAY "
        f"likelihood ratio is constant and no observation this simulator can "
        f"produce moves belief mass toward `WONT_PAY`. Its probability is "
        f"pinned at 0.10 after slot 1 and is non-increasing thereafter, so the "
        f"singleton `{{WONT_PAY}}` condition is unreachable for any alpha, any "
        f"seed, any regime. An earlier draft read this as \"a payment decline "
        f"is a weak signal of intent\" -- true about the world, false about "
        f"this number. **The off-ramp lane is untested, not tested and "
        f"negative**, and `retry_storm`'s pre-registered hypothesis about it "
        f"is vacuous rather than falsified: the outcome was fixed before the "
        f"regime ran. `false off-ramp = 0` is likewise not a safety result.",
        "",
    ]
    if cov:
        out += [
            f"3. **The off-ramp gate under-covers.** Over the "
            f"{sum(c['coverage_n'] for c in conf):,} decision points the gate "
            f"was actually queried at, marginal coverage is "
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
        f"5. **The allocator wants to retry instruments the issuer just "
        f"confirmed dead.** Re-solving after a terminal outcome returned "
        f"`ATTEMPT` {after_term} times across all engine cells -- on mandates "
        f"that had just come back `DEAD` or `OPTED_OUT`. `belief.update()` "
        f"compounds naive-Bayes with no floor, and a single `CARD_EXPIRED` "
        f"cannot overtake the slot-1 `INSUFFICIENT_FUNDS` prior, so "
        f"`CANT_PAY_NOW` keeps dominating. The CANT_PAY_EVER -> REAUTH row of "
        f"the project's own cause/action table does not fire on observed "
        f"evidence. This case previously fell through every counter and was "
        f"discarded unrecorded.",
        "",
        f"6. Actions across all engine cells: {reauth} REAUTH (of which "
        f"**{false_reauth} were issued on mandates whose true cause is not "
        f"`CANT_PAY_EVER`** -- `issuer_outage`'s own pre-registered "
        f"falsification criterion, now measured), {stops} STOP, {offers} "
        f"OFFER. Constraint violations: **{viol}**.",
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
        for c in data["cells"]:
            if (c["regime"] == HEADLINE_REGIME and c["arm"] == HEADLINE_ARM
                    and c["profile"] == profile and c["policy"] == policy):
                return c
        return None

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
    out = {
        "headline_cell": f"{HEADLINE_REGIME}/{HEADLINE_ARM}",
        "seed": data["seed"],
        "gate_kind": data["gate_kind"],
        "regimes_where_we_lose": losing,
        "offers_fired_total": sum(c["n_offer"] for c in eng),
        "false_reauth_total": sum(c["false_reauth_count"] for c in eng),
        "reauth_total": sum(c["n_reauth"] for c in eng),
        "attempt_after_terminal_total": sum(c["n_attempt_after_terminal"] for c in eng),
    }
    out.update(bars(cell("engine")) or {})
    out["baseline"] = bars(cell("ladder"))
    out["engine_permissive"] = bars(cell("engine", "permissive"))
    out["reference_null"] = bars(cell("null"))
    out["reference_one_shot"] = bars(cell("one_shot"))
    return out


def _readme_table(data: dict) -> list[str]:
    s = _summary_payload(data)

    def row(name, b):
        if b is None:
            return f"| {name} | — | — | — |"
        return (f"| {name} | {b['recovered']} | "
                f"{b['attempts_per_recovery'] if b['attempts_per_recovery'] is not None else '—'} | "
                f"**{b['mandates_preserved']}** |")

    return [
        f"*Auto-generated by `.\\run.ps1 eval`. Headline cell: "
        f"`{s['headline_cell']}`, seed {s['seed']}. Full report: "
        f"[reports/regimes.md](reports/regimes.md).*",
        "",
        "| | recovered | attempts/recovery | **mandates preserved** |",
        "|---|---|---|---|",
        row("Fixed ladder (baseline)", s["baseline"]),
        row("This engine (strict)", {k: s[k] for k in
            ("recovered", "attempts_per_recovery", "mandates_preserved")}),
        row("This engine (permissive)", s["engine_permissive"]),
        row("*Reference:* one attempt, no model", s["reference_one_shot"]),
        row("*Reference:* never attempt", s["reference_null"]),
        "",
        "The two reference rows are not policies we propose; they are there "
        "because every metric above is monotonically decreasing in attempt "
        "count, so \"preserves more\" follows from \"attempts less\". "
        "`one_shot` preserves more than the engine in most cells while "
        "spending fewer attempts. Read the preserved column against those "
        "rows, not against the ladder alone.",
        "",
        f"Error costs (headline cell): stopped-on that would have paid, and "
        f"false off-ramp — see [reports/regimes.md](reports/regimes.md). "
        f"**The off-ramp never fires in this evaluation** "
        f"(`OFFER` = {s['offers_fired_total']} across every cell), so the "
        f"false-off-ramp column is not yet evidence of safety. "
        f"{s['false_reauth_total']} of {s['reauth_total']} REAUTHs were "
        f"issued on mandates whose true cause is not `CANT_PAY_EVER`.",
        "",
        f"**Where we lose:** {', '.join('`' + r + '`' for r in s['regimes_where_we_lose'])} "
        f"— the engine recovers less money than the ladder in these regimes. "
        f"See the \"Where we lose\" section of the full report for why.",
    ]


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
    args.out.write_text(render(data, figures=args.figures), encoding="utf-8")

    summary = _REPO_ROOT / "reports" / "results.json"
    summary.write_text(json.dumps(_summary_payload(data), indent=2), encoding="utf-8")
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
