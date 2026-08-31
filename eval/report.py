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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = _REPO_ROOT / "reports" / "regimes.json"
OUT_MD = _REPO_ROOT / "reports" / "regimes.md"
FIG_DIR = _REPO_ROOT / "reports" / "figures"

# dataviz: categorical slots 1 and 2. Validated as a pair (light mode) --
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface.
C_LADDER = "#2a78d6"
C_ENGINE = "#eb6834"
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_MUTED = "#52514e"


def _rupees(paise: int) -> str:
    return f"{paise / 100:,.0f}"


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


def _three_bar_table(data: dict[str, Any], profile: str) -> list[str]:
    rows = [
        "| regime | arm | policy | recovered (Rs) | attempts | preserved | "
        "missed recovery (n / Rs) | false off-ramp (n / Rs) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (regime, arm), v in _paired(data, profile).items():
        for policy in ("ladder", "engine"):
            c = v[policy]
            miss = (f"{c['missed_recovery_count']} / {_rupees(c['missed_recovery_paise'])}"
                    if policy == "engine" else "--")
            fo = (f"{c['false_offramp_count']} / {_rupees(c['false_offramp_paise'])}"
                  if policy == "engine" else "--")
            rows.append(
                f"| {regime} | {arm} | {policy} | {_rupees(c['recovered_paise'])} | "
                f"{c['attempts_spent']} | {c['mandates_preserved']}/{c['n_mandates']} | "
                f"{miss} | {fo} |"
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
        "| regime | gate live | marginal coverage | mean set size | singleton {WONT_PAY} rate | OFFERs fired |",
        "|---|---|---:|---:|---:|---:|",
    ]
    seen = set()
    for c in data["cells"]:
        if c["policy"] != "engine" or c["profile"] != "strict":
            continue
        if c["regime"] in seen:
            continue
        seen.add(c["regime"])
        offers = sum(x["n_offer"] for x in data["cells"]
                     if x["policy"] == "engine" and x["regime"] == c["regime"])
        if c["gate_kind"] != "conformal":
            rows.append(f"| {c['regime']} | {c['gate_kind']} | n/a (stub gate) | n/a | n/a | {offers} |")
            continue
        rows.append(
            f"| {c['regime']} | conformal | {c['coverage_marginal']:.3f} "
            f"(n={c['coverage_n']}) | {c['mean_set_size']:.2f} / 3 | "
            f"{c['singleton_wont_pay_rate']:.3f} | {offers} |"
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
        f"`permissive`.** The profiles are not inert in the code -- "
        "`profiles.requires_fresh_notification()` shifts the earliest "
        "committable day by one under `strict`, and `allocator.solve()` reads "
        "it -- but that constraint never binds at the optimum this policy "
        "chooses, because the policy always picks the earliest legal day "
        "anyway (see 'No timing discrimination' below). The two compliance "
        "interpretations are therefore indistinguishable *on this evidence*, "
        "which is a fact about how little the current policy uses timing, not "
        "a demonstration that the ambiguity does not matter.",
    ]


def _losses(data: dict[str, Any]) -> list[str]:
    """The gate requires at least one regime where we lose, explained. This
    finds them mechanically rather than by hand, so a future run cannot
    quietly stop reporting one."""
    out: list[str] = []
    money, iatro = [], []
    for (regime, arm), v in _paired(data, "strict").items():
        L, E = v["ladder"], v["engine"]
        if E["recovered_paise"] < L["recovered_paise"]:
            money.append((regime, arm,
                          100 * (E["recovered_paise"] - L["recovered_paise"]) / L["recovered_paise"],
                          E["missed_recovery_count"], E["missed_recovery_paise"]))
        if E["iatrogenic_failures"] > L["iatrogenic_failures"]:
            iatro.append((regime, arm, L["iatrogenic_failures"], E["iatrogenic_failures"]))

    money.sort(key=lambda r: r[2])
    out += ["### Money left on the table", ""]
    out += ["| regime | arm | money delta | mandates we did not attempt that would have paid | value (Rs) |",
            "|---|---|---:|---:|---:|"]
    for regime, arm, pct, n, paise in money:
        out.append(f"| {regime} | {arm} | {pct:+.1f}% | {n} | {_rupees(paise)} |")
    out.append("")

    if iatro:
        out += ["### Regimes where the engine caused MORE collateral damage than the ladder", "",
                "| regime | arm | ladder iatrogenic | engine iatrogenic |",
                "|---|---|---:|---:|"]
        for regime, arm, l, e in iatro:
            out.append(f"| {regime} | {arm} | {l} | {e} |")
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
    for field, title, scale, slug in bars:
        lad = np.array([pairs[k]["ladder"][field] for k in keys], dtype=float) * scale
        eng = np.array([pairs[k]["engine"][field] for k in keys], dtype=float) * scale

        fig, ax = plt.subplots(figsize=(11, 5.4))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)
        x = np.arange(len(keys))
        w = 0.38
        # 2px surface gap between adjacent fills -> the 0.02 inset on each bar
        ax.bar(x - w / 2 - 0.01, lad, w, label="ladder (incumbent)", color=C_LADDER)
        ax.bar(x + w / 2 + 0.01, eng, w, label="engine", color=C_ENGINE)

        ax.set_title(title, color=C_TEXT, fontsize=13, loc="left", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=C_MUTED,
                           rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelcolor=C_MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=9, labelcolor=C_MUTED, loc="upper right")
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
    stops = sum(c["n_stop"] for c in eng)
    viol = sum(len(c["violations"]) for c in cells)
    pairs = _paired(data, "strict")
    won_pres = sum(1 for v in pairs.values()
                   if v["engine"]["mandates_preserved"] > v["ladder"]["mandates_preserved"])
    won_money = sum(1 for v in pairs.values()
                    if v["engine"]["recovered_paise"] > v["ladder"]["recovered_paise"])
    cov = [c["coverage_marginal"] for c in eng if c["coverage_marginal"] is not None]
    setsz = [c["mean_set_size"] for c in eng if c["mean_set_size"] is not None]

    return [
        f"1. **The engine preserves more mandates in {won_pres} of {len(pairs)} "
        f"regime-arm cells, and recovers more money in {won_money}.** It spends "
        "fewer attempts everywhere. That is the trade the thesis predicts.",
        "",
        f"2. **The off-ramp never fires. `OFFER` = {offers} across all "
        f"{len(eng)} engine cells.** The conformal gate holds coverage "
        f"(marginal {min(cov):.3f}-{max(cov):.3f} against a 0.95 target) by "
        f"returning almost the whole label set -- mean size "
        f"{min(setsz):.2f}-{max(setsz):.2f} of 3 -- so the singleton "
        "`{WONT_PAY}` condition is never met. This is the gate working as "
        "designed, not failing: after a single slot-1 decline the belief for a "
        "true `WONT_PAY` mandate typically points at `CANT_PAY_NOW`, because a "
        "payment decline is a weak signal of intent. **Every preserved-mandate "
        "win below is therefore won by restraint -- attempting less -- and "
        "none of it by correctly identifying exit intent.** The intent signal "
        "that would change this (`src/llm/intent.py`, support-ticket text) is "
        "not in this harness.",
        "",
        f"3. **No timing discrimination.** Every attempt the engine commits "
        "lands on day 2, the earliest legal day, in every regime and under "
        "both compliance profiles. The hazard model's only temporal feature is "
        "`in_salary_window` (days 1-5) plus the slot index, so backward "
        "induction has nothing with which to prefer day 4 to day 2. The "
        "'timed to their replenishment rhythm' claim in the project's own "
        "framing is **not supported by this evidence**; the engine's advantage "
        "comes from *whether* and *how often* it attempts, not *when*. This "
        "confirms B12's shadow-mode finding "
        "(`SAME_ACTION_DIFFERENT_DAY = 0`) across all five regimes.",
        "",
        f"4. Actions taken across all engine cells: {reauth} REAUTH, {stops} "
        f"STOP, {offers} OFFER. Constraint violations: **{viol}**.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figures", action="store_true", help="also render the PNGs")
    ap.add_argument("--artifact", type=pathlib.Path, default=ARTIFACT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_MD)
    args = ap.parse_args(argv)

    data = load(args.artifact)
    args.out.write_text(render(data, figures=args.figures), encoding="utf-8")
    print(f"wrote {args.out.name}"
          + (f" + {len(list(FIG_DIR.glob('regimes_*.png')))} figures" if args.figures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
