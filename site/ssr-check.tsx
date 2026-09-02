/**
 * Renders every non-canvas part of the landing page against the REAL staged
 * results.json and asserts that B15's gate holds.
 *
 *     npm run render-check
 *
 * Why this exists. B15's gate says the counters are "wired to real report
 * output, not hard-coded". `tsc` proves the props type-check and `vite build`
 * proves it bundles; neither can tell a figure read from results.json apart
 * from a number typed into a component -- and a previous draft of this page
 * did exactly that, carrying PLAN.md's storyboard placeholders ("ladder kills
 * 14") into shipped copy, wrong by a factor of six.
 *
 * So this check works from both ends: it asserts the REAL figures appear, and
 * it asserts the KNOWN-FABRICATED ones do not. The second half is the part
 * that would have caught the original bug.
 *
 * The Canvas cannot be server-rendered -- there is no WebGL in node -- but
 * every figure it displays is drawn from the same `deriveNarrative()` output
 * that the fallbacks and counters use, and both fallbacks ARE rendered here.
 */
import { renderToString } from "react-dom/server";
import fs from "node:fs";
import { CountersSection } from "./src/components/Counters";
import { ResultsSection } from "./src/components/ResultsSection";
import { CanvasFallback } from "./src/components/CanvasFallback";
import { ReducedMotionFallback } from "./src/components/ReducedMotionFallback";
import { deriveNarrative, type ReportData } from "./src/hooks/useReportData";

const R: ReportData = JSON.parse(
  fs.readFileSync("public/data/results.json", "utf8"),
);
const n = deriveNarrative(R);

const counters = renderToString(
  <CountersSection id="counters" narrative={n} data={R} />,
);

const results = renderToString(
  <ResultsSection
    id="results"
    recoveredPct={R.recovered_pct}
    recovered={R.recovered}
    attemptsPerRecovery={R.attempts_per_recovery}
    mandatesPreserved={R.mandates_preserved}
    ladderRecoveredPct={R.baseline.recovered_pct}
    ladderRecovered={R.baseline.recovered}
    ladderAttemptsPerRecovery={R.baseline.attempts_per_recovery}
    ladderMandatesPreserved={R.baseline.mandates_preserved}
    oneShotRecoveredPct={R.reference_one_shot.recovered_pct}
    oneShotRecovered={R.reference_one_shot.recovered}
    oneShotMandatesPreserved={R.reference_one_shot.mandates_preserved}
    signTestPreservesMore={R.sign_test.vs_ladder.preserves_more}
    signTestTotal={R.paired_comparisons}
    signTestRecoverMore={R.sign_test.vs_ladder.recovers_more}
    signTestSpendsFewerAttempts={R.sign_test.vs_ladder.spends_fewer_attempts}
    seedCount={n.seedCount}
  />,
);

const canvasFallback = renderToString(<CanvasFallback narrative={n} />);
const reducedMotion = renderToString(<ReducedMotionFallback narrative={n} />);

let failed = 0;
const need = (hay: string, s: string, label: string) => {
  if (!hay.includes(s)) {
    console.log(`MISSING in ${label}: ${s}`);
    failed++;
  }
};
const forbid = (hay: string, s: string, label: string) => {
  if (hay.includes(s)) {
    console.log(`FABRICATED FIGURE in ${label}: ${s}`);
    failed++;
  }
};

// --- the real figures must reach the output ---------------------------------
// Bar labels are plain text; the headline stats animate from 0 and so carry
// their figure in data-target/aria-label instead -- see AnimatedStat.
need(counters, R.mandates_preserved, "counters");
need(counters, R.baseline.mandates_preserved, "counters");
need(counters, `data-target="+${n.preservedDelta}"`, "counters");
need(counters, `data-target="−${n.attemptsSaved}"`, "counters");
// The engine loses the money bar, and the page must say so rather than
// quietly drawing only the bars it wins.
need(counters, "loses this bar", "counters");
need(counters, String(n.engineAttempts), "counters");
need(counters, String(n.ladderAttempts), "counters");
need(counters, R.recovered_pct, "counters");
need(counters, R.baseline.recovered_pct, "counters");
need(results, R.mandates_preserved, "results");
need(results, R.baseline.mandates_preserved, "results");
need(results, R.reference_one_shot.mandates_preserved, "results");
need(results, R.recovered, "results");
need(results, R.baseline.recovered, "results");

for (const [label, html] of [
  ["canvas-fallback", canvasFallback],
  ["reduced-motion", reducedMotion],
] as const) {
  need(html, String(n.ladderLost), label);
  need(html, String(n.engineLost), label);
  need(html, R.recovered_pct, label);
  need(html, R.baseline.recovered_pct, label);
  // Both fallbacks must state the trade, not just the win.
  need(html, String(n.ladderAttempts), label);
}

// --- limitations belong in the repo, not on the page ------------------------
// The "What this page is not showing you" block was moved to README.md's
// "What this can't do". This asserts it stays moved: the landing page sells
// what the engine does, and README is where the build is honest about what it
// lacks. If a caveat creeps back into the page, this fails.
for (const [label, html] of [
  ["counters", counters],
  ["results", results],
  ["canvas-fallback", canvasFallback],
  ["reduced-motion", reducedMotion],
] as const) {
  forbid(html, "What this page is not showing you", label);
  forbid(html, "untested", label);
  forbid(html, "Buildathon", label);
  forbid(html, "Track 03", label);
}

// --- the placeholders must NOT ----------------------------------------------
// PLAN.md's storyboard numbers, written months before B13 produced a result.
// If any of these reappears, someone has typed a number into a component.
for (const [label, html] of [
  ["counters", counters],
  ["results", results],
  ["canvas-fallback", canvasFallback],
  ["reduced-motion", reducedMotion],
] as const) {
  forbid(html, "14 mandates lost", label);
  forbid(html, "~160", label);
  forbid(html, "40 are declined", label);
  forbid(html, "ladder: 26", label);
  forbid(html, "ours: 38", label);
}

// A guard against the arithmetic silently inverting: the engine preserves
// MORE and recovers LESS. If that ever flips, the page's whole argument is
// backwards and every caption above is wrong.
if (!(n.enginePreserved > n.ladderPreserved)) {
  console.log("ASSERTION: engine no longer preserves more than the ladder");
  failed++;
}
if (n.preservedDelta !== n.enginePreserved - n.ladderPreserved) {
  console.log("ASSERTION: preservedDelta disagrees with its own operands");
  failed++;
}

if (failed) {
  process.exitCode = 1;
  console.log(`render-check FAILED with ${failed} problem(s)`);
} else {
  console.log(
    `rendered: counters ${counters.length} chars, results ${results.length} chars, ` +
      `fallbacks ${canvasFallback.length}+${reducedMotion.length} chars — ` +
      `engine ${n.enginePreserved}/${n.total} vs ladder ${n.ladderPreserved}/${n.total}, ` +
      `recovered ${n.engineRecoveredPct} vs ${n.ladderRecoveredPct}, offers fired ${n.offersFired}`,
  );
}
