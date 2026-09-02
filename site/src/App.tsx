import { Suspense, lazy } from 'react'
import { HeroSection } from './components/HeroSection'
import { CountersSection } from './components/Counters'
import { ResultsSection } from './components/ResultsSection'
import { Footer } from './components/Footer'
import { CanvasErrorBoundary } from './components/CanvasErrorBoundary'
import { CanvasFallback } from './components/CanvasFallback'
import { ReducedMotionFallback } from './components/ReducedMotionFallback'
import { useReportData } from './hooks/useReportData'
import { useReducedMotion } from './hooks/useReducedMotion'
import { useWebGLSupport } from './hooks/useWebGLSupport'
import './App.css'

// Lazy-load the three.js bundle -- by far the largest dependency, and the
// page's argument survives without it (see CanvasFallback).
const Scene = lazy(() =>
  import('./components/Scene').then((m) => ({ default: m.Scene })),
)

function LoadingSpinner() {
  return (
    <div className="scene-loading">
      <div className="spinner" />
      <p>Loading 3D scene…</p>
    </div>
  )
}

function App() {
  const report = useReportData()
  const prefersReducedMotion = useReducedMotion()
  const webgl = useWebGLSupport()

  // Nothing on this page invents a number, so nothing on this page renders
  // before the report is in hand. The loading and error states are explicit
  // rather than an empty page that looks intentional.
  if (report.status === 'loading') {
    return (
      <div className="app-root">
        <HeroSection id="hero" />
        <div className="report-status">Loading results…</div>
        <Footer />
      </div>
    )
  }

  if (report.status === 'error') {
    return (
      <div className="app-root">
        <HeroSection id="hero" />
        <div className="report-status report-status--error">
          <p>
            <strong>Could not load results.json</strong> — {report.message}
          </p>
          <p>
            Every figure on this page is read from <code>reports/results.json</code>
            , staged by <code>python scripts/dashboard_data.py site</code>. Rather
            than show placeholder numbers, it shows nothing. Run{' '}
            <code>.\run.ps1 site</code> to regenerate.
          </p>
        </div>
        <Footer />
      </div>
    )
  }

  const { data, narrative } = report

  return (
    <div className="app-root">
      <HeroSection id="hero" />

      {/* Three ways this renders, decided before the canvas is mounted:
          reduced motion gets the static storyboard; a machine without a
          usable GPU gets the HTML fallback (the probe, not the boundary,
          catches that -- see useWebGLSupport); everything else gets the
          scene, with the boundary still watching for a later failure. */}
      <div className="narrative-section" id="narrative">
        {prefersReducedMotion ? (
          <ReducedMotionFallback narrative={narrative} />
        ) : webgl === 'unavailable' ? (
          <CanvasFallback narrative={narrative} />
        ) : webgl === 'probing' ? (
          <LoadingSpinner />
        ) : (
          <CanvasErrorBoundary
            fallback={<CanvasFallback narrative={narrative} />}
          >
            <Suspense fallback={<LoadingSpinner />}>
              <Scene narrative={narrative} />
            </Suspense>
          </CanvasErrorBoundary>
        )}
      </div>

      <CountersSection id="counters" narrative={narrative} data={data} />

      <ResultsSection
        id="results"
        recoveredPct={data.recovered_pct}
        recovered={data.recovered}
        attemptsPerRecovery={data.attempts_per_recovery}
        mandatesPreserved={data.mandates_preserved}
        ladderRecoveredPct={data.baseline.recovered_pct}
        ladderRecovered={data.baseline.recovered}
        ladderAttemptsPerRecovery={data.baseline.attempts_per_recovery}
        ladderMandatesPreserved={data.baseline.mandates_preserved}
        oneShotRecoveredPct={data.reference_one_shot.recovered_pct}
        oneShotRecovered={data.reference_one_shot.recovered}
        oneShotMandatesPreserved={data.reference_one_shot.mandates_preserved}
        signTestPreservesMore={data.sign_test.vs_ladder.preserves_more}
        signTestTotal={data.paired_comparisons}
        signTestRecoverMore={data.sign_test.vs_ladder.recovers_more}
        signTestSpendsFewerAttempts={
          data.sign_test.vs_ladder.spends_fewer_attempts
        }
        offersFired={narrative.offersFired}
        seedCount={narrative.seedCount}
      />

      <Footer />
    </div>
  )
}

export default App
