import type { Narrative } from '../hooks/useReportData'

/**
 * Shown when WebGL fails, or when the browser can only offer a software
 * context (Scene.tsx asks for `failIfMajorPerformanceCaveat`, so that case
 * throws rather than rendering at single-digit fps).
 *
 * Pure HTML -- no three.js, no Canvas element. Same figures as the scene it
 * replaces, all of them from `reports/results.json`.
 */
export function CanvasFallback({ narrative: n }: { narrative: Narrative }) {
  return (
    <div className="canvas-fallback">
      <div className="fallback-card">
        <h2 className="fallback-title">
          What the engine does, without the animation
        </h2>
        <p className="fallback-intro">
          This browser can&rsquo;t run the 3D scene. The argument doesn&rsquo;t
          depend on it.
        </p>

        <div className="fallback-flow">
          <div className="fallback-step">
            <div className="fallback-icon">◻</div>
            <h3>{n.total} mandates</h3>
            <p>
              All of them authorised the payment. All of them reach a debit date.
            </p>
          </div>
          <div className="fallback-arrow">→</div>
          <div className="fallback-step fallback-step--danger">
            <div className="fallback-icon">⚡</div>
            <h3>Fixed ladder</h3>
            <p>
              Retries on a schedule, halts after three.{' '}
              <strong>{n.ladderLost}</strong> of {n.total} not preserved, on{' '}
              {n.ladderAttempts} attempts.
            </p>
          </div>
          <div className="fallback-arrow">→</div>
          <div className="fallback-step fallback-step--engine">
            <div className="fallback-icon">◈</div>
            <h3>This engine</h3>
            <p>
              Asks which of three things went wrong.{' '}
              <strong>{n.engineLost}</strong> of {n.total} not preserved, on{' '}
              {n.engineAttempts} attempts.
            </p>
          </div>
        </div>

        <div className="fallback-bottom">
          <div className="fallback-stat fallback-stat--primary">
            <span className="fallback-stat-value">+{n.preservedDelta}</span>
            <span className="fallback-stat-label">
              mandates preserved vs ladder
            </span>
          </div>
          <div className="fallback-stat">
            <span className="fallback-stat-value">
              {n.engineRecoveredPct} vs {n.ladderRecoveredPct}
            </span>
            <span className="fallback-stat-label">
              recovered — the engine recovers less
            </span>
          </div>
          <div className="fallback-stat">
            <span className="fallback-stat-value">
              {n.engineAttempts} vs {n.ladderAttempts}
            </span>
            <span className="fallback-stat-label">attempts spent</span>
          </div>
        </div>
      </div>
    </div>
  )
}
