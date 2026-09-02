import type { Narrative } from '../hooks/useReportData'

/**
 * Static three-frame storyboard shown when prefers-reduced-motion: reduce.
 * No animation, no Canvas -- pure HTML + CSS.
 *
 * It tells the SAME story as the animated scene, with the same figures from
 * `reports/results.json`. A fallback that quietly says something different
 * from the thing it replaces is worse than no fallback.
 */
export function ReducedMotionFallback({ narrative: n }: { narrative: Narrative }) {
  const dots = (kind: 'none' | 'ladder' | 'engine') =>
    Array.from({ length: n.total }, (_, i) => {
      let cls = 'mini-dot--grey'
      if (kind === 'ladder') {
        cls = i < n.ladderLost ? 'mini-dot--dead' : 'mini-dot--green'
      } else if (kind === 'engine') {
        if (i < n.engineLost) cls = 'mini-dot--dead'
        else if (i < n.ladderLost) cls = 'mini-dot--teal'
        else cls = 'mini-dot--green'
      }
      return <div key={i} className={`mini-dot ${cls}`} />
    })

  return (
    <div className="reduced-motion-fallback">
      <div className="storyboard">
        <div className="storyboard-frame">
          <div className="frame-number">1</div>
          <div className="frame-visual frame-approaching">
            <div className="mini-grid">{dots('none')}</div>
          </div>
          <div className="frame-caption">
            <strong>{n.total} mandates approach the debit date</strong>
            <br />
            Every one of them authorised this payment. Some will fail.
          </div>
        </div>

        <div className="storyboard-frame frame--danger">
          <div className="frame-number">2</div>
          <div className="frame-visual frame-ladder">
            <div className="mini-grid">{dots('ladder')}</div>
          </div>
          <div className="frame-caption">
            <strong>
              The fixed ladder: {n.ladderLost} of {n.total} not preserved
            </strong>
            <br />
            Retry on a schedule, halt after three. Recovered{' '}
            {n.ladderRecoveredPct} on {n.ladderAttempts} attempts, and never
            asked why any of them failed.
          </div>
        </div>

        <div className="storyboard-frame frame--engine">
          <div className="frame-number">3</div>
          <div className="frame-visual frame-engine">
            <div className="mini-grid">{dots('engine')}</div>
          </div>
          <div className="frame-caption">
            <strong>
              The engine: {n.engineLost} of {n.total} not preserved
            </strong>
            <br />
            {n.preservedDelta} more kept (teal) on {n.engineAttempts} attempts,
            and it recovered {n.engineRecoveredPct}, less than the ladder&rsquo;s{' '}
            {n.ladderRecoveredPct}. That is the trade.
          </div>
        </div>
      </div>
    </div>
  )
}
