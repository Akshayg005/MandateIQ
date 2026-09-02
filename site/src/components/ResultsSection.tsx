interface ResultsSectionProps {
  id: string
  recoveredPct: string
  recovered: string
  attemptsPerRecovery: number
  mandatesPreserved: string
  ladderRecoveredPct: string
  ladderRecovered: string
  ladderAttemptsPerRecovery: number
  ladderMandatesPreserved: string
  oneShotRecoveredPct: string
  oneShotRecovered: string
  oneShotMandatesPreserved: string
  signTestPreservesMore: number
  signTestTotal: number
  signTestRecoverMore: number
  signTestSpendsFewerAttempts: number
  offersFired: number
  seedCount: number
}

export function ResultsSection({
  id,
  recoveredPct,
  recovered,
  attemptsPerRecovery,
  mandatesPreserved,
  ladderRecoveredPct,
  ladderRecovered,
  ladderAttemptsPerRecovery,
  ladderMandatesPreserved,
  oneShotRecoveredPct,
  oneShotRecovered,
  oneShotMandatesPreserved,
  signTestPreservesMore,
  signTestTotal,
  signTestRecoverMore,
  signTestSpendsFewerAttempts,
  offersFired,
  seedCount,
}: ResultsSectionProps) {
  return (
    <section id={id} className="results-section">
      <h2 className="section-label">Full Results</h2>
      <p className="section-subtitle">
        All numbers from{' '}
        <code>reports/results.json</code> — reproducible via{' '}
        <code>.\run.ps1 eval</code>
      </p>

      <div className="results-table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th></th>
              <th>Recovered</th>
              <th>Attempts / Rec</th>
              <th className="col-highlight">Mandates Preserved</th>
            </tr>
          </thead>
          <tbody>
            <tr className="row-ladder">
              <td className="row-label">
                <span className="dot dot--ladder"></span>
                Fixed Ladder (incumbent)
              </td>
              <td>
                {ladderRecovered}{' '}
                <span className="pct">({ladderRecoveredPct})</span>
              </td>
              <td>{ladderAttemptsPerRecovery.toFixed(2)}</td>
              <td className="col-highlight">{ladderMandatesPreserved}</td>
            </tr>
            <tr className="row-engine">
              <td className="row-label">
                <span className="dot dot--engine"></span>
                This Engine
              </td>
              <td>
                {recovered}{' '}
                <span className="pct">({recoveredPct})</span>
              </td>
              <td>{attemptsPerRecovery.toFixed(2)}</td>
              <td className="col-highlight font-bold">{mandatesPreserved}</td>
            </tr>
            <tr className="row-ref">
              <td className="row-label">
                <span className="dot dot--ref"></span>
                One attempt, no model
              </td>
              <td>
                {oneShotRecovered}{' '}
                <span className="pct">({oneShotRecoveredPct})</span>
              </td>
              <td>—</td>
              <td className="col-highlight">{oneShotMandatesPreserved}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="sign-test-summary">
        <h3>
          Sign Test ({seedCount} seeds, {signTestTotal} paired comparisons)
        </h3>
        <p>
          Engine vs Ladder:{' '}
          <strong>
            preserves more in {signTestPreservesMore}/{signTestTotal}
          </strong>
          , recovers more in {signTestRecoverMore}/{signTestTotal}, spends fewer
          attempts in {signTestSpendsFewerAttempts}/{signTestTotal}.
        </p>
        <p className="sign-test-note">
          Deliberately recovering less this cycle to protect lifetime value is
          the thesis, not a bug.
        </p>
      </div>

      {/* The same disclosure the reviewer dashboard leads with. The scroll
          narrative above deliberately does not dramatise the off-ramp, because
          in every published run it never fired. */}
      <div className="honesty-note">
        <h3>What this page is not showing you</h3>
        <ul>
          <li>
            <code>OFFER</code> — the off-ramp, and the reason this system exists
            — fired <strong>{offersFired} times</strong>. The belief layer pins
            P(WONT_PAY) at 0.10, so the <code>{'{WONT_PAY}'}</code> singleton the
            conformal gate requires is unreachable for any alpha, seed or regime.
            The off-ramp lane is <strong>untested</strong>, not
            tested-and-negative. The animation above shows the two policies that
            did run, and no amber &ldquo;customer parks intact&rdquo; stream,
            because no such event occurred.
          </li>
          <li>
            Every figure here is a mean over {seedCount} seeds of the
            baseline/nominal cell. Per-mandate detail lives in the reviewer
            dashboard, which reads the seed-0 batch — the two must not be
            compared directly.
          </li>
          <li>
            Against <em>one attempt, no model</em> the engine does not win on
            preservation. That row is in the table above on purpose.
          </li>
        </ul>
      </div>
    </section>
  )
}
