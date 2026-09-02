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
  seedCount,
}: ResultsSectionProps) {
  return (
    <section id={id} className="results-section">
      <h2 className="section-label">Full Results</h2>
      <p className="section-subtitle">
        All numbers from{' '}
        <code>reports/results.json</code>, reproducible via{' '}
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
              <td>n/a</td>
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

      {/* The limitations that used to sit here now live in README.md's
          "What this can't do". The page shows what the engine does; the
          repository is where the build is honest about what it lacks. */}
    </section>
  )
}
