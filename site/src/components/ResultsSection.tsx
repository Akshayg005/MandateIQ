import {
  AttemptsPerRecovery,
  FixedLadder,
  OneShot,
  Preserved,
  Recovered,
  Seeds,
  SignTest,
  Synthetic,
} from '../glossary'

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
        Every number here comes from a saved run over <Synthetic />, and the
        whole run can be reproduced from the repository with one command.
        Hover or tap any underlined term for what it means.
      </p>

      <div className="results-table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th></th>
              <th>Money collected</th>
              <th>Attempts per recovery</th>
              <th className="col-highlight">Customers still subscribed</th>
            </tr>
          </thead>
          <tbody>
            <tr className="row-ladder">
              <td className="row-label">
                <span className="dot dot--ladder"></span>
                Fixed ladder — what Razorpay does today
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

      {/* Definitions live BELOW the table, not inside its cells. The table
          wrapper is an overflow-x scroller, and per the CSS overflow spec a
          container that clips one axis clips the other too -- so a popover
          opened from a <th> is cut off at the table's edge, which is exactly
          what happened when these were tried in the header row. Out here they
          have the whole page to open into, and the table keeps short headings
          that fit without a horizontal scrollbar. */}
      <p className="results-legend">
        What these mean: <Recovered />, <AttemptsPerRecovery />, customers{' '}
        <Preserved /> · the two comparisons are the <FixedLadder /> and{' '}
        <OneShot />.
      </p>

      <div className="sign-test-summary">
        <h3>
          Head to head — {seedCount} <Seeds />, <SignTest />, {signTestTotal}{' '}
          comparisons
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
          Deliberately collecting less this cycle to keep a customer paying for
          another year is the whole argument, not a bug. The full list of what
          this engine cannot do — including the one feature that only works on
          a made-up signal — is in the repository&rsquo;s README.
        </p>
      </div>

      {/* Caveats are not banished from this page -- see HowItWorks.tsx, which
          carries the off-ramp finding and opens each explanation on demand.
          What is deliberately NOT repeated here is the full eleven-item list:
          a reader at the results table wants the numbers explained, not a
          second copy of the README. The link above is the route to it. */}
    </section>
  )
}
