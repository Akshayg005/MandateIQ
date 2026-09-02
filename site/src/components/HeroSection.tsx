interface HeroSectionProps {
  id: string
}

/**
 * Left-aligned, not centred, and no gradient on the headline.
 *
 * The centred-hero-over-a-dark-mesh-gradient with a colour-swept headline is
 * the house style of every AI-generated landing page, and the previous draft
 * of this file was exactly that. An asymmetric column with plain white type
 * reads as something a person laid out.
 *
 * Deliberately carries no figures. Everything numeric on this page is read
 * from a saved report, and the hero renders before that fetch resolves;
 * rather than show a placeholder where a number will be, the hero makes the
 * argument and lets the scene below it carry the evidence.
 */
export function HeroSection({ id }: HeroSectionProps) {
  return (
    <section id={id} className="hero-section">
      <div className="hero-glow" aria-hidden="true" />

      <div className="hero-inner">
        <p className="hero-eyebrow">Subscription payments, India</p>

        <h1 className="hero-title">
          When a subscription payment fails,
          <br />
          most systems just try again.
        </h1>

        <p className="hero-subtitle">
          Sometimes the money is simply not there today. Sometimes the card is
          dead. And sometimes the customer has quietly decided to leave. Those
          three need three different answers, and retrying is only the right
          answer to one of them.
        </p>

        <div className="hero-actions">
          <a className="btn btn--primary" href="#how">
            See how it decides
          </a>
          <a className="btn btn--ghost" href="#results">
            Jump to the results
          </a>
        </div>
      </div>
    </section>
  )
}

/**
 * The three causes, as the reader's first real explanation of the product.
 * Given its own section rather than crammed into the hero, which keeps the
 * hero inside one viewport and gives these room to be read.
 */
export function CauseSection({ id }: { id: string }) {
  const causes = [
    {
      key: 'now',
      when: 'The money is not there today',
      then: 'Try again, timed to when they are actually paid',
      body: 'Salary lands, balances recover. Retrying on a fixed schedule ignores that; retrying on their rhythm does not.',
    },
    {
      key: 'ever',
      when: 'The card or account is dead',
      then: 'Stop, and ask them to re-authorise',
      body: 'Expired card, closed account, cancelled mandate. No number of retries fixes any of these, and each one annoys someone who already cannot pay.',
    },
    {
      key: 'wont',
      when: 'They want to leave',
      then: 'Offer a way out, and let them choose',
      body: 'Pause, downgrade, or cancel. The system never cancels on someone’s behalf. Grinding an exiting customer is what turns a lapsed subscription into a complaint.',
    },
  ]

  return (
    <section id={id} className="cause-section">
      <div className="cause-head">
        <h2>Three reasons a payment fails. Three different answers.</h2>
        <p>
          The engine works out which one it is looking at before it decides
          what to do, and it only ever gets four attempts per customer.
        </p>
      </div>

      <ol className="cause-list">
        {causes.map((c, i) => (
          <li key={c.key} className={`cause-row cause-row--${c.key}`}>
            <span className="cause-num">{String(i + 1).padStart(2, '0')}</span>
            <div className="cause-when">
              <h3>{c.when}</h3>
              <p>{c.body}</p>
            </div>
            <div className="cause-then">
              <span className="cause-then-label">What it does</span>
              <p>{c.then}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
