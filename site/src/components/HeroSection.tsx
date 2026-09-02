interface HeroSectionProps {
  id: string
}

/**
 * Deliberately carries no figures.
 *
 * Everything numeric on this page is read from reports/results.json, and the
 * hero renders before that fetch resolves. Rather than show a placeholder or
 * a skeleton where a number will be, the hero makes the argument -- which is
 * qualitative anyway -- and lets the scene below it carry the evidence.
 */
export function HeroSection({ id }: HeroSectionProps) {
  return (
    <section id={id} className="hero-section">
      <div className="hero-aurora" aria-hidden="true">
        <span className="aurora aurora--teal" />
        <span className="aurora aurora--blue" />
        <span className="aurora aurora--amber" />
      </div>

      <div className="hero-inner">
        <h1 className="hero-title">
          <span className="hero-line hero-line--muted">Every retry engine asks</span>
          <span className="hero-line hero-line--strike">
            &ldquo;will this succeed?&rdquo;
          </span>
          <span className="hero-line hero-line--accent">
            We ask which of three
            <br />
            things went wrong.
          </span>
        </h1>

        <p className="hero-subtitle">
          A decision engine for failed recurring debits under India&rsquo;s
          e-mandate framework. Four attempts exist, ever. Spending one on a
          customer who wants out is how you lose them.
        </p>

        <div className="cause-cards">
          <article className="cause-card cause-card--now">
            <header>
              <span className="cause-tag">CANT_PAY_NOW</span>
              <span className="cause-dot" />
            </header>
            <h3>Transient liquidity gap</h3>
            <p>
              The money isn&rsquo;t there today. Spend a slot, timed to their
              replenishment rhythm, not to a fixed ladder.
            </p>
          </article>

          <article className="cause-card cause-card--ever">
            <header>
              <span className="cause-tag">CANT_PAY_EVER</span>
              <span className="cause-dot" />
            </header>
            <h3>Instrument is dead</h3>
            <p>
              Expired card, closed account, revoked mandate. Stop retrying.
              Request re-authorisation instead.
            </p>
          </article>

          <article className="cause-card cause-card--wont">
            <header>
              <span className="cause-tag">WONT_PAY</span>
              <span className="cause-dot" />
            </header>
            <h3>Passive resistance</h3>
            <p>
              They want out. <strong>Offer</strong> an exit: pause, downgrade,
              cancel. The system never cancels. The customer decides.
            </p>
          </article>
        </div>

      </div>
    </section>
  )
}
