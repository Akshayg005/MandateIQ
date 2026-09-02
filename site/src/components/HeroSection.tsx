interface HeroSectionProps {
  id: string
}

export function HeroSection({ id }: HeroSectionProps) {
  return (
    <section id={id} className="hero-section">
      <div className="hero-badge">Razorpay AI Buildathon · Track 03</div>
      <h1 className="hero-title">
        <span className="hero-title-line">Every retry engine asks</span>
        <span className="hero-title-accent">"will this succeed?"</span>
      </h1>
      <p className="hero-subtitle">
        Ours asks <em>"which of three things went wrong?"</em>
        <br />
        — and sometimes concludes the right action is to let the customer go.
      </p>
      <div className="hero-scroll-cue" aria-hidden="true">
        <span className="scroll-text">Scroll to see the difference</span>
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          className="scroll-arrow"
        >
          <path
            d="M12 4v16m0 0l-6-6m6 6l6-6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </section>
  )
}
