import { useEffect, useRef, useState } from 'react'

interface CounterProps {
  target: number
  suffix?: string
  prefix?: string
  duration?: number
  /** Whether to trigger the animation */
  active: boolean
}

function AnimatedCounter({
  target,
  suffix = '',
  prefix = '',
  duration = 2000,
  active,
}: CounterProps) {
  const [value, setValue] = useState(0)
  const rafRef = useRef(0)

  useEffect(() => {
    if (!active) return
    const start = performance.now()

    function tick(now: number) {
      const elapsed = now - start
      const progress = Math.min(elapsed / duration, 1)
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(Math.round(eased * target))
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick)
      }
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [active, target, duration])

  // The animated text starts at 0 and counts up, so the real figure would
  // otherwise never appear in the served HTML -- invisible to a screen
  // reader, to a viewer whose IntersectionObserver never fires, and to
  // ssr-check.tsx, which is what enforces B15's "wired to real report output"
  // gate. data-target carries it for the check; aria-label carries it for
  // people, who should hear the number rather than a running count.
  return (
    <span
      className="counter-value"
      data-target={`${prefix}${target}${suffix}`}
      aria-label={`${prefix}${target}${suffix}`}
    >
      <span aria-hidden="true">
        {prefix}
        {value}
        {suffix}
      </span>
    </span>
  )
}

interface CountersSectionProps {
  id: string
  ladderPreserved: number
  enginePreserved: number
  total: number
  recoveredPct: string
  attemptsPerRecovery: number
  ladderRecoveredPct: string
  ladderAttemptsPerRecovery: number
  signTestPreservesMore: number
  signTestTotal: number
  seedCount: number
}

export function CountersSection({
  id,
  ladderPreserved,
  enginePreserved,
  total,
  recoveredPct,
  attemptsPerRecovery,
  ladderRecoveredPct,
  ladderAttemptsPerRecovery,
  signTestPreservesMore,
  signTestTotal,
  seedCount,
}: CountersSectionProps) {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          obs.disconnect()
        }
      },
      { threshold: 0.3 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section id={id} ref={ref} className="counters-section">
      <h2 className="section-label">The Three Bars</h2>
      <p className="section-subtitle">
        Across {signTestTotal} paired comparisons ({seedCount} seeds), the
        engine preserves more mandates in{' '}
        <strong>
          {signTestPreservesMore}/{signTestTotal}
        </strong>
        .
      </p>

      <div className="counters-grid">
        <div className="counter-card counter-card--primary">
          <div className="counter-label">Mandates Preserved</div>
          <div className="counter-row">
            <div className="counter-col">
              <div className="counter-sub-label">Ours</div>
              <div className="counter-big">
                <AnimatedCounter
                  target={enginePreserved}
                  active={visible}
                  suffix={`/${total}`}
                />
              </div>
            </div>
            <div className="counter-vs">vs</div>
            <div className="counter-col">
              <div className="counter-sub-label">Ladder</div>
              <div className="counter-big counter-big--dim">
                <AnimatedCounter
                  target={ladderPreserved}
                  active={visible}
                  suffix={`/${total}`}
                />
              </div>
            </div>
          </div>
          <div className="counter-delta">
            +
            <AnimatedCounter
              target={enginePreserved - ladderPreserved}
              active={visible}
            />{' '}
            mandates saved from unnecessary churn
          </div>
        </div>

        <div className="counter-card">
          <div className="counter-label">Recovery Rate</div>
          <div className="counter-row">
            <div className="counter-col">
              <div className="counter-sub-label">Ours</div>
              <div className="counter-medium">{recoveredPct}</div>
            </div>
            <div className="counter-vs">vs</div>
            <div className="counter-col">
              <div className="counter-sub-label">Ladder</div>
              <div className="counter-medium counter-medium--dim">
                {ladderRecoveredPct}
              </div>
            </div>
          </div>
          <div className="counter-note">
            Less money this cycle — by design. Lifetime value &gt; one
            transaction.
          </div>
        </div>

        <div className="counter-card">
          <div className="counter-label">Attempts per Recovery</div>
          <div className="counter-row">
            <div className="counter-col">
              <div className="counter-sub-label">Ours</div>
              <div className="counter-medium">
                {attemptsPerRecovery.toFixed(2)}
              </div>
            </div>
            <div className="counter-vs">vs</div>
            <div className="counter-col">
              <div className="counter-sub-label">Ladder</div>
              <div className="counter-medium counter-medium--dim">
                {ladderAttemptsPerRecovery.toFixed(2)}
              </div>
            </div>
          </div>
          <div className="counter-note">
            Fewer attempts = fewer chances to annoy a customer who would have
            paid.
          </div>
        </div>
      </div>
    </section>
  )
}
