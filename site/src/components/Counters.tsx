import { useEffect, useRef, useState } from 'react'
import { Reveal } from './Reveal'
import type { Narrative, ReportData } from '../hooks/useReportData'

/**
 * The three bars, as an actual chart.
 *
 * This project's headline metric is three bars -- recovered, attempts spent,
 * mandates preserved -- so the page draws three, not one with the other two
 * in prose. They are three SEPARATE paired charts because the measures have
 * three different units (mandates, paise, attempts); putting money and counts
 * on one axis, or on two axes of one chart, would be a lie about scale.
 *
 * Bar lengths come from raw magnitudes (paise, counts). Percentages appear as
 * labels only -- a percentage is not a length.
 *
 * The engine LOSES the middle chart, visibly and on purpose. A page that only
 * drew the bar it wins would be the same dishonesty this whole rewrite was
 * about.
 */

/** Validated on the dark surface #0f1117 via the dataviz palette checker:
 *  lightness band, chroma floor, CVD separation (ΔE 12.5 protan / 28.7
 *  tritan), normal-vision ΔE 24.3, contrast >= 3:1. Do not re-pick by eye. */
const ENGINE = '#0d9488'
const LADDER = '#d97706'

function useInView<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    // No IntersectionObserver (older browser, some test runners) means the
    // chart should be AT its value, never stuck at zero.
    if (typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setSeen(true)
          obs.disconnect()
        }
      },
      { threshold: 0.25 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return [ref, seen] as const
}

interface BarChartProps {
  title: string
  unit: string
  engineValue: number
  ladderValue: number
  engineLabel: string
  ladderLabel: string
  verdict: string
  engineWins: boolean
  active: boolean
}

/**
 * One paired horizontal bar. Both bars are direct-labelled, so identity never
 * rests on colour alone -- which is also why there are only six marks on this
 * page and no tooltip: the exact figure is already on screen beside each bar.
 */
function PairedBar({
  title,
  unit,
  engineValue,
  ladderValue,
  engineLabel,
  ladderLabel,
  verdict,
  engineWins,
  active,
}: BarChartProps) {
  const max = Math.max(engineValue, ladderValue, 1)
  const enginePct = (engineValue / max) * 100
  const ladderPct = (ladderValue / max) * 100

  return (
    <Reveal className="bar-chart-reveal">
    <figure className="bar-chart">
      <figcaption className="bar-chart-head">
        <h3>{title}</h3>
        <span className="bar-chart-unit">{unit}</span>
      </figcaption>

      <div className="bar-rows">
        <div className="bar-row">
          <span className="bar-name">This engine</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: active ? `${enginePct}%` : '0%',
                background: ENGINE,
              }}
            />
          </div>
          <span className="bar-value" style={{ color: 'var(--text)' }}>
            {engineLabel}
          </span>
        </div>

        <div className="bar-row">
          <span className="bar-name">Fixed ladder</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: active ? `${ladderPct}%` : '0%',
                background: LADDER,
              }}
            />
          </div>
          <span className="bar-value">{ladderLabel}</span>
        </div>
      </div>

      <p className={`bar-verdict${engineWins ? '' : ' bar-verdict--loss'}`}>
        {verdict}
      </p>
    </figure>
    </Reveal>
  )
}

export function CountersSection({
  id,
  narrative: n,
  data,
}: {
  id: string
  narrative: Narrative
  data: ReportData
}) {
  const [ref, seen] = useInView<HTMLElement>()

  return (
    <section id={id} ref={ref} className="counters-section">
      <Reveal className="section-head">
        <span className="section-kicker">The headline metric</span>
        <h2 className="section-title">Three bars, not one</h2>
        <p className="section-subtitle">
          Recovery rate alone is the incumbent&rsquo;s metric. Across{' '}
          {data.paired_comparisons} paired comparisons over {n.seedCount} seeds,
          this engine preserves more mandates in{' '}
          <strong>
            {data.sign_test.vs_ladder.preserves_more}/{data.paired_comparisons}
          </strong>{' '}
          and recovers more money in only{' '}
          <strong>
            {data.sign_test.vs_ladder.recovers_more}/{data.paired_comparisons}
          </strong>
          .
        </p>
      </Reveal>

      {/* Legend: two series, always present, never colour-alone. */}
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: ENGINE }} />
          This engine
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: LADDER }} />
          Fixed ladder (incumbent)
        </span>
      </div>

      <div className="bar-charts">
        <PairedBar
          title="Mandates preserved"
          unit={`of ${n.total}`}
          engineValue={n.enginePreserved}
          ladderValue={n.ladderPreserved}
          engineLabel={`${n.enginePreserved}/${n.total}`}
          ladderLabel={`${n.ladderPreserved}/${n.total}`}
          verdict={`${n.preservedDelta} more customers still have a live mandate next cycle.`}
          engineWins
          active={seen}
        />
        <PairedBar
          title="Money recovered"
          unit="this cycle"
          engineValue={n.engineRecoveredPaise}
          ladderValue={n.ladderRecoveredPaise}
          engineLabel={`${data.recovered} (${n.engineRecoveredPct})`}
          ladderLabel={`${data.baseline.recovered} (${n.ladderRecoveredPct})`}
          verdict="The engine loses this bar. Deliberately recovering less this cycle to protect lifetime value is the thesis, not a bug."
          engineWins={false}
          active={seen}
        />
        <PairedBar
          title="Attempts spent"
          unit={`of ${n.total} × 4 NPCI allows`}
          engineValue={n.engineAttempts}
          ladderValue={n.ladderAttempts}
          engineLabel={String(n.engineAttempts)}
          ladderLabel={String(n.ladderAttempts)}
          verdict={`${n.attemptsSaved} fewer attempts. Every attempt carries an opt-out the customer can use to kill the mandate outright.`}
          engineWins
          active={seen}
        />
      </div>

      <div className="headline-stat">
        <AnimatedStat
          target={n.preservedDelta}
          prefix="+"
          active={seen}
          label="mandates preserved versus the incumbent"
        />
        <AnimatedStat
          target={n.attemptsSaved}
          prefix="−"
          active={seen}
          label="attempts not spent on someone who was leaving"
        />
      </div>
    </section>
  )
}

/**
 * Counts up on entry. The figure also lives in data-target and aria-label,
 * because the animated text starts at 0 and would otherwise never appear in
 * the served HTML -- invisible to a screen reader, to a viewer whose observer
 * never fires, and to ssr-check.tsx, which enforces B15's "wired to real
 * report output" gate.
 */
function AnimatedStat({
  target,
  prefix = '',
  active,
  label,
  duration = 1400,
}: {
  target: number
  prefix?: string
  active: boolean
  label: string
  duration?: number
}) {
  const [value, setValue] = useState(0)
  const raf = useRef(0)

  useEffect(() => {
    if (!active) return
    if (typeof window === 'undefined' || !window.matchMedia) {
      setValue(target)
      return
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target)
      return
    }
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      setValue(Math.round((1 - Math.pow(1 - t, 3)) * target))
      if (t < 1) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [active, target, duration])

  return (
    <div className="headline-stat-item">
      <span
        className="headline-stat-value"
        data-target={`${prefix}${target}`}
        aria-label={`${prefix}${target} ${label}`}
      >
        <span aria-hidden="true">
          {prefix}
          {value}
        </span>
      </span>
      <span className="headline-stat-label" aria-hidden="true">
        {label}
      </span>
    </div>
  )
}
