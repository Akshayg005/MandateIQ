import { useEffect, useState } from 'react'

/**
 * The shape of `reports/results.json`, staged into `public/data/` by
 * `python scripts/dashboard_data.py site`.
 *
 * IMPORTANT: every headline figure here is a MEAN over the seeds listed in
 * `seeds` (8 of them), not a single run. `seed` is `seeds[0]` and is a legacy
 * field; do not present it as "the" seed. `reports/mandates.json` is the
 * seed-0 batch and is deliberately NOT staged for this page, because a seed-0
 * count beside an 8-seed mean under one label is exactly the kind of quiet
 * mismatch this project's honesty rules exist to prevent.
 */
export interface PolicyFigures {
  recovered_paise: number
  recovered_pct: string
  recovered: string
  attempts_spent: number
  attempts_per_recovery: number | null
  mandates_preserved: string
}

export interface ReportData extends PolicyFigures {
  attempts_per_recovery: number
  seeds: number[]
  paired_comparisons: number
  offers_fired_total: number
  sign_test: {
    vs_ladder: {
      preserves_more: number
      recovers_more: number
      spends_fewer_attempts: number
    }
  }
  baseline: PolicyFigures & { attempts_per_recovery: number }
  reference_one_shot: PolicyFigures
}

/** Parse "142/200" -> { preserved: 142, total: 200, lost: 58 }. */
export function parseFraction(s: string): {
  preserved: number
  total: number
  lost: number
} {
  const [n, d] = s.split('/')
  const preserved = Number.parseInt(n, 10)
  const total = Number.parseInt(d, 10)
  if (!Number.isFinite(preserved) || !Number.isFinite(total)) {
    throw new Error(`unparseable mandates_preserved: ${s}`)
  }
  return { preserved, total, lost: total - preserved }
}

/**
 * The figures the scroll narrative animates. Derived here, once, so that no
 * component invents a count of its own -- B15's gate is that the counters are
 * wired to real report output, and a constant in a component is how that
 * stops being true without anyone noticing.
 */
export interface Narrative {
  total: number
  enginePreserved: number
  engineLost: number
  ladderPreserved: number
  ladderLost: number
  engineRecoveredPct: string
  ladderRecoveredPct: string
  engineAttempts: number
  ladderAttempts: number
  /** Raw magnitudes, for bar lengths. Percentages are for labels, not widths. */
  engineRecoveredPaise: number
  ladderRecoveredPaise: number
  /** Mandates the engine preserves that the ladder does not. */
  preservedDelta: number
  /** Attempts the engine does NOT spend. Positive means it spends fewer. */
  attemptsSaved: number
  /** 0 in every published run. The off-ramp lane is untested, not negative. */
  offersFired: number
  seedCount: number
}

export function deriveNarrative(d: ReportData): Narrative {
  const engine = parseFraction(d.mandates_preserved)
  const ladder = parseFraction(d.baseline.mandates_preserved)
  return {
    total: engine.total,
    enginePreserved: engine.preserved,
    engineLost: engine.lost,
    ladderPreserved: ladder.preserved,
    ladderLost: ladder.lost,
    engineRecoveredPct: d.recovered_pct,
    ladderRecoveredPct: d.baseline.recovered_pct,
    engineAttempts: d.attempts_spent,
    ladderAttempts: d.baseline.attempts_spent,
    engineRecoveredPaise: d.recovered_paise,
    ladderRecoveredPaise: d.baseline.recovered_paise,
    preservedDelta: engine.preserved - ladder.preserved,
    attemptsSaved: d.baseline.attempts_spent - d.attempts_spent,
    offersFired: d.offers_fired_total,
    seedCount: d.seeds?.length ?? 1,
  }
}

export type ReportState =
  | { status: 'loading' }
  | { status: 'ready'; data: ReportData; narrative: Narrative }
  | { status: 'error'; message: string }

/**
 * Fetches results.json at runtime -- counters are NEVER hard-coded.
 *
 * Failure is returned, not swallowed. An earlier version caught and discarded
 * the error, which silently unmounted every section that took a number and
 * left a page that looked deliberately sparse rather than broken.
 */
export function useReportData(): ReportState {
  const [state, setState] = useState<ReportState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    // BASE_URL, not a leading slash: the page is served from a sub-path on
    // GitHub Pages and an absolute path 404s there.
    const url = `${import.meta.env.BASE_URL}data/results.json`

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.json()
      })
      .then((d: ReportData) => {
        if (cancelled) return
        setState({ status: 'ready', data: d, narrative: deriveNarrative(d) })
      })
      .catch((e: unknown) => {
        if (cancelled) return
        const message = e instanceof Error ? e.message : String(e)
        setState({ status: 'error', message })
      })

    return () => {
      cancelled = true
    }
  }, [])

  return state
}
