import { useEffect, useState } from 'react'

export type Reachability = 'checking' | 'up' | 'down'

/**
 * Is the other app actually serving?
 *
 * The two halves of this project run as separate servers, so "Open the data"
 * is a link to a different origin that may simply not be running -- and a
 * link that lands on a browser connection-error page looks like a broken
 * site rather than a stopped server. This probes first so the nav can say
 * which it is.
 *
 * `no-cors` because the dashboard sends no CORS headers and none are needed:
 * an opaque response still proves something answered, and a refused
 * connection still rejects. The response body is never read.
 */
export function useReachable(url: string, timeoutMs = 2500): Reachability {
  const [state, setState] = useState<Reachability>('checking')

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    fetch(url, { mode: 'no-cors', signal: controller.signal, cache: 'no-store' })
      .then(() => {
        if (!cancelled) setState('up')
      })
      .catch(() => {
        if (!cancelled) setState('down')
      })
      .finally(() => clearTimeout(timer))

    return () => {
      cancelled = true
      clearTimeout(timer)
      controller.abort()
    }
  }, [url, timeoutMs])

  return state
}
