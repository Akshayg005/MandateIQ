import { DASHBOARD_URL, REPO_URL } from '../links'
import { useReachable } from '../hooks/useReachable'

/**
 * One line, always visible, so the reader can reach the detail view without
 * hunting for it. The two apps are separate builds on separate ports, so
 * these are real links rather than routes.
 *
 * "Open the data" is probed before it is offered. It points at a second
 * server that may not be running, and a link that dumps the reader on a
 * connection-error page reads as a broken site rather than a stopped
 * process. When it is down the link stays visible but says so.
 *
 * Kept to a single row at desktop and well under 80px tall.
 */
export function SiteNav() {
  const dashboard = useReachable(DASHBOARD_URL)
  const down = dashboard === 'down'

  return (
    <nav className="site-nav" aria-label="Primary">
      <a className="nav-brand" href="#hero">
        <span className="nav-mark" aria-hidden="true" />
        MandateIQ
      </a>

      <div className="nav-links">
        <a className="nav-link" href="#how">
          How it works
        </a>
        <a className="nav-link" href="#results">
          Results
        </a>

        {down ? (
          <span
            className="nav-link nav-link--cta nav-link--down"
            role="link"
            aria-disabled="true"
            title="The data view runs as a separate app and is not started right now."
          >
            Data view offline
          </span>
        ) : (
          <a className="nav-link nav-link--cta" href={DASHBOARD_URL}>
            Open the data
          </a>
        )}

        <a
          className="nav-link nav-link--quiet"
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
        >
          Code
        </a>
      </div>
    </nav>
  )
}
