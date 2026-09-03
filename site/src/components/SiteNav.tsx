import { DASHBOARD_URL, REPO_URL } from '../links'
import { useReachable } from '../hooks/useReachable'

/**
 * Inline rather than a <use> against public/icons.svg: that sprite's github
 * symbol carries a hard-coded #08060d fill, which is invisible on this
 * page's near-black surface. `currentColor` inherits the link's colour
 * instead, including on hover, and saves a request for one 8-line path.
 */
function GithubIcon() {
  return (
    <svg
      className="nav-icon"
      viewBox="0 0 16 16"
      width="15"
      height="15"
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  )
}

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
          className="nav-link nav-link--quiet nav-link--icon"
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
        >
          <GithubIcon />
          GitHub
        </a>
      </div>
    </nav>
  )
}
