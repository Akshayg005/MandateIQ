import { DASHBOARD_URL, REPO_URL } from '../links'

/**
 * One line, always visible, so the reader can get to the detail view without
 * hunting for it. The two apps are separate builds on separate ports, so
 * these are real links rather than routes.
 *
 * Kept to a single row at desktop and a height well under 80px: a marketing
 * nav that eats a tenth of the viewport is a design smell, not a feature.
 */
export function SiteNav() {
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
        <a className="nav-link nav-link--cta" href={DASHBOARD_URL}>
          Open the data
        </a>
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
