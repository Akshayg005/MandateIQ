/**
 * Where the other half of this project lives.
 *
 * The two apps are separate Vite builds on separate ports, so neither can
 * route to the other through a client router -- the link has to be a real
 * URL. It is configurable rather than hardcoded because the dev ports are a
 * local convention: `.\run.ps1 up` serves the dashboard on 4317, but a
 * deployed copy will not.
 *
 * Set VITE_DASHBOARD_URL at build time to point somewhere else.
 */
export const DASHBOARD_URL =
  import.meta.env.VITE_DASHBOARD_URL ?? 'http://localhost:4317'

export const REPO_URL = 'https://github.com/Akshayg005/MandateIQ'
