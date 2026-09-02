/**
 * Where the other half of this project lives.
 *
 * The two apps are separate Vite builds on separate ports, so neither can
 * route to the other through a client router. Configurable rather than
 * hardcoded because the dev ports are a local convention: `.\run.ps1 up`
 * serves the landing page on 4318, a deployed copy will not.
 */
export const SITE_URL = import.meta.env.VITE_SITE_URL ?? "http://localhost:4318";

export const REPO_URL = "https://github.com/Akshayg005/MandateIQ";
