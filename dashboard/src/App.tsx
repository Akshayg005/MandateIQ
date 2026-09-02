/**
 * Shell. Two personas, one artifact set, no computation.
 *
 * The provenance strip in the header is not decoration: a screenshot of this
 * page has to be traceable to the run that produced it, so the freeze hash
 * and the commit are on screen at all times.
 */
import { useEffect, useState } from "react";
import Acquirer from "./Acquirer";
import Merchant from "./Merchant";
import { loadAll, type Data } from "./data";

type View = "merchant" | "acquirer";

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("merchant");

  useEffect(() => {
    loadAll().then(setData, (e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="empty">
        Could not load the report artifacts: <code>{error}</code>
        <br />
        Run <code>.\run.ps1 eval</code>, then{" "}
        <code>python scripts\dashboard_data.py</code>.
      </div>
    );
  }
  if (!data) return <div className="empty">loading…</div>;

  const { manifest, results, regimes, mandates } = data;

  return (
    <>
      <header className="top">
        <h1>Mandate Recovery Engine</h1>
        <span className="sub">
          decision engine for failed recurring debits · RBI e-mandate framework
          2026
        </span>
        <span className="prov">
          freeze {manifest.freeze_hash.slice(0, 12)} · commit{" "}
          {manifest.git_sha.slice(0, 12)} · {results.seeds.length} seeds ·{" "}
          {regimes.cells.length} cells
        </span>
      </header>

      <nav className="tabs">
        <button
          aria-selected={view === "merchant"}
          onClick={() => setView("merchant")}
        >
          Merchant
        </button>
        <button
          aria-selected={view === "acquirer"}
          onClick={() => setView("acquirer")}
        >
          Acquirer · clause 10(c)
        </button>
      </nav>

      <main>
        {view === "merchant" ? (
          <Merchant results={results} mandates={mandates?.mandates ?? null} />
        ) : (
          <Acquirer regimes={regimes} />
        )}
      </main>
    </>
  );
}
