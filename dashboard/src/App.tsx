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
import { SITE_URL } from "./links";

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
        The saved evaluation run could not be loaded, so there is nothing to
        show here. The repository README explains how to generate one.
      </div>
    );
  }
  if (!data) return <div className="empty">loading…</div>;

  const { manifest, results, regimes, mandates } = data;

  return (
    <>
      <header className="top">
        <a className="back" href={SITE_URL}>
          <span aria-hidden="true">&larr;</span> Overview
        </a>
        <div className="top-title">
          <h1>The data behind the numbers</h1>
          <span className="sub">
            Every figure on this page is read from a saved report. Nothing here
            is recalculated in the browser.
          </span>
        </div>
        <span className="prov" title="Which run produced these numbers">
          <b>run</b> {manifest.git_sha.slice(0, 7)}
          <b>frozen</b> {manifest.freeze_hash.slice(0, 8)}
          <b>seeds</b> {results.seeds.length}
          <b>cells</b> {regimes.cells.length}
        </span>
      </header>

      <nav className="tabs" aria-label="View">
        <button
          aria-selected={view === "merchant"}
          onClick={() => setView("merchant")}
        >
          What happened to the batch
          <em>
            One merchant
            {mandates ? `, ${mandates.mandates.length} mandates` : ""}
          </em>
        </button>
        <button
          aria-selected={view === "acquirer"}
          onClick={() => setView("acquirer")}
        >
          How it holds up under stress
          <em>Every regime, both rulebooks</em>
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
