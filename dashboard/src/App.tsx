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
import { ExplainMore } from "./Explain";
import { Cell, FreezeHash, Seed } from "./glossary";

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
        {/* Orientation before vocabulary. A reader arriving from the landing
            page has the argument and none of the words; the first table below
            uses "arm", "regime" and "conformal" without apology. This strip is
            where they get explained, and it is deliberately the first thing in
            <main> rather than a link to somewhere else. */}
        <section className="orientation">
          <h2>New here? What this page is</h2>
          <p>
            The landing page makes the argument. This is the evidence behind
            it: the saved output of an evaluation run, rendered exactly as it
            was written to disk. Nothing on this page is recalculated, averaged
            or rounded in your browser — if a figure appears here, it appears
            in the report file too.
          </p>
          <ExplainMore label="How to read it, and what the codes in the header mean">
            <p>
              <strong>Two views.</strong> &ldquo;What happened to the
              batch&rdquo; is one merchant&rsquo;s customers for one cycle, and
              is where the money and the per-customer detail live. &ldquo;How it
              holds up under stress&rdquo; is the compliance surface an acquirer
              cares about — attempt budgets, authentication limits, and the
              places the engine is wrong by its own criteria.
            </p>
            <p>
              <strong>The header codes.</strong> <em>run</em> is the commit this
              build came from. <em>frozen</em> is the <FreezeHash />.{" "}
              <em>seeds</em> is how many independent batches were run — each{" "}
              <Seed /> is reproducible. <em>cells</em> is how many <Cell /> the
              full sweep produced.
            </p>
            <p>
              <strong>Underlined words explain themselves.</strong> Hover one,
              or tap it on a touch screen. Every term this page assumes has a
              definition attached to it.
            </p>
          </ExplainMore>
        </section>

        {view === "merchant" ? (
          <Merchant results={results} mandates={mandates?.mandates ?? null} />
        ) : (
          <Acquirer regimes={regimes} />
        )}
      </main>
    </>
  );
}
