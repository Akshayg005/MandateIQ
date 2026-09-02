/**
 * Merchant view: what this cycle cost and what it preserved, then the batch.
 *
 * The three bars are shown for FOUR policies, not two. `null` (never attempt)
 * and `one_shot` (one attempt, no model, no belief, no gate) are reference
 * arms, and on two of the three bars one_shot beats the engine. Showing only
 * engine-vs-ladder would make the headline unfalsifiable, which is exactly
 * what B13's review found and corrected. Every number here is read from
 * reports/results.json; nothing on this page is computed.
 */
import { useMemo, useState } from "react";
import type { Bars, MandateRecord, Results } from "./data";
import Drilldown from "./Drilldown";

const PRESERVED_DENOM = 200;

function preservedCount(b: Bars): number {
  return Number(b.mandates_preserved.split("/")[0]);
}

function BarRow({
  label,
  width,
  value,
  reference,
}: {
  label: string;
  width: number;
  value: string;
  reference?: boolean;
}) {
  return (
    <>
      <span className="label">{label}</span>
      <span className="track">
        <span
          className={reference ? "fill ref" : "fill"}
          style={{ width: `${Math.max(0, Math.min(100, width))}%` }}
        />
      </span>
      <span className="value">{value}</span>
    </>
  );
}

function ThreeBars({ r }: { r: Results }) {
  const arms: Array<{ name: string; bars: Bars; ref: boolean }> = [
    { name: "engine (ours)", bars: r, ref: false },
    { name: "ladder (incumbent)", bars: r.baseline, ref: true },
    { name: "one_shot (no model)", bars: r.reference_one_shot, ref: true },
    { name: "null (never attempt)", bars: r.reference_null, ref: true },
  ];
  const maxRec = Math.max(...arms.map((a) => a.bars.recovered_paise));
  const maxAtt = Math.max(...arms.map((a) => a.bars.attempts_spent));

  return (
    <>
      <div className="bargroup">
        <h3>money recovered</h3>
        <div className="bars">
          {arms.map((a) => (
            <BarRow
              key={a.name}
              label={a.name}
              width={maxRec ? (a.bars.recovered_paise / maxRec) * 100 : 0}
              value={a.bars.recovered}
              reference={a.ref}
            />
          ))}
        </div>
      </div>

      <div className="bargroup">
        <h3>attempts spent (lower is better)</h3>
        <div className="bars">
          {arms.map((a) => (
            <BarRow
              key={a.name}
              label={a.name}
              width={maxAtt ? (a.bars.attempts_spent / maxAtt) * 100 : 0}
              value={String(a.bars.attempts_spent)}
              reference={a.ref}
            />
          ))}
        </div>
      </div>

      <div className="bargroup">
        <h3>mandates preserved (the bar the incumbent does not report)</h3>
        <div className="bars">
          {arms.map((a) => (
            <BarRow
              key={a.name}
              label={a.name}
              width={(preservedCount(a.bars) / PRESERVED_DENOM) * 100}
              value={a.bars.mandates_preserved}
              reference={a.ref}
            />
          ))}
        </div>
      </div>
    </>
  );
}

function Honesty({ r }: { r: Results }) {
  const st = r.sign_test.vs_one_shot;
  const n = r.paired_comparisons;
  return (
    <div className="limits">
      <h2>What this table does not say</h2>
      <ul>
        <li>
          Against the incumbent ladder the thesis holds and is stable: across{" "}
          {n} paired comparisons over {r.seeds.length} seeds, the engine
          preserves more in {r.sign_test.vs_ladder.preserves_more}/{n} and
          spends fewer attempts in{" "}
          {r.sign_test.vs_ladder.spends_fewer_attempts}/{n} — at a cost in
          money, winning there only {r.sign_test.vs_ladder.recovers_more}/{n}.
        </li>
        <li>
          <strong>
            Against <code>one_shot</code> — one attempt, no model — it does not.
          </strong>{" "}
          That policy preserves more in {st.preserves_fewer}/{n} and the engine
          spends more attempts in {st.spends_more_attempts}/{n}. The engine
          wins only on money, {st.recovers_more}/{n}. More seeds made this
          finding stronger, not weaker. The defensible claim is against the
          incumbent, not against every trivial baseline.
        </li>
        <li>
          <code>OFFER</code> fired {r.offers_fired_total} times — and that is
          arithmetic, not measurement. <code>cause_map</code> pins
          P(WONT_PAY) at 0.10 under both symbols the proxy alphabet can emit,
          so the {"{WONT_PAY}"} singleton the off-ramp requires is unreachable
          for any alpha, seed or regime. The off-ramp lane is{" "}
          <strong>untested</strong>, not tested-and-negative.
        </li>
        <li>
          {r.false_reauth_total} of {r.reauth_total} re-auth requests went to
          mandates whose true cause is <em>not</em> CANT_PAY_EVER — the
          issuer_outage regime&rsquo;s own pre-registered falsification
          criterion.
        </li>
        <li>
          {r.attempt_after_terminal_total} post-terminal re-solves returned
          ATTEMPT on instruments the issuer had just confirmed dead. The belief
          layer cannot conclude CANT_PAY_EVER from an observed dead instrument.
        </li>
        <li>
          Every attempt lands on day 2. There is no timing discrimination, so
          the <code>strict</code> and <code>permissive</code> profiles are
          provably the same function here.
        </li>
      </ul>
    </div>
  );
}

export default function Merchant({
  results,
  mandates,
}: {
  results: Results;
  mandates: MandateRecord[] | null;
}) {
  const [outcome, setOutcome] = useState("all");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);

  const categories = useMemo(
    () => Array.from(new Set((mandates ?? []).map((m) => m.category))).sort(),
    [mandates],
  );
  const outcomes = useMemo(
    () =>
      Array.from(
        new Set((mandates ?? []).map((m) => m.final_outcome ?? "NONE")),
      ).sort(),
    [mandates],
  );

  const rows = useMemo(
    () =>
      (mandates ?? []).filter(
        (m) =>
          (outcome === "all" || (m.final_outcome ?? "NONE") === outcome) &&
          (category === "all" || m.category === category),
      ),
    [mandates, outcome, category],
  );

  const chosen = rows.find((m) => m.mandate_id === selected) ?? null;

  return (
    <>
      <Honesty r={results} />

      <section className="panel">
        <h2>
          three bars · {results.headline_cell} · {results.seeds.length} seeds
        </h2>
        <div className="body">
          <ThreeBars r={results} />
          <p className="note">
            Recovery rate alone is the incumbent&rsquo;s metric. Preserved
            mandates is the bar this project exists to move, and{" "}
            <code>null</code> preserves 200/200 by never attempting — which is
            why it is on the chart.
          </p>
        </div>
      </section>

      {!mandates ? (
        <section className="panel">
          <div className="empty">
            <code>mandates.json</code> is not staged. Run{" "}
            <code>python -m eval.export_mandates</code> then{" "}
            <code>python scripts\dashboard_data.py</code>.
          </div>
        </section>
      ) : (
        <>
          <section className="panel">
            <h2>the batch — click a mandate for its decision trail</h2>
            <div className="body">
              <div className="filters">
                <label htmlFor="oc">outcome</label>
                <select id="oc" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                  <option value="all">all</option>
                  {outcomes.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
                <label htmlFor="ct">category</label>
                <select id="ct" value={category} onChange={(e) => setCategory(e.target.value)}>
                  <option value="all">all</option>
                  {categories.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <span className="count">
                  {rows.length} of {mandates.length} mandates
                </span>
              </div>

              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>mandate</th>
                      <th>category</th>
                      <th className="num">amount</th>
                      <th className="num">attempts</th>
                      <th>final action</th>
                      <th>outcome</th>
                      <th>AFA</th>
                      <th className="num">ledger rows</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((m) => (
                      <tr
                        key={m.mandate_id}
                        className="clickable"
                        aria-selected={m.mandate_id === selected}
                        onClick={() =>
                          setSelected(m.mandate_id === selected ? null : m.mandate_id)
                        }
                      >
                        <td className="mono">{m.mandate_id}</td>
                        <td>{m.category}</td>
                        <td className="num">{m.amount}</td>
                        <td className="num">{m.attempts_spent}</td>
                        <td className="mono">{m.final_action ?? "—"}</td>
                        <td className="mono">{m.final_outcome ?? "—"}</td>
                        <td className={m.above_afa ? "bad" : "dim"}>
                          {m.above_afa ? "above" : "under"}
                        </td>
                        <td className="num">{m.ledger.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          {chosen ? (
            <Drilldown m={chosen} />
          ) : (
            <section className="panel">
              <div className="empty">Select a mandate above to see its drill-down.</div>
            </section>
          )}
        </>
      )}
    </>
  );
}
