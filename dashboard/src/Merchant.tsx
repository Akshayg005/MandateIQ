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

function preservedCount(b: Bars): number {
  return Number(b.mandates_preserved.split("/")[0]);
}

/* The batch size comes from the report ("142/200"), never from a constant.
   A hardcoded 200 here would silently mis-scale every bar the day the
   evaluation runs a different batch size. */
function preservedDenom(b: Bars): number {
  const d = Number(b.mandates_preserved.split("/")[1]);
  return Number.isFinite(d) && d > 0 ? d : 1;
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
    { name: "This engine", bars: r, ref: false },
    { name: "Fixed retry schedule", bars: r.baseline, ref: true },
    { name: "Try once, then stop", bars: r.reference_one_shot, ref: true },
    { name: "Never retry at all", bars: r.reference_null, ref: true },
  ];
  const maxRec = Math.max(...arms.map((a) => a.bars.recovered_paise));
  const maxAtt = Math.max(...arms.map((a) => a.bars.attempts_spent));

  return (
    <>
      <div className="bargroup">
        <h3>Money collected this cycle</h3>
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
        <h3>Attempts used <em>(fewer is better)</em></h3>
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
        <h3>Customers still subscribed at the end <em>(the number the incumbent never reports)</em></h3>
        <div className="bars">
          {arms.map((a) => (
            <BarRow
              key={a.name}
              label={a.name}
              width={(preservedCount(a.bars) / preservedDenom(a.bars)) * 100}
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
      <h2>Read this before the numbers</h2>
      <p className="limits-lede">
        We ran the same {r.seeds.length} batches through every policy and
        compared them one batch at a time, {n} comparisons in all. Here is where
        this engine wins and where it does not.
      </p>
      <ul>
        <li>
          <strong>Against the incumbent, it holds up.</strong> It kept more
          customers in {r.sign_test.vs_ladder.preserves_more} of the {n}{" "}
          comparisons and used fewer attempts in{" "}
          {r.sign_test.vs_ladder.spends_fewer_attempts} of {n}. It collected
          more money in only {r.sign_test.vs_ladder.recovers_more} of {n}, which
          is the trade this project is arguing for.
        </li>
        <li>
          <strong>
            Against simply trying once and stopping, it does not.
          </strong>{" "}
          That policy has no model in it at all, and it kept more customers than
          this engine in {st.preserves_fewer} of {n} comparisons while this
          engine used more attempts in {st.spends_more_attempts} of {n}. The
          engine only comes out ahead on money, {st.recovers_more} of {n}.
          Running more batches made that result stronger, not weaker. The claim
          worth defending is against the incumbent, not against every simple
          rule you could write.
        </li>
        <li>
          <strong>
            The off-ramp never actually ran. It was chosen{" "}
            {r.offers_fired_total} times, across every batch and every stress
            test.
          </strong>{" "}
          Offering someone a graceful exit is the whole reason this project
          exists, and the safety check in front of it is set so tightly that it
          can never be satisfied by this data. So that path is untested, not
          tested and found wanting. It is the biggest gap here.
        </li>
        <li>
          <strong>
            It asks the wrong people to re-authorise, often.
          </strong>{" "}
          {r.false_reauth_total} of {r.reauth_total} re-authorisation requests
          went to customers whose payment had not actually died. That is the
          failure the issuer-outage stress test was written to catch, and it
          catches it.
        </li>
        <li>
          <strong>It keeps trying after a bank says stop.</strong>{" "}
          {r.attempt_after_terminal_total} times it decided to attempt again on
          a card or account the bank had just confirmed dead. The engine cannot
          yet learn that fact from the decline itself.
        </li>
        <li>
          <strong>Every attempt lands on the same day.</strong> The engine is
          not really choosing <em>when</em> to retry yet, only whether to. That
          also means the two compliance rulebooks it supports behave
          identically on this data, so this run does not tell them apart.
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
        <h2>Four policies, three ways to judge them</h2>
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
            <h2>Every mandate in the batch. Click one to see why.</h2>
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
