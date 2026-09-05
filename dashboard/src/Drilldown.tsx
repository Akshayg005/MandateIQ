/**
 * The per-mandate drill-down. B14's gate names five things it must show:
 * belief, chosen slot, binding constraint, conformal set, ledger trail.
 * Each has its own labelled block below, and each renders what the artifact
 * says rather than a tidied version of it.
 *
 * Three of those five are degenerate in the current engine, and this view is
 * deliberately built to make that visible instead of flattering:
 *
 *  - the chosen slot is ALWAYS day 2. There is no timing discrimination, so
 *    "why this day" has no interesting answer yet, and the panel says so.
 *  - the binding constraint is null on 299 of 316 decisions -- most choices
 *    are genuine value comparisons, not forced ones.
 *  - the conformal set is often, but NOT always, all three causes. Roughly
 *    a third of decisions are singletons. The {WONT_PAY} singleton -- the
 *    one that opens the off-ramp -- appears in two quite different places,
 *    and confusing them would misread the panel:
 *
 *      1. On a LIVE decision, where it actually fires OFFER (R5,
 *         2026-09-05). This export runs under the same synthetic WONT_PAY
 *         channel the published grid does -- a channel that reads the
 *         simulator's privileged true cause, disclosed as fabricated in
 *         reports/regimes.md and README's "What this can't do". This
 *         comment previously said "OFFER = 0 anyway"; that was true before
 *         R5 and is not true now, so it has been rewritten rather than
 *         left standing.
 *      2. On a RETROSPECTIVE post-opt-out re-solve (R2, 2026-09-04), where
 *         belief was already collapsed to near-certain WONT_PAY because
 *         the customer had left. OFFER cannot fire there: clause 6(c)
 *         denies every action but STOP once opted_out is true, singleton
 *         or not. A WONT_PAY singleton next to an OPTED_OUT binding
 *         constraint is the record of a decision already made, never an
 *         off-ramp opportunity missed.
 *
 *    The aggregate coverage/singleton RATE shown elsewhere (Acquirer view)
 *    excludes the retrospective queries entirely, which is why it and an
 *    individual drill-down can legitimately disagree. The set line renders
 *    the two causes that were EXCLUDED as struck-through, so an all-three
 *    set reads as "excluded nothing" rather than as a confident answer.
 *  - an OFFER decision now carries the actual pause/downgrade/cancel menu,
 *    rendered below the decision. The system offers; the customer decides
 *    (invariant 6). Nothing here executes any of those steps.
 */
import type { Cause, Decision, LedgerRow, MandateRecord } from "./data";
import { BindingConstraint, ConformalSet, Slot } from "./glossary";

const CAUSES: Cause[] = ["CANT_PAY_NOW", "CANT_PAY_EVER", "WONT_PAY"];

function actionTag(action: string) {
  const cls = action.toLowerCase();
  return <span className={`tag ${cls}`}>{action}</span>;
}

function BeliefBars({ d }: { d: Decision }) {
  return (
    <div>
      {CAUSES.map((c) => {
        const p = d.belief[c] ?? 0;
        const inSet = d.conformal_set.includes(c);
        return (
          <div className="belief" key={c}>
            <span className={inSet ? "" : "dim"}>{c}</span>
            <span className="track">
              <span
                className={inSet ? "fill inset" : "fill"}
                style={{ width: `${Math.round(p * 100)}%` }}
              />
            </span>
            <span className="p">{p.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

function ConformalLine({ d }: { d: Decision }) {
  if (d.conformal_set.length === 0) {
    return (
      <span className="setline">
        <span className="dim">
          {"{ }"} — empty set: no cause clears alpha. A real conformal answer,
          not a missing value.
        </span>
      </span>
    );
  }
  return (
    <span className="setline">
      {"{ "}
      {CAUSES.filter((c) => d.conformal_set.includes(c)).map((c, i) => (
        <span className="in" key={c}>
          {i > 0 ? ", " : ""}
          {c}
        </span>
      ))}
      {" }"}
      {d.conformal_set.length < 3 && (
        <>
          {"  excluded: "}
          {CAUSES.filter((c) => !d.conformal_set.includes(c)).map((c, i) => (
            <span className="out" key={c}>
              {i > 0 ? ", " : ""}
              {c}
            </span>
          ))}
        </>
      )}
      {d.conformal_set.length === 3 && (
        <span className="dim">{"  excluded: nothing — the set is uninformative here"}</span>
      )}
    </span>
  );
}

function DecisionBlock({ d }: { d: Decision }) {
  return (
    <div className="decision">
      <header>
        {actionTag(d.action)}
        <span className="slot">
          {d.chosen_slot !== null ? (
            <>
              <Slot /> {d.chosen_slot} · day {d.chosen_day}
            </>
          ) : (
            "no slot spent"
          )}
        </span>
        <span className="dim mono">
          {d.outcome ? `→ ${d.outcome}` : "not executed"}
        </span>
      </header>

      <BeliefBars d={d} />

      <dl className="kv" style={{ marginTop: 8 }}>
        <dt>
          <ConformalSet />
        </dt>
        <dd>
          <ConformalLine d={d} />
        </dd>
        <dt>
          <BindingConstraint />
        </dt>
        <dd>
          {d.binding_constraint ?? (
            <span className="dim">none — an unforced value comparison</span>
          )}
        </dd>
        <dt>amount</dt>
        <dd>{d.amount ?? <span className="dim">—</span>}</dd>
        <dt>solver</dt>
        <dd className="dim">{d.solver_version}</dd>
        <dt>decision_sha256</dt>
        <dd className="dim">{d.decision_sha256}</dd>
      </dl>

      {d.offer && (
        <div className="offer">
          <div className="dim">
            {"the off-ramp offered, in order — the customer chooses; this system never cancels"}
          </div>
          <ol>
            {d.offer.steps.map((s) => (
              <li key={s.kind}>
                <strong>{s.kind}</strong> — {s.description}
              </li>
            ))}
          </ol>
          <div className="dim">{`expires after day ${d.offer.expires_on_day}`}</div>
        </div>
      )}

      {d.action === "ATTEMPT" && d.outcome === null && (
        <p className="note">
          <strong>Post-terminal re-solve.</strong> The allocator was asked
          &ldquo;the instrument is dead, now what?&rdquo; and answered ATTEMPT.
          It was never committed and never sent — there is a <code>plan</code>{" "}
          row but no <code>committed_schedule</code> row and no ledger trail.
          This is a disclosed defect in the belief layer (B7/B8), shown rather
          than hidden.
        </p>
      )}
    </div>
  );
}

function LedgerTable({ rows }: { rows: LedgerRow[] }) {
  if (rows.length === 0) {
    return (
      <p className="note">
        No ledger rows — this mandate never spent a slot, so no debit was ever
        committed or sent.
      </p>
    );
  }
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>state</th>
            <th>action</th>
            <th className="num">amount</th>
            <th>outcome</th>
            <th>decline class</th>
            <th>reason</th>
            <th>idempotency key</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.idempotency_key}-${r.state}-${i}`}>
              <td className="num">{r.attempt_index}</td>
              <td className="mono">{r.state}</td>
              <td className="mono">{r.action}</td>
              <td className="num">{r.amount}</td>
              <td className="mono">{r.outcome ?? <span className="dim">—</span>}</td>
              <td className="mono">{r.decline_class ?? <span className="dim">—</span>}</td>
              <td className="mono">{r.reason ?? <span className="dim">—</span>}</td>
              <td className="mono dim">{r.idempotency_key.slice(0, 16)}…</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Drilldown({ m }: { m: MandateRecord }) {
  return (
    <div className="split">
      <section className="panel">
        <h2>
          {m.mandate_id} · decisions ({m.decisions.length})
        </h2>
        <div className="body">
          <dl className="kv" style={{ marginBottom: 12 }}>
            <dt>category</dt>
            <dd>{m.category}</dd>
            <dt>amount</dt>
            <dd>{m.amount}</dd>
            <dt>mandate ceiling</dt>
            <dd>{m.ceiling} <span className="dim">(clause 4c)</span></dd>
            <dt>AFA-free limit</dt>
            <dd>
              {m.afa_limit}{" "}
              <span className="dim">
                {m.afa_limit_paise > 1500000 ? "(clause 8b, elevated)" : "(clause 8a)"}
              </span>
            </dd>
            <dt>above the cliff</dt>
            <dd className={m.above_afa ? "bad" : ""}>
              {m.above_afa ? "YES — re-auth path, not silent retry" : "no"}
            </dd>
            <dt>profile</dt>
            <dd>{m.profile}</dd>
            <dt>attempts spent</dt>
            <dd>{m.attempts_spent} / 4 <span className="dim">(NPCI)</span></dd>
            <dt>final outcome</dt>
            <dd>{m.final_outcome ?? <span className="dim">—</span>}</dd>
            <dt>true cause</dt>
            <dd>
              {m.ground_truth.true_cause}{" "}
              <span className="dim">
                — simulator ground truth; the engine never sees this
              </span>
            </dd>
          </dl>

          {m.decisions.map((d) => (
            <DecisionBlock d={d} key={d.index} />
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>ledger trail</h2>
        <div className="body">
          <LedgerTable rows={m.ledger} />
          {m.ledger_note && <p className="note">{m.ledger_note}</p>}
          <p className="note">
            Rows written by <code>src/execute/commit.py</code> and{" "}
            <code>src/execute/executor.py</code> into the real schema — real
            idempotency keys, real INTENT → SENT → RESULT ordering, real{" "}
            <code>plan</code> foreign key.{" "}
            <span className="tag sim">simulated provider</span> Only the
            issuer&rsquo;s answer comes from the frozen simulator; decline
            strings carry a <code>[simulated:</code> marker so they can never be
            mistaken for observed issuer text.
          </p>
        </div>
      </section>
    </div>
  );
}
