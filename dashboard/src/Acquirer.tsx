/**
 * Acquirer view — clause 10(c) rendered as product.
 *
 * 10(c) makes the ACQUIRER responsible for its merchants' e-mandate
 * compliance, which is the answer to "why would a payment aggregator ship
 * this rather than a merchant". So this view is not a second copy of the
 * merchant numbers: it is the compliance surface. Attempts against the NPCI
 * cap, mandates sitting above the AFA-free cliff, opt-outs, re-auth requests
 * that went to live instruments, and the gate's measured coverage against its
 * own target.
 *
 * ONE SEED AT A TIME, NEVER AN AVERAGE. B13 established that the per-seed
 * sign test supersedes the averaged table, and averaging 8 seeds here would
 * be computing a number the report does not publish. The seed selector shows
 * cells as the sweep wrote them.
 */
import { useMemo, useState } from "react";
import type { Regimes } from "./data";
import { pct, ratio } from "./data";

const NPCI_CAP = 4;

export default function Acquirer({ regimes }: { regimes: Regimes }) {
  const [regime, setRegime] = useState("baseline");
  const [profile, setProfile] = useState("strict");
  const [seed, setSeed] = useState(regimes.seeds[0] ?? 0);

  const cells = useMemo(
    () =>
      regimes.cells.filter(
        (c) => c.regime === regime && c.profile === profile && c.seed === seed,
      ),
    [regimes, regime, profile, seed],
  );

  const spec = regimes.regimes[regime];
  const engineCells = cells.filter((c) => c.policy === "engine");
  const violations = cells.flatMap((c) =>
    c.violations.map((v) => ({ arm: c.arm, policy: c.policy, text: v })),
  );

  return (
    <>
      <section className="panel">
        <h2>portfolio compliance posture</h2>
        <div className="body">
          <div className="filters">
            <label htmlFor="rg">regime</label>
            <select id="rg" value={regime} onChange={(e) => setRegime(e.target.value)}>
              {Object.keys(regimes.regimes).map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
            <label htmlFor="pf">profile</label>
            <select id="pf" value={profile} onChange={(e) => setProfile(e.target.value)}>
              {regimes.profiles.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <label htmlFor="sd">seed</label>
            <select
              id="sd"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            >
              {regimes.seeds.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <span className="count">{cells.length} cells</span>
          </div>

          {spec && (
            <>
              <p className="note" style={{ marginTop: 0 }}>
                <strong>{regime}.</strong> {spec.story}
              </p>
              <p className="note">
                <strong>Pre-registered hypothesis.</strong> {spec.hypothesis}
              </p>
              {spec.approximation && (
                <p className="note">
                  <strong>What this regime cannot show.</strong>{" "}
                  {spec.approximation}
                </p>
              )}
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>attempt budget and the AFA cliff</h2>
        <div className="body">
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>arm</th>
                  <th>policy</th>
                  <th className="num">mandates</th>
                  <th className="num">attempts</th>
                  <th className="num">att/mandate</th>
                  <th className="num">above AFA</th>
                  <th className="num">re-auth</th>
                  <th className="num">offer</th>
                  <th className="num">stop</th>
                  <th className="num">opted out</th>
                  <th className="num">preserved</th>
                </tr>
              </thead>
              <tbody>
                {cells.map((c) => {
                  const per = c.n_mandates ? c.attempts_spent / c.n_mandates : 0;
                  return (
                    <tr key={`${c.arm}-${c.policy}`}>
                      <td>{c.arm}</td>
                      <td className="mono">{c.policy}</td>
                      <td className="num">{c.n_mandates}</td>
                      <td className="num">{c.attempts_spent}</td>
                      <td className={per > NPCI_CAP ? "num bad" : "num"}>
                        {per.toFixed(2)}
                      </td>
                      <td className="num">{c.n_above_afa}</td>
                      <td className="num">{c.n_reauth}</td>
                      <td className="num">{c.n_offer}</td>
                      <td className="num">{c.n_stop}</td>
                      <td className="num">{c.opted_out}</td>
                      <td className="num">
                        {c.mandates_preserved}/{c.n_mandates}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="note">
            NPCI allows 1 original + 3 retries = <strong>4 attempts, ever</strong>.
            &ldquo;att/mandate&rdquo; is the cell&rsquo;s mean; the per-mandate cap is
            enforced in the allocator and no cell may exceed it. Mandates above
            the AFA-free limit (₹15,000, or ₹1,00,000 for the clause 8(b)
            categories) are not silently retryable — they belong on the re-auth
            path.
          </p>
        </div>
      </section>

      <section className="panel">
        <h2>where the engine is wrong, by its own criteria</h2>
        <div className="body">
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>arm</th>
                  <th className="num">re-auth issued</th>
                  <th className="num">to live instruments</th>
                  <th className="num">false re-auth rate</th>
                  <th className="num">attempt after terminal</th>
                  <th className="num">missed recovery</th>
                  <th className="num">false off-ramp</th>
                </tr>
              </thead>
              <tbody>
                {engineCells.map((c) => (
                  <tr key={c.arm}>
                    <td>{c.arm}</td>
                    <td className="num">{c.n_reauth}</td>
                    <td className="num bad">{c.false_reauth_count}</td>
                    <td className="num">{ratio(c.false_reauth_count, c.n_reauth)}</td>
                    <td className="num">{c.n_attempt_after_terminal}</td>
                    <td className="num">{c.missed_recovery_count}</td>
                    <td className="num">{c.false_offramp_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            A re-auth request sent to an instrument that is in fact alive is a
            real cost to a real customer, so it is reported next to the
            recoveries rather than under them.{" "}
            <strong>Missed recovery is an upper bound, not a point estimate</strong>:
            the counterfactual always lands inside the days 1–5 salary window.
          </p>
        </div>
      </section>

      <section className="panel">
        <h2>conformal gate — measured coverage, not claimed</h2>
        <div className="body">
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>arm</th>
                  <th className="num">queries</th>
                  <th className="num">marginal coverage</th>
                  <th className="num">target</th>
                  <th className="num">mean set size</th>
                  <th className="num">singleton rate</th>
                  <th className="num">{"{WONT_PAY}"} rate</th>
                  <th>worst class</th>
                </tr>
              </thead>
              <tbody>
                {engineCells.map((c) => {
                  const target = 1 - regimes.gate_diagnostics.alpha;
                  const perClass = Object.entries(c.coverage_per_class);
                  const worst = perClass.length
                    ? perClass.reduce((a, b) => (a[1] <= b[1] ? a : b))
                    : null;
                  const under =
                    c.coverage_marginal !== null && c.coverage_marginal < target;
                  return (
                    <tr key={c.arm}>
                      <td>{c.arm}</td>
                      <td className="num">{c.coverage_n}</td>
                      <td className={under ? "num bad" : "num"}>
                        {pct(c.coverage_marginal, 1)}
                      </td>
                      <td className="num dim">{pct(target, 1)}</td>
                      <td className="num">
                        {c.mean_set_size === null ? "--" : c.mean_set_size.toFixed(2)}
                      </td>
                      <td className="num">{pct(c.singleton_rate, 1)}</td>
                      <td className="num">{pct(c.singleton_wont_pay_rate, 1)}</td>
                      <td className="num">
                        {worst ? `${worst[0]} ${pct(worst[1], 1)}` : "--"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="note">
            The gate is calibrated once, on baseline, and reused under every
            regime — a regime breaks exchangeability by construction, so
            coverage is <em>measured</em> per cell rather than assumed.{" "}
            <strong>It under-covers.</strong> The {"{WONT_PAY}"} singleton rate is
            zero everywhere, which is why the off-ramp never fires.
          </p>
        </div>
      </section>

      <section className="panel">
        <h2>invariant violations recorded by the sweep</h2>
        <div className="body">
          {violations.length === 0 ? (
            <p className="note good" style={{ margin: 0 }}>
              None in these cells. A violation here would mean a policy
              committed above a mandate ceiling or the allocator failed to
              solve — the sweep records them rather than dropping them.
            </p>
          ) : (
            <ul>
              {violations.map((v, i) => (
                <li key={i} className="mono">
                  <span className="dim">
                    {v.arm}/{v.policy}:
                  </span>{" "}
                  {v.text}
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </>
  );
}
