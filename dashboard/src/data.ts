/**
 * Types and loaders for the three report artifacts.
 *
 * THE RULE THIS FILE ENFORCES: the dashboard renders `reports/` and computes
 * nothing. There is no arithmetic in this module and none anywhere downstream
 * of it beyond counting rows and grouping them.
 *
 * In particular there is NO CURRENCY FORMATTING here, and there must never
 * be. `src/core/money.py` is the only place in this project allowed to render
 * a rupee figure (CLAUDE.md invariant 2), and `scripts/guard_invariants.py`
 * cannot scan TypeScript -- so a `formatRupees()` written in this directory
 * would be the exact violation the guard was widened to catch in
 * eval/report.py, in a language where nothing would catch it. Every money
 * value therefore arrives pre-formatted from Python as a sibling `amount` /
 * `recovered` string, and paise fields are used only for comparisons.
 */

export type Cause = "CANT_PAY_NOW" | "CANT_PAY_EVER" | "WONT_PAY";

export interface Manifest {
  generated_at: string;
  git_sha: string;
  freeze_hash: string;
  staged: string[];
  missing: string[];
}

/** One policy's three bars, as eval/report.py renders them. */
export interface Bars {
  recovered_paise: number;
  recovered_pct: string;
  recovered: string;
  attempts_spent: number;
  attempts_per_recovery: number | null;
  mandates_preserved: string;
}

export interface SignTest {
  preserves_more: number;
  preserves_fewer?: number;
  recovers_more: number;
  recovers_less?: number;
  spends_fewer_attempts: number;
  spends_more_attempts?: number;
}

export interface Results extends Bars {
  headline_cell: string;
  seed: number;
  seeds: number[];
  paired_comparisons: number;
  sign_test: { vs_ladder: SignTest; vs_one_shot: SignTest };
  gate_kind: string;
  regimes_where_we_lose: string[];
  offers_fired_total: number;
  /**
   * R5. The off-ramp's own error costs, so the Merchant view reads the PAIR
   * rather than a bare count. `offramp_scored_total` is the exact
   * denominator `false_offramp_total` was measured against -- never
   * `offers_fired_total`, which can differ if a post-terminal re-solve ever
   * returns OFFER. Optional: an older results.json (pre-R5) has neither.
   */
  offramp_scored_total?: number;
  false_offramp_total?: number;
  true_offramp_total?: number;
  false_reauth_total: number;
  reauth_total: number;
  attempt_after_terminal_total: number;
  baseline: Bars;
  engine_permissive: Bars;
  reference_null: Bars;
  reference_one_shot: Bars;
}

/** One (regime, arm, profile, policy, seed) cell of the 1024-cell sweep. */
export interface Cell {
  regime: string;
  arm: string;
  profile: string;
  policy: string;
  seed: number;
  gate_kind: string;
  n_mandates: number;
  billable_paise: number;
  recovered_paise: number;
  attempts_spent: number;
  mandates_preserved: number;
  recovered: number;
  dead: number;
  opted_out: number;
  censored: number;
  iatrogenic_failures: number;
  n_attempt: number;
  n_offer: number;
  n_reauth: number;
  n_stop: number;
  n_above_afa: number;
  n_attempt_after_terminal: number;
  missed_recovery_count: number;
  false_offramp_count: number;
  false_reauth_count: number;
  coverage_marginal: number | null;
  coverage_n: number;
  singleton_rate: number | null;
  singleton_wont_pay_rate: number | null;
  mean_set_size: number | null;
  coverage_per_class: Record<string, number>;
  violations: string[];
}

export interface RegimeSpec {
  story: string;
  hypothesis: string;
  approximation: string;
  overlay: unknown;
}

export interface Regimes {
  schema: number;
  seeds: number[];
  gate_kind: string;
  gate_diagnostics: { alpha: number; n_calib: number; calib_seed: number };
  arms: string[];
  profiles: string[];
  regimes: Record<string, RegimeSpec>;
  cells: Cell[];
}

/** One solve() call, as the allocator produced it. The drill-down's subject. */
export interface Decision {
  index: number;
  action: "ATTEMPT" | "REAUTH" | "OFFER" | "STOP";
  chosen_slot: number | null;
  chosen_day: number | null;
  amount_paise: number | null;
  amount: string | null;
  belief: Record<Cause, number>;
  belief_json: string;
  conformal_set: Cause[];
  binding_constraint: string | null;
  solver_version: string;
  decision_sha256: string;
  /**
   * The actual pause -> downgrade -> cancel menu, present iff `action` is
   * "OFFER" (R5). Before R5 src/policy/offramp.py had no caller anywhere,
   * so a chosen OFFER produced no Offer object and this panel could only
   * say that something was offered, never what. The order is load-bearing:
   * least drastic and most reversible first, so a customer who only needed
   * a pause is never shown "cancel" as the headline option.
   */
  offer: Offer | null;
  outcome: string | null;
}

export interface OffRampStep {
  kind: "PAUSE" | "DOWNGRADE" | "CANCEL";
  description: string;
}

export interface Offer {
  expires_on_day: number;
  steps: OffRampStep[];
}

/** A ledger row exactly as src/ledger/store.py read it back. */
export interface LedgerRow {
  idempotency_key: string;
  attempt_index: number;
  action: string;
  state: "INTENT" | "SENT" | "RESULT" | "FAILED";
  amount_paise: number;
  amount: string;
  outcome: string | null;
  decline_class: string | null;
  reason: string | null;
  provider_ref: string | null;
  profile: string;
  decision_sha256: string;
}

export interface MandateRecord {
  mandate_id: string;
  cycle_id: number;
  category: string;
  amount_paise: number;
  amount: string;
  ceiling_paise: number;
  ceiling: string;
  afa_limit_paise: number;
  afa_limit: string;
  above_afa: boolean;
  profile: string;
  decisions: Decision[];
  attempts_spent: number;
  final_action: string | null;
  final_outcome: string | null;
  ledger: LedgerRow[];
  ledger_note: string | null;
  ground_truth: { true_cause: Cause };
}

export interface Mandates {
  schema: number;
  cell: { regime: string; arm: string; profile: string; seed: number };
  cycle_start: string;
  ledger_provenance: string;
  mandates: MandateRecord[];
}

export interface Data {
  manifest: Manifest;
  results: Results;
  regimes: Regimes;
  mandates: Mandates | null;
}

async function getJson<T>(name: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export async function loadAll(): Promise<Data> {
  const [manifest, results, regimes] = await Promise.all([
    getJson<Manifest>("manifest.json"),
    getJson<Results>("results.json"),
    getJson<Regimes>("regimes.json"),
  ]);
  // mandates.json needs Postgres to regenerate, so a reviewer can legitimately
  // have the aggregate views without it. Absence is reported on the page, not
  // crashed on.
  let mandates: Mandates | null = null;
  try {
    mandates = await getJson<Mandates>("mandates.json");
  } catch {
    mandates = null;
  }
  return { manifest, results, regimes, mandates };
}

/** Percent, for a value that is already a rate in [0,1]. Not money. */
export function pct(x: number | null, digits = 1): string {
  return x === null || Number.isNaN(x) ? "--" : `${(x * 100).toFixed(digits)}%`;
}

export function ratio(a: number, b: number): string {
  return b === 0 ? "--" : `${((a / b) * 100).toFixed(1)}%`;
}
