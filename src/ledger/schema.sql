-- Append-only ledger + committed schedule + lease + plan + mandate lifecycle.
-- Postgres 16. See PLAN_DETAIL.md §3 for the write-ordering protocol and the
-- idempotency-key derivation this schema exists to support.
--
-- `ledger` has no UPDATE path, by construction: every column that needs to
-- change after the fact lives in a different table (committed_schedule,
-- mandate_lifecycle, attempt_lease). Enforcement is at the application layer
-- (src/ledger/store.py never issues UPDATE/DELETE against ledger) plus this
-- file never containing one -- there is deliberately no DB trigger.

-- The Plan that authorised a debit. Created before `ledger` so ledger's FK
-- can reference it. Without this table decision_sha256 hashes something no
-- table stores, and the audit trail is decorative.
CREATE TABLE plan (
  decision_sha256    TEXT        PRIMARY KEY,
  mandate_id         TEXT        NOT NULL,
  cycle_id           INT         NOT NULL,
  profile            TEXT        NOT NULL,
  belief_json        TEXT        NOT NULL,
  conformal_set      TEXT        NOT NULL,
  binding_constraint TEXT,
  solver_version     TEXT        NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ledger (
  ledger_id        BIGSERIAL PRIMARY KEY,
  idempotency_key  TEXT        NOT NULL,
  mandate_id       TEXT        NOT NULL,
  cycle_id         INT         NOT NULL,
  attempt_index    SMALLINT    NOT NULL,      -- 1..4 -- NPCI: 1 original + 3 retries, ever
  action           TEXT        NOT NULL,      -- ATTEMPT | OFFER | REAUTH | STOP
  state            TEXT        NOT NULL,      -- INTENT | SENT | RESULT | FAILED
  amount_paise     BIGINT      NOT NULL,      -- integer paise, never float
  provider_ref     TEXT,                      -- razorpay id, null on INTENT
  outcome          TEXT,                      -- null until RESULT
  decline_class    TEXT,
  reason           TEXT,                      -- UNCONFIRMED | UNRESOLVED_FINAL | ...
  profile          TEXT        NOT NULL,      -- which compliance interpretation authorised this
  payload_sha256   TEXT        NOT NULL,
  decision_sha256  TEXT        NOT NULL REFERENCES plan (decision_sha256),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (amount_paise >= 0),
  CHECK (attempt_index BETWEEN 1 AND 4)
);
CREATE UNIQUE INDEX ledger_intent_once
  ON ledger (idempotency_key) WHERE state = 'INTENT';
CREATE INDEX ledger_key ON ledger (idempotency_key);

-- Committed >=24h ahead (clause 6a). The amount the executor sends is READ
-- from here, never recomputed. A row is never updated in place except to be
-- voided; a changed decision is a NEW row at generation+1.
--
-- committed_at is deliberately NOT DB-defaulted (no `DEFAULT now()`). The
-- executor must derive it from src/core/clock.now() -- the one freezable
-- clock in the codebase -- not from Postgres's own wall clock, or a frozen
-- test loses control over what gets written here. The 24h CHECK below only
-- proves the two application-supplied columns are self-consistent; it
-- cannot prove committed_at was honestly "now" when the row was written --
-- that honesty is an executor-discipline invariant, reviewed at B9, not
-- something this schema can enforce without a trigger.
-- decision_sha256 (B9): added when src/execute/commit.py was written --
-- committed_schedule is the only durable record an executor process (which
-- may be a different process than the one that called solve(), any amount
-- of time later, per the crash-recovery design this whole layer exists
-- for) reads before writing a `ledger` row -- and `ledger.decision_sha256`
-- is NOT NULL REFERENCES plan. Without this column here, attaching the
-- correct plan to a ledger row would mean joining plan and
-- committed_schedule by (mandate_id, cycle_id) and nearest committed_at,
-- which is exactly the kind of timing-heuristic join B1's plan table
-- exists to make unnecessary -- two solve() calls in the same cycle (or
-- the same frozen-clock instant, which tests can and do produce) would
-- make that join ambiguous. A direct FK column is unambiguous. Logged in
-- DECISIONS.md, 2026-08-30, B9 -- schema.sql is not eval/frozen/ and this
-- is an additive column, not a rewrite of anything B1's gate certified
-- (money/clock/ids tests, no UPDATE path on ledger); both still hold.
CREATE TABLE committed_schedule (
  idempotency_key TEXT PRIMARY KEY,
  mandate_id TEXT NOT NULL, cycle_id INT NOT NULL, attempt_index SMALLINT NOT NULL,
  generation SMALLINT NOT NULL DEFAULT 0,
  action TEXT NOT NULL, amount_paise BIGINT NOT NULL,
  profile TEXT NOT NULL,
  decision_sha256 TEXT NOT NULL REFERENCES plan (decision_sha256),
  scheduled_for TIMESTAMPTZ NOT NULL, committed_at TIMESTAMPTZ NOT NULL,
  notification_sent_at TIMESTAMPTZ,
  voided_at TIMESTAMPTZ, void_reason TEXT,
  CHECK (scheduled_for >= committed_at + INTERVAL '24 hours'),
  CHECK (amount_paise >= 0),
  CHECK (attempt_index BETWEEN 1 AND 4)
);
CREATE UNIQUE INDEX committed_one_live_per_slot
  ON committed_schedule (mandate_id, cycle_id, attempt_index)
  WHERE voided_at IS NULL;

-- Append-only mandate state. Current state = latest row by effective_at.
-- This is the route clause 6(c) requires: an opt-out riding on the T-24h
-- notification has to land somewhere the executor will read.
CREATE TABLE mandate_lifecycle (
  event_id     TEXT        PRIMARY KEY,       -- provider event id; also the dedupe key
  mandate_id   TEXT        NOT NULL,
  state        TEXT        NOT NULL,          -- CREATED|ACTIVE|PAUSED|REVOKED|EXPIRED|COMPLETED
  source       TEXT        NOT NULL,          -- WEBHOOK | RECONCILE | INTERNAL
  effective_at TIMESTAMPTZ NOT NULL,
  recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX mandate_lifecycle_current ON mandate_lifecycle (mandate_id, effective_at DESC);

-- Mutable state kept OUT of the ledger. An optimisation over
-- ledger_intent_once, not the concurrency control.
CREATE TABLE attempt_lease (
  idempotency_key TEXT PRIMARY KEY,
  owner TEXT NOT NULL, expires_at TIMESTAMPTZ NOT NULL
);

-- B3: ingest. Event-level idempotency, type-agnostic -- backs dedupe.py
-- only. Deliberately has no decline-specific columns: coupling this to
-- DeclineClass/Cause would force dedupe.py to import src/classify/, which
-- is exactly the scope leak this table exists to avoid.
CREATE TABLE webhook_event (
  event_id    TEXT        PRIMARY KEY,
  event_type  TEXT        NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- B3: the classified landing zone the gate needs. NOT `ledger` -- ledger
-- rows key off decision_sha256 REFERENCES plan, representing this system's
-- OWN decisions to move money, and no plan row can exist before B8's
-- allocator. A bare observed decline has no decision to attach to; forcing
-- it through `ledger` would be a category error. See DECISIONS.md for the
-- B3 gate rebinding this table's name is now part of.
--
-- mandate_id is nullable: a payment.failed body carries no reliable link
-- back to a mandate (Razorpay's Payment entity has no subscription_id), so
-- webhook.py resolves it from payload.payment.entity.notes, falling back to
-- a sibling subscription.entity.id, falling back to NULL. An honest NULL,
-- reported as a rate, is preferred over a guessed mandate_id.
CREATE TABLE ingested_event (
  event_id           TEXT        PRIMARY KEY,
  event_type         TEXT        NOT NULL,
  mandate_id         TEXT,
  provider_ref       TEXT,
  decline_code       TEXT,
  decline_text       TEXT,
  decline_class      TEXT,
  cause_prior        TEXT,                 -- JSON dict[Cause,float] from cause_map.prior()
  -- Which version of decline_taxonomy.classify() / cause_map.prior() wrote
  -- decline_class / cause_prior on THIS row. "the taxonomy will grow all
  -- week" (.claude/skills/new-failure-class/SKILL.md) -- without this, a
  -- future re-read of an old row can't tell which ruleset judged it,
  -- mirroring the exact reason B11's normaliser output must be versioned
  -- before it can touch a belief (PLAN_DETAIL.md B11 gate). Nullable: not
  -- every future writer of this table need touch classification at all.
  taxonomy_version   TEXT,
  prior_version      TEXT,
  amount_paise       BIGINT,
  raw_payload_sha256 TEXT        NOT NULL,
  received_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (amount_paise IS NULL OR amount_paise >= 0)
);
CREATE INDEX ingested_event_mandate ON ingested_event (mandate_id, received_at DESC);

-- B11: one row per (event, normaliser prompt version). Append-only like
-- every other table here -- no UPDATE, by the same application-layer
-- discipline this file's header describes for `ledger` -- but for a
-- DIFFERENT reason than ledger's: a re-normalisation under a NEW
-- normalizer_version (a prompt edit) is a genuinely NEW fact, not a
-- correction of the old one, so the primary key includes it. Overwriting
-- the old row on a prompt change would destroy exactly the audit trail
-- this table exists to hold -- which prompt version produced which verdict,
-- on which historical decline. ON CONFLICT DO NOTHING (src/ledger/store.py)
-- makes a retried write under the SAME version a silent no-op instead.
--
-- This table -- not a column added to ingested_event -- because the
-- normaliser runs strictly AFTER ingest (only on the UNKNOWN it leaves
-- unresolved, decline_taxonomy.py's own docstring), so filling a column on
-- an existing row would require an UPDATE, and because ingested_event's
-- own decline_class must stay the deterministic taxonomy's verdict --
-- overwriting it would destroy the UNKNOWN rate as a reported metric
-- (decline_taxonomy.py: "a reported metric, not a swallowed one").
--
-- This is the durable form of PLAN_DETAIL.md's B11 gate clause 3:
-- "normaliser output is versioned in the ledger before it can touch a
-- belief" -- src/policy/belief.update()'s required source_version
-- parameter is what a caller must read back FROM this table before a
-- normalised decline is allowed to update a Belief at all.
-- confidence: the model's own self-reported confidence for `value`, kept
-- rather than only consumed and discarded by normalizer.py's UNKNOWN
-- override. A verdict a merchant can dispute needs to show WHY, not just
-- what -- "MANDATE_REVOKED, model X, prompt-hash Y" with no confidence is
-- not disputable (payments-domain review, 2026-08-31). NORMALIZE_TOOL
-- requires confidence on every call, so NOT NULL rather than nullable.
CREATE TABLE normalized_decline (
  event_id           TEXT        NOT NULL REFERENCES ingested_event (event_id),
  value              TEXT        NOT NULL,   -- DeclineClass.value
  confidence         DOUBLE PRECISION NOT NULL,
  normalizer_version TEXT        NOT NULL,
  model_id           TEXT        NOT NULL,
  raw_sha256         TEXT        NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (event_id, normalizer_version),
  CHECK (confidence >= 0.0 AND confidence <= 1.0)
);
CREATE INDEX normalized_decline_event ON normalized_decline (event_id, created_at DESC);

-- shadow_ledger (B12) --------------------------------------------------------
--
-- Shadow mode decides WITHOUT executing: it runs the allocator over a batch
-- and records what it WOULD have done beside what the fixed T+1/T+2/T+3
-- ladder would have done. No money moves, no provider is called, and no row
-- in `ledger` or `committed_schedule` is ever written by this path. That
-- separation is the entire point -- PLAN_DETAIL.md section 6 calls B12 a
-- "read-only observer of the B8 policy" -- so this is a distinct table
-- rather than a flag on `ledger`. A nullable `is_shadow` column on the real
-- ledger would put unexecuted decisions one forgotten WHERE clause away
-- from being counted as real money.
--
-- decision_sha256 is deliberately PLAIN TEXT with NO foreign key to `plan`,
-- unlike ledger.decision_sha256 which is NOT NULL REFERENCES plan. Carrying
-- that FK would force shadow mode to write real `plan` rows in order to
-- observe, which is exactly the side effect it exists to avoid. The hash is
-- still recorded so a shadow decision can be reproduced and compared against
-- a later real one; it is a fingerprint here, not a reference.
--
-- Additive DDL, appended after the B11 freeze of the tables above, same
-- precedent as the additive column recorded further up this file. Nothing
-- under eval/frozen/ is touched.
CREATE TABLE shadow_ledger (
  run_id                    TEXT        NOT NULL,  -- one shadow run over one batch
  mandate_id                TEXT        NOT NULL,
  cycle_id                  INTEGER     NOT NULL,
  profile                   TEXT        NOT NULL,  -- Profile.value
  -- what the incumbent ladder would have done
  ladder_action             TEXT        NOT NULL,  -- Action.value
  ladder_slot               INTEGER     NOT NULL,
  ladder_day                INTEGER     NOT NULL,
  ladder_committed_attempts INTEGER     NOT NULL,
  -- what we would have done
  our_action                TEXT        NOT NULL,  -- Action.value
  our_slot                  INTEGER,               -- NULL unless our_action = ATTEMPT
  our_day                   INTEGER,
  binding_constraint        TEXT,
  conformal_set             TEXT        NOT NULL,  -- sorted Cause.value list, comma-joined
  belief_json               TEXT        NOT NULL,
  decision_sha256           TEXT        NOT NULL,  -- fingerprint, NOT a FK -- see above
  divergence                TEXT        NOT NULL,
  agrees                    BOOLEAN     NOT NULL,
  created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, mandate_id, cycle_id),
  CHECK (ladder_slot BETWEEN 2 AND 4),
  CHECK (our_slot IS NULL OR our_slot BETWEEN 2 AND 4),
  -- an ATTEMPT must name a slot and a day; anything else must not
  CHECK ((our_action = 'ATTEMPT') = (our_slot IS NOT NULL AND our_day IS NOT NULL))
);
CREATE INDEX shadow_ledger_run ON shadow_ledger (run_id, divergence);
