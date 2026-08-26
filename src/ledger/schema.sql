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
CREATE TABLE committed_schedule (
  idempotency_key TEXT PRIMARY KEY,
  mandate_id TEXT NOT NULL, cycle_id INT NOT NULL, attempt_index SMALLINT NOT NULL,
  generation SMALLINT NOT NULL DEFAULT 0,
  action TEXT NOT NULL, amount_paise BIGINT NOT NULL,
  profile TEXT NOT NULL,
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
