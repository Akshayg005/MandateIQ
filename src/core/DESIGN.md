# src/core/ — deterministic primitives only

Nothing in this directory (or in `src/model/` or `src/policy/`) may import
`anthropic`, `openai`, or any other LLM client. Enforced by
`scripts/guard_invariants.py` on every edit.

Nothing here uses `float` for money. Integer paise only. `money.py` is the
only module that formats currency for display.

Nothing here calls `datetime.now()` directly — use `clock.py` so tests can
freeze time. The 24-hour commitment lag is untestable otherwise.

If a task in this directory seems to require an LLM, the task belongs in
`src/llm/` and the result is passed in as a plain value.
