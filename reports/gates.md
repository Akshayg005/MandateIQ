# Day gates

Tick only when the stated condition is actually verified, not when the code
exists. `show_state.py` reads this at every session start, and `checkpoint.py` summarises it into STATE.md at every session end.

- [ ] **D1** baseline produces a number; both guards fire on a deliberate violation; FREEZE_HASH recorded
- [ ] **D2** a real test-mode `payment.failed` lands in the ledger with a classified cause
- [ ] **D3** model beats the ladder on nominal recovery; split is mandate-level; no future-encoding features
- [ ] **D4** reliability diagram roughly diagonal; empirical conformal coverage matches nominal
- [ ] **D5** zero constraint violations across the eval; both profiles produce numbers; compliance-auditor all-VERIFIED
- [ ] **D6** 50 induced kills, zero double-charges, zero lost jobs, ledger complete
- [ ] **D7** off-ramp fires only on singleton conformal sets; golden set passes; no LLM import in core
- [ ] **D8** benchmark table in DECISIONS.md incl. variance column; shadow mode produces a delta log
- [ ] **D9** every number reproducible by one command; at least one regime where we lose, explained
- [ ] **D10** merchant + acquirer views; per-mandate drill-down shows belief, slot, binding constraint, ledger
- [ ] **D11** landing page 60fps on a mid laptop; reduced-motion fallback; canvas-failure fallback
- [ ] **D12** README has "What this can't do" with >=4 items; video under 5:00; three takes max
