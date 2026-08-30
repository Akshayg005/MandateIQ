"""LLM edge: language-only tools, never decision-core logic.

This package interfaces with Gemini for:
  - Decline normalization (transient linguistics)
  - Exit intent scoring (weak signal, gated by conformal prediction)
  - Batch narratives (once per batch, merchant-facing prose)

All structured output goes through forced function calling (tools.py).
No decision is made by an LLM -- this package returns scores and labels
that feed into src/policy/gate.py and src/policy/offramp.py.
"""
