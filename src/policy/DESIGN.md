# src/policy/ — the decision layer

No LLM imports. No float money.

Every constant here carries its RBI clause reference in a docstring. If you
cannot cite it, it does not belong here — it is a tuning parameter and goes
in a config file instead.

The allocator solves EXACTLY via backward induction over 4 slots. Do not
introduce RL, do not introduce a heuristic "good enough" search. The state
space is small; solve it properly and say so.

`offramp.py` never executes a cancellation. It constructs an OFFER.
