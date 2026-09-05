# Adding a failure class to the taxonomy

Decline strings are not standardised across issuers, so the taxonomy grows
continuously. Adding a class touches six places; missing one produces a class
that exists in the mapper but is invisible to the model.

Checklist -- do all six:

1. **Taxonomy** -- add the code and its canonical string variants to
   `src/classify/decline_taxonomy.py`.
2. **Cause mapping** -- map it in `src/classify/cause_map.py` to exactly one
   of `CANT_PAY_NOW` / `CANT_PAY_EVER` / `WONT_PAY`. If it is genuinely
   ambiguous, map it to `CANT_PAY_NOW` (the safe default: we retry rather
   than offer an exit) and add a comment saying why.
3. **Simulator** -- add a generator for it in `eval/simulator.py` with a
   realistic base rate. Do NOT touch `eval/frozen/` -- the class goes in the
   generator code, its prevalence comes from the frozen config.
4. **Model features** -- confirm `person_period.py` one-hot encodes the new
   class and that an unseen class at inference maps to an explicit "unknown"
   bucket rather than silently becoming all-zeros.
5. **Golden set** -- add at least two real-world string variants to
   `eval/golden/declines.jsonl` with the expected class.
6. **Test** -- a test asserting the raw string maps to the right cause.

Then run `.\run.ps1 lint` and `.\run.ps1 eval` (`./run.sh` on POSIX), and
confirm the change did not move the headline numbers in a way you cannot
explain.
