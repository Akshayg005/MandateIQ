# POSTMORTEM — what broke during the build

The rubric line is *"Failure recovery — what broke, and what you did about
it."* That is asking about this build, not runtime resilience. Entries are
written **at the moment of breakage**, before the cause is known. Do not
backfill a tidy story, and do not delete an entry because it turned out to
be your own mistake — those are the valuable ones.

Use the `/log-incident` skill. Format:

## Incident 0 — template (delete when the first real one lands)
**When:** Day 0, 21:40
**Symptom:** what was observed, in the terms it was first noticed
**Root cause:** what was actually wrong
**Why it wasn't caught earlier:** the gap in tests, types, or guards
**Fix:** what changed, with the commit hash
**Guard added:** the test or hook that makes this class of bug impossible to
reintroduce — or "none, accepted risk" with the reason
