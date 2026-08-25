# src/model/ — statistical core

No LLM imports. No float money.

The model is a **discrete-time competing-risks** survival model:
person-period reshape → multinomial logit cause-specific hazards →
cumulative incidence per cause → isotonic calibration → split conformal.

Three rules that are easy to get wrong and expensive to get wrong:

1. **Censoring is not a missing value.** A mandate that exhausts its four
   attempts without resolving is right-censored, not a negative label.
   Dropping those rows biases everything toward fast-resolving cases.
2. **No feature may encode the future.** Anything computed from an attempt
   at slot k must not appear in the row for slot < k. Leakage here is
   invisible and inflates every number in the report.
3. **Cause-specific vs subdistribution hazards answer different questions.**
   Cause-specific explains mechanism to the merchant. Subdistribution
   predicts this mandate's risk for the allocator. Do not mix them up.

Do NOT reach for DeepHit, transformers, or deep survival models. Insufficient
data, and it destroys the architectural argument.
