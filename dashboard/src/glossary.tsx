/**
 * Every term this dashboard uses that a non-specialist would not know.
 *
 * This view was built for a reviewer who already knows the domain, and it
 * shows: "conformal gate — measured coverage, not claimed" is a correct
 * heading and an unreadable one. The numbers do not get simpler, but nobody
 * should have to already know the vocabulary to find out what they mean.
 *
 * HOUSE STYLE, same as the landing page's glossary:
 *   - Explain the thing, not the jargon for the thing.
 *   - No term is defined using another undefined term.
 *   - Give the number where there is one.
 *   - Where the honest answer is unflattering, it goes in the definition.
 */
import { Explain } from "./Explain";

/* --- the evaluation grid -------------------------------------------------- */

export function Regime() {
  return (
    <Explain term="regime" wide>
      A stress scenario the whole batch is run under — a payday delay, an
      issuer outage, a festival spending spike. Each one is a set of overrides
      on the frozen simulator, and each was written down with its expected
      outcome <em>before</em> it was run, so a result cannot be reinterpreted
      after the fact.
    </Explain>
  );
}

export function Arm() {
  return (
    <Explain term="arm" wide>
      How wrong the simulator is allowed to be about the world.{" "}
      <em>nominal</em> is the world the model assumes.{" "}
      <em>misspecified</em> breaks that assumption on purpose.{" "}
      <em>coupled</em> makes customers share a bank balance, so one
      customer&rsquo;s retry can cause another&rsquo;s failure.
    </Explain>
  );
}

export function Profile() {
  return (
    <Explain term="profile" wide>
      Which reading of the RBI rules is being enforced. The circular never says
      whether a <em>retry</em> needs its own fresh 24-hour notice, so both
      readings ship: <em>strict</em> assumes it does, <em>permissive</em>{" "}
      assumes it does not. On this evaluation the two produce identical
      numbers, because nothing here discriminates on timing — that is a defect
      in the compliance model, not a finding about the policy.
    </Explain>
  );
}

export function Seed() {
  return (
    <Explain term="seed" wide>
      The starting number for the random generator that builds a batch of
      customers. The same seed always produces the same batch, so any figure
      here can be reproduced exactly. Eight seeds are run; this page shows one
      at a time rather than an average, because an average would be a number
      the report does not publish.
    </Explain>
  );
}

export function Cell() {
  return (
    <Explain term="cells" wide>
      One row of the evaluation grid: a single combination of regime, arm,
      profile, policy and seed. The full sweep is 1,024 of them.
    </Explain>
  );
}

/* --- the rules ------------------------------------------------------------ */

export function NpciCap() {
  return (
    <Explain term="4 attempts, ever" wide>
      India&rsquo;s payment network allows one original charge plus three
      retries per billing cycle — four in total, not four per week. Once spent,
      nothing more can be tried this cycle. The per-customer cap is enforced in
      the allocator, so no cell can exceed it; the column here is the
      cell&rsquo;s mean, which is why it reads below four.
    </Explain>
  );
}

export function Afa() {
  return (
    <Explain term="AFA-free limit" wide>
      Above a certain amount the customer has to authenticate the payment
      themselves — Additional Factor of Authentication — so it cannot be
      charged silently. The limit is ₹15,000 per transaction, or ₹1,00,000 for
      insurance premiums, mutual fund subscriptions and credit card bills.
      Anything above it belongs on the re-authorisation path, not on a retry.
    </Explain>
  );
}

export function OptedOut() {
  return (
    <Explain term="opted out" wide>
      The customer used the cancel button that RBI requires in every
      pre-transaction notice — either for that one payment or for the whole
      mandate. It is counted as its own outcome and never folded into
      &ldquo;declined&rdquo;, because a decline is a bank saying no and this is
      a customer saying no.
    </Explain>
  );
}

export function Reauth() {
  return (
    <Explain term="re-auth" wide>
      A request asking the customer to re-authorise the mandate — used when the
      instrument looks dead (expired card, closed account) or when the amount
      is above the limit that can be charged without them. It is the correct
      action for a dead instrument and a real annoyance for a live one, which
      is why the next table counts the ones that went to the wrong place.
    </Explain>
  );
}

export function Offer() {
  return (
    <Explain term="offer" wide>
      An exit offered to a customer who appears to want out: pause, then
      downgrade, then cancel. The system never cancels anything itself.{" "}
      <strong>This column is 0 in every cell.</strong> The safety gate below
      never opened on this data, so the off-ramp is untested rather than tested
      and found wanting.
    </Explain>
  );
}

/* --- the model ------------------------------------------------------------ */

export function Belief() {
  return (
    <Explain term="belief" wide>
      The current estimate of how likely each of the three explanations is —
      can&rsquo;t pay now, can&rsquo;t pay ever, won&rsquo;t pay — for one
      customer at one moment. It starts from what the bank said and is updated
      after every attempt.
    </Explain>
  );
}

export function ConformalSet() {
  return (
    <Explain term="conformal set" wide>
      Rather than one confident answer, the model returns the <em>set</em> of
      explanations it cannot rule out at a chosen confidence level (95% here).
      Often that is all three, which means it ruled out nothing. An exit is
      only ever offered when the set has shrunk to exactly one — &ldquo;wants
      to leave&rdquo; — which on this data never happens.
    </Explain>
  );
}

export function Coverage() {
  return (
    <Explain term="marginal coverage" wide>
      How often the set above actually contained the true answer, measured
      rather than assumed. The target is 95%. Falling below it means the sets
      are too small and the guarantee is not being honoured; well above it
      means they are too big to be useful.
    </Explain>
  );
}

export function BindingConstraint() {
  return (
    <Explain term="binding constraint" wide>
      Which rule, if any, forced the decision rather than the maths choosing
      it — the attempt budget being exhausted, the amount sitting above the
      authentication limit, the mandate&rsquo;s own ceiling. It is empty on
      most decisions, which means they were genuine value comparisons and not
      forced moves.
    </Explain>
  );
}

export function Slot() {
  return (
    <Explain term="slot" wide>
      Which of the four allowed attempts this is, and which day it lands on.{" "}
      <strong>Every attempt here lands on day 2</strong> — there is no timing
      discrimination in the current engine, so &ldquo;why this day&rdquo; has
      no interesting answer yet.
    </Explain>
  );
}

/* --- the error columns ---------------------------------------------------- */

export function FalseReauth() {
  return (
    <Explain term="false re-auth" wide>
      A re-authorisation request sent to an instrument that was in fact alive —
      the customer could have paid and was asked to re-enrol instead. It is a
      real cost to a real person, so it is reported beside the recoveries
      rather than underneath them.
    </Explain>
  );
}

export function AttemptAfterTerminal() {
  return (
    <Explain term="attempt after terminal" wide>
      The allocator asking to retry an instrument the issuer has already
      confirmed dead. It means the belief layer cannot conclude &ldquo;this
      instrument is gone&rdquo; from a bank message that says exactly that.
    </Explain>
  );
}

export function MissedRecovery() {
  return (
    <Explain term="missed recovery" wide>
      Money that a more aggressive policy would have collected and this one did
      not. It is an <strong>upper bound, not a point estimate</strong>: the
      comparison always lands inside the days 1–5 window when salaries arrive,
      which is the most favourable possible assumption for the alternative.
    </Explain>
  );
}

export function FalseOfframp() {
  return (
    <Explain term="false off-ramp" wide>
      An exit offered to somebody who would have kept paying — the exact harm
      this project exists to avoid. It reads 0 everywhere, and that is{" "}
      <strong>not evidence of safety</strong>: no exit was ever offered at all,
      so nothing could have been offered wrongly.
    </Explain>
  );
}

export function Iatrogenic() {
  return (
    <Explain term="iatrogenic failures" wide>
      Failures the system caused itself: one customer&rsquo;s retry drains a
      shared balance and a different customer&rsquo;s payment then fails. Only
      measurable in the <em>coupled</em> arm, where customers share an account.
    </Explain>
  );
}

/* --- provenance ----------------------------------------------------------- */

export function FreezeHash() {
  return (
    <Explain term="freeze hash" wide>
      The commit that froze the evaluation protocol — the simulator, the
      scoring rules and the success criteria — before any decision logic was
      written. It is on screen so a screenshot of this page can be traced back
      to the exact run and the exact rules that produced it, and so nobody has
      to take on trust that the test was not tuned to the result.
    </Explain>
  );
}
