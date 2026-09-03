/**
 * Every term this page uses that a first-time reader would not know.
 *
 * One file, so a definition cannot drift between the two places it appears,
 * and so the list of "things we assume the reader knows" is reviewable as a
 * list rather than scattered through six components.
 *
 * HOUSE STYLE for these definitions:
 *   - Explain the thing, not the jargon for the thing.
 *   - No term is defined using another undefined term.
 *   - Where a number is involved, give the number.
 *   - Where the honest answer is unflattering, it goes in the definition.
 *     A tooltip that quietly omits the caveat is worse than no tooltip.
 */
import { Explain } from './components/Explain'

export function Mandate() {
  return (
    <Explain term="mandate" wide>
      A standing permission you give a business to pull money from your account
      on a schedule — the thing behind every subscription, SIP and EMI. In India
      these run on UPI AutoPay or a card, and the rules for them are set by the
      RBI. When we say a mandate was <em>preserved</em>, it means the customer
      still has that subscription at the end of the cycle.
    </Explain>
  )
}

export function Mandates() {
  return (
    <Explain term="mandates" wide>
      Standing permissions to charge a customer on a schedule — subscriptions,
      SIPs, EMIs. Preserving one means the customer is still subscribed at the
      end of the cycle.
    </Explain>
  )
}

export function FixedLadder() {
  return (
    <Explain term="fixed ladder" wide>
      What almost every payment system does today, and what this engine is
      measured against. When a payment fails it simply tries again on a fixed
      schedule — one day later, two days later, three days later — and then
      gives up. It never asks <em>why</em> the payment failed. This is
      Razorpay&rsquo;s documented behaviour, so the comparison here is against
      the real incumbent rather than a weak stand-in.
    </Explain>
  )
}

export function FourAttempts() {
  return (
    <Explain term="four attempts" wide>
      India&rsquo;s payment network (NPCI) allows one original charge plus three
      retries per billing cycle. Four in total — not four per week, four ever.
      Once they are spent, nothing more can be tried until the next cycle, which
      is why deciding <em>which</em> attempts to spend matters more than
      scheduling them.
    </Explain>
  )
}

export function PreNotification() {
  return (
    <Explain term="24-hour notice" wide>
      RBI rules require the bank to tell the customer at least 24 hours before
      any automatic debit. So every attempt has to be committed a full day
      before it happens — you cannot watch a customer&rsquo;s balance and pounce
      when the salary lands. The system has to forecast instead of react.
    </Explain>
  )
}

export function OptOut() {
  return (
    <Explain term="opt out" wide>
      That mandatory 24-hour notice comes with a button to cancel — either just
      this one payment, or the entire subscription. So every retry is also a
      fresh invitation to leave. This is why the count of attempts is a cost and
      not just an efficiency number.
    </Explain>
  )
}

export function Preserved() {
  return (
    <Explain term="preserved" wide>
      The customer still has an active subscription at the end of the cycle —
      they did not cancel, and their payment permission was not revoked or left
      dead. This is the number the engine is built to protect, and it is the one
      most retry systems never report.
    </Explain>
  )
}

export function SignTest() {
  return (
    <Explain term="compared one batch at a time" wide>
      Rather than averaging everything into a single figure, each batch of
      customers is run through both policies and the two results are compared
      head to head. Reporting &ldquo;wins in 256 of 256&rdquo; is harder to
      fluke than reporting a single average, because one lucky batch cannot
      carry the number.
    </Explain>
  )
}

export function Seeds() {
  return (
    <Explain term="batches" wide>
      The evaluation is run eight separate times with different randomly
      generated sets of customers, so a result cannot depend on one convenient
      set. Each run is reproducible: the same starting number always produces
      the same customers.
    </Explain>
  )
}

export function AttemptsPerRecovery() {
  return (
    <Explain term="attempts per recovery" wide>
      How many charge attempts were spent for each payment that eventually
      succeeded. Lower is better — it means less money collected per unit of
      customer irritation, and fewer chances for someone to cancel.
    </Explain>
  )
}

export function Synthetic() {
  return (
    <Explain term="simulated customers" wide>
      These numbers come from a simulator, not from live traffic. It was built
      and frozen before any of the decision logic was written, so the rules
      could not be tuned to it. What it can show is whether taking the real
      constraints seriously beats ignoring them. What it cannot show is how
      much money this would make in production.
    </Explain>
  )
}

export function OneShot() {
  return (
    <Explain term="one attempt, no model" wide>
      A deliberately dumb comparison: charge once, on day two, and never try
      again. No prediction, no logic at all. It is included because it beats
      this engine on two of the three measures — and a results page that only
      showed comparisons the engine wins would not be worth reading.
    </Explain>
  )
}

export function Recovered() {
  return (
    <Explain term="money collected" wide>
      The rupee value of failed payments that were successfully charged during
      the cycle. This is the number every other retry system leads with. Here it
      is deliberately one of three, because collecting the most money this month
      and keeping the most customers are not the same goal.
    </Explain>
  )
}
