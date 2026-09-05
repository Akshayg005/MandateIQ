/**
 * How the engine actually decides, for someone who has never seen a payments
 * system.
 *
 * The scroll narrative above shows the engine behaving differently from the
 * ladder; it does not say HOW. Without this section a first-time reader
 * reaches the results with no model of what produced them, and a chart you
 * cannot account for is just a claim with a rectangle next to it.
 *
 * Every step leads with the plain sentence and keeps the mechanism behind a
 * disclosure, so the section reads in about twenty seconds and rewards anyone
 * who wants the next layer. Nothing here is hidden: the disclosures are
 * <details>, so their text is in the DOM, findable by in-page search, and
 * present in the server-rendered HTML.
 *
 * The fourth card is a limit, not a step, and it is styled differently on
 * purpose. Until R5 (2026-09-05) it said the off-ramp -- the reason this
 * project exists -- never fired in any published run. It fires now, but only
 * because the evaluation feeds it a FABRICATED signal that reads the
 * simulator's own hidden answer, so the card still leads with the limit
 * rather than the number. A reader who finds the README's version afterwards
 * must not find the two disagreeing; that is the whole reason this card
 * exists, and it is why the card was rewritten rather than deleted when the
 * result changed.
 */
import { Reveal } from './Reveal'
import { ExplainMore } from './Explain'
import {
  FourAttempts,
  OptOut,
  PreNotification,
} from '../glossary'

export function HowItWorks({ id }: { id: string }) {
  return (
    <section id={id} className="how-section">
      <Reveal className="section-head">
        <h2 className="section-title">How it decides</h2>
        <p className="section-subtitle">
          Four steps, and a limit. Each one opens if you want the mechanism
          underneath it.
        </p>
      </Reveal>

      <div className="how-steps">
        <Reveal className="how-step">
          <span className="how-step-n">1</span>
          <h3>Read what the bank said</h3>
          <p>
            A failed payment comes back with a message from the customer&rsquo;s
            bank, and every bank writes it differently. That message is turned
            into one label from a fixed list.
          </p>
          <ExplainMore label="What the AI does here — and what it isn't allowed to do">
            <p>
              This is one of only three places a language model is used, and its
              entire job is translation: it reads free text like{' '}
              <em>&ldquo;INSUFF FUNDS&rdquo;</em> or{' '}
              <em>&ldquo;a/c closed&rdquo;</em> and returns a single label from
              a closed list.
            </p>
            <p>
              It never returns a probability, a rupee amount or a retry date.
              Those come from statistics, and the repository has an automated
              check that <strong>fails the build</strong> if a language model is
              imported anywhere near the code that decides about money.
            </p>
          </ExplainMore>
        </Reveal>

        <Reveal className="how-step">
          <span className="how-step-n">2</span>
          <h3>Work out which of three things went wrong</h3>
          <p>
            Can&rsquo;t pay right now. Can&rsquo;t pay ever — the card is dead.
            Or won&rsquo;t pay: they want out and are letting it fail. Each one
            needs a completely different response.
          </p>
          <ExplainMore label="How can it tell them apart?">
            <p>
              It cannot, not with certainty, and that is treated as the central
              problem rather than an inconvenience. A statistical model
              estimates how likely each of the three is, given what the bank
              said, how much is owed, how many times it has already failed and
              where the customer is in their month.
            </p>
            <p>
              Crucially it produces a <strong>set</strong> of possibilities it
              cannot rule out — often all three — rather than one confident
              answer. What happens next depends on how small that set is.
            </p>
          </ExplainMore>
        </Reveal>

        <Reveal className="how-step">
          <span className="how-step-n">3</span>
          <h3>Decide whether to spend one of four attempts</h3>
          <p>
            There are only <FourAttempts /> per customer per cycle, ever. So the
            question is not &ldquo;when should we retry?&rdquo; but
            &ldquo;is this attempt worth one of the four we will never get
            back?&rdquo;
          </p>
          <ExplainMore label="Why timing cannot be reactive">
            <p>
              Indian rules require <PreNotification /> before any automatic
              debit, so an attempt has to be committed a day before it lands.
              Watching an account and charging the moment a salary arrives is
              not possible here.
            </p>
            <p>
              Worse, that notice comes with a button to{' '}
              <OptOut />. So an attempt is not free even when it fails quietly —
              it is another prompt to cancel. The engine works out the best use
              of the remaining attempts by playing the cycle forward to its end
              and reasoning backwards from there.
            </p>
          </ExplainMore>
        </Reveal>

        <Reveal className="how-step">
          <span className="how-step-n">4</span>
          <h3>Or stop, and offer a way out</h3>
          <p>
            If the evidence says the customer wants to leave, the right move is
            to stop charging and offer them a choice: pause, downgrade, or
            cancel. The system never cancels anything itself.
          </p>
          <ExplainMore label="What stops it offering an exit to someone who wanted to stay?">
            <p>
              A safety check that has to be passed before any exit is offered.
              The model must have ruled out both other explanations — the
              remaining set has to contain <strong>only</strong>{' '}
              &ldquo;wants to leave&rdquo; — at a confidence level chosen in
              advance. Anything less and the customer stays in the normal retry
              path.
            </p>
            <p>
              The reason for the caution is that a wrong guess here cancels
              someone who was happy to keep paying, which is the exact harm this
              project exists to prevent.
            </p>
          </ExplainMore>
        </Reveal>

        <Reveal className="how-step how-step--caveat">
          <span className="how-step-n">!</span>
          <h3>And the limit: that off-ramp only fires on a made-up signal</h3>
          <p>
            Until recently the safety check above never opened. Not once, in
            any published run. It opens now — but only because the evaluation
            hands it a bank message it invented, using the hidden answer it is
            supposed to be guessing. So the feature this project exists for is
            tested-and-imperfect rather than untested, which is better; it is
            still not evidence that a real signal would work.
          </p>
          <ExplainMore label="What changed, what did not, and why say so here">
            <p>
              The simulated bank messages used to be too coarse to ever push
              the &ldquo;wants to leave&rdquo; estimate high enough for the
              check to pass, at any setting — arithmetic rather than a
              finding. A new message type fixed that, and the evaluation now
              emits it by peeking at each customer&rsquo;s real reason for
              failing, which no live system can do.
            </p>
            <p>
              So the honest reading is the shape of the curve, not the
              headline. When that made-up signal is good, roughly one offer in
              ten goes to someone who would have paid anyway. When it is no
              better than a coin flip, that rises to between a third and three
              quarters. The report publishes every point on that curve,
              including the worthless ones, because a table showing only good
              signals would prove nothing.
            </p>
            <p>
              It sits on this page because it is the single most important thing
              to know about the results below, and a reader who found it in the
              repository afterwards would be right to wonder what else was left
              off.{' '}
              <strong>
                The full list of what this cannot do is in the README.
              </strong>
            </p>
          </ExplainMore>
        </Reveal>
      </div>
    </section>
  )
}
