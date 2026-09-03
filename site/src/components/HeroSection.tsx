import { motion, useScroll, useTransform, useReducedMotion } from 'motion/react'
import { useRef } from 'react'
import { Reveal } from './Reveal'

interface HeroSectionProps {
  id: string
}

/**
 * Left-aligned, not centred, and no gradient on the headline.
 *
 * The centred-hero-over-a-dark-mesh-gradient with a colour-swept headline is
 * the house style of every AI-generated landing page, and the previous draft
 * of this file was exactly that. An asymmetric column with plain white type
 * reads as something a person laid out.
 *
 * Deliberately carries no figures. Everything numeric on this page is read
 * from a saved report, and the hero renders before that fetch resolves;
 * rather than show a placeholder where a number will be, the hero makes the
 * argument and lets the scene below it carry the evidence.
 */
export function HeroSection({ id }: HeroSectionProps) {
  const ref = useRef<HTMLElement>(null)
  const reduce = useReducedMotion()

  // Parallax on the way out only: the hero is the first thing on screen, so
  // there is no "entering" half to animate. Transform and opacity only, both
  // compositor properties, so this costs nothing per scroll event.
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  })
  const glowY = useTransform(scrollYProgress, [0, 1], ['0%', '38%'])
  const copyY = useTransform(scrollYProgress, [0, 1], ['0%', '22%'])
  const fade = useTransform(scrollYProgress, [0, 0.85], [1, 0])

  // The entrance is staggered by hand rather than with <Reveal>, because the
  // hero is already in view at load: whileInView would fire everything on the
  // same frame and the sequence -- which is the reading order of the argument
  // -- would be lost.
  const rise = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 18 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] as const },
        }

  return (
    <section id={id} className="hero-section" ref={ref}>
      <motion.div
        className="hero-glow"
        aria-hidden="true"
        style={reduce ? undefined : { y: glowY }}
      />

      <motion.div
        className="hero-inner"
        style={reduce ? undefined : { y: copyY, opacity: fade }}
      >
        <motion.p className="hero-eyebrow" {...rise(0)}>
          Subscription payments, India
        </motion.p>

        <motion.h1 className="hero-title" {...rise(0.08)}>
          When a subscription payment fails,
          <br />
          most systems just try again.
        </motion.h1>

        <motion.p className="hero-subtitle" {...rise(0.16)}>
          Sometimes the money is simply not there today. Sometimes the card is
          dead. And sometimes the customer has quietly decided to leave. Those
          three need three different answers, and retrying is only the right
          answer to one of them.
        </motion.p>

        <motion.div className="hero-actions" {...rise(0.24)}>
          <a className="btn btn--primary" href="#how">
            See how it decides
          </a>
          <a className="btn btn--ghost" href="#results">
            Jump to the results
          </a>
        </motion.div>

        <motion.div className="hero-scroll-cue" aria-hidden="true" {...rise(0.34)}>
          <span className="hero-scroll-line" />
          Scroll
        </motion.div>
      </motion.div>
    </section>
  )
}

/**
 * The three causes, as the reader's first real explanation of the product.
 * Given its own section rather than crammed into the hero, which keeps the
 * hero inside one viewport and gives these room to be read.
 */
export function CauseSection({ id }: { id: string }) {
  const reduceList = useReducedMotion()
  const causes = [
    {
      key: 'now',
      when: 'The money is not there today',
      then: 'Try again, timed to when they are actually paid',
      body: 'Salary lands, balances recover. Retrying on a fixed schedule ignores that; retrying on their rhythm does not.',
    },
    {
      key: 'ever',
      when: 'The card or account is dead',
      then: 'Stop, and ask them to re-authorise',
      body: 'Expired card, closed account, cancelled mandate. No number of retries fixes any of these, and each one annoys someone who already cannot pay.',
    },
    {
      key: 'wont',
      when: 'They want to leave',
      then: 'Offer a way out, and let them choose',
      body: 'Pause, downgrade, or cancel. The system never cancels on someone’s behalf. Grinding an exiting customer is what turns a lapsed subscription into a complaint.',
    },
  ]

  return (
    <section id={id} className="cause-section">
      <Reveal className="cause-head">
        <h2>Three reasons a payment fails. Three different answers.</h2>
        <p>
          The engine works out which one it is looking at before it decides
          what to do, and it only ever gets four attempts per customer.
        </p>
      </Reveal>

      <ol className="cause-list">
        {causes.map((c, i) => (
          <motion.li
            key={c.key}
            className={`cause-row cause-row--${c.key}`}
            initial={reduceList ? false : { opacity: 0, y: 22 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{
              duration: 0.65,
              delay: reduceList ? 0 : i * 0.1,
              ease: [0.16, 1, 0.3, 1],
            }}
          >
            <span className="cause-num">{String(i + 1).padStart(2, '0')}</span>
            <div className="cause-when">
              <h3>{c.when}</h3>
              <p>{c.body}</p>
            </div>
            <div className="cause-then">
              <span className="cause-then-label">What it does</span>
              <p>{c.then}</p>
            </div>
          </motion.li>
        ))}
      </ol>
    </section>
  )
}
