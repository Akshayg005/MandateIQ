import { motion, useScroll, useSpring, useReducedMotion } from 'motion/react'

/**
 * A hairline at the top of the page showing how far through the argument the
 * reader is.
 *
 * Motivated by the shape of this page specifically: the scene alone is 520vh,
 * so a reader part-way down it has no way to tell whether they are near the
 * end or nowhere near it, and "how much more of this is there" is exactly the
 * question that makes someone leave.
 *
 * Driven by a spring on scaleX, which is a compositor-only property -- no
 * layout, no paint, no React render per scroll event. Under reduced motion
 * the spring is dropped and the bar tracks scroll directly: the information
 * is still wanted, the springiness is what is not.
 */
export function ScrollProgress() {
  const reduce = useReducedMotion()
  const { scrollYProgress } = useScroll()
  const smooth = useSpring(scrollYProgress, {
    stiffness: 180,
    damping: 30,
    restDelta: 0.001,
  })

  return (
    <motion.div
      className="scroll-progress"
      style={{ scaleX: reduce ? scrollYProgress : smooth }}
      aria-hidden="true"
    />
  )
}
