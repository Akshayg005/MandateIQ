import { motion, useReducedMotion } from 'motion/react'
import type { ReactNode } from 'react'

/**
 * Scroll-entry reveal, built on Motion's `whileInView`.
 *
 * Deliberately not GSAP and not a scroll listener: this needs no pinning and
 * no scrub, so the lighter tool is the right one. Motion handles the observer,
 * runs the transform off the main React render path, and collapses to a plain
 * static render when the reader prefers reduced motion.
 *
 * The motion is motivated rather than decorative: it stages a section's
 * contents in reading order, which is the order the argument is meant to land
 * in. Anything that does not benefit from sequence should not be wrapped.
 */
export function Reveal({
  children,
  delay = 0,
  y = 20,
  className,
}: {
  children: ReactNode
  delay?: number
  y?: number
  className?: string
}) {
  const reduce = useReducedMotion()
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{
        duration: 0.65,
        delay: reduce ? 0 : delay,
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      {children}
    </motion.div>
  )
}

/** Staggers its children in sequence. Same reduced-motion contract. */
export function RevealStagger({
  children,
  step = 0.08,
  className,
}: {
  children: ReactNode[]
  step?: number
  className?: string
}) {
  return (
    <div className={className}>
      {children.map((child, i) => (
        <Reveal key={i} delay={i * step}>
          {child}
        </Reveal>
      ))}
    </div>
  )
}
