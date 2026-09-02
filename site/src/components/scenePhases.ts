/**
 * The six beats of the scroll narrative, in one place.
 *
 * The cubes, the camera and the captions all read from here. When these lived
 * as three separate lists of magic numbers they drifted, and a caption
 * describing the ladder appeared over cubes that had already been rewound.
 */
export const PHASES = [
  { start: 0.0, key: 'approach' },
  { start: 0.15, key: 'ladder' },
  { start: 0.32, key: 'ladder-result' },
  { start: 0.45, key: 'rewind' },
  { start: 0.55, key: 'engine' },
  { start: 0.8, key: 'trade' },
] as const

export type PhaseKey = (typeof PHASES)[number]['key']

export function phaseFor(p: number): number {
  let i = 0
  for (let k = 0; k < PHASES.length; k++) if (p >= PHASES[k].start) i = k
  return i
}

/** Frame-rate independent damping. delta in seconds, rate in "per second". */
export function damp(current: number, target: number, rate: number, delta: number) {
  return current + (target - current) * (1 - Math.exp(-rate * delta))
}
