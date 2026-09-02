import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import {
  EffectComposer,
  Bloom,
  ChromaticAberration,
  Vignette,
} from '@react-three/postprocessing'
import { Vector2, Vector3, MathUtils } from 'three'
import { MandateCubes } from './MandateCubes'
import { GridFloor } from './GridFloor'
import { PHASES, phaseFor, damp } from './scenePhases'
import type { Narrative } from '../hooks/useReportData'

/**
 * Scroll progress lives in refs, not in state.
 *
 * An earlier version called setProgress() on every scroll event, re-rendering
 * the whole R3F tree at scroll frequency -- the standard way to lose the
 * 60fps this block's gate asks for. Now the scroll handler writes a raw
 * target, `useFrame` damps a smoothed value toward it on the render loop the
 * canvas is already running, and React is told only when the narrative PHASE
 * changes: six re-renders for the whole page instead of one per scroll pixel.
 *
 * The damping is also what makes it feel expensive. Raw scroll position is
 * jittery and frame-rate dependent; an exponential follow is smooth at any
 * refresh rate and gives the batch weight, as though 200 mandates take a
 * moment to respond.
 */

/** Camera positions per narrative beat, interpolated continuously. */
const CAM_KEYS: [number, number, number][] = [
  [0, 2.4, 15], // approach -- wide and cold, nothing has happened yet
  [3.4, 1.6, 11.5], // ladder   -- swing in as the retries start
  [0.6, -0.4, 8.4], // result   -- drop to the level of what was lost
  [-3.6, 3.2, 13], // rewind   -- pull back and up
  [2.2, 0.9, 11.5], // engine -- push in on the second run
  // Far enough back that the whole resolved lattice is in frame. This is the
  // shot the page exists to deliver; cropping it defeats the point.
  [0, 0.2, 12.5], // trade   -- square on to the formation
]

function sampleCamera(p: number, out: Vector3) {
  const last = PHASES.length - 1
  let i = 0
  for (let k = 0; k < PHASES.length; k++) if (p >= PHASES[k].start) i = k
  const a = CAM_KEYS[i]
  const b = CAM_KEYS[Math.min(i + 1, last)]
  const start = PHASES[i].start
  const end = i < last ? PHASES[i + 1].start : 1
  const t = end > start ? MathUtils.clamp((p - start) / (end - start), 0, 1) : 0
  const e = t * t * (3 - 2 * t)
  return out.set(
    MathUtils.lerp(a[0], b[0], e),
    MathUtils.lerp(a[1], b[1], e),
    MathUtils.lerp(a[2], b[2], e),
  )
}

function CameraRig({
  progressRef,
  pointerRef,
}: {
  progressRef: React.RefObject<number>
  pointerRef: React.RefObject<{ x: number; y: number }>
}) {
  const { camera } = useThree()
  const target = useMemo(() => new Vector3(), [])
  const current = useMemo(() => new Vector3(0, 2.4, 15), [])

  // Mutating the camera inside useFrame is the react-three-fiber idiom, and
  // is the point of this design: the previous version drove the camera from
  // useEffect on a state prop, re-rendering the tree on every scroll event.
  useFrame((_, delta) => {
    sampleCamera(progressRef.current ?? 0, target)
    // A little parallax from the pointer -- small enough to read as depth
    // rather than as a control the reader is expected to operate.
    const ptr = pointerRef.current ?? { x: 0, y: 0 }
    target.x += ptr.x * 1.1
    target.y += ptr.y * 0.7

    const d = Math.min(delta, 0.05)
    current.x = damp(current.x, target.x, 3.2, d)
    current.y = damp(current.y, target.y, 3.2, d)
    current.z = damp(current.z, target.z, 3.2, d)
    camera.position.copy(current)
    camera.lookAt(0, 0, 0)
  })
  return null
}

/** Damps the smoothed progress toward the raw scroll target, once per frame. */
function ProgressDamper({
  targetRef,
  progressRef,
}: {
  targetRef: React.RefObject<number>
  progressRef: React.RefObject<number>
}) {
  useFrame((_, delta) => {
    progressRef.current = damp(
      progressRef.current ?? 0,
      targetRef.current ?? 0,
      6,
      Math.min(delta, 0.05),
    )
  })
  return null
}

interface SceneProps {
  narrative: Narrative
}

export function Scene({ narrative }: SceneProps) {
  const targetRef = useRef(0)
  const progressRef = useRef(0)
  const pointerRef = useRef({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)
  const [phase, setPhase] = useState(0)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const viewportH = window.innerHeight
    const range = rect.height + viewportH
    const p = Math.max(0, Math.min(1, (viewportH - rect.top) / range))
    targetRef.current = p
    // React hears about phase changes only -- six of them, not one per pixel.
    setPhase((prev) => {
      const next = phaseFor(p)
      return next === prev ? prev : next
    })
  }, [])

  const handlePointer = useCallback((e: PointerEvent) => {
    pointerRef.current = {
      x: (e.clientX / window.innerWidth - 0.5) * 2,
      y: -(e.clientY / window.innerHeight - 0.5) * 2,
    }
  }, [])

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', handleScroll, { passive: true })
    window.addEventListener('pointermove', handlePointer, { passive: true })
    handleScroll()
    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
      window.removeEventListener('pointermove', handlePointer)
    }
  }, [handleScroll, handlePointer])

  // Small. At 0.0006+ the additive grid lines fringe red/cyan hard enough to
  // read as a broken render rather than as a lens.
  const aberration = useMemo(() => new Vector2(0.00022, 0.0003), [])

  // The ref goes on the TALL outer element, never on the sticky child: a
  // pinned element's rect.top stays at 0 while it is stuck, so measuring the
  // child would freeze progress at 0 for the whole scene.
  return (
    <div ref={containerRef} className="scene-scroll">
      <div className="scene-container">
      <Canvas
        // Capped rather than uncapped: a 3x retina display would otherwise
        // render 9x the fragments for no visible gain on a scene this dark.
        dpr={[1, 1.75]}
        gl={{
          antialias: true,
          powerPreference: 'high-performance',
          // true, deliberately. On a machine with no usable GPU the browser
          // would otherwise hand back a software context that renders this
          // scene at single-digit fps; failing here throws, the error
          // boundary catches it, and the reader gets the HTML fallback
          // instead of a canvas that merely appears broken.
          failIfMajorPerformanceCaveat: true,
        }}
        camera={{ position: [0, 2.4, 15], fov: 46, near: 0.1, far: 140 }}
      >
        <color attach="background" args={['#07080c']} />
        <fogExp2 attach="fog" args={['#07080c', 0.026]} />

        <ambientLight intensity={0.35} />
        <directionalLight position={[6, 8, 5]} intensity={1.1} />
        <pointLight position={[-7, 3, 2]} intensity={40} distance={30} color="#3b82f6" />
        <pointLight position={[7, -3, 4]} intensity={30} distance={30} color="#f59e0b" />
        <pointLight position={[0, 0, 6]} intensity={18} distance={22} color="#14b8a6" />

        <ProgressDamper targetRef={targetRef} progressRef={progressRef} />
        <CameraRig progressRef={progressRef} pointerRef={pointerRef} />
        <GridFloor progressRef={progressRef} />
        <MandateCubes progressRef={progressRef} narrative={narrative} />

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.55}
            luminanceSmoothing={0.85}
            intensity={1.15}
            mipmapBlur
          />
          <ChromaticAberration offset={aberration} />
          <Vignette eskil={false} offset={0.28} darkness={0.82} />
        </EffectComposer>
      </Canvas>

        <div className="scene-overlay">
          <PhaseLabel phase={phase} narrative={narrative} />
        </div>
        <ScenePhaseRail phase={phase} />
      </div>
    </div>
  )
}

/** Six ticks down the side, so the reader knows where they are in the story. */
function ScenePhaseRail({ phase }: { phase: number }) {
  return (
    <div className="phase-rail" aria-hidden="true">
      {PHASES.map((ph, i) => (
        <span
          key={ph.key}
          className={`phase-tick${i === phase ? ' phase-tick--on' : ''}${
            i < phase ? ' phase-tick--past' : ''
          }`}
        />
      ))}
    </div>
  )
}

/** Every figure in every caption is read from the report. */
function PhaseLabel({
  phase,
  narrative: n,
}: {
  phase: number
  narrative: Narrative
}) {
  const labels: { kicker: string; text: string; sub: string; mod: string }[] = [
    {
      kicker: 'The batch',
      text: `${n.total} mandates approach the debit date`,
      sub: 'Every one of them authorised this payment. Some will fail.',
      mod: '',
    },
    {
      kicker: 'The incumbent',
      text: 'The fixed ladder retries on a schedule, then halts',
      sub: `${n.ladderAttempts} attempts spent. It never asks why any single mandate failed.`,
      mod: ' phase-label--danger',
    },
    {
      kicker: 'The cost',
      text: `${n.ladderLost} of ${n.total} not preserved`,
      sub: `It recovered ${n.ladderRecoveredPct} of the money — and lost the customers it took it from.`,
      mod: ' phase-label--danger',
    },
    {
      kicker: 'Rewind',
      text: 'Ask a different question',
      sub: 'Not "will a retry succeed?" but "which of three things went wrong?"',
      mod: ' phase-label--rewind',
    },
    {
      kicker: 'This engine',
      text: `${n.engineLost} of ${n.total} not preserved`,
      sub: `${n.preservedDelta} more mandates kept than the ladder, on ${n.engineAttempts} attempts instead of ${n.ladderAttempts}.`,
      mod: ' phase-label--engine',
    },
    {
      kicker: 'The trade',
      text: `It recovered ${n.engineRecoveredPct}. The ladder recovered ${n.ladderRecoveredPct}.`,
      sub: 'Stated plainly, because it is the point. Mandates preserved is the bar this system optimises — deliberately recovering less this cycle to protect lifetime value is the thesis, not a bug.',
      mod: ' phase-label--result',
    },
  ]
  const l = labels[phase] ?? labels[0]
  return (
    <div className={`phase-label${l.mod}`} key={phase}>
      <span className="phase-kicker">{l.kicker}</span>
      <h2 className="phase-title">{l.text}</h2>
      <p className="phase-sub">{l.sub}</p>
    </div>
  )
}
