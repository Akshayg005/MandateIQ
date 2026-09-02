import { useRef, useEffect, useState, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import { MandateCubes } from './MandateCubes'
import type { Narrative } from '../hooks/useReportData'

/**
 * Scroll progress lives in a ref, not in state.
 *
 * The previous version called setProgress() on every scroll event, which
 * re-rendered the whole R3F tree at scroll frequency -- the standard way to
 * lose the 60fps this block's gate asks for. The canvas is already running a
 * render loop, so `useFrame` samples the ref there and React is told only
 * when the narrative PHASE changes: six re-renders for the whole page instead
 * of one per scroll pixel.
 */

function CameraRig({ progressRef }: { progressRef: React.RefObject<number> }) {
  const { camera } = useThree()
  // Mutating the camera inside useFrame is the react-three-fiber idiom, and
  // is the point of this change: the previous version drove the camera from
  // useEffect on a state prop, re-rendering the tree on every scroll event.
  useFrame(() => {
    const p = progressRef.current ?? 0
    camera.position.x = Math.sin(p * Math.PI * 0.5) * 1.5
    camera.position.y = 2 - p
    camera.position.z = 12 - p * 3
    camera.lookAt(0, 0, 0)
  })
  return null
}

/** Phase boundaries, shared with MandateCubes' comment block. */
const PHASE_STARTS = [0, 0.15, 0.32, 0.45, 0.55, 0.8]

function phaseFor(p: number): number {
  let i = 0
  for (let k = 0; k < PHASE_STARTS.length; k++) if (p >= PHASE_STARTS[k]) i = k
  return i
}

interface SceneProps {
  narrative: Narrative
}

export function Scene({ narrative }: SceneProps) {
  const progressRef = useRef(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const [phase, setPhase] = useState(0)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const viewportH = window.innerHeight
    const range = rect.height + viewportH
    const p = Math.max(0, Math.min(1, (viewportH - rect.top) / range))
    progressRef.current = p
    // React hears about phase changes only -- six of them, not one per pixel.
    setPhase((prev) => {
      const next = phaseFor(p)
      return next === prev ? prev : next
    })
  }, [])

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('resize', handleScroll, { passive: true })
    handleScroll()
    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('resize', handleScroll)
    }
  }, [handleScroll])

  return (
    <div ref={containerRef} className="scene-container">
      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          powerPreference: 'high-performance',
          // true, deliberately. On a machine with no usable GPU the browser
          // would otherwise hand back a software context that renders this
          // scene at single-digit fps; failing here throws, the error
          // boundary catches it, and the reader gets the HTML fallback
          // instead of a canvas that appears broken.
          failIfMajorPerformanceCaveat: true,
        }}
        camera={{ position: [0, 2, 12], fov: 50, near: 0.1, far: 100 }}
      >
        <color attach="background" args={['#0a0b0f']} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <pointLight position={[-5, 3, 0]} intensity={0.3} color="#3b82f6" />
        <pointLight position={[5, -2, 3]} intensity={0.3} color="#f59e0b" />

        <CameraRig progressRef={progressRef} />
        <MandateCubes progressRef={progressRef} narrative={narrative} />

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.6}
            luminanceSmoothing={0.9}
            intensity={0.4}
          />
        </EffectComposer>
      </Canvas>

      <div className="scene-overlay">
        <PhaseLabel phase={phase} narrative={narrative} />
      </div>
    </div>
  )
}

/** Every figure in every label is read from the report. */
function PhaseLabel({
  phase,
  narrative: n,
}: {
  phase: number
  narrative: Narrative
}) {
  const labels: { text: string; sub: string; mod: string }[] = [
    {
      text: `${n.total} mandates approach the debit date`,
      sub: 'Every one of them authorised this payment. Some will fail.',
      mod: '',
    },
    {
      text: 'The fixed ladder: retry on a schedule, halt after three',
      sub: `${n.ladderLost} of ${n.total} mandates not preserved. ${n.ladderAttempts} attempts spent.`,
      mod: ' phase-label--danger',
    },
    {
      text: `${n.ladderLost} customers gone`,
      sub: `It recovered ${n.ladderRecoveredPct} of the money. It did not ask why any of them failed.`,
      mod: ' phase-label--danger',
    },
    {
      text: 'Rewind. Ask a different question.',
      sub: 'Not "will a retry succeed?" but "which of three things went wrong?"',
      mod: ' phase-label--rewind',
    },
    {
      text: `The engine loses ${n.engineLost}`,
      sub: `${n.preservedDelta} more mandates kept than the ladder, on ${n.engineAttempts} attempts instead of ${n.ladderAttempts}.`,
      mod: ' phase-label--engine',
    },
    {
      text: `And it recovered less: ${n.engineRecoveredPct} against ${n.ladderRecoveredPct}`,
      sub: 'That is the trade, stated plainly. Mandates preserved is the bar this system optimises — deliberately recovering less this cycle to protect lifetime value is the thesis, not a bug.',
      mod: ' phase-label--result',
    },
  ]
  const l = labels[phase] ?? labels[0]
  return (
    <div className={`phase-label${l.mod}`} key={phase}>
      <h2 className="phase-title">{l.text}</h2>
      <p className="phase-sub">{l.sub}</p>
    </div>
  )
}
