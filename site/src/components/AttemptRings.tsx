import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, MathUtils, type Mesh, type MeshBasicMaterial } from 'three'

/**
 * Four rings on the floor, lit in sequence as the run advances.
 *
 * WHAT THIS IS AND IS NOT. It is the attempt budget, which is a structural
 * constant of the problem: NPCI allows 1 original debit plus 3 retries, four
 * ever, per mandate (see DESIGN.md). Four is not a figure read from a report
 * and it is not a per-mandate claim about what either policy did -- it is the
 * shape of the box both policies are playing inside, which is the thing the
 * page keeps asserting in prose and never showed.
 *
 * Which is also why they carry no labels and no counts. The moment a ring
 * says "3 of 4 spent" it is making a quantitative claim, and every
 * quantitative claim on this page has to come from results.json. These stay
 * geometry.
 *
 * Cost: four ring meshes with basic materials, no lighting, no shadows. They
 * light by lerping one colour and one opacity per frame, which is four
 * material writes -- not a per-frame cost worth worrying about beside the
 * 200-instance lattice.
 */

const SLOTS = 4
const RING_Y = -4.42 // just above GridFloor's plane at -4.5, so it reads as on it
const SPACING = 3.4

/**
 * Set back from the lattice rather than directly beneath it. At z = 0 the
 * camera's low angle put the rings on the very bottom edge of the frame --
 * cropped in the engine shot and out of frame entirely in the ladder shot,
 * which is a decoration nobody ever sees. Pushing them away from the camera
 * lifts them up the screen into the lower third, where they sit under the
 * formation instead of under the viewport.
 */
const RING_Z = -5.5

/** Amber while the ladder is spending them, teal on the engine's pass. The
 *  same two series colours the charts use, for the same two policies. */
const LADDER: [number, number, number] = [0.85, 0.46, 0.04]
const ENGINE: [number, number, number] = [0.05, 0.72, 0.65]

function Ring({
  index,
  progressRef,
}: {
  index: number
  progressRef: React.RefObject<number>
}) {
  const meshRef = useRef<Mesh>(null)
  const matRef = useRef<MeshBasicMaterial>(null)
  const lit = useRef(0)

  const x = useMemo(() => (index - (SLOTS - 1) / 2) * SPACING, [index])

  useFrame((_, delta) => {
    const mat = matRef.current
    const mesh = meshRef.current
    if (!mat || !mesh) return

    const p = progressRef.current ?? 0
    const inEngineArm = p > 0.5

    // Each arm spends its slots across its own stretch of the scroll. The
    // rings light left to right as that stretch is crossed.
    const armP = inEngineArm
      ? MathUtils.clamp((p - 0.55) / 0.25, 0, 1)
      : MathUtils.clamp((p - 0.15) / 0.17, 0, 1)

    // Slot i is lit once the arm has crossed i/4 of its run.
    const threshold = index / SLOTS
    const target = armP > threshold ? 1 : 0

    // Damped rather than switched, so a ring comes up rather than blinking.
    lit.current += (target - lit.current) * (1 - Math.exp(-7 * Math.min(delta, 0.05)))

    const c = inEngineArm ? ENGINE : LADDER
    mat.color.setRGB(c[0], c[1], c[2])
    mat.opacity = 0.06 + lit.current * 0.5

    // A lit ring sits fractionally proud of the floor and breathes.
    const s = 1 + lit.current * 0.09
    mesh.scale.setScalar(s)
  })

  return (
    <mesh
      ref={meshRef}
      position={[x, RING_Y, RING_Z]}
      rotation={[-Math.PI / 2, 0, 0]}
      frustumCulled={false}
    >
      <ringGeometry args={[0.92, 1.06, 48]} />
      <meshBasicMaterial
        ref={matRef}
        transparent
        opacity={0.06}
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </mesh>
  )
}

export function AttemptRings({
  progressRef,
}: {
  progressRef: React.RefObject<number>
}) {
  return (
    <group>
      {Array.from({ length: SLOTS }, (_, i) => (
        <Ring key={i} index={i} progressRef={progressRef} />
      ))}
    </group>
  )
}
