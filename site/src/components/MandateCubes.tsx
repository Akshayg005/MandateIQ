import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { InstancedMesh } from 'three'
import { Color, Matrix4, Quaternion, Vector3, MathUtils } from 'three'
import type { Narrative } from '../hooks/useReportData'

/**
 * The scroll narrative, as one instanced draw call.
 *
 * EVERY COUNT COMES FROM `reports/results.json`. There are no constants for
 * how many mandates survive, because an earlier draft of this file carried
 * PLAN.md's storyboard placeholders ("ladder kills 14") as if they were
 * results, and they were wrong by a factor of six -- the ladder loses 90.
 *
 * What the cubes assert, precisely: under the fixed ladder `ladderLost` of
 * `total` mandates are not preserved; under the engine, `engineLost` are.
 * The cubes are COUNTS, not identified mandates -- the engine's survivors are
 * drawn as a superset of the ladder's because the sign test says the engine
 * preserves more in every one of its paired comparisons, not because mandate
 * #7 was traced through both arms. The figures are means over 8 seeds.
 *
 * The visual grammar, so the motion carries the argument:
 *   chaos -> order    surviving mandates resolve into a lattice; the batch
 *                     ends more organised than it started
 *   tumble and fall   a mandate that is not preserved leaves the formation
 *   teal in overdrive the mandates the engine keeps and the ladder does not,
 *                     pushed above 1.0 so the bloom pass blows them out
 *
 * Scroll progress arrives as a ref, sampled on the render loop the canvas is
 * already running, so scrolling never re-renders the React tree.
 */

const COL_NEUTRAL = new Color('#5b6472')
const COL_GREEN = new Color('#22c55e')
const COL_DUST = new Color('#141519')
// Deliberately > 1.0: HDR values survive tone mapping (disabled on this
// material) and are what the bloom threshold actually catches.
const COL_SAVED = new Color('#14b8a6').multiplyScalar(2.6)

const LATTICE_COLS = 16
const LATTICE_GAP = 0.62

interface MandateCubesProps {
  progressRef: React.RefObject<number>
  narrative: Narrative
}

export function MandateCubes({ progressRef, narrative }: MandateCubesProps) {
  const meshRef = useRef<InstancedMesh>(null)
  const { total, engineLost, ladderLost } = narrative

  const cubeData = useMemo(() => {
    // `seed` is local to this callback and lives only for one layout pass;
    // the mulberry32 state never escapes it. Fixed, so the scene -- and any
    // screenshot of it -- is identical run to run.
    let seed = 42
    function rand() {
      seed |= 0
      seed = (seed + 0x6d2b79f5) | 0
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }

    // Survivor rank determines the lattice slot, and differs per arm because
    // the two policies preserve different numbers.
    let ladderRank = 0
    let engineRank = 0
    const ladderKept = total - ladderLost
    const engineKept = total - engineLost

    const latticeSlot = (rank: number, kept: number) => {
      const cols = Math.min(LATTICE_COLS, Math.max(1, kept))
      const rows = Math.ceil(kept / cols)
      const c = rank % cols
      const r = Math.floor(rank / cols)
      return {
        x: (c - (cols - 1) / 2) * LATTICE_GAP,
        y: (r - (rows - 1) / 2) * LATTICE_GAP,
      }
    }

    return Array.from({ length: total }, (_, i) => {
      const lostByEngine = i < engineLost
      const lostByLadder = i < ladderLost
      const ladderSlot = lostByLadder ? null : latticeSlot(ladderRank++, ladderKept)
      const engineSlot = lostByEngine ? null : latticeSlot(engineRank++, engineKept)
      return {
        lostByEngine,
        lostByLadder,
        ladderSlot,
        engineSlot,
        startX: (rand() - 0.5) * 22,
        startY: (rand() - 0.5) * 12,
        startZ: -18 - rand() * 26,
        spin: rand() * Math.PI * 2,
        spinRate: 0.25 + rand() * 0.5,
        axis: new Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize(),
        bob: rand() * Math.PI * 2,
        // Losses cascade rather than snapping out together. Ordered by index
        // so the wave reads as a sweep across the batch.
        stagger: (i / Math.max(1, total)) * 0.55,
        fallSpin: 2 + rand() * 4,
        drift: (rand() - 0.5) * 4,
      }
    })
  }, [total, engineLost, ladderLost])

  const m4 = useMemo(() => new Matrix4(), [])
  const pos = useMemo(() => new Vector3(), [])
  const scl = useMemo(() => new Vector3(), [])
  const quat = useMemo(() => new Quaternion(), [])
  const col = useMemo(() => new Color(), [])
  const clock = useRef(0)

  useFrame((_, delta) => {
    const mesh = meshRef.current
    if (!mesh) return
    clock.current += delta
    const t = clock.current
    const p = progressRef.current ?? 0

    const driftP = MathUtils.clamp(p / 0.15, 0, 1)
    const ladderP = MathUtils.clamp((p - 0.15) / 0.17, 0, 1)
    const rewindP = MathUtils.clamp((p - 0.45) / 0.1, 0, 1)
    const engineP = MathUtils.clamp((p - 0.55) / 0.25, 0, 1)

    const inEngineArm = p > 0.5
    // The rewind un-runs the ladder before the engine's pass begins.
    const armRaw = inEngineArm ? engineP : ladderP * (1 - rewindP)

    for (let i = 0; i < total; i++) {
      const d = cubeData[i]
      const dies = inEngineArm ? d.lostByEngine : d.lostByLadder
      const slot = inEngineArm ? d.engineSlot : d.ladderSlot

      // Per-cube time, staggered, so the batch resolves as a wave.
      const arm = MathUtils.clamp((armRaw - d.stagger) / (1 - d.stagger), 0, 1)
      const eased = arm * arm * (3 - 2 * arm)

      // Approach: from the far scatter into a loose cloud.
      const cloudX = d.startX * 0.42
      const cloudY = d.startY * 0.42
      const cloudZ = MathUtils.lerp(d.startZ, 0, driftP)

      let x = cloudX
      let y = cloudY + Math.sin(d.bob + t * 0.7) * 0.18
      let z = cloudZ
      let scale = 1
      let angle = d.spin + t * d.spinRate

      if (dies) {
        // Leaves the formation: tumbles, falls, dims.
        x += d.drift * eased
        y -= eased * eased * 14
        z += eased * 2
        angle += eased * d.fallSpin
        scale = Math.max(0.05, 1 - eased * 0.85)
      } else if (slot) {
        // Resolves into the lattice.
        x = MathUtils.lerp(cloudX, slot.x, eased)
        y = MathUtils.lerp(y, slot.y, eased)
        z = MathUtils.lerp(cloudZ, 0, eased)
        // Settled cubes stop tumbling and square up to the camera.
        angle = MathUtils.lerp(angle, 0, eased)
        // A slow breath, so the formation is alive rather than frozen.
        scale = 1 + eased * 0.12 + Math.sin(d.bob + t * 1.4) * 0.02 * eased
      }

      if (p < 0.15) {
        col.copy(COL_NEUTRAL)
      } else if (dies) {
        col.copy(COL_NEUTRAL).lerp(COL_DUST, eased)
      } else if (inEngineArm && d.lostByLadder) {
        // Kept by the engine, lost by the ladder: the delta, in overdrive.
        col.copy(COL_NEUTRAL).lerp(COL_SAVED, eased)
      } else {
        col.copy(COL_NEUTRAL).lerp(COL_GREEN, eased)
      }

      quat.setFromAxisAngle(d.axis, angle)
      pos.set(x, y, z)
      scl.setScalar(scale * 0.2)
      m4.compose(pos, quat, scl)
      mesh.setMatrixAt(i, m4)
      mesh.setColorAt(i, col)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, total]}
      frustumCulled={false}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        toneMapped={false}
        roughness={0.28}
        metalness={0.35}
      />
    </instancedMesh>
  )
}
