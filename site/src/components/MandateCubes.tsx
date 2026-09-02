import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { InstancedMesh } from 'three'
import { Color, Matrix4, Vector3, MathUtils } from 'three'
import type { Narrative } from '../hooks/useReportData'

/**
 * The scroll narrative, as instanced cubes.
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
 * preserves more in 256 of 256 paired comparisons, not because mandate #7 was
 * traced through both arms. The figures are means over 8 seeds.
 *
 * Reads scroll progress from a ref rather than a prop so that scrolling does
 * not re-render the React tree: `useFrame` samples the ref on the render loop
 * the canvas is already running.
 */

const COL_NEUTRAL = new Color('#6b7280') // approaching the debit date
const COL_GREEN = new Color('#22c55e') // preserved
const COL_DUST = new Color('#1f2023') // not preserved
const COL_SAVED = new Color('#14b8a6') // preserved by the engine, lost by the ladder

interface MandateCubesProps {
  progressRef: React.RefObject<number>
  narrative: Narrative
}

export function MandateCubes({ progressRef, narrative }: MandateCubesProps) {
  const meshRef = useRef<InstancedMesh>(null)
  const { total, engineLost, ladderLost } = narrative

  // Deterministic layout: a fixed seed so the scene is identical run to run,
  // which is also what makes a screenshot reproducible.
  const cubeData = useMemo(() => {
    // `seed` is local to this useMemo callback and lives only for the length
    // of one layout pass; the mulberry32 state never escapes it.
    let seed = 42
    function rand() {
      seed |= 0
      seed = (seed + 0x6d2b79f5) | 0
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }

    return Array.from({ length: total }, (_, i) => ({
      // i < engineLost        -> lost under both policies
      // engineLost <= i < ladderLost -> lost by the ladder, PRESERVED by the engine
      // i >= ladderLost       -> preserved under both
      lostByEngine: i < engineLost,
      lostByLadder: i < ladderLost,
      startX: (rand() - 0.5) * 16,
      startY: (rand() - 0.5) * 10,
      startZ: -15 - rand() * 20,
      phase: rand() * Math.PI * 2,
      driftX: (rand() - 0.5) * 3,
      driftY: (rand() - 0.5) * 2,
    }))
  }, [total, engineLost, ladderLost])

  const tempMatrix = useMemo(() => new Matrix4(), [])
  const tempPos = useMemo(() => new Vector3(), [])
  const tempColor = useMemo(() => new Color(), [])

  useFrame(() => {
    const mesh = meshRef.current
    if (!mesh) return
    const p = progressRef.current ?? 0

    // 0.00-0.15  drift toward the debit date
    // 0.15-0.32  the fixed ladder runs; `ladderLost` go dark
    // 0.32-0.45  hold the ladder's result
    // 0.45-0.55  rewind
    // 0.55-0.80  the engine runs; `engineLost` go dark
    // 0.80-1.00  hold: the difference, and what it cost
    const ladderP = MathUtils.clamp((p - 0.15) / 0.17, 0, 1)
    const rewindP = MathUtils.clamp((p - 0.45) / 0.1, 0, 1)
    const engineP = MathUtils.clamp((p - 0.55) / 0.25, 0, 1)
    const driftP = MathUtils.clamp(p / 0.15, 0, 1)

    // Which arm is on screen, and how far through its run.
    const inEngineArm = p > 0.5
    const armP = inEngineArm ? engineP : ladderP

    for (let i = 0; i < total; i++) {
      const d = cubeData[i]
      let x = d.startX
      let y = d.startY
      let z = d.startZ + driftP * (15 + -d.startZ)
      let scale = 1

      y += Math.sin(d.phase + p * 4) * 0.15

      const dies = inEngineArm ? d.lostByEngine : d.lostByLadder

      if (armP > 0) {
        if (dies) {
          // Three impacts under the ladder, and the collapse that follows.
          // The engine spends fewer attempts, so it shakes less.
          const impacts = inEngineArm ? 1 : 3
          const hit = MathUtils.clamp(armP * impacts, 0, impacts)
          const shake = Math.sin(hit * 20) * 0.3 * (1 - armP)
          x += shake
          y += shake * 0.5 - armP * 2
          scale = Math.max(0.06, 1 - armP * 0.9)
        } else {
          // Preserved: settles forward, intact.
          z += armP * 3
          x += armP * d.driftX * 0.4
          y += armP * d.driftY * 0.2
          scale = 1 + armP * 0.08
        }
      }

      // Rewind restores everything before the engine's run.
      if (rewindP > 0 && rewindP < 1) {
        scale = MathUtils.lerp(scale, 1, rewindP)
        y = MathUtils.lerp(y, d.startY, rewindP * 0.6)
      }

      if (p < 0.15) {
        tempColor.copy(COL_NEUTRAL)
      } else if (dies) {
        tempColor.copy(COL_NEUTRAL).lerp(COL_DUST, armP)
      } else if (inEngineArm && d.lostByLadder) {
        // The mandates the engine keeps and the ladder does not: the delta.
        tempColor.copy(COL_NEUTRAL).lerp(COL_SAVED, armP)
      } else {
        tempColor.copy(COL_NEUTRAL).lerp(COL_GREEN, armP)
      }

      tempPos.set(x, y, z)
      tempMatrix.makeScale(scale * 0.18, scale * 0.18, scale * 0.18)
      tempMatrix.setPosition(tempPos)
      mesh.setMatrixAt(i, tempMatrix)
      mesh.setColorAt(i, tempColor)
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
        roughness={0.35}
        metalness={0.1}
      />
    </instancedMesh>
  )
}
