import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, type Points, type ShaderMaterial } from 'three'

/**
 * A slow drift of motes through the volume the cubes move in.
 *
 * Purpose is depth, not decoration. The scene is a dark void with one lattice
 * in it, so nothing except the grid floor told the eye how far away anything
 * was; the camera could swing through its whole arc and the formation read as
 * flat. Motes at a spread of depths parallax against that motion and give the
 * space a size.
 *
 * ONE draw call, and no per-frame CPU work. Positions are uploaded once as a
 * static buffer and animated entirely in the vertex shader from a single time
 * uniform -- the alternative, walking a few thousand positions on the CPU
 * every frame, is exactly the kind of cost that put a 283ms frame in this
 * scene once already.
 *
 * `sizeAttenuation` is done by hand in the shader (gl_PointSize scaled by
 * 1/-mvPosition.z) so a mote at the back is genuinely smaller rather than
 * uniformly sized, which is what sells the depth.
 */

const COUNT = 650

const vertex = /* glsl */ `
  uniform float uTime;
  uniform float uProgress;
  uniform float uPixelRatio;
  attribute float aScale;
  attribute float aPhase;
  varying float vFade;

  void main() {
    vec3 p = position;

    // Each mote drifts on its own phase. Slow: this should be felt as
    // atmosphere, never watched as an animation.
    p.x += sin(uTime * 0.11 + aPhase) * 0.9;
    p.y += cos(uTime * 0.09 + aPhase * 1.7) * 0.7;

    // The field drifts toward the camera as the story advances, so the
    // volume feels like it is being travelled through rather than orbited.
    p.z = mod(p.z + uProgress * 14.0 + uTime * 0.35 + 40.0, 80.0) - 40.0;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;

    // CLAMPED, and this matters more than it looks. gl_PointSize grows as
    // 1/-z, so a mote drifting near the camera becomes a very large quad;
    // with additive blending and no depth write, that is pure overdraw, and
    // measured as six dropped frames across one pass of the scene. The alpha
    // already fades these out (see vFade) but the fragments are still shaded
    // before the discard, so the size is what has to be capped, not just the
    // opacity.
    gl_PointSize = min(aScale * uPixelRatio * (26.0 / -mv.z), 7.0);

    // Fade at both ends of the depth range so motes are never seen popping
    // in or out at the wrap boundary.
    vFade = smoothstep(-40.0, -22.0, mv.z) * smoothstep(2.0, -6.0, mv.z);
  }
`

const fragment = /* glsl */ `
  precision mediump float;
  uniform vec3 uColor;
  varying float vFade;

  void main() {
    // Round, soft-edged mote from the point coordinate. Discarding outside
    // the radius keeps them from reading as squares at large sizes.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = dot(d, d);
    if (r > 0.25) discard;
    float alpha = smoothstep(0.25, 0.0, r) * vFade * 0.5;
    if (alpha < 0.004) discard;
    gl_FragColor = vec4(uColor, alpha);
  }
`

export function DustField({
  progressRef,
  pixelRatio = 1,
}: {
  progressRef: React.RefObject<number>
  pixelRatio?: number
}) {
  const ref = useRef<Points>(null)
  const matRef = useRef<ShaderMaterial>(null)

  const { positions, scales, phases } = useMemo(() => {
    // Same mulberry32 as MandateCubes, and for the same reason: the scene --
    // and any screenshot of it -- must be identical run to run.
    let seed = 1337
    const rand = () => {
      seed |= 0
      seed = (seed + 0x6d2b79f5) | 0
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }

    const positions = new Float32Array(COUNT * 3)
    const scales = new Float32Array(COUNT)
    const phases = new Float32Array(COUNT)
    for (let i = 0; i < COUNT; i++) {
      positions[i * 3] = (rand() - 0.5) * 54
      positions[i * 3 + 1] = (rand() - 0.5) * 30
      positions[i * 3 + 2] = (rand() - 0.5) * 80
      scales[i] = 0.5 + rand() * 1.6
      phases[i] = rand() * Math.PI * 2
    }
    return { positions, scales, phases }
  }, [])

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uProgress: { value: 0 },
      uPixelRatio: { value: pixelRatio },
      uColor: { value: [0.42, 0.55, 0.72] },
    }),
    [pixelRatio],
  )

  useFrame((_, delta) => {
    const m = matRef.current
    if (!m) return
    m.uniforms.uTime.value += delta
    m.uniforms.uProgress.value = progressRef.current ?? 0
  })

  return (
    <points ref={ref} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute attach="attributes-aScale" args={[scales, 1]} />
        <bufferAttribute attach="attributes-aPhase" args={[phases, 1]} />
      </bufferGeometry>
      <shaderMaterial
        ref={matRef}
        vertexShader={vertex}
        fragmentShader={fragment}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </points>
  )
}
