import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, DoubleSide, type ShaderMaterial } from 'three'

/**
 * The ground plane: an infinite fading grid with a slow scan sweep.
 *
 * A shader rather than geometry, so the whole floor is one draw call and one
 * quad regardless of how far it appears to stretch. Lines are drawn from the
 * fragment's world position, and faded by distance so the horizon dissolves
 * into the fog instead of ending in a visible edge.
 */

const vertex = /* glsl */ `
  varying vec2 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`

const fragment = /* glsl */ `
  precision highp float;
  varying vec2 vWorld;
  uniform float uTime;
  uniform float uProgress;
  uniform vec3  uColor;
  uniform vec3  uSweep;

  // Distance to the nearest grid line, in pixels, so line weight stays
  // constant as the camera moves rather than aliasing into moire.
  float gridLine(vec2 p, float spacing) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) / fwidth(p / spacing);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    float fine  = gridLine(vWorld, 1.0)  * 0.35;
    float major = gridLine(vWorld, 5.0)  * 0.65;
    float lines = clamp(fine + major, 0.0, 1.0);

    // Radial fade: the grid exists near the action and nowhere else.
    float d = length(vWorld);
    float fade = smoothstep(46.0, 6.0, d);

    // A sweep that travels outward once per policy run, so the floor reacts
    // to the narrative instead of sitting inert under it.
    float ring = abs(d - (uProgress * 60.0 + uTime * 0.6));
    float sweep = smoothstep(3.5, 0.0, ring) * 0.5;

    vec3 col = uColor * lines + uSweep * sweep * lines;
    float alpha = (lines * 0.5 + sweep * 0.35) * fade;
    if (alpha < 0.002) discard;
    gl_FragColor = vec4(col, alpha);
  }
`

export function GridFloor({ progressRef }: { progressRef: React.RefObject<number> }) {
  const matRef = useRef<ShaderMaterial>(null)
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uProgress: { value: 0 },
      uColor: { value: [0.18, 0.23, 0.32] },
      uSweep: { value: [0.08, 0.72, 0.65] },
    }),
    [],
  )

  useFrame((_, delta) => {
    const m = matRef.current
    if (!m) return
    m.uniforms.uTime.value += delta
    m.uniforms.uProgress.value = progressRef.current ?? 0
  })

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -4.5, 0]} frustumCulled={false}>
      <planeGeometry args={[120, 120]} />
      <shaderMaterial
        ref={matRef}
        vertexShader={vertex}
        fragmentShader={fragment}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        side={DoubleSide}
        blending={AdditiveBlending}
      />
    </mesh>
  )
}
