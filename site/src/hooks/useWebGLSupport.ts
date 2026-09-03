import { useEffect, useState } from 'react'

/**
 * `full`      — a hardware context on the strict terms the scene prefers.
 * `degraded`  — a hardware context the browser would only grant on looser
 *               terms. The scene still runs, at reduced cost.
 * `unavailable` — no context at all, or a software rasteriser. HTML fallback.
 */
export type WebGLTier = 'probing' | 'full' | 'degraded' | 'unavailable'

/**
 * Decide whether to mount the canvas at all, BEFORE mounting it.
 *
 * Why this exists rather than relying on CanvasErrorBoundary. Verified in
 * headless Chromium with `--disable-gpu`: react-three-fiber does not throw
 * during React's render pass when context creation fails, so the error
 * boundary never fires. What the reader got was a black rectangle with the
 * narrative captions floating over nothing -- strictly worse than either the
 * scene or the fallback, and it defeated B15's canvas-failure criterion while
 * every build, lint and render check stayed green.
 *
 * WHY THIS IS TIERED RATHER THAN A YES/NO. The first version asked for a
 * context with `failIfMajorPerformanceCaveat: true` and treated a refusal as
 * "no WebGL". That flag does not mean "this machine has no GPU"; it means
 * "the browser is being conservative right now", which it also is with
 * hardware acceleration toggled off, a driver on the blocklist, a stale GPU
 * process, or simply a profile carrying different flags from a fresh one.
 * Measured on this project: the same laptop (Intel Iris Xe) renders the scene
 * at a steady 59.9fps under a clean Chrome profile and was refused a context
 * outright under an everyday one. Those readers were shown the no-WebGL
 * fallback on hardware that runs the scene fine.
 *
 * So the strict ask is now only the FIRST question. If it is refused we ask
 * again on looser terms and inspect what we were given: a real GPU renders
 * the scene (at reduced cost), and only a software rasteriser -- which would
 * genuinely crawl -- falls through to the HTML fallback. That keeps the
 * fallback for the case it was written for without spending it on machines
 * that never needed it.
 *
 * The boundary is kept as a second line of defence: this probe covers context
 * creation, not a shader that fails to compile later.
 */

/** Renderer strings that mean "there is no GPU behind this context". */
const SOFTWARE =
  /swiftshader|llvmpipe|softpipe|software|basic render|mesa offscreen/i

interface Probed {
  gl: WebGLRenderingContext | WebGL2RenderingContext | null
  renderer: string
}

function acquire(failIfMajorPerformanceCaveat: boolean): Probed {
  const canvas = document.createElement('canvas')
  const attrs: WebGLContextAttributes = {
    failIfMajorPerformanceCaveat,
    powerPreference: 'high-performance',
  }
  const gl = (canvas.getContext('webgl2', attrs) ??
    canvas.getContext('webgl', attrs)) as
    | WebGLRenderingContext
    | WebGL2RenderingContext
    | null
  if (!gl) return { gl: null, renderer: '' }

  // Both, because which one carries the real string varies by browser and by
  // whether the debug extension is exposed at all.
  const dbg = gl.getExtension('WEBGL_debug_renderer_info')
  const unmasked = dbg
    ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) ?? '')
    : ''
  const plain = String(gl.getParameter(gl.RENDERER) ?? '')
  return { gl, renderer: `${unmasked} ${plain}` }
}

/** Hand the context back: browsers cap how many may be live, and these exist
 *  only to answer the question. */
function release(gl: WebGLRenderingContext | WebGL2RenderingContext | null) {
  gl?.getExtension('WEBGL_lose_context')?.loseContext()
}

function probe(): { tier: Exclude<WebGLTier, 'probing'>; why: string } {
  try {
    const strict = acquire(true)
    if (strict.gl) {
      const software = SOFTWARE.test(strict.renderer)
      release(strict.gl)
      return software
        ? { tier: 'unavailable', why: `software renderer: ${strict.renderer}` }
        : { tier: 'full', why: strict.renderer }
    }

    // Refused on strict terms. That is not an answer about the hardware, so
    // ask again without the caveat flag and look at what we get.
    const loose = acquire(false)
    if (!loose.gl) {
      return { tier: 'unavailable', why: 'no WebGL context on any terms' }
    }
    const software = SOFTWARE.test(loose.renderer)
    release(loose.gl)
    return software
      ? { tier: 'unavailable', why: `software renderer: ${loose.renderer}` }
      : { tier: 'degraded', why: `conservative context: ${loose.renderer}` }
  } catch (e) {
    return { tier: 'unavailable', why: `probe threw: ${String(e)}` }
  }
}

export function useWebGLSupport(): WebGLTier {
  const [tier, setTier] = useState<WebGLTier>('probing')
  useEffect(() => {
    const { tier: t, why } = probe()
    // Left in deliberately. When a reader reports "I only see the fallback",
    // this one line is the difference between diagnosing it and guessing.
    console.info(`[scene] WebGL tier: ${t} — ${why}`)
    setTier(t)
  }, [])
  return tier
}
