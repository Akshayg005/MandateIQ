import { useEffect, useState } from 'react'

export type WebGLSupport = 'probing' | 'ok' | 'unavailable'

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
 * The probe asks for exactly what Scene.tsx asks for, including
 * `failIfMajorPerformanceCaveat`, so a machine that would get a software
 * renderer at single-digit fps is reported unavailable and reads the HTML
 * fallback instead.
 *
 * The boundary is kept as a second line of defence: this probe covers context
 * creation, not a shader that fails to compile later.
 */
function probe(): boolean {
  try {
    const canvas = document.createElement('canvas')
    const attrs: WebGLContextAttributes = {
      failIfMajorPerformanceCaveat: true,
      powerPreference: 'high-performance',
    }
    const gl =
      canvas.getContext('webgl2', attrs) ?? canvas.getContext('webgl', attrs)
    if (!gl) return false
    // Hand the context back immediately; browsers cap how many may be live,
    // and this one exists only to answer the question.
    const lose = (gl as WebGLRenderingContext).getExtension('WEBGL_lose_context')
    lose?.loseContext()
    return true
  } catch {
    return false
  }
}

export function useWebGLSupport(): WebGLSupport {
  const [support, setSupport] = useState<WebGLSupport>('probing')
  useEffect(() => {
    setSupport(probe() ? 'ok' : 'unavailable')
  }, [])
  return support
}
