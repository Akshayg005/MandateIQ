/* Copied from site/src/components/Explain.tsx, deliberately.

   The two apps are separate Vite builds with separate dependency trees and
   separate palettes; wiring a shared package between them would mean a
   workspace, a build step and a versioning story, to share eighty lines that
   have no reason to change. If the interaction contract below is ever
   revised, revise it in both -- there are exactly two copies and this comment
   is in both of them.
*/
/**
 * A term that explains itself on hover, focus, or click.
 *
 * WHY THIS EXISTS. Everything on this page is either a payments term
 * ("mandate", "AutoPay"), a regulatory one ("pre-notification", "AFA"), or a
 * statistical one ("sign test", "seed"). A first-time reader needs all three
 * and wants none of them in their way. Putting the definitions inline turns
 * the page into a textbook; leaving them out turns it into a page only the
 * author can read.
 *
 * So the definition is present, always, and costs nothing until asked for.
 * Nothing is hidden -- every definition is in the DOM and in the accessible
 * tree from first render, which is also what lets the SSR render-check see
 * them.
 *
 * INTERACTION CONTRACT, and it has to be all three:
 *   - hover        : mouse users, no click needed
 *   - focus        : keyboard users, Tab reaches it because it is a <button>
 *   - click / tap   : touch devices, where hover does not exist at all
 * Escape closes. A click outside closes. Clicking the term again closes it,
 * so a tap is not a one-way door.
 *
 * It is a <button>, not a <span> with handlers, so it is reachable by Tab and
 * announced as interactive without any aria-role theatre.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'

// useLayoutEffect logs a warning when React renders on the server, and this
// component IS server-rendered -- ssr-check.tsx renders the whole page to a
// string. There is no layout to measure without a window, so fall back to
// useEffect there. The effect below is a no-op in that environment anyway.
const useMeasureEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect


export function Explain({
  term,
  children,
  wide = false,
}: {
  /** The word as it appears in the sentence. */
  term: string
  /** The plain-language explanation. Written for someone who has never seen
   *  a payments system. */
  children: React.ReactNode
  /** For definitions that need more than a line or two. */
  wide?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [pinned, setPinned] = useState(false)
  const id = useId()
  const wrapRef = useRef<HTMLSpanElement>(null)
  const popRef = useRef<HTMLSpanElement>(null)

  // A pinned popover (opened by click/tap) survives the mouse leaving, so a
  // reader can move the pointer onto the text to read it. An unpinned one
  // follows the pointer.
  useEffect(() => {
    if (!pinned) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setPinned(false)
        setOpen(false)
      }
    }
    const onDown = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) {
        setPinned(false)
        setOpen(false)
      }
    }
    document.addEventListener('keydown', onKey)
    document.addEventListener('mousedown', onDown)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('mousedown', onDown)
    }
  }, [pinned])


  // Nudge the popover back inside the viewport.
  //
  // It is centred on its term, so a term near either edge of the window puts
  // half the popover off-screen -- seen for real on the acquirer view, where
  // the first column legend sits ~100px from the left and its definition was
  // cut in half. Measuring is the only honest way to know: the term's position
  // depends on line wrapping, font metrics and the reader's window width, none
  // of which a static class can predict.
  //
  // A layout effect, so the correction lands in the same frame the popover
  // becomes visible and the reader never sees it jump.
  useMeasureEffect(() => {
    const el = popRef.current
    if (!el) return
    if (!(open || pinned)) {
      el.style.marginLeft = ''
      return
    }
    el.style.marginLeft = ''
    const rect = el.getBoundingClientRect()
    const GUTTER = 10
    if (rect.left < GUTTER) el.style.marginLeft = `${GUTTER - rect.left}px`
    else if (rect.right > window.innerWidth - GUTTER)
      el.style.marginLeft = `${window.innerWidth - GUTTER - rect.right}px`
  }, [open, pinned])

  return (
    <span className="explain" ref={wrapRef}>
      <button
        type="button"
        className={`explain-term${open || pinned ? ' is-open' : ''}`}
        aria-expanded={open || pinned}
        aria-describedby={id}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => !pinned && setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={() => {
          setPinned((p) => !p)
          setOpen(true)
        }}
      >
        {term}
        <span className="explain-marker" aria-hidden="true">
          ?
        </span>
      </button>
      {/* Always rendered. Visibility is CSS, so the text is in the accessible
          tree and in the server-rendered HTML whether or not it is on screen. */}
      <span
        id={id}
        ref={popRef}
        role="tooltip"
        className={`explain-pop${open || pinned ? ' is-open' : ''}${
          wide ? ' explain-pop--wide' : ''
        }`}
      >
        {children}
      </span>
    </span>
  )
}

/**
 * A block-level version: a heading you can click to reveal a paragraph.
 *
 * Used where the explanation is a few sentences rather than a phrase -- "how
 * does it actually decide?" -- and where showing it by default would bury the
 * thing the section is actually about.
 *
 * <details>/<summary> rather than a hand-rolled disclosure: it is keyboard
 * operable, screen-reader announced, findable by in-page search even while
 * collapsed, and it needs no JavaScript to work.
 */
export function ExplainMore({
  label,
  children,
  defaultOpen = false,
}: {
  label: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="explain-more" open={defaultOpen}>
      <summary>
        <span className="explain-more-label">{label}</span>
      </summary>
      <div className="explain-more-body">{children}</div>
    </details>
  )
}
