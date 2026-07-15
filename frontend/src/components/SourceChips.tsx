import type { SourceChip } from '../lib/types'

interface SourceChipsProps {
  sources: SourceChip[]
  /** research reported nothing cleared the similarity threshold. */
  noSources: boolean
  /** research node is currently active (drives the "retrieving" placeholder). */
  active: boolean
}

// Keep the panel under the research node compact; overflow collapses to a count.
const MAX_VISIBLE = 6

function basename(path: string): string {
  const file = path.split(/[\\/]/).pop() || path
  return file.replace(/\.(pdf|txt|md|json|csv)$/i, '')
}

/**
 * Live list of retrieved source chunks under the research PipelineNode. Chips
 * populate as `source_retrieved` detail events arrive; a distinct muted state
 * renders when the run reported no sources above the similarity threshold.
 */
export function SourceChips({ sources, noSources, active }: SourceChipsProps) {
  // Nothing to show yet and nothing to say — stay out of the way when idle.
  if (!noSources && sources.length === 0 && !active) return null

  const visible = sources.slice(0, MAX_VISIBLE)
  const overflow = sources.length - visible.length

  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-ink-faint">
        sources{sources.length > 0 && !noSources ? ` · ${sources.length}` : ''}
      </span>

      {noSources ? (
        <span
          className="inline-flex w-fit items-center gap-1 rounded border border-dashed border-hairline bg-surface-raised/40 px-1.5 py-0.5 font-mono text-[9px] text-ink-faint/80"
          title="No retrieval query scored above the similarity threshold"
        >
          <span className="h-1 w-1 rounded-full bg-ink-faint/60" />
          no sources above threshold
        </span>
      ) : sources.length === 0 ? (
        <span className="font-mono text-[9px] text-ink-faint/70">retrieving…</span>
      ) : (
        <div className="flex flex-wrap gap-1">
          {visible.map((s, i) => (
            <span
              key={`${s.sourceDocument}-${i}`}
              className="anim-rise-in inline-flex max-w-[150px] items-center gap-1 rounded border border-teal/25 bg-teal/5 px-1.5 py-0.5 font-mono text-[9px] text-ink-muted"
              title={`${s.sourceDocument}${s.similarityScore != null ? ` · score ${s.similarityScore.toFixed(4)}` : ''}`}
            >
              <span className="truncate">{basename(s.sourceDocument)}</span>
              {s.similarityScore != null && (
                <span className="shrink-0 tabular-nums text-teal/90">
                  {s.similarityScore.toFixed(2)}
                </span>
              )}
            </span>
          ))}
          {overflow > 0 && (
            <span className="inline-flex items-center rounded border border-hairline bg-surface-raised/40 px-1.5 py-0.5 font-mono text-[9px] text-ink-faint">
              +{overflow}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
