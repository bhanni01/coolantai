import type { NodeLayout } from '../lib/pipeline'
import type { NodeRuntime } from '../lib/types'

interface PipelineNodeProps {
  layout: NodeLayout
  runtime: NodeRuntime
  processing: boolean // true when this node's active state is a revision (copper)
}

const BADGE_TONE: Record<NonNullable<NodeRuntime['badge']>['tone'], string> = {
  teal: 'border-teal/40 text-teal',
  copper: 'border-copper/40 text-copper',
  amber: 'border-amber/50 text-amber',
  red: 'border-red/50 text-red',
}

export function PipelineNode({ layout, runtime, processing }: PipelineNodeProps) {
  const { status, badge } = runtime
  const active = status === 'active'
  const done = status === 'done'

  const accent: 'teal' | 'copper' = processing ? 'copper' : 'teal'

  const shell = [
    'group relative flex h-full w-full flex-col justify-center rounded-lg border px-3.5 py-2 transition-colors duration-300',
    active
      ? processing
        ? 'border-copper/70 bg-copper/10 anim-node-pulse-copper'
        : 'border-teal/70 bg-teal/10 anim-node-pulse'
      : done
        ? 'border-teal/35 bg-surface-raised'
        : 'border-hairline bg-surface-raised/60',
  ].join(' ')

  const statusTextClass = active
    ? processing
      ? 'text-copper'
      : 'text-teal'
    : done
      ? 'text-teal/80'
      : 'text-ink-faint'

  return (
    <div className={shell}>
      {/* status dot */}
      <div className="mb-1 flex items-center gap-2">
        <StatusDot status={status} accent={accent} />
        <span
          className={`font-mono text-[9px] uppercase tracking-[0.16em] ${statusTextClass}`}
        >
          {active ? (processing ? 'revising' : 'running') : done ? 'done' : 'idle'}
        </span>
        {badge && (
          <span
            className={`ml-auto rounded-md border px-1.5 py-px font-mono text-[10px] leading-none ${BADGE_TONE[badge.tone]}`}
            title="flag count"
          >
            {badge.text}
          </span>
        )}
      </div>

      <div
        className={`font-display text-[13px] font-medium leading-tight ${
          status === 'idle' ? 'text-ink-muted' : 'text-ink'
        }`}
      >
        {layout.label}
      </div>
      <div className="mt-0.5 flex items-center justify-between gap-2">
        <span className="truncate text-[11px] text-ink-faint">{layout.sub}</span>
        <span className="shrink-0 font-mono text-[9px] text-ink-faint/80">{layout.tier}</span>
      </div>
    </div>
  )
}

function StatusDot({
  status,
  accent,
}: {
  status: NodeRuntime['status']
  accent: 'teal' | 'copper'
}) {
  if (status === 'active') {
    return (
      <span className="relative flex h-2.5 w-2.5 items-center justify-center">
        <span
          className={`absolute h-2.5 w-2.5 rounded-full border-2 border-transparent anim-spin ${
            accent === 'copper' ? 'border-t-copper border-r-copper' : 'border-t-teal border-r-teal'
          }`}
        />
      </span>
    )
  }
  return (
    <span
      className={`h-2 w-2 rounded-full ${
        status === 'done' ? 'bg-teal' : 'bg-ink-faint/50'
      }`}
    />
  )
}
