import { useEffect, useRef } from 'react'
import { Panel } from './Panel'
import type { LogEntry, LogLevel } from '../lib/types'

const LEVEL_STYLE: Record<LogLevel, { dot: string; text: string; tag: string }> = {
  info: { dot: 'bg-teal', text: 'text-ink', tag: 'text-teal/80' },
  success: { dot: 'bg-teal', text: 'text-ink', tag: 'text-teal' },
  revision: { dot: 'bg-copper', text: 'text-ink', tag: 'text-copper' },
  warn: { dot: 'bg-amber', text: 'text-ink', tag: 'text-amber' },
  error: { dot: 'bg-red', text: 'text-ink', tag: 'text-red' },
}

function fmtTime(d: Date) {
  return d.toLocaleTimeString('en-US', { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0')
}

export function LiveLog({ logs }: { logs: LogEntry[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logs])

  return (
    <Panel
      title="Live Event Log"
      eyebrow="graph.astream"
      className="min-h-0"
      bodyClassName="min-h-0"
      right={
        <span className="font-mono text-[11px] text-ink-faint">
          {logs.length} event{logs.length === 1 ? '' : 's'}
        </span>
      }
    >
      <div ref={scrollRef} className="scroll-slim h-full overflow-y-auto px-4 py-3">
        {logs.length === 0 ? (
          <div className="flex h-full items-center justify-center py-10 font-mono text-xs text-ink-faint">
            awaiting run…
          </div>
        ) : (
          <ul className="space-y-1.5">
            {logs.map((log) => {
              const s = LEVEL_STYLE[log.level]
              return (
                <li key={log.id} className="anim-log-in flex items-start gap-2.5 font-mono text-[12px] leading-relaxed">
                  <span className="mt-1.5 shrink-0 text-[10px] text-ink-faint tabular-nums">{fmtTime(log.ts)}</span>
                  <span className={`mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full ${s.dot}`} />
                  <span className={`shrink-0 ${s.tag}`}>[{log.node}]</span>
                  <span className={`min-w-0 ${s.text}`}>{log.message}</span>
                  {log.loop > 0 && (
                    <span className="ml-auto shrink-0 rounded border border-copper/40 px-1 text-[9px] text-copper">
                      L{log.loop}
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </Panel>
  )
}
