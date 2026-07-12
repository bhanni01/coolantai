import { useEffect, useRef, useState } from 'react'
import type { RunPhase } from '../lib/types'

const MAX = 3

export function LoopCounter({ loop, phase }: { loop: number; phase: RunPhase }) {
  const [bump, setBump] = useState(false)
  const prev = useRef(loop)

  useEffect(() => {
    if (loop !== prev.current) {
      prev.current = loop
      setBump(true)
      const t = setTimeout(() => setBump(false), 500)
      return () => clearTimeout(t)
    }
  }, [loop])

  const isRevision = loop > 0
  const label = phase === 'idle' ? 'awaiting run' : isRevision ? 'revision loop' : 'initial pass'

  return (
    <div className="flex flex-col rounded-xl border border-hairline bg-surface/70 p-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
          {label}
        </span>
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            isRevision ? 'bg-copper' : phase === 'idle' ? 'bg-ink-faint/50' : 'bg-teal'
          }`}
        />
      </div>

      <div className="mt-2 flex items-baseline gap-1.5">
        <span
          className={`font-mono text-4xl font-medium leading-none ${
            bump ? 'anim-loop-bump' : ''
          } ${isRevision ? 'text-copper' : phase === 'idle' ? 'text-ink-muted' : 'text-teal'}`}
        >
          {loop}
        </span>
        <span className="font-mono text-lg text-ink-faint">/ {MAX}</span>
      </div>

      {/* segmented progress toward the revision cap */}
      <div className="mt-3 flex gap-1.5">
        {Array.from({ length: MAX }).map((_, i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
              i < loop ? 'bg-copper' : 'bg-hairline'
            }`}
          />
        ))}
      </div>
    </div>
  )
}
