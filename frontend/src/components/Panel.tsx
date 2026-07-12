import type { ReactNode } from 'react'

interface PanelProps {
  title: string
  eyebrow?: string
  right?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}

export function Panel({ title, eyebrow, right, children, className = '', bodyClassName = '' }: PanelProps) {
  return (
    <section
      className={`flex flex-col rounded-xl border border-hairline bg-surface/70 backdrop-blur-sm shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset,0_18px_40px_-24px_rgba(0,0,0,0.7)] ${className}`}
    >
      <header className="flex items-center justify-between gap-3 border-b border-hairline px-4 py-3">
        <div className="min-w-0">
          {eyebrow && (
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
              {eyebrow}
            </div>
          )}
          <h2 className="truncate font-display text-sm font-medium text-ink">{title}</h2>
        </div>
        {right}
      </header>
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  )
}
