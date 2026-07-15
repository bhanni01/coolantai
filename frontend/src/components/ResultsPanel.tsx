import { Panel } from './Panel'
import type { CandidateResult, CrossCheck, RunResult, Verdict } from '../lib/types'

const VERDICT_STYLE: Record<Verdict, string> = {
  accept: 'border-teal/40 text-teal',
  revise: 'border-copper/40 text-copper',
  reject: 'border-red/40 text-red',
}

function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) >= 100) return String(Math.round(v))
  if (Number.isInteger(v)) return String(v)
  return String(parseFloat(v.toFixed(3)))
}

interface ResultsPanelProps {
  result: RunResult | null
  /** Live cross-check detail events, keyed by candidate id. */
  crossChecks: Record<string, CrossCheck[]>
}

export function ResultsPanel({ result, crossChecks }: ResultsPanelProps) {
  return (
    <Panel
      title="Results"
      eyebrow="ranked shortlist · DOE · usage"
      className="min-h-0"
      bodyClassName="min-h-0"
      right={
        result ? (
          <span className="font-mono text-[11px] text-teal">{result.candidates.length} shortlisted</span>
        ) : (
          <span className="font-mono text-[11px] text-ink-faint">pending</span>
        )
      }
    >
      {!result ? (
        <div className="flex h-full min-h-[220px] items-center justify-center px-6 text-center font-mono text-xs text-ink-faint">
          results appear here once the run completes
        </div>
      ) : (
        <div className="scroll-slim h-full space-y-5 overflow-y-auto px-4 py-4">
          {/* Ranked candidates */}
          <div className="space-y-2.5">
            {result.candidates.map((c, i) => (
              <CandidateCard
                key={c.id}
                candidate={c}
                rank={i + 1}
                checks={crossChecks[c.id] ?? []}
              />
            ))}
          </div>

          {/* DOE plan */}
          <div className="anim-rise-in rounded-lg border border-hairline bg-surface-raised/50 p-3.5">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
                lab validation plan
              </span>
              <span className="rounded border border-teal/30 px-1.5 py-px font-mono text-[10px] text-teal">
                {result.doe.design_type}
              </span>
            </div>
            <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
              <Metric label="runs" value={String(result.doe.runs)} />
              <Metric label="replicates" value={`×${result.doe.replicates}`} />
              <Metric label="factors" value={String(result.doe.factors.length)} />
            </div>
            <div className="mt-2.5 space-y-1">
              {result.doe.factors.map((f) => (
                <div key={f.name} className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="text-ink-muted">{f.name}</span>
                  <span className="text-ink-faint">({f.unit})</span>
                  <span className="ml-auto text-ink">{f.levels.join(' · ')}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Token / cost */}
          <TokenReadout result={result} />
        </div>
      )}
    </Panel>
  )
}

function CandidateCard({
  candidate,
  rank,
  checks,
}: {
  candidate: CandidateResult
  rank: number
  checks: CrossCheck[]
}) {
  const { compliance } = candidate
  return (
    <div className="anim-rise-in rounded-lg border border-hairline bg-surface-raised/50 p-3.5">
      <div className="flex items-center gap-2.5">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-teal/30 font-mono text-xs text-teal">
          {rank}
        </span>
        <div className="min-w-0">
          <div className="truncate font-display text-sm font-medium text-ink">{candidate.name}</div>
          <div className="font-mono text-[10px] text-ink-faint">{candidate.id}</div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <ValidationBadge checks={checks} />
          <span className={`rounded border px-1.5 py-px font-mono text-[10px] uppercase ${VERDICT_STYLE[candidate.verdict]}`}>
            {candidate.verdict}
          </span>
          <div className="text-right">
            <div className="font-mono text-lg leading-none text-teal">{candidate.overall.toFixed(1)}</div>
            <div className="font-mono text-[9px] text-ink-faint">/ 10</div>
          </div>
        </div>
      </div>

      {/* property estimates */}
      <div className="mt-3 grid grid-cols-2 gap-1.5">
        {candidate.properties.map((p) => (
          <div key={p.name} className="flex items-center justify-between rounded border border-hairline/70 bg-graphite/40 px-2 py-1">
            <span className="truncate font-mono text-[10px] text-ink-faint">{p.name}</span>
            <span className="ml-2 flex shrink-0 items-baseline gap-1">
              <span className={`font-mono text-[11px] ${p.meets === false ? 'text-red' : 'text-ink'}`}>{p.value}</span>
              <span className="font-mono text-[9px] text-ink-faint">{p.unit}</span>
            </span>
          </div>
        ))}
      </div>

      {/* compliance summary */}
      <div className="mt-2.5 flex items-center gap-3 font-mono text-[10px]">
        <span className="text-ink-faint uppercase tracking-wider">compliance</span>
        <span className="text-teal">{compliance.pass} pass</span>
        {compliance.needs_review > 0 && <span className="text-amber">{compliance.needs_review} review</span>}
        <span className={compliance.fail > 0 ? 'text-red' : 'text-ink-faint'}>{compliance.fail} fail</span>
      </div>

      {/* cross-check against extracted reference profiles */}
      {checks.length > 0 && (
        <div className="mt-2.5 border-t border-hairline/60 pt-2.5">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            reference cross-check
          </div>
          <div className="space-y-1">
            {checks.map((c, i) => {
              const conflict = c.status === 'conflict'
              return (
                <div
                  key={`${c.property}-${i}`}
                  className="flex items-center gap-2 font-mono text-[10px]"
                  title={`${c.status} vs ${c.referenceSource}`}
                >
                  <span
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${conflict ? 'bg-amber' : 'bg-teal'}`}
                  />
                  <span className="truncate text-ink-muted">{c.property}</span>
                  <span className="ml-auto shrink-0 tabular-nums text-ink">
                    {fmtNum(c.estimate)}
                  </span>
                  <span className="shrink-0 text-ink-faint">vs</span>
                  <span className="shrink-0 tabular-nums text-ink-muted">
                    {fmtNum(c.referenceValue)}
                    {c.unit ? ` ${c.unit}` : ''}
                  </span>
                  <span className={`shrink-0 uppercase ${conflict ? 'text-amber' : 'text-teal'}`}>
                    {conflict ? 'conflict' : 'ok'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function ValidationBadge({ checks }: { checks: CrossCheck[] }) {
  if (checks.length === 0) return null
  const conflicts = checks.filter((c) => c.status === 'conflict').length
  const validated = checks.length - conflicts
  const hasConflict = conflicts > 0
  return (
    <span
      className={`rounded border px-1.5 py-px font-mono text-[10px] ${
        hasConflict ? 'border-amber/50 text-amber' : 'border-teal/40 text-teal'
      }`}
      title={`${validated} validated · ${conflicts} conflict against reference profiles`}
    >
      {hasConflict ? `⚠ ${conflicts}` : `✓ ${validated}`}
    </span>
  )
}

function TokenReadout({ result }: { result: RunResult }) {
  const max = Math.max(...result.tokenCost.by_node.map((n) => n.prompt + n.completion))
  return (
    <div className="anim-rise-in rounded-lg border border-hairline bg-surface-raised/50 p-3.5">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">token &amp; cost</span>
        <div className="flex items-baseline gap-4">
          <Metric label="tokens" value={result.tokenCost.total_tokens.toLocaleString()} />
          <Metric label="usd" value={`$${result.tokenCost.total_cost_usd.toFixed(4)}`} accent />
        </div>
      </div>
      <div className="space-y-1.5">
        {result.tokenCost.by_node.map((n) => {
          const total = n.prompt + n.completion
          return (
            <div key={n.node} className="flex items-center gap-2">
              <span className="w-36 shrink-0 truncate font-mono text-[10px] text-ink-muted">{n.node}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-graphite">
                <div className="h-full rounded-full bg-teal/60" style={{ width: `${(total / max) * 100}%` }} />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-[10px] text-ink-faint tabular-nums">
                {total.toLocaleString()}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={`font-mono text-sm ${accent ? 'text-copper' : 'text-ink'}`}>{value}</span>
      <span className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">{label}</span>
    </div>
  )
}
