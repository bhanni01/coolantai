import { useMemo, useState } from 'react'
import { LiveLog } from './components/LiveLog'
import { LoopCounter } from './components/LoopCounter'
import { Panel } from './components/Panel'
import { PipelineDiagram } from './components/PipelineDiagram'
import { ProfileForm } from './components/ProfileForm'
import { ResultsPanel } from './components/ResultsPanel'
import { USE_MOCK, useRun } from './hooks/useRun'
import { DEFAULT_PROFILE } from './lib/profile'
import type { RunPhase } from './lib/types'

const TOTAL_NODES = 6

export default function App() {
  const {
    phase,
    nodes,
    logs,
    loop,
    result,
    error,
    sources,
    noSourcesAboveThreshold,
    crossChecks,
    start,
    reset,
  } = useRun()
  const [profile, setProfile] = useState(DEFAULT_PROFILE)

  const doneCount = useMemo(
    () => Object.values(nodes).filter((n) => n.status === 'done').length,
    [nodes],
  )

  const runLabel =
    phase === 'idle'
      ? 'Run Pipeline'
      : phase === 'running'
        ? 'Restart'
        : phase === 'error'
          ? 'Retry'
          : 'Run Again'

  return (
    <div className="mx-auto flex min-h-screen max-w-[1400px] flex-col gap-5 px-5 py-6 lg:px-8">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg border border-teal/30 bg-teal/10">
            <div className="h-3.5 w-3.5 rounded-sm bg-teal" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-display text-xl font-semibold tracking-tight text-ink">
                Coolant Formulation Copilot
              </h1>
              <span
                className={`rounded border px-1.5 py-px font-mono text-[9px] uppercase tracking-wider ${
                  USE_MOCK ? 'border-copper/40 text-copper' : 'border-teal/40 text-teal'
                }`}
              >
                {USE_MOCK ? 'mock' : 'live'}
              </span>
            </div>
            <p className="font-sans text-xs text-ink-muted">
              PFAS-free data-center coolant · multi-agent LangGraph pipeline
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <StatusPill phase={phase} />
          {phase !== 'idle' && (
            <button
              onClick={reset}
              className="rounded-lg border border-hairline px-3 py-2 font-sans text-sm text-ink-muted transition-colors hover:border-ink-faint hover:text-ink"
            >
              Reset
            </button>
          )}
          <button
            onClick={() => start(profile)}
            className="rounded-lg border border-teal/50 bg-teal/15 px-4 py-2 font-sans text-sm font-medium text-teal transition-colors hover:bg-teal/25"
          >
            {runLabel}
          </button>
        </div>
      </header>

      {/* Error banner */}
      {phase === 'error' && error && (
        <div className="anim-rise-in flex items-start gap-3 rounded-xl border border-red/40 bg-red/10 px-4 py-3">
          <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-red" />
          <div>
            <div className="font-display text-sm font-medium text-red">Run failed</div>
            <div className="font-mono text-xs text-ink-muted">{error}</div>
            {!USE_MOCK && (
              <div className="mt-1 font-mono text-[11px] text-ink-faint">
                Check the backend is running (uvicorn api.main:app) — or set VITE_USE_MOCK=true for the offline demo.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Datacenter profile — the only run input; the backend resolves the spec */}
      <Panel
        title="Datacenter Profile"
        eyebrow="closed selections → server-resolved target spec"
      >
        <ProfileForm profile={profile} onChange={setProfile} disabled={phase === 'running'} />
      </Panel>

      {/* Pipeline diagram */}
      <Panel title="Pipeline" eyebrow="research → generator → evaluate ∥ → critic ↺ → planner">
        <PipelineDiagram
          nodes={nodes}
          sources={sources}
          noSourcesAboveThreshold={noSourcesAboveThreshold}
        />
      </Panel>

      {/* Stat row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <LoopCounter loop={loop} phase={phase} />
        <ProgressCard doneCount={doneCount} phase={phase} />
      </div>

      {/* Log + results */}
      <div className="grid min-h-[440px] flex-1 grid-cols-1 gap-5 xl:grid-cols-2">
        <LiveLog logs={logs} />
        <ResultsPanel result={result} crossChecks={crossChecks} />
      </div>

      <footer className="pt-1 text-center font-mono text-[10px] text-ink-faint">
        {USE_MOCK
          ? 'driven by a mock state machine · offline demo fallback (VITE_USE_MOCK=true)'
          : 'live · streaming node events from the FastAPI backend over SSE'}
      </footer>
    </div>
  )
}

function StatusPill({ phase }: { phase: RunPhase }) {
  const map: Record<RunPhase, { label: string; dot: string; text: string; border: string }> = {
    idle: { label: 'idle', dot: 'bg-ink-faint/60', text: 'text-ink-muted', border: 'border-hairline' },
    running: { label: 'running', dot: 'bg-copper anim-node-pulse-copper', text: 'text-copper', border: 'border-copper/40' },
    complete: { label: 'complete', dot: 'bg-teal', text: 'text-teal', border: 'border-teal/40' },
    error: { label: 'error', dot: 'bg-red', text: 'text-red', border: 'border-red/40' },
  }
  const s = map[phase]
  return (
    <div className={`flex items-center gap-2 rounded-full border px-3 py-1.5 ${s.border}`}>
      <span className={`h-2 w-2 rounded-full ${s.dot}`} />
      <span className={`font-mono text-[11px] uppercase tracking-wider ${s.text}`}>{s.label}</span>
    </div>
  )
}

function ProgressCard({ doneCount, phase }: { doneCount: number; phase: RunPhase }) {
  const pct = phase === 'complete' ? 100 : Math.round((doneCount / TOTAL_NODES) * 100)
  return (
    <div className="flex flex-col rounded-xl border border-hairline bg-surface/70 p-4">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">nodes completed</span>
        <span className="font-mono text-[10px] text-ink-faint">{pct}%</span>
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="font-mono text-4xl font-medium leading-none text-teal">{doneCount}</span>
        <span className="font-mono text-lg text-ink-faint">/ {TOTAL_NODES}</span>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-hairline">
        <div className="h-full rounded-full bg-teal transition-[width] duration-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
