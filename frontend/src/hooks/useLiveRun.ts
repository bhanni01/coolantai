import { useCallback, useEffect, useRef, useState } from 'react'
import { eventsUrl, mapCompleteResult, postRun } from '../lib/api'
import type { RawCompleteResult } from '../lib/api'
import type { DatacenterProfile } from '../lib/profile'
import type {
  CrossCheck,
  LogEntry,
  LogLevel,
  NodeName,
  NodeRuntime,
  NodeStateMap,
  RunController,
  RunResult,
  SourceChip,
} from '../lib/types'

const ALL_NODES: NodeName[] = [
  'research',
  'generator',
  'property_estimator',
  'compliance_checker',
  'critic',
  'experiment_planner',
]

function initialNodes(): NodeStateMap {
  return Object.fromEntries(
    ALL_NODES.map((n) => [n, { status: 'idle', loop: 0 }]),
  ) as NodeStateMap
}

// If no event arrives for this long, treat the run as stalled rather than
// letting the UI sit silently. Comfortably longer than the slowest real node.
const STALL_MS = 45_000

// Backend node event shape (api/runner.py node_event).
interface NodeEvent {
  node_name: NodeName
  status: 'active' | 'completed'
  output_summary: string
  timestamp: string
  loop_iteration: number
}

// Backend detail event shape (api/runner.py, custom stream). Additive alongside
// the node status events; arrives between a node's 'active' and 'completed'.
interface DetailEvent {
  node_name: NodeName
  detail_type: 'source_retrieved' | 'cross_check'
  payload: Record<string, unknown>
  timestamp: string
  loop_iteration: number
}

function complianceBadge(summary: string): NodeRuntime['badge'] {
  const fail = Number(/(\d+)\s*fail/.exec(summary)?.[1] ?? 0)
  const review = Number(/(\d+)\s*needs_review/.exec(summary)?.[1] ?? 0)
  if (fail > 0) return { text: String(fail), tone: 'red' }
  if (review > 0) return { text: String(review), tone: 'amber' }
  return { text: '0', tone: 'teal' }
}

function deriveLevel(node: NodeName, summary: string, loop: number): LogLevel {
  if (node === 'compliance_checker') {
    if (/[1-9]\d*\s*fail/.test(summary)) return 'error'
    if (/[1-9]\d*\s*needs_review/.test(summary)) return 'warn'
    return 'info'
  }
  if (node === 'critic') {
    const verdict = /\((accept|revise|reject)\)/.exec(summary)?.[1]
    return verdict === 'accept' ? 'success' : 'revision'
  }
  if (node === 'experiment_planner') return 'success'
  if (node === 'generator' && loop > 0) return 'revision'
  return 'info'
}

/**
 * Drives the pipeline UI from the live FastAPI SSE stream. Emits the same
 * RunController shape as useMockRun, so the presentational components are
 * unchanged. Handles POST failure, a server error event, connection loss, and
 * a stalled stream — all surface as phase 'error' with a message.
 */
export function useLiveRun(): RunController {
  const [phase, setPhase] = useState<RunController['phase']>('idle')
  const [nodes, setNodes] = useState<NodeStateMap>(initialNodes)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loop, setLoop] = useState(0)
  const [result, setResult] = useState<RunResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sources, setSources] = useState<SourceChip[]>([])
  const [noSourcesAboveThreshold, setNoSources] = useState(false)
  const [crossChecks, setCrossChecks] = useState<Record<string, CrossCheck[]>>({})

  const runIdRef = useRef(0) // bumps on every start/reset to invalidate stale async
  const logIdRef = useRef(0)
  const loopRef = useRef(0) // latest loop, for closures that outlive a render
  const esRef = useRef<EventSource | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const stallRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const terminatedRef = useRef(false)

  const setLoopValue = useCallback((v: number) => {
    loopRef.current = v
    setLoop(v)
  }, [])

  const teardown = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
    abortRef.current?.abort()
    abortRef.current = null
    if (stallRef.current) clearTimeout(stallRef.current)
    stallRef.current = null
  }, [])

  const appendLog = useCallback((entry: Omit<LogEntry, 'id' | 'ts'>) => {
    setLogs((prev) => [...prev, { ...entry, id: ++logIdRef.current, ts: new Date() }])
  }, [])

  const start = useCallback((profile: DatacenterProfile) => {
    const myRun = ++runIdRef.current
    const live = () => runIdRef.current === myRun
    const fail = (message: string) => {
      if (!live() || terminatedRef.current) return
      terminatedRef.current = true
      teardown()
      appendLog({ node: 'system', message, level: 'error', loop: 0 })
      setError(message)
      setPhase('error')
    }
    const armStall = () => {
      if (stallRef.current) clearTimeout(stallRef.current)
      stallRef.current = setTimeout(
        () => fail('No events received for 45s — the run appears stalled.'),
        STALL_MS,
      )
    }

    teardown()
    terminatedRef.current = false
    setPhase('running')
    setNodes(initialNodes())
    setLogs([])
    setLoopValue(0)
    setResult(null)
    setError(null)
    setSources([])
    setNoSources(false)
    setCrossChecks({})
    appendLog({
      node: 'system',
      message: `Run started — ${profile.cooling_method} · ${profile.rack_density} · ${profile.climate_zone} · ${profile.regulatory_region} · optimize ${profile.optimization_priority}`,
      level: 'info',
      loop: 0,
    })

    const abort = new AbortController()
    abortRef.current = abort
    // POST the profile; the backend resolves the full spec server-side.
    postRun(profile, abort.signal)
      .then((runId) => {
        if (!live()) return
        armStall()
        const es = new EventSource(eventsUrl(runId))
        esRef.current = es

        es.onopen = () => armStall()

        es.addEventListener('node', (e) => {
          if (!live()) return
          armStall()
          const ev = JSON.parse((e as MessageEvent).data) as NodeEvent
          setLoopValue(ev.loop_iteration)
          if (ev.status === 'active') {
            setNodes((prev) => ({
              ...prev,
              [ev.node_name]: { status: 'active', loop: ev.loop_iteration },
            }))
          } else {
            const badge =
              ev.node_name === 'compliance_checker'
                ? complianceBadge(ev.output_summary)
                : undefined
            setNodes((prev) => ({
              ...prev,
              [ev.node_name]: { status: 'done', loop: ev.loop_iteration, badge },
            }))
            appendLog({
              node: ev.node_name,
              message: ev.output_summary,
              level: deriveLevel(ev.node_name, ev.output_summary, ev.loop_iteration),
              loop: ev.loop_iteration,
            })
          }
        })

        // Fine-grained detail events, additive to the node status events. They
        // arrive while their parent node is still 'active', so they never touch
        // node/loop/phase state — only the dedicated detail channels.
        es.addEventListener('detail', (e) => {
          if (!live()) return
          armStall()
          const ev = JSON.parse((e as MessageEvent).data) as DetailEvent
          const p = ev.payload
          if (ev.detail_type === 'source_retrieved') {
            if (p.no_sources_above_threshold === true) {
              setNoSources(true)
              return
            }
            setSources((prev) => [
              ...prev,
              {
                sourceDocument: String(p.source_document ?? 'unknown'),
                similarityScore:
                  typeof p.similarity_score === 'number' ? p.similarity_score : null,
              },
            ])
          } else if (ev.detail_type === 'cross_check') {
            const candidateId = String(p.candidate_id ?? '')
            const check: CrossCheck = {
              candidateId,
              property: String(p.property ?? ''),
              estimate: Number(p.estimate),
              unit: String(p.unit ?? ''),
              referenceValue: Number(p.reference_value),
              referenceSource: String(p.reference_source ?? ''),
              status: p.status === 'conflict' ? 'conflict' : 'validated',
            }
            setCrossChecks((prev) => ({
              ...prev,
              [candidateId]: [...(prev[candidateId] ?? []), check],
            }))
          }
        })

        es.addEventListener('complete', (e) => {
          if (!live() || terminatedRef.current) return
          terminatedRef.current = true
          const data = JSON.parse((e as MessageEvent).data) as { result: RawCompleteResult }
          const mapped = mapCompleteResult(data.result)
          setResult(mapped)
          appendLog({
            node: 'system',
            message: `Run complete — ${mapped.shortlist.length} candidate(s) shortlisted`,
            level: 'success',
            loop: loopRef.current,
          })
          setPhase('complete')
          teardown()
        })

        // Named server 'error' events carry .data; native connection errors do not.
        es.addEventListener('error', (e) => {
          const data = (e as MessageEvent).data
          if (data) {
            const payload = JSON.parse(data) as { message?: string }
            fail(payload.message ?? 'Backend reported an error.')
          } else if (!terminatedRef.current) {
            // Connection dropped / could not connect (and not a normal close).
            fail('Lost connection to the backend event stream.')
          }
        })
      })
      .catch((err) => {
        if (abort.signal.aborted) return // superseded by a restart/reset
        fail(`Could not start run: ${err instanceof Error ? err.message : String(err)}`)
      })
  }, [appendLog, teardown, setLoopValue])

  const reset = useCallback(() => {
    runIdRef.current++
    terminatedRef.current = true
    teardown()
    setPhase('idle')
    setNodes(initialNodes())
    setLogs([])
    setLoopValue(0)
    setResult(null)
    setError(null)
    setSources([])
    setNoSources(false)
    setCrossChecks({})
  }, [teardown, setLoopValue])

  useEffect(() => () => {
    runIdRef.current++
    teardown()
  }, [teardown])

  return {
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
  }
}
