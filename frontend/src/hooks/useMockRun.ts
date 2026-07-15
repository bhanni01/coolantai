import { useCallback, useEffect, useRef, useState } from 'react'
import { MOCK_RESULT, TIMELINE } from '../lib/mockRun'
import type {
  CrossCheck,
  LogEntry,
  NodeName,
  NodeStateMap,
  RunController,
  RunPhase,
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

const DOWNSTREAM_OF_GENERATOR: NodeName[] = [
  'property_estimator',
  'compliance_checker',
  'critic',
]

function initialNodes(): NodeStateMap {
  return Object.fromEntries(
    ALL_NODES.map((n) => [n, { status: 'idle', loop: 0 }]),
  ) as NodeStateMap
}

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/**
 * Drives the pipeline UI from a scripted timeline on real timers — the
 * guaranteed-offline fallback. Implements the same RunController contract as
 * useLiveRun (its `error` channel is always null), so the two are swappable.
 */
export function useMockRun(): RunController {
  const [phase, setPhase] = useState<RunPhase>('idle')
  const [nodes, setNodes] = useState<NodeStateMap>(initialNodes)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loop, setLoop] = useState(0)
  const [result, setResult] = useState<RunResult | null>(null)
  const [sources, setSources] = useState<SourceChip[]>([])
  const [crossChecks, setCrossChecks] = useState<Record<string, CrossCheck[]>>({})

  const runIdRef = useRef(0)
  const logIdRef = useRef(0)

  const appendLogs = useCallback(
    (entries: Omit<LogEntry, 'id' | 'ts'>[]) => {
      setLogs((prev) => [
        ...prev,
        ...entries.map((e) => ({ ...e, id: ++logIdRef.current, ts: new Date() })),
      ])
    },
    [],
  )

  const play = useCallback(
    async (myRun: number) => {
      const live = () => runIdRef.current === myRun

      setPhase('running')
      setNodes(initialNodes())
      setLogs([])
      setResult(null)
      setLoop(0)
      setSources([])
      setCrossChecks({})
      appendLogs([
        { node: 'system', message: 'Run started — spec "DC-Coolant-A"', level: 'info', loop: 0 },
      ])

      await delay(300)
      if (!live()) return

      for (const beat of TIMELINE) {
        if (!live()) return

        // Activate this beat's nodes; on a revision, reset the downstream
        // nodes first so the loop reads as a genuine re-run.
        setNodes((prev) => {
          const next: NodeStateMap = { ...prev }
          if (beat.activate.includes('generator') && beat.loop > 0) {
            for (const n of DOWNSTREAM_OF_GENERATOR) {
              next[n] = { status: 'idle', loop: next[n].loop }
            }
          }
          for (const n of beat.activate) {
            next[n] = { ...next[n], status: 'active', loop: beat.loop, badge: undefined }
          }
          return next
        })
        setLoop(beat.loop)

        // Stream source chips in while the node is still active — mirrors the
        // live `source_retrieved` detail events arriving mid-node.
        if (beat.sources?.length) {
          const chips = beat.sources
          const step = beat.durationMs / (chips.length + 1)
          for (const chip of chips) {
            await delay(step)
            if (!live()) return
            setSources((prev) => [...prev, chip])
          }
          await delay(step)
        } else {
          await delay(beat.durationMs)
        }
        if (!live()) return

        setNodes((prev) => {
          const next: NodeStateMap = { ...prev }
          for (const n of beat.activate) {
            next[n] = { status: 'done', loop: beat.loop, badge: beat.badges?.[n] }
          }
          return next
        })
        // Cross-checks land as the property_estimator beat completes, keyed by
        // candidate id (mirrors the live `cross_check` detail events).
        if (beat.crossChecks?.length) {
          setCrossChecks((prev) => {
            const next = { ...prev }
            for (const c of beat.crossChecks!) {
              next[c.candidateId] = [...(next[c.candidateId] ?? []), c]
            }
            return next
          })
        }
        appendLogs(beat.logs.map((l) => ({ ...l, loop: beat.loop })))

        await delay(180)
        if (!live()) return
      }

      setResult(MOCK_RESULT)
      appendLogs([
        { node: 'system', message: 'Run complete — 3 candidates shortlisted', level: 'success', loop: 3 },
      ])
      setPhase('complete')
    },
    [appendLogs],
  )

  // The scripted timeline ignores the submitted profile — it exists only to
  // satisfy the shared RunController contract.
  const start = useCallback(() => {
    const id = ++runIdRef.current
    void play(id)
  }, [play])

  const reset = useCallback(() => {
    runIdRef.current++ // cancel any in-flight run
    setPhase('idle')
    setNodes(initialNodes())
    setLogs([])
    setLoop(0)
    setResult(null)
    setSources([])
    setCrossChecks({})
  }, [])

  // Cancel on unmount.
  useEffect(() => () => void runIdRef.current++, [])

  return {
    phase,
    nodes,
    logs,
    loop,
    result,
    error: null,
    sources,
    noSourcesAboveThreshold: false,
    crossChecks,
    start,
    reset,
  }
}
