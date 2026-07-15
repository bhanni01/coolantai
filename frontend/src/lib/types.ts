// Shared types for the dashboard. These intentionally mirror the FastAPI
// SSE event / complete-result shapes (api/runner.py) so the mock state machine
// can be swapped for the real /run/{id}/events stream without touching the UI.

import type { DatacenterProfile } from './profile'

export type NodeName =
  | 'research'
  | 'generator'
  | 'property_estimator'
  | 'compliance_checker'
  | 'critic'
  | 'experiment_planner'

export type NodeStatus = 'idle' | 'active' | 'done'

export type RunPhase = 'idle' | 'running' | 'complete' | 'error'

export type LogLevel = 'info' | 'success' | 'revision' | 'warn' | 'error'

export type Verdict = 'accept' | 'revise' | 'reject'

export type ComplianceStatus = 'pass' | 'needs_review' | 'fail'

export interface LogEntry {
  id: number
  ts: Date
  node: NodeName | 'system'
  message: string
  level: LogLevel
  loop: number
}

export interface NodeRuntime {
  status: NodeStatus
  loop: number
  /** Small count badge, e.g. compliance flags, tinted by tone. */
  badge?: { text: string; tone: 'teal' | 'copper' | 'amber' | 'red' }
}

export type NodeStateMap = Record<NodeName, NodeRuntime>

export interface PropertyEstimate {
  name: string
  value: string // pre-formatted for the mono readout
  unit: string
  meets: boolean | null
}

export type CrossCheckStatus = 'validated' | 'conflict'

/** One `source_retrieved` detail event (api/runner.py → research node). */
export interface SourceChip {
  sourceDocument: string
  similarityScore: number | null
}

/** One `cross_check` detail event (property_estimator → cross_check_estimates). */
export interface CrossCheck {
  candidateId: string
  property: string
  estimate: number
  unit: string
  referenceValue: number
  referenceSource: string
  status: CrossCheckStatus
}

export interface CandidateResult {
  id: string
  name: string
  overall: number
  verdict: Verdict
  properties: PropertyEstimate[]
  compliance: { pass: number; needs_review: number; fail: number }
}

export interface DOEFactor {
  name: string
  unit: string
  levels: number[]
}

export interface DOEPlan {
  design_type: string
  runs: number
  replicates: number
  factors: DOEFactor[]
}

export interface NodeTokenCost {
  node: NodeName
  prompt: number
  completion: number
  cost: number
}

export interface TokenCost {
  total_tokens: number
  total_cost_usd: number
  by_node: NodeTokenCost[]
}

export interface RunResult {
  shortlist: string[]
  candidates: CandidateResult[]
  doe: DOEPlan
  tokenCost: TokenCost
}

// The single shape App consumes. useMockRun and useLiveRun both implement it,
// so PipelineDiagram / LiveLog / LoopCounter / ResultsPanel are agnostic to
// which one is driving.
export interface RunController {
  phase: RunPhase
  nodes: NodeStateMap
  logs: LogEntry[]
  loop: number
  result: RunResult | null
  error: string | null
  /** Live source chunks retrieved by the research node, in arrival order. */
  sources: SourceChip[]
  /** True when research reported that nothing cleared the similarity threshold. */
  noSourcesAboveThreshold: boolean
  /** Live property cross-checks emitted by property_estimator, keyed by candidate id. */
  crossChecks: Record<string, CrossCheck[]>
  start: (profile: DatacenterProfile) => void
  reset: () => void
}
