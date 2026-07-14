import type { DatacenterProfile } from './profile'
import type {
  CandidateResult,
  ComplianceStatus,
  DOEPlan,
  NodeName,
  RunResult,
  TokenCost,
  Verdict,
} from './types'

// Same-origin by default: the FastAPI app serves the built frontend, so /run
// etc. are relative paths. In dev the Vite proxy forwards them to the backend.
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? ''

/** POST the datacenter profile; returns the run_id the events stream is keyed on. */
export async function postRun(
  profile: DatacenterProfile,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
    signal,
  })
  if (!res.ok) {
    throw new Error(`POST /run failed: ${res.status} ${res.statusText}`)
  }
  const body = (await res.json()) as { run_id?: string }
  if (!body.run_id) throw new Error('POST /run returned no run_id')
  return body.run_id
}

export function eventsUrl(runId: string): string {
  return `${API_BASE}/run/${runId}/events`
}

const NODE_ORDER: NodeName[] = [
  'research',
  'generator',
  'property_estimator',
  'compliance_checker',
  'critic',
  'experiment_planner',
]

function fmtValue(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Math.abs(v) >= 100) return String(Math.round(v))
  if (Number.isInteger(v)) return String(v)
  return String(parseFloat(v.toFixed(3)))
}

// Shapes of the backend `complete` event's result.state (a GraphState dump).
interface RawEstimate {
  candidate_id: string
  property: string
  value: number
  unit: string
  meets_target: boolean | null
}
interface RawFlag {
  candidate_id: string
  status: ComplianceStatus
}
interface RawScore {
  candidate_id: string
  overall: number
  verdict: Verdict
}
interface RawCandidate {
  id: string
  name: string
}
interface RawState {
  shortlist: string[]
  candidates: RawCandidate[]
  critic_scores: RawScore[]
  property_estimates: RawEstimate[]
  compliance_flags: RawFlag[]
  experiment_plan: {
    design_type: string
    replicates: number
    runs: unknown[]
    factors: { name: string; unit: string; levels: number[] }[]
  } | null
}
interface RawTokenCost {
  total_tokens: number
  total_cost_usd: number
  by_node: Record<
    string,
    { prompt_tokens: number; completion_tokens: number; cost_usd: number }
  >
}
export interface RawCompleteResult {
  shortlist: string[]
  state: RawState
  token_cost: RawTokenCost
}

/**
 * Assemble the ResultsPanel view-model from the backend complete event. The
 * ranked candidates carry only formulation data, so scores, property estimates,
 * and compliance flags are joined in from the embedded GraphState by id.
 */
export function mapCompleteResult(result: RawCompleteResult): RunResult {
  const state = result.state
  const nameById = new Map(state.candidates.map((c) => [c.id, c.name]))
  const scoreById = new Map(state.critic_scores.map((s) => [s.candidate_id, s]))

  const estByCand = new Map<string, RawEstimate[]>()
  for (const e of state.property_estimates) {
    const arr = estByCand.get(e.candidate_id) ?? []
    arr.push(e)
    estByCand.set(e.candidate_id, arr)
  }
  const flagsByCand = new Map<string, RawFlag[]>()
  for (const f of state.compliance_flags) {
    const arr = flagsByCand.get(f.candidate_id) ?? []
    arr.push(f)
    flagsByCand.set(f.candidate_id, arr)
  }

  const candidates: CandidateResult[] = state.shortlist.map((id) => {
    const score = scoreById.get(id)
    const estimates = estByCand.get(id) ?? []
    const flags = flagsByCand.get(id) ?? []
    return {
      id,
      name: nameById.get(id) ?? id,
      overall: score?.overall ?? 0,
      verdict: score?.verdict ?? 'revise',
      // Only the spec-target properties (meets_target set) — matches the layout.
      properties: estimates
        .filter((e) => e.meets_target !== null)
        .map((e) => ({
          name: e.property,
          value: fmtValue(e.value),
          unit: e.unit,
          meets: e.meets_target,
        })),
      compliance: {
        pass: flags.filter((f) => f.status === 'pass').length,
        needs_review: flags.filter((f) => f.status === 'needs_review').length,
        fail: flags.filter((f) => f.status === 'fail').length,
      },
    }
  })

  const plan = state.experiment_plan
  const doe: DOEPlan = plan
    ? {
        design_type: plan.design_type,
        runs: plan.runs.length,
        replicates: plan.replicates,
        factors: plan.factors,
      }
    : { design_type: 'none', runs: 0, replicates: 1, factors: [] }

  const byNode = result.token_cost.by_node
  const tokenCost: TokenCost = {
    total_tokens: result.token_cost.total_tokens,
    total_cost_usd: result.token_cost.total_cost_usd,
    by_node: NODE_ORDER.filter((n) => byNode[n]).map((n) => ({
      node: n,
      prompt: byNode[n].prompt_tokens,
      completion: byNode[n].completion_tokens,
      cost: byNode[n].cost_usd,
    })),
  }

  return { shortlist: result.shortlist, candidates, doe, tokenCost }
}
