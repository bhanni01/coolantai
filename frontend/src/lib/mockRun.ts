import type { LogLevel, NodeName, RunResult, Verdict } from './types'

// One step of the scripted run. A step activates one or more nodes, holds them
// "active" for durationMs, then marks them done and emits the log lines.
export interface Beat {
  activate: NodeName[]
  loop: number
  durationMs: number
  verdict?: Verdict
  logs: { node: NodeName | 'system'; message: string; level: LogLevel }[]
  badges?: Partial<
    Record<NodeName, { text: string; tone: 'teal' | 'copper' | 'amber' | 'red' }>
  >
}

// A representative run mirroring an actual pipeline execution: an initial pass
// that is rejected, three revision loops (cap = 3), then planning. Timings are
// compressed for review.
export const TIMELINE: Beat[] = [
  {
    activate: ['research'],
    loop: 0,
    durationMs: 1100,
    logs: [{ node: 'research', message: 'Retrieved 3 findings from Chroma (2 sources cited)', level: 'info' }],
  },
  {
    activate: ['generator'],
    loop: 0,
    durationMs: 1150,
    logs: [{ node: 'generator', message: 'Proposed 3 candidate formulations (initial pass)', level: 'info' }],
  },
  {
    activate: ['property_estimator', 'compliance_checker'],
    loop: 0,
    durationMs: 1300,
    badges: { compliance_checker: { text: '2', tone: 'amber' } },
    logs: [
      { node: 'property_estimator', message: '8 property estimates computed (mixing rules + tables)', level: 'info' },
      { node: 'compliance_checker', message: '15 compliance flags — 2 needs_review', level: 'warn' },
    ],
  },
  {
    activate: ['critic'],
    loop: 0,
    durationMs: 1050,
    verdict: 'reject',
    logs: [{ node: 'critic', message: 'Top candidate cand-0-rev0 scored 3.0/10 — rejected', level: 'revision' }],
  },

  // Revision 1
  {
    activate: ['generator'],
    loop: 1,
    durationMs: 1050,
    logs: [{ node: 'generator', message: 'Revised formulations — 3 new candidates (revision 1)', level: 'revision' }],
  },
  {
    activate: ['property_estimator', 'compliance_checker'],
    loop: 1,
    durationMs: 1200,
    badges: { compliance_checker: { text: '3', tone: 'amber' } },
    logs: [
      { node: 'property_estimator', message: '16 property estimates (cross-checked vs Shell S5 X)', level: 'info' },
      { node: 'compliance_checker', message: '34 compliance flags — 3 needs_review', level: 'warn' },
    ],
  },
  {
    activate: ['critic'],
    loop: 1,
    durationMs: 1000,
    verdict: 'revise',
    logs: [{ node: 'critic', message: 'cand-3-rev1 scored 6.0/10 — revise (EPDM seal risk)', level: 'revision' }],
  },

  // Revision 2
  {
    activate: ['generator'],
    loop: 2,
    durationMs: 1000,
    logs: [{ node: 'generator', message: 'Revised formulations — 3 new candidates (revision 2)', level: 'revision' }],
  },
  {
    activate: ['property_estimator', 'compliance_checker'],
    loop: 2,
    durationMs: 1150,
    badges: { compliance_checker: { text: '1', tone: 'red' } },
    logs: [
      { node: 'property_estimator', message: '24 property estimates computed', level: 'info' },
      { node: 'compliance_checker', message: '49 compliance flags — 1 fail, 4 needs_review', level: 'error' },
    ],
  },
  {
    activate: ['critic'],
    loop: 2,
    durationMs: 1000,
    verdict: 'revise',
    logs: [{ node: 'critic', message: 'cand-3-rev1 still leads at 6.0/10 — revise', level: 'revision' }],
  },

  // Revision 3 (cap)
  {
    activate: ['generator'],
    loop: 3,
    durationMs: 1000,
    logs: [{ node: 'generator', message: 'Revised formulations — 3 new candidates (revision 3)', level: 'revision' }],
  },
  {
    activate: ['property_estimator', 'compliance_checker'],
    loop: 3,
    durationMs: 1150,
    badges: { compliance_checker: { text: '0', tone: 'teal' } },
    logs: [
      { node: 'property_estimator', message: '24 property estimates computed', level: 'info' },
      { node: 'compliance_checker', message: '70 compliance flags — 0 fail', level: 'info' },
    ],
  },
  {
    activate: ['critic'],
    loop: 3,
    durationMs: 1050,
    verdict: 'revise',
    logs: [
      { node: 'critic', message: 'cand-3-rev1 scored 6.0/10 — revise', level: 'revision' },
      { node: 'system', message: 'Revision cap reached (3/3) — routing to planner', level: 'warn' },
    ],
  },

  // Planning
  {
    activate: ['experiment_planner'],
    loop: 3,
    durationMs: 1300,
    logs: [{ node: 'experiment_planner', message: 'full_factorial DOE — 18 runs × 2 replicates', level: 'success' }],
  },
]

export const MOCK_RESULT: RunResult = {
  shortlist: ['cand-3-rev1', 'cand-6-rev2', 'cand-9-rev3'],
  candidates: [
    {
      id: 'cand-3-rev1',
      name: 'ThermaSafe-1',
      overall: 6.0,
      verdict: 'revise',
      compliance: { pass: 12, needs_review: 3, fail: 0 },
      properties: [
        { name: 'thermal_conductivity', value: '0.150', unit: 'W/m·K', meets: true },
        { name: 'flash_point', value: '310', unit: '°C', meets: true },
        { name: 'kinematic_viscosity', value: '48', unit: 'cSt', meets: true },
        { name: 'cost', value: '10.4', unit: 'USD/L', meets: true },
      ],
    },
    {
      id: 'cand-6-rev2',
      name: 'ThermaSafe-2',
      overall: 5.5,
      verdict: 'revise',
      compliance: { pass: 11, needs_review: 4, fail: 0 },
      properties: [
        { name: 'thermal_conductivity', value: '0.148', unit: 'W/m·K', meets: true },
        { name: 'flash_point', value: '295', unit: '°C', meets: true },
        { name: 'kinematic_viscosity', value: '52', unit: 'cSt', meets: false },
        { name: 'cost', value: '11.2', unit: 'USD/L', meets: true },
      ],
    },
    {
      id: 'cand-9-rev3',
      name: 'ThermaSafe-3',
      overall: 5.0,
      verdict: 'revise',
      compliance: { pass: 10, needs_review: 5, fail: 0 },
      properties: [
        { name: 'thermal_conductivity', value: '0.141', unit: 'W/m·K', meets: true },
        { name: 'flash_point', value: '260', unit: '°C', meets: true },
        { name: 'kinematic_viscosity', value: '45', unit: 'cSt', meets: true },
        { name: 'cost', value: '9.8', unit: 'USD/L', meets: true },
      ],
    },
  ],
  doe: {
    design_type: 'full_factorial',
    runs: 18,
    replicates: 2,
    factors: [
      { name: 'temperature', unit: '°C', levels: [25, 50, 75] },
      { name: 'pressure', unit: 'kPa', levels: [101, 150] },
    ],
  },
  tokenCost: {
    total_tokens: 39810,
    total_cost_usd: 0.0696,
    by_node: [
      { node: 'research', prompt: 11454, completion: 520, cost: 0.002 },
      { node: 'generator', prompt: 4371, completion: 1583, cost: 0.0268 },
      { node: 'property_estimator', prompt: 4108, completion: 741, cost: 0.0011 },
      { node: 'compliance_checker', prompt: 8459, completion: 813, cost: 0.0018 },
      { node: 'critic', prompt: 6524, completion: 614, cost: 0.0224 },
      { node: 'experiment_planner', prompt: 1577, completion: 3522, cost: 0.0155 },
    ],
  },
}
