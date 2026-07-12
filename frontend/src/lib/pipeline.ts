import type { NodeName } from './types'

// Diagram is drawn in a fixed coordinate space and scaled responsively via the
// SVG viewBox, so edges and node cards share one coordinate system.
export const DIAGRAM_W = 1065
export const DIAGRAM_H = 440

export interface NodeLayout {
  id: NodeName
  label: string
  sub: string
  tier: 'gpt-4o' | 'gpt-4o-mini'
  x: number
  y: number
  w: number
  h: number
}

export const NODE_LAYOUT: NodeLayout[] = [
  { id: 'research', label: 'Research', sub: 'RAG retrieval', tier: 'gpt-4o-mini', x: 20, y: 174, w: 150, h: 92 },
  { id: 'generator', label: 'Generator', sub: 'propose candidates', tier: 'gpt-4o', x: 230, y: 174, w: 150, h: 92 },
  { id: 'property_estimator', label: 'Property Estimator', sub: 'deterministic tools', tier: 'gpt-4o-mini', x: 440, y: 64, w: 180, h: 88 },
  { id: 'compliance_checker', label: 'Compliance Checker', sub: 'PFAS / material', tier: 'gpt-4o-mini', x: 440, y: 288, w: 180, h: 88 },
  { id: 'critic', label: 'Critic', sub: 'score & route', tier: 'gpt-4o', x: 690, y: 174, w: 140, h: 92 },
  { id: 'experiment_planner', label: 'Experiment Planner', sub: 'DOE plan', tier: 'gpt-4o', x: 895, y: 174, w: 150, h: 92 },
]

export const NODE_BY_ID: Record<NodeName, NodeLayout> = Object.fromEntries(
  NODE_LAYOUT.map((n) => [n.id, n]),
) as Record<NodeName, NodeLayout>

export interface EdgeDef {
  id: string
  from: NodeName
  to: NodeName
  kind: 'forward' | 'revise'
  /** SVG path in diagram coordinates. */
  d: string
}

// Anchor helpers (right/left/bottom edge centers).
const rc = (id: NodeName) => {
  const n = NODE_BY_ID[id]
  return { x: n.x + n.w, y: n.y + n.h / 2 }
}
const lc = (id: NodeName) => {
  const n = NODE_BY_ID[id]
  return { x: n.x, y: n.y + n.h / 2 }
}
const bc = (id: NodeName) => {
  const n = NODE_BY_ID[id]
  return { x: n.x + n.w / 2, y: n.y + n.h }
}

function curve(a: { x: number; y: number }, b: { x: number; y: number }) {
  const mx = (a.x + b.x) / 2
  return `M ${a.x} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x} ${b.y}`
}

export const EDGES: EdgeDef[] = [
  { id: 'research-generator', from: 'research', to: 'generator', kind: 'forward', d: curve(rc('research'), lc('generator')) },
  { id: 'generator-estimator', from: 'generator', to: 'property_estimator', kind: 'forward', d: curve(rc('generator'), lc('property_estimator')) },
  { id: 'generator-compliance', from: 'generator', to: 'compliance_checker', kind: 'forward', d: curve(rc('generator'), lc('compliance_checker')) },
  { id: 'estimator-critic', from: 'property_estimator', to: 'critic', kind: 'forward', d: curve(rc('property_estimator'), lc('critic')) },
  { id: 'compliance-critic', from: 'compliance_checker', to: 'critic', kind: 'forward', d: curve(rc('compliance_checker'), lc('critic')) },
  { id: 'critic-planner', from: 'critic', to: 'experiment_planner', kind: 'forward', d: curve(rc('critic'), lc('experiment_planner')) },
  // Revise loop: critic bottom → down under the parallel pair → generator bottom.
  {
    id: 'critic-generator',
    from: 'critic',
    to: 'generator',
    kind: 'revise',
    d: `M ${bc('critic').x} ${bc('critic').y} C ${bc('critic').x} 428, ${bc('generator').x} 428, ${bc('generator').x} ${bc('generator').y}`,
  },
]

/** Left-to-right execution order (parallel pair grouped) for narrow layouts. */
export const NODE_ORDER: NodeName[] = [
  'research',
  'generator',
  'property_estimator',
  'compliance_checker',
  'critic',
  'experiment_planner',
]
