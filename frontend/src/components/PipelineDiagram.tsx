import { DIAGRAM_H, DIAGRAM_W, EDGES, NODE_LAYOUT } from '../lib/pipeline'
import type { EdgeDef } from '../lib/pipeline'
import type { NodeStateMap } from '../lib/types'
import { PipelineNode } from './PipelineNode'

const TEAL = '#4fc3b0'
const COPPER = '#d98e4a'
const HAIR = '#2b373d'
const TEAL_DIM = '#2f6f66'

interface EdgeVisual {
  color: string
  flowing: boolean
  strong: boolean
}

function edgeVisual(edge: EdgeDef, nodes: NodeStateMap): EdgeVisual {
  const src = nodes[edge.from]
  const dst = nodes[edge.to]

  if (edge.kind === 'revise') {
    const flowing = src.status === 'done' && dst.status === 'active' && dst.loop > 0
    return { color: COPPER, flowing, strong: flowing }
  }

  // Forward edge.
  const flowing = src.status === 'done' && dst.status === 'active'
  const done = src.status === 'done' && dst.status === 'done'
  const revisionContext = dst.status === 'active' && dst.loop > 0
  return {
    color: flowing ? (revisionContext ? COPPER : TEAL) : done ? TEAL_DIM : HAIR,
    flowing,
    strong: flowing || done,
  }
}

export function PipelineDiagram({ nodes }: { nodes: NodeStateMap }) {
  return (
    <div className="w-full px-4 py-5">
      <svg
        viewBox={`0 0 ${DIAGRAM_W} ${DIAGRAM_H}`}
        className="h-auto w-full"
        role="img"
        aria-label="Coolant formulation pipeline"
      >
        <defs>
          <marker id="arrow-teal" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={TEAL} />
          </marker>
          <marker id="arrow-copper" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={COPPER} />
          </marker>
          <marker id="arrow-dim" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={HAIR} />
          </marker>
        </defs>

        {/* Edges under the nodes. */}
        <g fill="none" strokeWidth={1.75} strokeLinecap="round">
          {EDGES.map((edge) => {
            const v = edgeVisual(edge, nodes)
            const marker =
              v.color === TEAL ? 'url(#arrow-teal)' : v.color === COPPER ? 'url(#arrow-copper)' : 'url(#arrow-dim)'
            return (
              <path
                key={edge.id}
                d={edge.d}
                stroke={v.color}
                markerEnd={marker}
                opacity={v.strong ? 1 : edge.kind === 'revise' ? 0.35 : 0.55}
                className={v.flowing ? 'anim-edge-flow' : undefined}
                strokeDasharray={!v.flowing && edge.kind === 'revise' ? '2 6' : undefined}
              />
            )
          })}
        </g>

        {/* "revise" label on the loop-back edge. */}
        <text
          x={(NODE_LAYOUT[1].x + NODE_LAYOUT[1].w / 2 + NODE_LAYOUT[4].x + NODE_LAYOUT[4].w / 2) / 2}
          y={422}
          textAnchor="middle"
          fill={COPPER}
          opacity={0.7}
          className="font-mono"
          fontSize={10}
          letterSpacing={2}
        >
          REVISE ↺ (max 3)
        </text>

        {/* Node cards via foreignObject so HTML nodes share the edge coordinate space. */}
        {NODE_LAYOUT.map((layout) => {
          const runtime = nodes[layout.id]
          const processing = runtime.status === 'active' && runtime.loop > 0
          return (
            <foreignObject key={layout.id} x={layout.x} y={layout.y} width={layout.w} height={layout.h}>
              <PipelineNode layout={layout} runtime={runtime} processing={processing} />
            </foreignObject>
          )
        })}
      </svg>
    </div>
  )
}
