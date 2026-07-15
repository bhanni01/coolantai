"""Run bookkeeping: drive graph.astream() and buffer per-node SSE events.

A run is started (non-blocking) by POST /run and consumed by GET
/run/{id}/events. Each RunSession buffers every event it produces, so a client
that connects after the run started still replays from the beginning and then
follows live events until the terminal 'complete' (or 'error') event.

Event payloads reuse the pipeline's own schemas — node updates carry the live
GraphState channels, and the complete event embeds the full GraphState dump
(GraphState.model_dump) plus the observability token/cost breakdown. Nothing
is redefined here.

Alongside the per-node status events, `detail` events carry fine-grained
signals emitted mid-node through streaming.emit_detail (one per retrieved
source chunk, one per cross-checked property estimate); they arrive between
their parent node's 'active' and 'completed' events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from coolant_copilot.observability import TokenCostTracker, format_run_summary
from coolant_copilot.state import CriticWeights, GraphState, TargetSpec

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_update(node: str, update: dict) -> str:
    """One-line summary of what a node contributed (mirrors main.py's status
    lines; reads the live GraphState objects in the update, never re-parses)."""
    if node == "research":
        return f"{len(update.get('research_findings', []))} research finding(s)"
    if node == "generator":
        candidates = update.get("candidates", [])
        revision = max((c.revision for c in candidates), default=0)
        label = f"revision {revision}" if revision else "initial pass"
        return f"{len(candidates)} candidate(s) so far ({label})"
    if node == "property_estimator":
        return f"{len(update.get('property_estimates', []))} property estimate(s)"
    if node == "compliance_checker":
        flags = update.get("compliance_flags", [])
        fails = sum(1 for f in flags if f.status == "fail")
        review = sum(1 for f in flags if f.status == "needs_review")
        return f"{len(flags)} compliance flag(s) — {fails} fail, {review} needs_review"
    if node == "critic":
        scores = update.get("critic_scores", [])
        if not scores:
            return "no new candidates to score"
        top = max(scores, key=lambda s: s.overall)
        return f"top candidate {top.candidate_id} scored {top.overall}/10 ({top.verdict})"
    if node == "experiment_planner":
        plan = update.get("experiment_plan")
        if plan is None:
            return "no plan produced"
        return (
            f"{plan.design_type} DOE with {len(plan.runs)} run(s) "
            f"× {plan.replicates} replicate(s)"
        )
    return "done"


def _token_cost(tracker: TokenCostTracker, revision_count: int) -> dict:
    return {
        "total_tokens": tracker.total_tokens,
        "total_cost_usd": round(tracker.total_cost, 6),
        "revision_loops": revision_count,
        "by_node": {
            node: {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "cost_usd": round(u.cost, 6),
                "calls": u.calls,
            }
            for node, u in tracker.by_node.items()
        },
        "summary_text": format_run_summary(tracker, revision_count),
    }


class RunSession:
    """Buffered, replayable event stream for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.events: list[dict] = []
        self.done = False
        self._new = asyncio.Event()

    def emit(self, event: dict) -> None:
        self.events.append(event)
        self._new.set()

    def finish(self) -> None:
        self.done = True
        self._new.set()

    async def stream(self):
        """Yield buffered events, then follow live ones until the run finishes."""
        cursor = 0
        while True:
            while cursor < len(self.events):
                yield self.events[cursor]
                cursor += 1
            if self.done:
                return
            self._new.clear()
            # Re-check after clearing to avoid a lost-wakeup race.
            if cursor < len(self.events) or self.done:
                continue
            await self._new.wait()


class RunRegistry:
    def __init__(self) -> None:
        self.sessions: dict[str, RunSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self, run_id: str) -> RunSession:
        session = RunSession(run_id)
        self.sessions[run_id] = session
        return session

    def get(self, run_id: str) -> RunSession | None:
        return self.sessions.get(run_id)

    def track_task(self, run_id: str, task: asyncio.Task) -> None:
        # Hold a reference so the background task isn't garbage-collected.
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(run_id, None))


async def execute_run(
    graph,
    spec: TargetSpec,
    session: RunSession,
    critic_weights: CriticWeights | None = None,
) -> None:
    """Drive graph.astream() and emit one event per node, then a terminal event.

    Never raises to the caller: any failure becomes an 'error' event and the
    session is finished so consumers unblock.
    """
    tracker = TokenCostTracker()
    initial = GraphState(
        target_spec=spec, critic_weights=critic_weights or CriticWeights()
    )
    values = dict(initial)

    def node_event(node: str, status: str, summary: str) -> dict:
        return {
            "type": "node",
            "node_name": node,
            "status": status,
            "output_summary": summary,
            "timestamp": _now(),
            "loop_iteration": int(values.get("revision_count", 0)),
        }

    try:
        # Combined modes: the "debug" stream emits a `task` event when each node
        # *starts* (so the two fan-out evaluators announce themselves in the same
        # superstep — genuine concurrency, not inferred), the "updates" stream
        # carries each node's result dict when it *completes*, and the "custom"
        # stream carries fine-grained detail chunks written mid-node via
        # streaming.emit_detail (per retrieved source, per cross-checked
        # property) — additive alongside the node status events.
        async for mode, chunk in graph.astream(
            initial,
            config={
                "run_name": f"coolant-copilot:{spec.name}",
                "recursion_limit": 50,
                "callbacks": [tracker],
                "tags": ["coolant-copilot", "api"],
            },
            stream_mode=["updates", "debug", "custom"],
        ):
            if mode == "debug":
                if chunk.get("type") == "task":
                    node = chunk.get("payload", {}).get("name", "")
                    if node and not node.startswith("__"):
                        session.emit(node_event(node, "active", ""))
                continue

            if mode == "custom":
                if isinstance(chunk, dict) and chunk.get("detail_type"):
                    session.emit(
                        {
                            "type": "detail",
                            "node_name": chunk.get("node_name", ""),
                            "detail_type": chunk["detail_type"],
                            "payload": chunk.get("payload", {}),
                            "timestamp": _now(),
                            "loop_iteration": int(values.get("revision_count", 0)),
                        }
                    )
                continue

            # mode == "updates"
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                update = update or {}
                values.update(update)
                session.emit(node_event(node, "completed", summarize_update(node, update)))

        final = GraphState.model_validate(values)
        by_id = {c.id: c for c in final.candidates}
        ranked = [by_id[cid].model_dump(mode="json") for cid in final.shortlist if cid in by_id]
        session.emit(
            {
                "type": "complete",
                "status": "complete",
                "timestamp": _now(),
                "result": {
                    "shortlist": final.shortlist,
                    "ranked_candidates": ranked,
                    "experiment_plan": (
                        final.experiment_plan.model_dump(mode="json")
                        if final.experiment_plan
                        else None
                    ),
                    # Full GraphState, reusing the pipeline schema verbatim.
                    "state": final.model_dump(mode="json"),
                    "token_cost": _token_cost(tracker, final.revision_count),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - surface as an event, don't crash the server
        logger.exception("run %s failed", session.run_id)
        session.emit(
            {
                "type": "error",
                "status": "error",
                "timestamp": _now(),
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        session.finish()


def format_sse(event: dict) -> str:
    """Serialize one event as an SSE frame (named event + JSON data)."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
