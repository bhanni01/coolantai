"""FastAPI service exposing the LangGraph coolant pipeline.

Endpoints:
  POST /run                 — accept a DatacenterProfile body (closed dropdown
                              selections; never a raw TargetSpec), resolve it
                              to the full spec + critic weights server-side,
                              start a run, and return a run_id immediately
                              (does not block on the graph).
  GET  /run/{run_id}/events — SSE stream: one event per node as
                              graph.astream() yields, then a terminal
                              'complete' (or 'error') event.

Run it with:  uvicorn api.main:app --reload   (from the repo root)

The pipeline itself is untouched; this only reuses build_graph, GraphState and
TargetSpec. The graph is provided through the get_graph dependency so tests can
inject a fake-wired graph.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from coolant_copilot.schemas.datacenter_profile import (
    DatacenterProfile,
    resolve_critic_weights,
    resolve_target_spec,
)

from .pipeline import get_graph
from .runner import RunRegistry, RunSession, execute_run, format_sse

app = FastAPI(title="Coolant Formulation Copilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = RunRegistry()

# SSE responses must not be buffered or cached by intermediaries.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run", status_code=202)
async def start_run(profile: DatacenterProfile, graph=Depends(get_graph)) -> dict:
    """Resolve the profile server-side, start a background run, return its id."""
    spec = resolve_target_spec(profile)
    weights = resolve_critic_weights(profile)
    run_id = uuid4().hex
    session = registry.create(run_id)
    task = asyncio.create_task(execute_run(graph, spec, session, critic_weights=weights))
    registry.track_task(run_id, task)
    return {"run_id": run_id, "events_url": f"/run/{run_id}/events"}


@app.get("/run/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    """Stream this run's node events as Server-Sent Events."""
    session: RunSession | None = registry.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id '{run_id}'")

    async def event_source():
        async for event in session.stream():
            yield format_sse(event)

    return StreamingResponse(
        event_source(), media_type="text/event-stream", headers=_SSE_HEADERS
    )
