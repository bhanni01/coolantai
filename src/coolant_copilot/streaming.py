"""Fine-grained detail events on LangGraph's "custom" stream.

emit_detail() lets node code (and the deterministic tools it drives) surface
per-item signals — e.g. one event per retrieved chunk — while the node is
still running. The API runner consumes them from graph.astream(stream_mode
including "custom") and forwards them as `detail` SSE events, additive
alongside the per-node status events.

Outside a LangGraph runnable context (direct node calls in tests, plain
function use) there is no stream writer, so emit_detail is a silent no-op;
under graph.invoke() the writer exists but discards, so the CLI path is
unaffected either way. Callers can therefore emit unconditionally.
"""

from langgraph.config import get_stream_writer


def emit_detail(node_name: str, detail_type: str, payload: dict) -> None:
    """Emit one detail event for `node_name` onto the custom stream."""
    try:
        writer = get_stream_writer()
    except (RuntimeError, LookupError):
        # No stream writer: either fully outside a LangGraph context
        # (RuntimeError — plain function/node calls in tests) or inside a bare
        # Runnable .invoke() with no pregel runtime, where get_stream_writer()
        # raises KeyError('__pregel_runtime') (LookupError). Both are no-ops.
        return
    writer({"node_name": node_name, "detail_type": detail_type, "payload": payload})
