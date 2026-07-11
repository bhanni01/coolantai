from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from coolant_copilot.observability import (
    TokenCostTracker,
    count_tokens,
    estimate_cost,
    format_run_summary,
    get_encoding,
)


def test_get_encoding_falls_back_for_unknown_model():
    # A bogus model name must not raise; it falls back to a default encoding.
    enc = get_encoding("not-a-real-model-xyz")
    assert enc.encode("hello world")


def test_count_tokens_positive():
    assert count_tokens("the quick brown fox", "gpt-4o-mini") > 0


def test_estimate_cost_prefers_most_specific_model_prefix():
    # gpt-4o-mini must not be priced as gpt-4o despite the shared prefix.
    mini = estimate_cost("gpt-4o-mini", 1_000_000, 0)
    full = estimate_cost("gpt-4o", 1_000_000, 0)
    assert mini == 0.15
    assert full == 2.50
    assert mini < full


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost("mystery-model", 1000, 1000) == 0.0


def _llm_result_with_usage(prompt: int, completion: int) -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        },
    )


def test_tracker_attributes_usage_to_node_from_metadata():
    tracker = TokenCostTracker()
    run_id = uuid4()

    tracker.on_chat_model_start(
        serialized={},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "research", "ls_model_name": "gpt-4o-mini"},
    )
    tracker.on_llm_end(_llm_result_with_usage(1000, 500), run_id=run_id)

    usage = tracker.by_node["research"]
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert usage.calls == 1
    assert usage.cost == estimate_cost("gpt-4o-mini", 1000, 500)


def test_tracker_extracts_usage_from_usage_metadata_fallback():
    tracker = TokenCostTracker()
    run_id = uuid4()
    tracker.on_chat_model_start(
        serialized={},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "critic", "ls_model_name": "gpt-4o"},
    )
    # No llm_output.token_usage — usage rides on the message instead.
    result = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(
                    content="x",
                    usage_metadata={
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "total_tokens": 240,
                    },
                )
            )
        ]],
        llm_output={},
    )
    tracker.on_llm_end(result, run_id=run_id)

    assert tracker.by_node["critic"].prompt_tokens == 200
    assert tracker.by_node["critic"].completion_tokens == 40


def test_tracker_unknown_node_when_metadata_missing():
    tracker = TokenCostTracker()
    run_id = uuid4()
    tracker.on_chat_model_start(serialized={}, messages=[], run_id=run_id, metadata={})
    tracker.on_llm_end(_llm_result_with_usage(10, 5), run_id=run_id)

    assert tracker.by_node["unknown"].total_tokens == 15


def test_format_run_summary_reports_nodes_totals_and_revisions():
    tracker = TokenCostTracker()
    r1, r2 = uuid4(), uuid4()
    tracker.on_chat_model_start(
        serialized={}, messages=[], run_id=r1,
        metadata={"langgraph_node": "research", "ls_model_name": "gpt-4o-mini"},
    )
    tracker.on_llm_end(_llm_result_with_usage(1000, 200), run_id=r1)
    tracker.on_chat_model_start(
        serialized={}, messages=[], run_id=r2,
        metadata={"langgraph_node": "generator", "ls_model_name": "gpt-4o"},
    )
    tracker.on_llm_end(_llm_result_with_usage(2000, 800), run_id=r2)

    summary = format_run_summary(tracker, revision_count=2)

    assert "research" in summary
    assert "generator" in summary
    assert "TOTAL" in summary
    assert "Revision loops: 2" in summary
    # Totals reflect both nodes.
    assert tracker.total_tokens == 4000
    assert f"{tracker.total_tokens:,}" in summary
