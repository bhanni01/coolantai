"""Token counting and per-node token/cost accounting for a graph run.

When a LANGSMITH_API_KEY is configured, LangSmith tracing already records every
node and every LLM call as nested runs, so per-node token and cost breakdown is
visible in the LangSmith UI for a single run. This module provides the same
accounting locally: `TokenCostTracker` is a LangChain callback handler that
attributes each LLM call's token usage to the LangGraph node that issued it and
estimates its dollar cost, so `main.py` can print a run summary without querying
LangSmith. Passing the tracker in the graph's `config["callbacks"]` is also what
guarantees instrumentation reaches every node, not just the root invoke.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per 1M tokens as (input, output). OpenAI list prices;
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def get_encoding(model: str) -> "tiktoken.Encoding":
    """tiktoken encoding for a model, falling back to cl100k_base if unknown."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    return len(get_encoding(model).encode(text))


def _match_price(model: str) -> tuple[float, float]:
    for name in sorted(PRICING, key=len, reverse=True):
        if model.startswith(name):
            return PRICING[name]
    return (0.0, 0.0)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar estimate for one call given its model and token split."""
    in_rate, out_rate = _match_price(model)
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


def _extract_usage(response: LLMResult) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from an LLMResult."""
    output = response.llm_output or {}
    usage = output.get("token_usage") or output.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is not None or completion is not None:
        return int(prompt or 0), int(completion or 0)

    prompt = completion = 0
    for generations in response.generations:
        for gen in generations:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                prompt += meta.get("input_tokens", 0)
                completion += meta.get("output_tokens", 0)
    return prompt, completion


@dataclass
class NodeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenCostTracker(BaseCallbackHandler):
    """Accumulates token usage and cost per LangGraph node across a run. """

    def __init__(self) -> None:
        self.by_node: dict[str, NodeUsage] = defaultdict(NodeUsage)
       
        self._pending: dict[UUID, tuple[str, str]] = {}

    def _start(self, run_id: UUID, metadata: dict | None, invocation_params: dict | None,
               serialized: dict | None) -> None:
        metadata = metadata or {}
        invocation_params = invocation_params or {}
        serialized_kwargs = (serialized or {}).get("kwargs", {})
        node = metadata.get("langgraph_node", "unknown")
        model = (
            metadata.get("ls_model_name")
            or invocation_params.get("model")
            or invocation_params.get("model_name")
            or serialized_kwargs.get("model")
            or serialized_kwargs.get("model_name")
            or "unknown"
        )
        self._pending[run_id] = (node, model)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id, kwargs.get("metadata"), kwargs.get("invocation_params"), serialized)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id, kwargs.get("metadata"), kwargs.get("invocation_params"), serialized)

    def on_llm_end(self, response, *, run_id, **kwargs):
        node, model = self._pending.pop(run_id, ("unknown", "unknown"))
        prompt_tokens, completion_tokens = _extract_usage(response)
        usage = self.by_node[node]
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.cost += estimate_cost(model, prompt_tokens, completion_tokens)
        usage.calls += 1

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.by_node.values())

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.by_node.values())


def format_run_summary(tracker: TokenCostTracker, revision_count: int) -> str:
    """Human-readable per-node token/cost table plus run totals."""
    lines = [
        "Token & cost summary",
        "--------------------",
        f"{'node':<22}{'prompt':>10}{'completion':>12}{'cost (USD)':>13}",
    ]
    for node in sorted(tracker.by_node):
        u = tracker.by_node[node]
        lines.append(
            f"{node:<22}{u.prompt_tokens:>10,}{u.completion_tokens:>12,}{u.cost:>13.4f}"
        )
    total_prompt = sum(u.prompt_tokens for u in tracker.by_node.values())
    total_completion = sum(u.completion_tokens for u in tracker.by_node.values())
    lines.append(
        f"{'TOTAL':<22}{total_prompt:>10,}{total_completion:>12,}{tracker.total_cost:>13.4f}"
    )
    lines += [
        "",
        f"Total tokens: {tracker.total_tokens:,}  |  "
        f"Estimated cost: ${tracker.total_cost:.4f}  |  "
        f"Revision loops: {revision_count}",
    ]
    return "\n".join(lines)
