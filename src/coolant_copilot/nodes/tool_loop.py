"""Shared bind_tools driver for tool-calling nodes.

Runs the classic loop: model proposes tool calls, we execute them and feed
ToolMessages back, until the model stops calling tools or the budget is
exhausted. Every requested tool_call id gets a ToolMessage (an OpenAI
requirement), but calls beyond the budget are answered with a refusal
instead of being executed.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def run_tool_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    user_prompt: str,
    max_tool_calls: int,
) -> tuple[list[BaseMessage], int]:
    """Drive a bind_tools loop. Returns (full message transcript, executed calls)."""
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    executed = 0

    while True:
        response = model.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            if executed >= max_tool_calls:
                content = "Tool-call budget exhausted; finish with what you have."
            elif call["name"] not in tools_by_name:
                content = f"Unknown tool '{call['name']}'."
            else:
                try:
                    result = tools_by_name[call["name"]].invoke(call["args"])
                    content = result if isinstance(result, str) else json.dumps(result)
                except Exception as exc:  # tool bugs must not kill the node
                    logger.warning("tool %s failed: %s", call["name"], exc)
                    content = f"Tool error: {exc}"
                executed += 1
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        if executed >= max_tool_calls:
            # One final model turn sees the refusals and wraps up; if it keeps
            # asking for tools anyway, the next iteration refuses them too and
            # we exit here.
            final = model.invoke(messages)
            messages.append(final)
            if getattr(final, "tool_calls", None):
                for call in final.tool_calls:
                    messages.append(
                        ToolMessage(
                            content="Tool-call budget exhausted.", tool_call_id=call["id"]
                        )
                    )
            break

    return messages, executed
