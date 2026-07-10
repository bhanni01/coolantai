from langchain_core.messages import AIMessage

from coolant_copilot.state import (
    Candidate,
    Component,
    PropertyName,
    PropertyTarget,
    ResearchFinding,
    TargetSpec,
)


def tool_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    """Shorthand for the dict form of an AIMessage tool call."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class FakeToolCallingLLM:
    """Scripted fake for bind_tools nodes, optionally with structured calls.

    tool_messages are returned one per bound-model invoke; when exhausted, an
    AIMessage with no tool calls ends the loop. structured_responses behave
    like FakeStructuredLLM (last one repeats).
    """

    def __init__(self, tool_messages=None, structured_responses=None):
        self.tool_messages = list(tool_messages or [])
        self.structured_responses = list(structured_responses or [])
        self.bound_tools: list[list] = []
        self.schemas: list = []
        self.message_log: list = []
        self.tool_invokes = 0
        self.structured_invokes = 0

    def bind_tools(self, tools):
        self.bound_tools.append(list(tools))
        parent = self

        class _Bound:
            def invoke(self, messages):
                parent.tool_invokes += 1
                parent.message_log.append(messages)
                if parent.tool_messages:
                    return parent.tool_messages.pop(0)
                return AIMessage(content="done")

        return _Bound()

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        parent = self

        class _Structured:
            def invoke(self, messages):
                parent.structured_invokes += 1
                parent.message_log.append(messages)
                if len(parent.structured_responses) > 1:
                    return parent.structured_responses.pop(0)
                return parent.structured_responses[0]

        return _Structured()


class FakeStructuredLLM:
    """Stands in for a chat model: records the schemas and prompts it was
    given and returns canned structured responses.

    Accepts a single response or a list consumed one per invoke; the last
    response repeats once the list is exhausted (so a looping node can be
    faked with a single entry).

    Also supports the research node's bind_tools ReAct phase: `bind_tools`
    returns a bound model that issues one search_chroma call per configured
    query, then stops. Tool-loop turns are intentionally NOT recorded in
    message_log, which tracks with_structured_output prompts only.
    """

    def __init__(self, responses, search_queries=("coolant literature",)):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.search_queries = list(search_queries)
        self.schemas = []
        self.message_log = []
        self.calls = 0

    def bind_tools(self, tools):
        queries = list(self.search_queries)

        class _Bound:
            def invoke(self, messages):
                if queries:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            tool_call("search_chroma", {"query": queries.pop(0)},
                                      f"search-{len(queries)}")
                        ],
                    )
                return AIMessage(content="done gathering")

        return _Bound()

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        self.calls += 1
        self.message_log.append(messages)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    @property
    def last_user_prompt(self) -> str:
        return self.message_log[-1][-1][1]


def make_target_spec(**overrides) -> TargetSpec:
    defaults = dict(
        name="DC-Coolant-A",
        application="single_phase_immersion",
        description="PFAS-free single-phase immersion coolant for hyperscale racks",
        property_targets=[
            PropertyTarget(
                property=PropertyName.THERMAL_CONDUCTIVITY,
                min_value=0.13,
                unit="W/m·K",
                priority="must",
            ),
            PropertyTarget(
                property=PropertyName.FLASH_POINT,
                min_value=150,
                unit="°C",
                priority="must",
            ),
        ],
        regulatory_regions=["US", "EU"],
    )
    return TargetSpec(**{**defaults, **overrides})


def make_candidate(candidate_id: str = "cand-0-rev0", revision: int = 0) -> Candidate:
    return Candidate(
        id=candidate_id,
        name="Synthetic ester blend",
        components=[
            Component(
                name="Pentaerythritol tetraoleate",
                cas_number="19321-40-5",
                role="base_fluid",
                weight_fraction=0.95,
            ),
            Component(name="BHT", cas_number="128-37-0", role="antioxidant", weight_fraction=0.05),
        ],
        rationale="Ester base fluids offer high flash point and no fluorine chemistry.",
        source_refs=["rf-0"],
        revision=revision,
    )


def make_finding(finding_id: str = "rf-0") -> ResearchFinding:
    return ResearchFinding(
        id=finding_id,
        source="esters.txt",
        summary="Ester fluids show 0.14-0.16 W/m·K conductivity and flash points above 250 °C.",
        relevance="Supports the thermal conductivity and flash point targets.",
    )
