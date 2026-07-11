from conftest import FakeToolCallingLLM, make_candidate, make_target_spec, tool_call
from langchain_core.messages import AIMessage

from coolant_copilot.nodes.property_estimator import make_property_estimator_node
from coolant_copilot.schemas.extraction import (
    BaseChemistry,
    ExtractedFluidProfile,
    ExtractedProperty,
)
from coolant_copilot.state import GraphState, PropertyName


def ester_profile() -> ExtractedFluidProfile:
    return ExtractedFluidProfile(
        fluid_name="Shell S5 X",
        base_chemistry=BaseChemistry.SYNTHETIC_ESTER,
        source_document="shell-s5x-datasheet.pdf",
        properties=[
            ExtractedProperty(
                property=PropertyName.THERMAL_CONDUCTIVITY,
                value=0.14,
                unit="W/m·K",
                confidence=1.0,
            )
        ],
    )


def state_with_candidate() -> GraphState:
    return GraphState(target_spec=make_target_spec(), candidates=[make_candidate("cand-0-rev0")])


def test_llm_tool_call_captures_deterministic_estimates():
    llm = FakeToolCallingLLM(
        tool_messages=[
            AIMessage(
                content="",
                tool_calls=[tool_call("estimate_properties", {"candidate_id": "cand-0-rev0"})],
            ),
        ]
    )

    update = make_property_estimator_node([ester_profile()], llm)(state_with_candidate())

    estimates = update["property_estimates"]
    assert {e.candidate_id for e in estimates} == {"cand-0-rev0"}
    tc = next(e for e in estimates if e.property is PropertyName.THERMAL_CONDUCTIVITY)
    assert tc.method == "linear_mixing_rule"  # computed by the tool, not the LLM
    assert tc.reference_check == "validated"  # cross-checked against the profile
    # Both tools were bound.
    assert [t.name for t in llm.bound_tools[0]] == [
        "estimate_properties",
        "get_reference_fluid_properties",
    ]


def test_reference_lookup_tool_returns_profile_data():
    llm = FakeToolCallingLLM(
        tool_messages=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("get_reference_fluid_properties", {"base_chemistry": "synthetic_ester"}, "c1"),
                    tool_call("estimate_properties", {"candidate_id": "cand-0-rev0"}, "c2"),
                ],
            ),
        ]
    )

    update = make_property_estimator_node([ester_profile()], llm)(state_with_candidate())

    # The lookup happened inside the loop; final state still carries estimates.
    assert update["property_estimates"]
    assert llm.tool_invokes == 2  # tool round + wrap-up round


def test_fallback_covers_candidates_the_llm_skipped():
    llm = FakeToolCallingLLM(tool_messages=[AIMessage(content="nothing to do")])

    update = make_property_estimator_node([], llm)(state_with_candidate())

    # LLM never called the tool, but estimates exist anyway (code fallback).
    assert {e.candidate_id for e in update["property_estimates"]} == {"cand-0-rev0"}


def test_no_new_candidates_is_a_noop():
    llm = FakeToolCallingLLM()
    state = GraphState(target_spec=make_target_spec())

    assert make_property_estimator_node([], llm)(state) == {}
    assert llm.tool_invokes == 0


def test_existing_estimates_are_preserved():
    llm = FakeToolCallingLLM()
    state = state_with_candidate()
    first = make_property_estimator_node([], llm)(state)

    state = state.model_copy(
        update={
            **first,
            "candidates": state.candidates + [make_candidate("cand-1-rev1", revision=1)],
        }
    )
    second = make_property_estimator_node([], FakeToolCallingLLM())(state)

    assert {e.candidate_id for e in second["property_estimates"]} == {
        "cand-0-rev0",
        "cand-1-rev1",
    }
    # cand-0 results carried over unchanged, not recomputed/duplicated.
    assert [e for e in second["property_estimates"] if e.candidate_id == "cand-0-rev0"] == [
        e for e in first["property_estimates"] if e.candidate_id == "cand-0-rev0"
    ]
