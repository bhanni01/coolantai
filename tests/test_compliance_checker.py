from conftest import FakeToolCallingLLM, make_candidate, make_target_spec, tool_call
from langchain_core.messages import AIMessage

from coolant_copilot.nodes.compliance_checker import make_compliance_checker_node
from coolant_copilot.state import GraphState
from coolant_copilot.tools.compliance import check_material_compatibility


def state_with_candidate(**spec_overrides) -> GraphState:
    return GraphState(
        target_spec=make_target_spec(**spec_overrides),
        candidates=[make_candidate("cand-0-rev0")],
    )


def test_llm_tool_calls_capture_deterministic_flags():
    llm = FakeToolCallingLLM(
        tool_messages=[
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("check_pfas_definition", {"candidate_id": "cand-0-rev0"}, "c1"),
                    tool_call(
                        "check_material_compatibility",
                        {"candidate_id": "cand-0-rev0", "seal_materials": ["EPDM", "FKM"]},
                        "c2",
                    ),
                ],
            ),
        ]
    )

    update = make_compliance_checker_node(llm)(state_with_candidate())

    flags = update["compliance_flags"]
    regulations = {f.regulation for f in flags}
    # PFAS per region + material compat for exactly the seals the LLM passed.
    assert {"EU-PFAS-2024", "TSCA-8a7", "MATERIAL-COMPAT-EPDM", "MATERIAL-COMPAT-FKM"} <= regulations
    assert "MATERIAL-COMPAT-NBR" not in regulations
    # Ester base fluid: EPDM swells (fail), FKM is fine (pass).
    assert next(f.status for f in flags if f.regulation == "MATERIAL-COMPAT-EPDM") == "fail"
    assert next(f.status for f in flags if f.regulation == "MATERIAL-COMPAT-FKM") == "pass"


def test_fallback_runs_all_checks_when_llm_skips():
    llm = FakeToolCallingLLM(tool_messages=[AIMessage(content="pass")])

    update = make_compliance_checker_node(llm)(state_with_candidate())

    regulations = {f.regulation for f in update["compliance_flags"]}
    assert {
        "EU-PFAS-2024",
        "TSCA-8a7",
        "MATERIAL-COMPAT-EPDM",
        "MATERIAL-COMPAT-FKM",
        "MATERIAL-COMPAT-NBR",
    } <= regulations


def test_exclusion_screening_always_runs_in_code():
    # The LLM has no exclusion tool; BHT is excluded by spec and must be
    # flagged regardless of what the LLM does.
    llm = FakeToolCallingLLM()
    update = make_compliance_checker_node(llm)(
        state_with_candidate(excluded_substances=["bht"])
    )

    excluded = [f for f in update["compliance_flags"] if f.regulation == "SPEC-EXCLUDED"]
    assert len(excluded) == 1
    assert excluded[0].status == "fail"
    assert excluded[0].component_name == "BHT"


def test_no_new_candidates_is_a_noop():
    llm = FakeToolCallingLLM()
    assert make_compliance_checker_node(llm)(GraphState(target_spec=make_target_spec())) == {}
    assert llm.tool_invokes == 0


class TestMaterialCompatibilityTable:
    def test_unknown_seal_needs_review(self):
        flags = check_material_compatibility(make_candidate(), ["KALREZ"])
        assert flags[0].regulation == "MATERIAL-COMPAT-KALREZ"
        assert flags[0].status == "needs_review"

    def test_default_seals_when_none_given(self):
        flags = check_material_compatibility(make_candidate(), None)
        assert {f.regulation for f in flags} == {
            "MATERIAL-COMPAT-EPDM",
            "MATERIAL-COMPAT-FKM",
            "MATERIAL-COMPAT-NBR",
        }
