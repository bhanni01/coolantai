from conftest import FakeStructuredLLM, make_candidate, make_target_spec

from coolant_copilot.nodes.experiment_planner import (
    MAX_PLAN_CANDIDATES,
    FactorSetting,
    PlanDraft,
    RunDraft,
    make_experiment_planner_node,
)
from coolant_copilot.state import DOEFactor, GraphState, PropertyName


def make_plan_draft(runs: list[RunDraft], replicates: int = 2) -> PlanDraft:
    return PlanDraft(
        objective="Validate thermal performance.",
        design_type="full_factorial",
        factors=[DOEFactor(name="temperature", unit="°C", levels=[25, 45, 65])],
        runs=runs,
        replicates=replicates,
        safety_notes="Standard lab PPE.",
    )


def make_run(index: int) -> RunDraft:
    return RunDraft(
        candidate_index=index,
        factor_settings=[FactorSetting(factor="temperature", level=45.0)],
        responses_to_measure=[PropertyName.THERMAL_CONDUCTIVITY],
    )


def state_with(shortlist: list[str], n_candidates: int = 4) -> GraphState:
    return GraphState(
        target_spec=make_target_spec(),
        candidates=[make_candidate(f"cand-{i}-rev0") for i in range(n_candidates)],
        shortlist=shortlist,
    )


def test_runs_numbered_and_mapped_to_shortlist_ids():
    state = state_with(["cand-2-rev0", "cand-0-rev0"])
    fake = FakeStructuredLLM(make_plan_draft([make_run(0), make_run(1), make_run(99)]))

    update = make_experiment_planner_node(llm=fake)(state)

    plan = update["experiment_plan"]
    # Index 0 = best shortlisted candidate; invalid index 99 dropped.
    assert [(r.run_number, r.candidate_id) for r in plan.runs] == [
        (1, "cand-2-rev0"),
        (2, "cand-0-rev0"),
    ]
    assert fake.schemas == [PlanDraft]


def test_only_top_shortlist_candidates_are_offered():
    shortlist = [f"cand-{i}-rev0" for i in range(4)]  # more than MAX_PLAN_CANDIDATES
    state = state_with(shortlist)
    fake = FakeStructuredLLM(make_plan_draft([make_run(0)]))

    make_experiment_planner_node(llm=fake)(state)

    for i in range(MAX_PLAN_CANDIDATES):
        assert f"cand-{i}-rev0" in fake.last_user_prompt
    assert "cand-3-rev0" not in fake.last_user_prompt


def test_empty_shortlist_falls_back_to_all_candidates():
    # Cap-exhausted path can arrive without any accepted candidate.
    state = state_with([], n_candidates=2)
    fake = FakeStructuredLLM(make_plan_draft([make_run(1)]))

    update = make_experiment_planner_node(llm=fake)(state)

    assert update["experiment_plan"].runs[0].candidate_id == "cand-1-rev0"


def test_replicates_clamped_to_at_least_one():
    state = state_with(["cand-0-rev0"])
    fake = FakeStructuredLLM(make_plan_draft([make_run(0)], replicates=0))

    update = make_experiment_planner_node(llm=fake)(state)

    assert update["experiment_plan"].replicates == 1
