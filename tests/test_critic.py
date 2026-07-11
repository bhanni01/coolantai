from conftest import FakeStructuredLLM, make_candidate, make_target_spec

from coolant_copilot.nodes.critic import CriticDraft, ScoreDraft, make_critic_node
from coolant_copilot.state import CriticScore, CriticWeights, GraphState


def make_score_draft(index: int, level: float = 7.0, verdict: str = "accept") -> ScoreDraft:
    # All four subscores equal `level`, so the code-computed weighted overall
    # is exactly `level` under any weights — keeps ordering assertions simple.
    return ScoreDraft(
        candidate_index=index,
        performance=level,
        compliance_risk=level,
        practicality=level,
        lifespan=level,
        verdict=verdict,
        feedback="ok" if verdict == "accept" else "improve viscosity",
    )


def prior_score(candidate_id: str, overall: float, verdict: str = "revise") -> CriticScore:
    return CriticScore(
        candidate_id=candidate_id,
        performance=overall,
        compliance_risk=8,
        practicality=6,
        overall=overall,
        verdict=verdict,
        feedback="prior",
    )


def test_scores_map_to_candidate_ids_and_invalid_indices_drop():
    state = GraphState(
        target_spec=make_target_spec(),
        candidates=[make_candidate("cand-0-rev0"), make_candidate("cand-1-rev0")],
    )
    fake = FakeStructuredLLM(
        CriticDraft(
            scores=[
                make_score_draft(0, level=8.0),
                make_score_draft(1, level=6.0, verdict="revise"),
                make_score_draft(1, level=2.0),  # duplicate index dropped
                make_score_draft(99),  # out of range dropped
            ],
            feedback_for_generator="Blend B needs a lower-viscosity co-base.",
        )
    )

    update = make_critic_node(llm=fake)(state)

    assert [(s.candidate_id, s.overall) for s in update["critic_scores"]] == [
        ("cand-0-rev0", 8.0),
        ("cand-1-rev0", 6.0),
    ]
    assert update["revision_feedback"] == "Blend B needs a lower-viscosity co-base."
    assert fake.schemas == [CriticDraft]


def test_shortlist_ranked_in_code_excluding_rejects():
    state = GraphState(
        target_spec=make_target_spec(),
        candidates=[
            make_candidate("cand-0-rev0"),
            make_candidate("cand-1-rev0"),
            make_candidate("cand-2-rev0"),
        ],
    )
    fake = FakeStructuredLLM(
        CriticDraft(
            scores=[
                make_score_draft(0, level=6.0),
                make_score_draft(1, level=9.0),
                make_score_draft(2, level=1.0, verdict="reject"),
            ],
            feedback_for_generator="",
        )
    )

    update = make_critic_node(llm=fake)(state)

    assert update["shortlist"] == ["cand-1-rev0", "cand-0-rev0"]


def test_scores_accumulate_and_only_unscored_candidates_are_prompted():
    state = GraphState(
        target_spec=make_target_spec(),
        candidates=[make_candidate("cand-0-rev0"), make_candidate("cand-1-rev1", revision=1)],
        critic_scores=[prior_score("cand-0-rev0", overall=5.0)],
    )
    fake = FakeStructuredLLM(
        CriticDraft(scores=[make_score_draft(0, level=8.5)], feedback_for_generator="")
    )

    update = make_critic_node(llm=fake)(state)

    # Index 0 refers to the first *unscored* candidate: cand-1-rev1.
    assert [(s.candidate_id, s.overall) for s in update["critic_scores"]] == [
        ("cand-0-rev0", 5.0),
        ("cand-1-rev1", 8.5),
    ]
    assert update["shortlist"] == ["cand-1-rev1", "cand-0-rev0"]
    assert "[candidate 0]" in fake.last_user_prompt
    assert "[candidate 1]" not in fake.last_user_prompt


def test_missing_property_estimates_surface_as_gaps_in_prompt():
    state = GraphState(target_spec=make_target_spec(), candidates=[make_candidate("cand-0-rev0")])
    fake = FakeStructuredLLM(
        CriticDraft(scores=[make_score_draft(0)], feedback_for_generator="")
    )

    make_critic_node(llm=fake)(state)

    # No evaluator output in state: every spec target is a data gap.
    assert "no estimate available" in fake.last_user_prompt
    assert "thermal_conductivity" in fake.last_user_prompt


def test_critic_weights_reorder_the_shortlist():
    """Same candidates and subscores, different optimization weights → the
    code-computed overall flips the ranking (weights drive the shortlist)."""

    def contrasting_draft() -> CriticDraft:
        return CriticDraft(
            scores=[
                # cand-0: strong performance, poor practicality (cost proxy).
                ScoreDraft(candidate_index=0, performance=9, compliance_risk=6,
                           practicality=2, lifespan=5, verdict="accept", feedback=""),
                # cand-1: the mirror image.
                ScoreDraft(candidate_index=1, performance=2, compliance_risk=6,
                           practicality=9, lifespan=5, verdict="accept", feedback=""),
            ],
            feedback_for_generator="",
        )

    def run_with(weights: CriticWeights) -> list[str]:
        state = GraphState(
            target_spec=make_target_spec(),
            critic_weights=weights,
            candidates=[make_candidate("cand-0-rev0"), make_candidate("cand-1-rev0")],
        )
        return make_critic_node(llm=FakeStructuredLLM(contrasting_draft()))(state)["shortlist"]

    performance_first = run_with(
        CriticWeights(performance=0.55, cost=0.15, compliance=0.15, lifespan=0.15)
    )
    cost_first = run_with(
        CriticWeights(performance=0.15, cost=0.55, compliance=0.15, lifespan=0.15)
    )

    assert performance_first == ["cand-0-rev0", "cand-1-rev0"]
    assert cost_first == ["cand-1-rev0", "cand-0-rev0"]


def test_no_unscored_candidates_skips_llm():
    state = GraphState(
        target_spec=make_target_spec(),
        candidates=[make_candidate("cand-0-rev0")],
        critic_scores=[prior_score("cand-0-rev0", overall=5.0)],
    )
    fake = FakeStructuredLLM(CriticDraft(scores=[], feedback_for_generator=""))

    update = make_critic_node(llm=fake)(state)

    assert fake.calls == 0
    assert update == {}
