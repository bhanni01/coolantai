from conftest import FakeStructuredLLM, make_candidate, make_finding, make_target_spec

from coolant_copilot.nodes.generator import CandidateDraft, GenerationDraft, make_generator_node
from coolant_copilot.state import Component, CriticScore, GraphState


def make_draft(name: str = "Ester blend", source_ref_ids: list[str] | None = None) -> CandidateDraft:
    return CandidateDraft(
        name=name,
        components=[
            Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=0.95),
            Component(name="BHT", role="antioxidant", weight_fraction=0.05),
        ],
        rationale="High flash point ester.",
        source_ref_ids=source_ref_ids if source_ref_ids is not None else ["rf-0"],
    )


def initial_state() -> GraphState:
    return GraphState(target_spec=make_target_spec(), research_findings=[make_finding("rf-0")])


def test_initial_generation_assigns_rev0_ids_and_keeps_counter():
    fake = FakeStructuredLLM(GenerationDraft(candidates=[make_draft("A"), make_draft("B")]))

    update = make_generator_node(llm=fake)(initial_state())

    assert [c.id for c in update["candidates"]] == ["cand-0-rev0", "cand-1-rev0"]
    assert all(c.revision == 0 for c in update["candidates"])
    assert "revision_count" not in update
    assert update["revision_feedback"] is None
    assert fake.schemas == [GenerationDraft]


def test_revision_appends_increments_counter_and_feeds_back_critique():
    state = initial_state().model_copy(
        update={
            "candidates": [make_candidate("cand-0-rev0")],
            "critic_scores": [
                CriticScore(
                    candidate_id="cand-0-rev0",
                    performance=4,
                    compliance_risk=9,
                    practicality=6,
                    overall=5,
                    verdict="revise",
                    feedback="Viscosity too high.",
                )
            ],
            "revision_count": 0,
            "revision_feedback": "Lower the viscosity; consider PAO co-base.",
        }
    )
    fake = FakeStructuredLLM(GenerationDraft(candidates=[make_draft("A v2")]))

    update = make_generator_node(llm=fake)(state)

    assert [c.id for c in update["candidates"]] == ["cand-0-rev0", "cand-1-rev1"]
    assert update["candidates"][1].revision == 1
    assert update["revision_count"] == 1
    assert update["revision_feedback"] is None
    # The critique and the prior candidate made it into the prompt.
    assert "Lower the viscosity" in fake.last_user_prompt
    assert "cand-0-rev0" in fake.last_user_prompt


def test_weight_fractions_are_renormalized():
    draft = make_draft()
    draft.components = [
        Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=0.9),
        Component(name="BHT", role="antioxidant", weight_fraction=0.3),
    ]
    fake = FakeStructuredLLM(GenerationDraft(candidates=[draft]))

    update = make_generator_node(llm=fake)(initial_state())

    total = sum(c.weight_fraction for c in update["candidates"][0].components)
    assert abs(total - 1.0) < 1e-6


def test_invalid_drafts_are_skipped_not_raised():
    bad = make_draft("no base fluid")
    bad.components = [Component(name="BHT", role="antioxidant", weight_fraction=1.0)]
    fake = FakeStructuredLLM(GenerationDraft(candidates=[bad, make_draft("good")]))

    update = make_generator_node(llm=fake)(initial_state())

    assert [c.name for c in update["candidates"]] == ["good"]


def test_hallucinated_source_refs_are_dropped():
    fake = FakeStructuredLLM(
        GenerationDraft(candidates=[make_draft(source_ref_ids=["rf-0", "rf-99"])])
    )

    update = make_generator_node(llm=fake)(initial_state())

    assert update["candidates"][0].source_refs == ["rf-0"]
