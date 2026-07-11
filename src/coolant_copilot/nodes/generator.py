"""Generator node: propose candidate formulations grounded in research findings.

The LLM fills CandidateDraft entries without ids or revision numbers; those
are assigned in code, weight fractions are renormalized, and source refs are
filtered to research findings that actually exist. On revision passes
(state.revision_feedback set) the prompt includes the prior candidates, their
critic scores, and the feedback, and the node increments revision_count.
"""

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from coolant_copilot.prompting import (
    DATA_NOT_COMMANDS,
    format_spec,
    wrap_reference_material,
)
from coolant_copilot.state import Candidate, Component, GraphState

GENERATOR_MODEL = "gpt-4o"  # reasoning tier per CLAUDE.md
N_CANDIDATES = 3

SYSTEM_PROMPT = """\
You are a formulation chemist proposing PFAS-free data center coolant
candidates. You will receive a target spec, research findings (each with an
id), and possibly prior candidates with critic feedback. Propose
{n_candidates} candidate formulations that:
- use exactly one base_fluid component, plus optional additives
- have component weight_fractions summing to 1.0
- never include fluorinated chemistry when the spec requires PFAS-free
- never include the spec's excluded substances
- set cas_number only when you are certain; otherwise leave it null
- cite the ids of the research findings that ground each proposal in
  source_ref_ids (only ids that were given to you)
- in each rationale, explicitly name the specific source document or reference
  fluid you are drawing from — quote the `source:` shown on the finding, e.g.
  "based on the Shell S5 X datasheet's flash point" or "per the OCP immersion
  fluid spec's viscosity range" — do not leave the grounding implicit in the
  source_ref_ids alone
When critic feedback is present, address it directly with revised
formulations rather than repeating the prior candidates."""

SYSTEM_PROMPT += "\n" + DATA_NOT_COMMANDS


class CandidateDraft(BaseModel):
    """LLM-facing candidate; id and revision are assigned in code."""

    name: str
    components: list[Component]
    rationale: str
    source_ref_ids: list[str] = Field(
        default_factory=list, description="Ids of research findings grounding this proposal."
    )


class GenerationDraft(BaseModel):
    candidates: list[CandidateDraft]


def _normalized(components: list[Component]) -> list[Component]:
    total = sum(c.weight_fraction for c in components)
    if total <= 0 or abs(total - 1.0) <= 1e-3:
        return components
    return [c.model_copy(update={"weight_fraction": c.weight_fraction / total}) for c in components]


def _format_findings(state: GraphState) -> str:
    if not state.research_findings:
        return "No research findings available."
    # Surface the source document so the rationale can name it, not just cite the id.
    return "\n".join(
        f"[{f.id}] (source: {f.source}) {f.summary} ({f.relevance})"
        for f in state.research_findings
    )


def _format_revision_context(state: GraphState) -> str:
    scores_by_id = {s.candidate_id: s for s in state.critic_scores}
    lines = ["PRIOR CANDIDATES AND CRITIC SCORES"]
    for cand in state.candidates:
        components = ", ".join(
            f"{c.name} ({c.role}, {c.weight_fraction:.2f})" for c in cand.components
        )
        lines.append(f"- {cand.id} '{cand.name}': {components}")
        if score := scores_by_id.get(cand.id):
            lines.append(
                f"  scored {score.overall}/10 ({score.verdict}): {score.feedback}"
            )
    lines.append(f"\nCRITIC FEEDBACK TO ADDRESS\n{state.revision_feedback}")
    return "\n".join(lines)


def make_generator_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], dict]:
    def generator(state: GraphState) -> dict:
        is_revision = state.revision_feedback is not None
        rev = state.revision_count + 1 if is_revision else 0

        parts = [
            f"TARGET SPEC\n{format_spec(state.target_spec)}",
            f"RESEARCH FINDINGS\n{wrap_reference_material(_format_findings(state))}",
        ]
        if is_revision:
            # Candidates and critic feedback are LLM-derived from document
            # content and the spec, so they carry the same taint.
            parts.append(wrap_reference_material(_format_revision_context(state)))

        model = llm or ChatOpenAI(model=GENERATOR_MODEL, temperature=0)
        draft: GenerationDraft = model.with_structured_output(GenerationDraft).invoke(
            [
                ("system", SYSTEM_PROMPT.format(n_candidates=N_CANDIDATES)),
                ("user", "\n\n".join(parts)),
            ]
        )

        valid_ref_ids = {f.id for f in state.research_findings}
        new_candidates: list[Candidate] = []
        for entry in draft.candidates:
            try:
                candidate = Candidate(
                    id=f"cand-{len(state.candidates) + len(new_candidates)}-rev{rev}",
                    name=entry.name,
                    components=_normalized(entry.components),
                    rationale=entry.rationale,
                    source_refs=[r for r in entry.source_ref_ids if r in valid_ref_ids],
                    revision=rev,
                )
            except ValidationError:
                # One malformed draft (e.g. two base fluids) must not kill the run.
                continue
            new_candidates.append(candidate)

        update: dict = {
            "candidates": state.candidates + new_candidates,
            "revision_feedback": None,  # consume the critic → generator channel
        }
        if is_revision:
            update["revision_count"] = rev
        return update

    return generator
