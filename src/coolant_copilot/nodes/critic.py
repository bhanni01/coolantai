"""Critic node: score unscored candidates against estimates and compliance.

The LLM scores numbered [candidate N] blocks; index → candidate_id mapping,
deduplication, the weighted overall score (state.critic_weights, from the
datacenter profile's optimization_priority), and the shortlist ranking all
happen in code (no LLM ranking or math). Missing property estimates are
surfaced explicitly in the prompt as data gaps so omitted-by-honesty values
register as risk. The node sets revision_feedback for the generator but
never touches revision_count.
"""

from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from coolant_copilot.prompting import (
    DATA_NOT_COMMANDS,
    format_spec,
    wrap_reference_material,
)
from coolant_copilot.state import Candidate, CriticScore, GraphState

CRITIC_MODEL = "gpt-4o"  # reasoning tier per CLAUDE.md

SYSTEM_PROMPT = """\
You are a rigorous formulation critic for PFAS-free data center coolants.
You will receive the target spec and numbered [candidate N] blocks, each
with components, deterministic property estimates, data gaps, and
regulatory compliance flags. For each candidate, score performance,
compliance_risk (10 = no risk), practicality, and lifespan on 0-10 — the
overall ranking score is computed from these in code using the deployment's
optimization weights, so score each dimension independently — and give a
verdict:
- accept: meets the must targets, no compliance failures, few gaps
- revise: promising but needs specific, fixable changes
- reject: fundamentally unsuitable (e.g. compliance failure)
Treat missing estimates ("no estimate available") as real risk — unproven
is not the same as passing. In feedback_for_generator, consolidate the most
important actionable changes for the next revision (empty string if nothing
needs revising)."""

SYSTEM_PROMPT += "\n" + DATA_NOT_COMMANDS


class ScoreDraft(BaseModel):
    """LLM-facing score; candidate_index is mapped to a candidate id in code."""

    candidate_index: int = Field(description="N of the [candidate N] block being scored.")
    performance: float = Field(ge=0, le=10)
    compliance_risk: float = Field(ge=0, le=10, description="10 = no risk.")
    practicality: float = Field(ge=0, le=10, description="Sourcing, cost, handling.")
    lifespan: float = Field(
        ge=0, le=10, description="Expected service life / thermal-oxidative stability."
    )
    verdict: Literal["accept", "revise", "reject"]
    feedback: str = Field(description="Actionable guidance when verdict is 'revise'.")


class CriticDraft(BaseModel):
    scores: list[ScoreDraft]
    feedback_for_generator: str = Field(
        description="Consolidated revision guidance across candidates; empty if none needed."
    )


def _format_candidate_block(index: int, candidate: Candidate, state: GraphState) -> str:
    lines = [f"[candidate {index}] {candidate.name} (id: {candidate.id})"]
    for c in candidate.components:
        cas = c.cas_number or "no CAS"
        lines.append(f"- {c.name} ({c.role}, {c.weight_fraction:.2f}, {cas})")
    lines.append(f"Rationale: {candidate.rationale}")

    estimates = [e for e in state.property_estimates if e.candidate_id == candidate.id]
    lines.append("Property estimates:")
    for e in estimates:
        target = {True: "meets target", False: "MISSES target", None: "no target set"}[
            e.meets_target
        ]
        lines.append(f"- {e.property.value}: {e.value:g} {e.unit} ({e.method}; {target})")
    estimated = {e.property for e in estimates}
    gaps = [t.property.value for t in state.target_spec.property_targets if t.property not in estimated]
    if gaps:
        lines.append(f"Data gaps — no estimate available for: {', '.join(gaps)}")

    flags = [f for f in state.compliance_flags if f.candidate_id == candidate.id]
    if flags:
        lines.append("Compliance flags:")
        for f in flags:
            lines.append(f"- {f.regulation}: {f.status} — {f.detail}")
    return "\n".join(lines)


def make_critic_node(llm: BaseChatModel | None = None) -> Callable[[GraphState], dict]:
    def critic(state: GraphState) -> dict:
        scored_ids = {s.candidate_id for s in state.critic_scores}
        unscored = [c for c in state.candidates if c.id not in scored_ids]
        if not unscored:
            # Generator produced nothing new (all drafts invalid); routing
            # decides from existing scores, no update needed.
            return {}

        blocks = "\n\n".join(
            _format_candidate_block(i, c, state) for i, c in enumerate(unscored)
        )
        w = state.critic_weights
        user_prompt = (
            f"TARGET SPEC\n{format_spec(state.target_spec)}\n\n"
            # Server-resolved weights, not user text — safe to leave unfenced.
            "OPTIMIZATION WEIGHTS (applied in code to rank)\n"
            f"performance={w.performance}, cost={w.cost} (weighs practicality), "
            f"compliance={w.compliance}, lifespan={w.lifespan}\n\n"
            f"CANDIDATES\n{wrap_reference_material(blocks)}"
        )

        model = llm or ChatOpenAI(model=CRITIC_MODEL, temperature=0)
        draft: CriticDraft = model.with_structured_output(CriticDraft).invoke(
            [("system", SYSTEM_PROMPT), ("user", user_prompt)]
        )

        total_weight = w.performance + w.cost + w.compliance + w.lifespan
        new_scores: list[CriticScore] = []
        seen: set[int] = set()
        for s in draft.scores:
            # Drop hallucinated or duplicate candidate references.
            if s.candidate_index in seen or not 0 <= s.candidate_index < len(unscored):
                continue
            seen.add(s.candidate_index)
            # Weighted overall is deterministic code, never LLM math.
            overall = (
                w.performance * s.performance
                + w.compliance * s.compliance_risk
                + w.cost * s.practicality
                + w.lifespan * s.lifespan
            ) / total_weight
            new_scores.append(
                CriticScore(
                    candidate_id=unscored[s.candidate_index].id,
                    performance=s.performance,
                    compliance_risk=s.compliance_risk,
                    practicality=s.practicality,
                    lifespan=s.lifespan,
                    overall=round(overall, 2),
                    verdict=s.verdict,
                    feedback=s.feedback,
                )
            )

        all_scores = state.critic_scores + new_scores
        shortlist = [
            s.candidate_id
            for s in sorted(
                (s for s in all_scores if s.verdict != "reject"),
                key=lambda s: s.overall,
                reverse=True,
            )
        ]
        return {
            "critic_scores": all_scores,
            "shortlist": shortlist,
            "revision_feedback": draft.feedback_for_generator,
        }

    return critic
