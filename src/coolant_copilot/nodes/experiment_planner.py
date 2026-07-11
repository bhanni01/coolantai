"""Experiment planner node: produce the DOE lab validation plan.

Plans for the top shortlisted candidates (all candidates as a fallback when
the revision cap is exhausted without an accept). The LLM references
candidates by index; run numbers and candidate ids are assigned in code.
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
from coolant_copilot.state import (
    DOEFactor,
    ExperimentPlan,
    ExperimentRun,
    GraphState,
    PropertyName,
)

PLANNER_MODEL = "gpt-4o"  # reasoning tier per CLAUDE.md
MAX_PLAN_CANDIDATES = 3

SYSTEM_PROMPT = """\
You are a design-of-experiments planner for coolant lab validation. You will
receive a target spec and numbered [candidate N] blocks with their estimated
properties and data gaps. Produce a DOE plan that:
- prioritizes verifying the spec's 'must' targets and any data gaps
- chooses an appropriate design_type for the factor count
- references candidates by their block number in candidate_index
- includes concrete factor levels within realistic lab operating ranges
- notes real safety considerations for the chemistries involved."""

SYSTEM_PROMPT += "\n" + DATA_NOT_COMMANDS


class FactorSetting(BaseModel):
    """One factor level; OpenAI strict structured output can't express dicts."""

    factor: str
    level: float


class RunDraft(BaseModel):
    """LLM-facing run; run_number and candidate_id are assigned in code."""

    candidate_index: int = Field(description="N of the [candidate N] block this run tests.")
    factor_settings: list[FactorSetting] = Field(description="Factor levels for this run.")
    responses_to_measure: list[PropertyName]


class PlanDraft(BaseModel):
    objective: str
    design_type: Literal[
        "full_factorial",
        "fractional_factorial",
        "box_behnken",
        "central_composite",
        "plackett_burman",
    ]
    factors: list[DOEFactor]
    runs: list[RunDraft]
    replicates: int
    safety_notes: str


def _format_candidate_block(index: int, candidate_id: str, state: GraphState) -> str:
    candidate = next(c for c in state.candidates if c.id == candidate_id)
    lines = [f"[candidate {index}] {candidate.name} (id: {candidate.id})"]
    for c in candidate.components:
        lines.append(f"- {c.name} ({c.role}, {c.weight_fraction:.2f})")
    estimates = [e for e in state.property_estimates if e.candidate_id == candidate_id]
    for e in estimates:
        lines.append(f"- estimated {e.property.value}: {e.value:g} {e.unit} ({e.method})")
    estimated = {e.property for e in estimates}
    gaps = [
        t.property.value for t in state.target_spec.property_targets if t.property not in estimated
    ]
    if gaps:
        lines.append(f"Data gaps to measure: {', '.join(gaps)}")
    return "\n".join(lines)


def make_experiment_planner_node(
    llm: BaseChatModel | None = None,
) -> Callable[[GraphState], dict]:
    def experiment_planner(state: GraphState) -> dict:
        selected = state.shortlist[:MAX_PLAN_CANDIDATES] or [c.id for c in state.candidates]

        blocks = "\n\n".join(
            _format_candidate_block(i, cid, state) for i, cid in enumerate(selected)
        )
        user_prompt = (
            f"TARGET SPEC\n{format_spec(state.target_spec)}\n\n"
            f"CANDIDATES\n{wrap_reference_material(blocks)}"
        )

        model = llm or ChatOpenAI(model=PLANNER_MODEL, temperature=0)
        draft: PlanDraft = model.with_structured_output(PlanDraft).invoke(
            [("system", SYSTEM_PROMPT), ("user", user_prompt)]
        )

        runs: list[ExperimentRun] = []
        for r in draft.runs:
            # Drop hallucinated candidate references.
            if not 0 <= r.candidate_index < len(selected):
                continue
            runs.append(
                ExperimentRun(
                    run_number=len(runs) + 1,
                    candidate_id=selected[r.candidate_index],
                    factor_settings={fs.factor: fs.level for fs in r.factor_settings},
                    responses_to_measure=r.responses_to_measure,
                )
            )

        plan = ExperimentPlan(
            objective=draft.objective,
            design_type=draft.design_type,
            factors=draft.factors,
            runs=runs,
            replicates=max(1, draft.replicates),
            safety_notes=draft.safety_notes,
        )
        return {"experiment_plan": plan}

    return experiment_planner
