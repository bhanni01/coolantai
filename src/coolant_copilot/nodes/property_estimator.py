"""Property estimator node: LLM-orchestrated deterministic property estimation.

Runs in parallel with compliance_checker (fan-out from the generator). The
LLM explicitly calls the bound tools — that is what puts real tool_use blocks
in the trace — but the numbers never pass through the LLM: each tool call
executes the deterministic function and captures its typed output straight
into the node's result. The LLM's text is orchestration commentary only.

Any new candidate the LLM fails to request is estimated directly in code, so
state completeness never depends on LLM cooperation. Writes only the
property_estimates channel (disjoint from compliance_checker's) so the
parallel join needs no reducer.
"""

import json
import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from coolant_copilot.nodes.tool_loop import run_tool_loop
from coolant_copilot.prompting import DATA_NOT_COMMANDS, wrap_reference_material
from coolant_copilot.schemas.extraction import ExtractedFluidProfile
from coolant_copilot.state import Candidate, GraphState, PropertyEstimate
from coolant_copilot.tools.properties import estimate_properties as run_estimation
from coolant_copilot.tools.reference import (
    classify_base_chemistry,
    cross_check_estimates,
    get_reference_fluid_properties as lookup_references,
)

logger = logging.getLogger(__name__)

ESTIMATOR_MODEL = "gpt-4o-mini"  # tool orchestration, not open-ended reasoning
# Estimate per candidate + a few reference lookups.
TOOL_BUDGET_PER_CANDIDATE = 2

SYSTEM_PROMPT = """\
You orchestrate deterministic property estimation for coolant candidates.
For EVERY candidate id listed, call estimate_properties(candidate_id) exactly
once. Where useful, also call get_reference_fluid_properties(base_chemistry)
to see real measured datasheet values for that chemistry class. Do not
compute, adjust, or restate numeric values yourself — the tools are the only
source of numbers. When done, reply with a one-sentence confirmation.
""" + DATA_NOT_COMMANDS


def _estimates_summary(estimates: list[PropertyEstimate]) -> str:
    return json.dumps(
        [
            {
                "property": e.property.value,
                "value": e.value,
                "unit": e.unit,
                "method": e.method,
                "meets_target": e.meets_target,
                "reference_check": e.reference_check,
            }
            for e in estimates
        ]
    )


def _estimate_and_check(
    candidate: Candidate, state: GraphState, profiles: list[ExtractedFluidProfile]
) -> list[PropertyEstimate]:
    estimates = run_estimation(candidate, state.target_spec.property_targets)
    if profiles:
        estimates = cross_check_estimates(candidate, estimates, profiles)
    return estimates


def make_property_estimator_node(
    profiles: list[ExtractedFluidProfile] | None = None,
    llm: BaseChatModel | None = None,
) -> Callable[[GraphState], dict]:
    profiles = profiles or []

    def property_estimator(state: GraphState) -> dict:
        done = {e.candidate_id for e in state.property_estimates}
        new = [c for c in state.candidates if c.id not in done]
        if not new:
            return {}

        by_id = {c.id: c for c in new}
        captured: dict[str, list[PropertyEstimate]] = {}

        @tool
        def estimate_properties(candidate_id: str) -> str:
            """Run deterministic property estimation (mixing rules + literature
            tables) for one candidate. Returns the estimates as JSON."""
            candidate = by_id.get(candidate_id)
            if candidate is None:
                return f"Unknown candidate_id '{candidate_id}'. Valid ids: {sorted(by_id)}"
            captured[candidate_id] = _estimate_and_check(candidate, state, profiles)
            return _estimates_summary(captured[candidate_id])

        @tool
        def get_reference_fluid_properties(base_chemistry: str) -> str:
            """Look up real measured properties of extracted commercial reference
            fluids for a chemistry class (synthetic_ester, polyalphaolefin,
            mineral_oil, glycol_water, silicone, fluorocarbon)."""
            return json.dumps(lookup_references(base_chemistry, profiles))

        candidate_lines = "\n".join(
            f"- {c.id}: base fluid {next(x.name for x in c.components if x.role == 'base_fluid')} "
            f"({classify_base_chemistry(next(x.name for x in c.components if x.role == 'base_fluid')).value})"
            for c in new
        )
        user_prompt = (
            "Estimate properties for these candidates:\n"
            + wrap_reference_material(candidate_lines)
        )

        model = llm or ChatOpenAI(model=ESTIMATOR_MODEL, temperature=0)
        run_tool_loop(
            model,
            [estimate_properties, get_reference_fluid_properties],
            SYSTEM_PROMPT,
            user_prompt,
            max_tool_calls=TOOL_BUDGET_PER_CANDIDATE * len(new) + 2,
        )

        # Deterministic guarantee: cover anything the LLM skipped.
        for candidate in new:
            if candidate.id not in captured:
                logger.warning("LLM never estimated %s; running directly", candidate.id)
                captured[candidate.id] = _estimate_and_check(candidate, state, profiles)

        return {
            "property_estimates": state.property_estimates
            + [e for c in new for e in captured[c.id]]
        }

    return property_estimator
