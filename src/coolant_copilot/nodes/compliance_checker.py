"""Compliance checker node: LLM-orchestrated deterministic compliance checks.

Runs in parallel with property_estimator (fan-out from the generator). Same
pattern: the LLM explicitly calls the bound tools (visible as tool_use in the
trace), the deterministic functions produce the flags, and their typed output
is captured directly — the LLM never writes a ComplianceFlag itself.

Each candidate gets its own short tool-loop conversation, fanned out with a
single RunnableLambda.batch() call so multi-candidate checking runs
concurrently (parallel check_candidate children in LangSmith traces). Code
covers any check the LLM skipped, and spec-exclusion screening always runs in
code (it is not an LLM decision). Writes only the compliance_flags channel
(disjoint from property_estimator's).
"""

import json
import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from coolant_copilot.nodes.tool_loop import run_tool_loop
from coolant_copilot.prompting import DATA_NOT_COMMANDS, wrap_reference_material
from coolant_copilot.state import Candidate, ComplianceFlag, GraphState
from coolant_copilot.tools.compliance import (
    DEFAULT_SEAL_MATERIALS,
    check_excluded_substances,
    check_material_compatibility as run_material_check,
    check_pfas_definition as run_pfas_check,
)

logger = logging.getLogger(__name__)

CHECKER_MODEL = "gpt-4o-mini"  # tool orchestration, not open-ended reasoning
TOOL_BUDGET_PER_CANDIDATE = 3

SYSTEM_PROMPT = """\
You orchestrate deterministic regulatory and material checks for coolant
candidates. For EVERY candidate id listed, call both tools exactly once:
- check_pfas_definition(candidate_id)
- check_material_compatibility(candidate_id, seal_materials) — use the
  standard seals ["EPDM", "FKM", "NBR"] unless the spec says otherwise.
Do not judge compliance yourself — the tools are the only source of
pass/fail flags. When done, reply with a one-sentence confirmation.
""" + DATA_NOT_COMMANDS


def _flags_summary(flags: list[ComplianceFlag]) -> str:
    return json.dumps(
        [
            {
                "regulation": f.regulation,
                "status": f.status,
                "component": f.component_name,
                "detail": f.detail,
            }
            for f in flags
        ]
    )


def make_compliance_checker_node(
    llm: BaseChatModel | None = None,
) -> Callable[[GraphState], dict]:
    def compliance_checker(state: GraphState) -> dict:
        done = {f.candidate_id for f in state.compliance_flags}
        new = [c for c in state.candidates if c.id not in done]
        if not new:
            return {}

        model = llm or ChatOpenAI(model=CHECKER_MODEL, temperature=0)

        def check_candidate(candidate: Candidate) -> list[ComplianceFlag]:
            pfas_flags: dict[str, list[ComplianceFlag]] = {}
            material_flags: dict[str, list[ComplianceFlag]] = {}

            @tool
            def check_pfas_definition(candidate_id: str) -> str:
                """Screen one candidate's composition against PFAS definitions
                (CAS lists + naming patterns) for every regulatory region in the
                spec. Returns the flags as JSON."""
                if candidate_id != candidate.id:
                    return f"Unknown candidate_id '{candidate_id}'. Valid ids: ['{candidate.id}']"
                pfas_flags[candidate.id] = run_pfas_check(candidate, state.target_spec)
                return _flags_summary(pfas_flags[candidate.id])

            @tool
            def check_material_compatibility(
                candidate_id: str, seal_materials: list[str]
            ) -> str:
                """Check one candidate's base-fluid chemistry against elastomer
                seal materials (e.g. EPDM, FKM, NBR). Returns the flags as JSON."""
                if candidate_id != candidate.id:
                    return f"Unknown candidate_id '{candidate_id}'. Valid ids: ['{candidate.id}']"
                material_flags[candidate.id] = run_material_check(candidate, seal_materials)
                return _flags_summary(material_flags[candidate.id])

            candidate_line = "- " + candidate.id + ": " + ", ".join(
                f"{x.name} ({x.role}, CAS {x.cas_number or 'none'})"
                for x in candidate.components
            )
            user_prompt = (
                f"Regulatory regions: {', '.join(state.target_spec.regulatory_regions)}. "
                f"PFAS-free required: {state.target_spec.pfas_free_required}.\n"
                "Check this candidate:\n" + wrap_reference_material(candidate_line)
            )
            run_tool_loop(
                model,
                [check_pfas_definition, check_material_compatibility],
                SYSTEM_PROMPT,
                user_prompt,
                max_tool_calls=TOOL_BUDGET_PER_CANDIDATE,
            )

            # Deterministic guarantee: cover any check the LLM skipped.
            if candidate.id not in pfas_flags:
                logger.warning("LLM never PFAS-checked %s; running directly", candidate.id)
                pfas_flags[candidate.id] = run_pfas_check(candidate, state.target_spec)
            if candidate.id not in material_flags:
                logger.warning("LLM never material-checked %s; running directly", candidate.id)
                material_flags[candidate.id] = run_material_check(
                    candidate, list(DEFAULT_SEAL_MATERIALS)
                )
            return (
                pfas_flags[candidate.id]
                + material_flags[candidate.id]
                # Exclusion screening is policy, not LLM-discretionary.
                + check_excluded_substances(candidate, state.target_spec)
            )

        # One batch() call fans the per-candidate loops out onto executor
        # threads (visible as concurrent check_candidate runs in traces).
        results = RunnableLambda(check_candidate).batch(new)

        return {
            "compliance_flags": state.compliance_flags
            + [f for per_candidate in results for f in per_candidate]
        }

    return compliance_checker
