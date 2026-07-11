"""Research node: ReAct-style retrieval loop, then distill a ResearchBrief.

Phase 1 — gathering: the LLM drives bound tools (search_chroma,
get_extracted_fluid_profile), capped at MAX_TOOL_CALLS executed calls. Every
chunk a search returns is accumulated in code; the LLM chooses queries but
cannot invent chunks.

Phase 2 — distillation: the gathered chunks go through the token-budget
selector and one structured call fills BriefDraft. Findings reference chunks
by index; sources on the final ResearchFindings are mapped from retrieval
metadata in code, never written by the LLM.
"""

import json
import logging
from typing import Callable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from coolant_copilot.nodes.tool_loop import run_tool_loop
from coolant_copilot.observability import get_encoding
from coolant_copilot.prompting import (
    DATA_NOT_COMMANDS,
    format_spec,
    wrap_reference_material,
)
from coolant_copilot.schemas.extraction import ExtractedFluidProfile
from coolant_copilot.state import GraphState, ResearchBrief, ResearchFinding, TargetSpec

logger = logging.getLogger(__name__)

RESEARCH_MODEL = "gpt-4o-mini"  # cheap extraction tier
K_PER_SEARCH = 4
MAX_TOOL_CALLS = 3
# Retrieved context is capped by tokens, not just count: take gathered chunks
# in retrieval order but stop adding once the cumulative context reaches this.
MAX_CONTEXT_TOKENS = 3000

SEARCH_SYSTEM_PROMPT = f"""\
You are a formulation research assistant for PFAS-free data center coolants.
Gather literature for the target spec using the tools:
- search_chroma(query): search the ingested literature; issue focused,
  *distinct* queries (chemistry classes, key properties, compliance).
- get_extracted_fluid_profile(fluid_name): fetch the extracted datasheet
  profile of a specific commercial fluid you saw mentioned.
You have a hard budget of {MAX_TOOL_CALLS} tool calls total — spend them on
different angles rather than rephrasing one query. When the budget is spent
or you have enough, reply with a one-sentence confirmation.
{DATA_NOT_COMMANDS}"""

SUMMARY_SYSTEM_PROMPT = f"""\
You are a formulation research assistant for PFAS-free data center coolants.
You will receive a target spec and numbered source chunks retrieved from a
literature database. Distill them into a research brief:
- overview: what the sources collectively say that matters for this spec
- findings: one entry per chunk that is actually relevant; set chunk_index to
  the number of the [chunk N] block it comes from, and skip irrelevant chunks
- gaps: important questions for this spec that the chunks do not answer
Only report what the chunks support — do not add outside knowledge.
{DATA_NOT_COMMANDS}"""


class FindingDraft(BaseModel):
    chunk_index: int = Field(description="N of the [chunk N] block this finding comes from.")
    summary: str = Field(description="What this chunk says, in 1-2 sentences.")
    relevance: str = Field(description="Why it matters for the target spec.")


class BriefDraft(BaseModel):
    """LLM-facing schema; mapped to ResearchBrief in code."""

    overview: str
    findings: list[FindingDraft]
    gaps: list[str]


def build_query(spec: TargetSpec) -> str:
    targets = ", ".join(
        f"{t.property.value} {t.min_value or ''}-{t.max_value or ''} {t.unit}".strip()
        for t in spec.property_targets
    )
    return (
        f"PFAS-free {spec.application.replace('_', ' ')} coolant. "
        f"{spec.description} Target properties: {targets}"
    )


def _chunk_block(index: int, chunk: Document) -> str:
    header = f"[chunk {index}] (source: {chunk.metadata.get('source', 'unknown')}"
    if "page" in chunk.metadata:
        header += f", page {chunk.metadata['page']}"
    return header + f")\n{chunk.page_content}"


def _select_within_budget(
    chunks: list[Document], max_tokens: int = MAX_CONTEXT_TOKENS
) -> tuple[list[Document], list[str], int]:
    """Take chunks in gathering order until the token budget is reached.

    Always keeps at least the top chunk, even if it alone exceeds the budget.
    Returns (selected_chunks, formatted_blocks, total_context_tokens).
    """
    enc = get_encoding(RESEARCH_MODEL)
    selected: list[Document] = []
    blocks: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks):
        block = _chunk_block(i, chunk)
        block_tokens = len(enc.encode(block))
        if selected and total + block_tokens > max_tokens:
            break
        selected.append(chunk)
        blocks.append(block)
        total += block_tokens
    return selected, blocks, total


def research(
    spec: TargetSpec,
    vectorstore: VectorStore,
    llm: BaseChatModel | None = None,
    profiles: list[ExtractedFluidProfile] | None = None,
) -> ResearchBrief:
    """ReAct gathering loop over the vectorstore, then one structured brief."""
    profiles = profiles or []
    gathered: list[Document] = []
    seen: set[tuple] = set()
    profile_notes: list[str] = []

    @tool
    def search_chroma(query: str) -> str:
        """Search the ingested coolant literature. Returns matching chunks;
        chunks already returned by earlier searches are not repeated."""
        chunks = vectorstore.similarity_search(query, k=K_PER_SEARCH)
        new_blocks: list[str] = []
        for chunk in chunks:
            key = (chunk.metadata.get("source"), chunk.metadata.get("page"), chunk.page_content)
            if key in seen:
                continue
            seen.add(key)
            gathered.append(chunk)
            new_blocks.append(_chunk_block(len(gathered) - 1, chunk))
        if not new_blocks:
            return "No new results for this query."
        return wrap_reference_material("\n\n".join(new_blocks))

    @tool
    def get_extracted_fluid_profile(fluid_name: str) -> str:
        """Fetch the extracted datasheet profile (real measured properties) of
        a named commercial fluid, if one was extracted during ingestion."""
        needle = fluid_name.strip().lower()
        matches = [p for p in profiles if needle in p.fluid_name.lower()]
        if not matches:
            available = ", ".join(p.fluid_name for p in profiles) or "none"
            return f"No extracted profile matches '{fluid_name}'. Available: {available}."
        note = matches[0].model_dump_json()
        profile_notes.append(note)
        return wrap_reference_material(note)

    model = llm or ChatOpenAI(model=RESEARCH_MODEL, temperature=0)
    user_prompt = (
        f"TARGET SPEC\n{format_spec(spec)}\n\n"
        f"A reasonable first query would be: {build_query(spec)}"
    )
    _, executed = run_tool_loop(
        model,
        [search_chroma, get_extracted_fluid_profile],
        SEARCH_SYSTEM_PROMPT,
        user_prompt,
        max_tool_calls=MAX_TOOL_CALLS,
    )
    logger.info("research loop: %d tool call(s), %d chunk(s) gathered", executed, len(gathered))

    if not gathered:
        return ResearchBrief(
            overview="No relevant sources found in the literature database.",
            findings=[],
            gaps=["Literature database returned nothing for this spec; ingest sources first."],
        )

    # Cap the gathered context by token count; chunk_index below refers to
    # positions in `selected`, which is what the LLM actually sees.
    selected, blocks, context_tokens = _select_within_budget(gathered)
    logger.info(
        "research context: %d/%d gathered chunks, %d tokens (budget %d)",
        len(selected),
        len(gathered),
        context_tokens,
        MAX_CONTEXT_TOKENS,
    )

    chunk_blocks = "\n\n".join(blocks)
    parts = [
        f"TARGET SPEC\n{format_spec(spec)}",
        f"SOURCE CHUNKS\n{wrap_reference_material(chunk_blocks)}",
    ]
    if profile_notes:
        parts.append(
            "EXTRACTED FLUID PROFILES (context only — findings must cite chunks)\n"
            + wrap_reference_material("\n".join(profile_notes))
        )

    draft: BriefDraft = model.with_structured_output(BriefDraft).invoke(
        [("system", SUMMARY_SYSTEM_PROMPT), ("user", "\n\n".join(parts))]
    )

    findings: list[ResearchFinding] = []
    seen_indexes: set[int] = set()
    for f in draft.findings:
        # Drop hallucinated or duplicate chunk references.
        if f.chunk_index in seen_indexes or not 0 <= f.chunk_index < len(selected):
            continue
        seen_indexes.add(f.chunk_index)
        findings.append(
            ResearchFinding(
                id=f"rf-{f.chunk_index}",
                source=selected[f.chunk_index].metadata.get("source", "unknown"),
                summary=f.summary,
                relevance=f.relevance,
            )
        )

    return ResearchBrief(overview=draft.overview, findings=findings, gaps=draft.gaps)


def make_research_node(
    vectorstore: VectorStore,
    llm: BaseChatModel | None = None,
    profiles: list[ExtractedFluidProfile] | None = None,
) -> Callable[[GraphState], dict]:
    """Adapt research() to the graph contract; findings are the inter-node channel."""

    def research_node(state: GraphState) -> dict:
        brief = research(state.target_spec, vectorstore, llm=llm, profiles=profiles)
        return {"research_findings": brief.findings}

    return research_node
