from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import VectorStore

from coolant_copilot.ingestion import get_vectorstore, ingest
from coolant_copilot.nodes.research import (
    MAX_CONTEXT_TOKENS,
    MMR_SEARCH_KWARGS,
    BriefDraft,
    FindingDraft,
    _select_within_budget,
    research,
)
from coolant_copilot.state import (
    PropertyName,
    PropertyTarget,
    ResearchBrief,
    TargetSpec,
)


class FakeStructuredLLM:
    """Stands in for a chat model across the research node's two phases.

    `bind_tools` returns a bound model that issues one search_chroma call per
    configured query then stops (driving the ReAct gathering loop);
    `with_structured_output` records the schema/prompt and returns a canned
    distillation response. `calls` counts only the structured-output phase.
    """

    def __init__(self, response, search_queries=("ester coolant thermal conductivity",)):
        self.response = response
        self.search_queries = list(search_queries)
        self.schema = None
        self.messages = None
        self.calls = 0

    def bind_tools(self, tools):
        queries = list(self.search_queries)

        class _Bound:
            def invoke(self, messages):
                if queries:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "search_chroma",
                            "args": {"query": queries.pop(0)},
                            "id": f"search-{len(queries)}",
                            "type": "tool_call",
                        }],
                    )
                return AIMessage(content="done gathering")

        return _Bound()

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self.response


@pytest.fixture
def spec() -> TargetSpec:
    return TargetSpec(
        name="DC-Coolant-A",
        application="single_phase_immersion",
        description="PFAS-free single-phase immersion coolant for hyperscale racks.",
        property_targets=[
            PropertyTarget(
                property=PropertyName.THERMAL_CONDUCTIVITY,
                min_value=0.13,
                unit="W/m·K",
                priority="must",
            ),
            PropertyTarget(
                property=PropertyName.FLASH_POINT,
                min_value=150,
                unit="°C",
                priority="must",
            ),
        ],
        regulatory_regions=["US", "EU"],
    )


@pytest.fixture
def vectorstore(tmp_path: Path):
    """Chroma store with 10 single-chunk documents (more than TOP_K=8)."""
    src = tmp_path / "sources"
    src.mkdir()
    (src / "esters.txt").write_text(
        "Synthetic ester base fluids reach thermal conductivities of 0.14-0.16 "
        "W/m-K and flash points above 250 C, with no fluorinated chemistry."
    )
    for i in range(9):
        (src / f"filler_{i}.txt").write_text(
            f"Filler document {i} about unrelated lubricant additive packages."
        )
    embeddings = DeterministicFakeEmbedding(size=64)
    ingest(src, embeddings, persist_dir=str(tmp_path / "chroma_db"))
    return get_vectorstore(embeddings, persist_dir=str(tmp_path / "chroma_db"))


def test_research_returns_brief_with_sources_from_metadata(spec, vectorstore):
    fake = FakeStructuredLLM(
        BriefDraft(
            overview="Esters look promising for this spec.",
            findings=[
                FindingDraft(chunk_index=0, summary="Esters hit 0.14-0.16 W/m-K.", relevance="Meets the conductivity target."),
                FindingDraft(chunk_index=2, summary="Additive packages exist.", relevance="Background."),
            ],
            gaps=["No data on long-term material compatibility."],
        )
    )

    brief = research(spec, vectorstore, llm=fake)

    assert isinstance(brief, ResearchBrief)
    assert brief.overview == "Esters look promising for this spec."
    assert brief.gaps == ["No data on long-term material compatibility."]
    assert [f.id for f in brief.findings] == ["rf-0", "rf-2"]
    # Sources come from retrieval metadata, not from the LLM.
    ingested_files = {f"filler_{i}.txt" for i in range(9)} | {"esters.txt"}
    assert all(f.source in ingested_files for f in brief.findings)


def test_research_uses_structured_output_and_prompts_from_gathered_chunks(spec, vectorstore):
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    research(spec, vectorstore, llm=fake)

    assert fake.schema is BriefDraft
    system, user = fake.messages
    assert system[0] == "system"
    # Spec details made it into the prompt.
    assert "thermal_conductivity" in user[1]
    assert "single_phase_immersion" in user[1]
    # Chunks gathered through the search_chroma tool loop are numbered in the
    # distillation prompt (the ReAct refactor replaced the old top-8 slice).
    assert "[chunk 0]" in user[1]


def test_research_drops_invalid_and_duplicate_chunk_refs(spec, vectorstore):
    fake = FakeStructuredLLM(
        BriefDraft(
            overview="x",
            findings=[
                FindingDraft(chunk_index=1, summary="ok", relevance="ok"),
                FindingDraft(chunk_index=1, summary="duplicate", relevance="dup"),
                FindingDraft(chunk_index=99, summary="hallucinated", relevance="bad"),
                FindingDraft(chunk_index=-1, summary="negative", relevance="bad"),
            ],
            gaps=[],
        )
    )

    brief = research(spec, vectorstore, llm=fake)

    assert [f.id for f in brief.findings] == ["rf-1"]
    assert brief.findings[0].summary == "ok"


def _doc(text: str, source: str) -> Document:
    return Document(text, metadata={"source": source})


def test_select_within_budget_stops_at_token_cap():
    # Each chunk is ~2000 tokens, so two already blow the 3000-token budget.
    big = "word " * 2000
    chunks = [_doc(big, f"doc{i}.txt") for i in range(5)]

    selected, blocks, total = _select_within_budget(chunks)

    assert len(selected) < len(chunks)  # budget stopped us before all five
    assert len(blocks) == len(selected)
    # Total is under budget, or we kept exactly the one mandatory top chunk.
    assert total <= MAX_CONTEXT_TOKENS or len(selected) == 1


def test_select_within_budget_keeps_at_least_top_chunk():
    huge = "word " * 10000  # single chunk larger than the whole budget
    selected, blocks, total = _select_within_budget([_doc(huge, "big.txt")])

    assert len(selected) == 1
    assert total > MAX_CONTEXT_TOKENS


def test_select_within_budget_keeps_all_when_small():
    chunks = [_doc(f"short chunk {i}", f"doc{i}.txt") for i in range(8)]

    selected, blocks, total = _select_within_budget(chunks)

    assert len(selected) == 8
    assert total < MAX_CONTEXT_TOKENS


def test_research_maps_findings_to_selected_chunk_positions(spec, vectorstore):
    """Findings index into the budget-selected chunks, not the raw retrieval."""
    fake = FakeStructuredLLM(
        BriefDraft(
            overview="ok",
            findings=[FindingDraft(chunk_index=0, summary="s", relevance="r")],
            gaps=[],
        )
    )

    brief = research(spec, vectorstore, llm=fake)

    assert [f.id for f in brief.findings] == ["rf-0"]


class SpyVectorStore(VectorStore):
    """Records retriever calls; relevance pairs drive the threshold gate."""

    def __init__(self, docs=None, relevance=None):
        self.docs = docs if docs is not None else [
            Document("Ester fluids reach 0.15 W/m·K.", metadata={"source": "esters.txt"})
        ]
        self.relevance = relevance  # list[(Document, score)] for threshold search
        self.mmr_kwargs: list[dict] = []
        self.threshold_kwargs: list[dict] = []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError

    def similarity_search(self, query, k=4, **kwargs):
        return list(self.docs)

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, lambda_mult=0.5, **kwargs):
        self.mmr_kwargs.append({"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
        return list(self.docs)

    def similarity_search_with_relevance_scores(self, query, k=4, score_threshold=None, **kwargs):
        self.threshold_kwargs.append({"k": k, "score_threshold": score_threshold})
        pairs = self.relevance if self.relevance is not None else [(d, 1.0) for d in self.docs]
        if score_threshold is not None:
            pairs = [(d, s) for d, s in pairs if s >= score_threshold]
        return pairs


def test_search_chroma_retrieves_via_mmr(spec):
    store = SpyVectorStore()
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    research(spec, store, llm=fake)

    # Every search_chroma call went through the MMR retriever with the
    # configured diversity kwargs — no plain similarity_search.
    assert store.mmr_kwargs
    assert all(kw == MMR_SEARCH_KWARGS for kw in store.mmr_kwargs)


def test_below_threshold_is_distinct_from_empty_collection(spec):
    # Store has content, but nothing scores above the threshold.
    store = SpyVectorStore(relevance=[])
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    brief = research(spec, store, llm=fake, score_threshold=0.5)

    assert fake.calls == 0  # nothing gathered → no distillation call
    assert brief.findings == []
    # The gate was consulted with the configured threshold...
    assert store.threshold_kwargs
    assert all(kw["score_threshold"] == 0.5 for kw in store.threshold_kwargs)
    # ...and nothing was fetched through MMR for the gated queries.
    assert store.mmr_kwargs == []
    # Distinct wording from the empty-collection fallback.
    assert "above threshold" in brief.overview
    assert brief.overview != "No relevant sources found in the literature database."


def test_threshold_pass_still_returns_chunks(spec):
    store = SpyVectorStore(relevance=None)  # everything scores 1.0
    fake = FakeStructuredLLM(
        BriefDraft(
            overview="ok",
            findings=[FindingDraft(chunk_index=0, summary="s", relevance="r")],
            gaps=[],
        )
    )

    brief = research(spec, store, llm=fake, score_threshold=0.5)

    assert store.threshold_kwargs  # gate consulted
    assert store.mmr_kwargs  # and retrieval proceeded via MMR
    assert [f.id for f in brief.findings] == ["rf-0"]


def test_seed_retrieval_runs_before_the_tool_loop(spec):
    # Even if the LLM never issues a search, the deterministic seed
    # retrieval (RunnableParallel: context=retriever, spec=passthrough)
    # already gathered chunks for the brief.
    store = SpyVectorStore()
    fake = FakeStructuredLLM(
        BriefDraft(
            overview="seeded",
            findings=[FindingDraft(chunk_index=0, summary="s", relevance="r")],
            gaps=[],
        ),
        search_queries=(),  # LLM contributes no queries at all
    )

    brief = research(spec, store, llm=fake)

    assert store.mmr_kwargs == [MMR_SEARCH_KWARGS]  # exactly the seed query
    assert [f.id for f in brief.findings] == ["rf-0"]
    assert brief.overview == "seeded"


def capture_details(monkeypatch) -> list[dict]:
    """Route streaming.emit_detail chunks into a list (normally they only
    exist inside a LangGraph custom stream)."""
    events: list[dict] = []
    monkeypatch.setattr(
        "coolant_copilot.streaming.get_stream_writer", lambda: events.append
    )
    return events


def test_research_emits_source_retrieved_detail_per_gathered_chunk(spec, monkeypatch):
    events = capture_details(monkeypatch)
    store = SpyVectorStore()  # one doc; relevance defaults to 1.0
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    research(spec, store, llm=fake)

    details = [e for e in events if e["detail_type"] == "source_retrieved"]
    # The single doc is gathered once (dedup across seed + tool queries).
    assert len(details) == 1
    assert details[0]["node_name"] == "research"
    assert details[0]["payload"] == {
        "source_document": "esters.txt",
        "similarity_score": 1.0,
    }


def test_research_emits_one_muted_detail_when_threshold_filters_everything(spec, monkeypatch):
    events = capture_details(monkeypatch)
    store = SpyVectorStore(relevance=[])  # content exists, nothing clears the gate
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    research(spec, store, llm=fake, score_threshold=0.5)

    details = [e for e in events if e["detail_type"] == "source_retrieved"]
    assert details == [
        {
            "node_name": "research",
            "detail_type": "source_retrieved",
            "payload": {"no_sources_above_threshold": True},
        }
    ]


def test_source_detail_score_is_none_when_store_has_no_score_support(spec, monkeypatch):
    events = capture_details(monkeypatch)

    class NoScoreStore(SpyVectorStore):
        def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
            raise NotImplementedError

    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    research(spec, NoScoreStore(), llm=fake)

    details = [e for e in events if e["detail_type"] == "source_retrieved"]
    assert len(details) == 1
    assert details[0]["payload"]["similarity_score"] is None


def test_research_empty_store_skips_llm(spec, tmp_path):
    embeddings = DeterministicFakeEmbedding(size=64)
    empty_store = get_vectorstore(embeddings, persist_dir=str(tmp_path / "empty_db"))
    fake = FakeStructuredLLM(BriefDraft(overview="x", findings=[], gaps=[]))

    brief = research(spec, empty_store, llm=fake)

    assert fake.calls == 0
    assert brief.findings == []
    assert "No relevant sources" in brief.overview
