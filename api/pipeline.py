"""Build the production LangGraph exactly as main.py does — no pipeline changes.

The compiled graph is stateless (state is passed per-invoke), so one instance
is built lazily on first use and reused across runs.
"""

from __future__ import annotations

from dotenv import load_dotenv

_GRAPH = None


def build_production_graph():
    """Construct the real graph (Chroma vectorstore + extracted profiles).

    Requires OPENAI_API_KEY and a populated ./chroma_db, same as `python
    main.py`. Imported lazily so the API module imports without a key or store.
    """
    load_dotenv()

    from langchain_openai import OpenAIEmbeddings

    from coolant_copilot.extraction import load_profiles
    from coolant_copilot.graph import build_graph
    from coolant_copilot.ingestion import (
        DEFAULT_COLLECTION,
        DEFAULT_PERSIST_DIR,
        get_vectorstore,
    )

    vectorstore = get_vectorstore(
        # Must match scripts/ingest.py, like main.py.
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_dir=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION,
    )
    profiles = load_profiles("data/extracted")
    return build_graph(vectorstore, reference_profiles=profiles)


def get_graph():
    """FastAPI dependency: the shared compiled graph (built once, cached).

    Overridden in tests via ``app.dependency_overrides`` with a fake-wired
    graph, so the real vectorstore/LLMs are never touched under test.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_production_graph()
    return _GRAPH
