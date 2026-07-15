from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from coolant_copilot.ingestion import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_documents,
    get_vectorstore,
    ingest,
)

SAMPLE_TEXT = (
    "Propylene glycol aqueous mixtures are widely used heat transfer fluids. "
    "A 50/50 propylene glycol water blend has a thermal conductivity of "
    "approximately 0.36 W/m-K at 25 C and contains no fluorinated compounds, "
    "making it inherently PFAS-free."
)


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "sources"
    src.mkdir()
    (src / "propylene_glycol.txt").write_text(SAMPLE_TEXT)
    return src


@pytest.fixture
def persist_dir(tmp_path: Path) -> str:
    return str(tmp_path / "chroma_db")


def test_ingest_and_similarity_search(source_dir, persist_dir, embeddings):
    counts = ingest(source_dir, embeddings, persist_dir=persist_dir)
    assert counts["added"] > 0

    vectorstore = get_vectorstore(embeddings, persist_dir=persist_dir)
    results = vectorstore.similarity_search("propylene glycol thermal conductivity", k=1)

    assert len(results) == 1
    assert "propylene glycol" in results[0].page_content.lower()
    assert results[0].metadata["source"] == "propylene_glycol.txt"


def test_reingest_is_idempotent(source_dir, persist_dir, embeddings):
    first = ingest(source_dir, embeddings, persist_dir=persist_dir)
    second = ingest(source_dir, embeddings, persist_dir=persist_dir)

    assert first["added"] > 0
    assert second["added"] == 0
    assert second["skipped"] == first["added"]

    vectorstore = get_vectorstore(embeddings, persist_dir=persist_dir)
    assert len(vectorstore.get()["ids"]) == first["added"]


def test_new_file_adds_only_new_chunks(source_dir, persist_dir, embeddings):
    first = ingest(source_dir, embeddings, persist_dir=persist_dir)

    (source_dir / "esters.txt").write_text(
        "Synthetic ester base fluids offer high flash points for immersion cooling."
    )
    second = ingest(source_dir, embeddings, persist_dir=persist_dir)

    assert second["added"] > 0
    assert second["skipped"] == first["added"]


def test_chunking_overlaps_split_boundaries():
    # The splitter must carry a 15-20% token overlap so sentences at split
    # boundaries appear in both neighboring chunks.
    assert 0.15 <= CHUNK_OVERLAP / CHUNK_SIZE <= 0.20

    long_doc = Document(
        " ".join(f"Sentence number {i} about coolant chemistry." for i in range(400)),
        metadata={"source": "long.txt"},
    )
    chunks = chunk_documents([long_doc])

    assert len(chunks) > 1
    for left, right in zip(chunks, chunks[1:]):
        # The head of each chunk repeats text from the tail of the previous.
        assert right.page_content[:40] in left.page_content


def test_missing_source_dir_raises(tmp_path, embeddings):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nope", embeddings, persist_dir=str(tmp_path / "db"))
