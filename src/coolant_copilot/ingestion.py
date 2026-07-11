"""Chunk and embed source documents into a persistent Chroma collection.

Idempotency: every chunk gets a deterministic id derived from its source,
page, and content hash. Chunks whose ids already exist in the collection are
skipped entirely, so re-running never duplicates chunks or re-embeds
unchanged content. Edited files produce new hashes and are ingested as new
chunks.
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from coolant_copilot.injection_audit import scan_documents

DEFAULT_COLLECTION = "coolant_sources"
DEFAULT_PERSIST_DIR = "./chroma_db"
# Chunk sizes are measured in *tokens* of the embedding model's own tokenizer
# (via tiktoken), not characters, so a chunk lines up with what the embedder
# actually consumes. Must stay well under the model's 8191-token input limit.
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 75  # tokens

TEXT_SUFFIXES = {".txt", ".md"}


def load_documents(source_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        # Paths in metadata are relative to the source dir so chunk ids
        # survive moves of the repo itself.
        source = str(path.relative_to(source_dir))
        if path.suffix.lower() == ".pdf":
            for page_num, page in enumerate(PdfReader(path).pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(
                        Document(text, metadata={"source": source, "page": page_num})
                    )
        elif path.suffix.lower() in TEXT_SUFFIXES:
            docs.append(Document(path.read_text(), metadata={"source": source}))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    # from_tiktoken_encoder measures chunk_size/overlap in tokens of the named
    # model's tokenizer, so chunk boundaries match the embedder's token counts.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def chunk_id(chunk: Document) -> str:
    key = (
        f"{chunk.metadata.get('source', '')}"
        f"|{chunk.metadata.get('page', '')}"
        f"|{chunk.page_content}"
    )
    return hashlib.sha256(key.encode()).hexdigest()


def get_vectorstore(
    embeddings: Embeddings,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest(
    source_dir: str | Path,
    embeddings: Embeddings,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, int]:
    """Ingest all supported files under source_dir. Returns chunk counts."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    chunks = chunk_documents(load_documents(source_dir))

    # Deduplicate within the batch: identical content on the same source/page
    # (e.g. repeated boilerplate) hashes to the same id.
    by_id: dict[str, Document] = {chunk_id(c): c for c in chunks}

    vectorstore = get_vectorstore(embeddings, persist_dir, collection_name)
    existing = set(vectorstore.get(ids=list(by_id))["ids"])
    new = {cid: c for cid, c in by_id.items() if cid not in existing}

    if new:
        # Visibility only: suspicious content is logged, never rejected.
        scan_documents(new.values())
        vectorstore.add_documents(list(new.values()), ids=list(new))

    return {"total": len(by_id), "added": len(new), "skipped": len(existing)}
