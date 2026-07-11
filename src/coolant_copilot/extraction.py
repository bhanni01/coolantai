"""Structured extraction of fluid product profiles from source documents.

One LLM call per source document decides whether the document describes a
specific fluid product and, if so, extracts an ExtractedFluidProfile. Pure
regulatory/background documents are skipped. Decisions are cached on disk
(profile JSONs plus a _skipped.json registry), so re-running ingestion never
repeats an LLM call for an already-processed document.
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from coolant_copilot.prompting import DATA_NOT_COMMANDS, wrap_reference_material
from coolant_copilot.schemas.extraction import (
    BaseChemistry,
    ExtractedFluidProfile,
    ExtractedProperty,
)
"""using gpt 4 o mini here just for price perspective, for a better result more expensive model can be used """
EXTRACTION_MODEL = "gpt-4o-mini"  
MAX_DOC_CHARS = 24_000
SKIP_REGISTRY = "_skipped.json"

SYSTEM_PROMPT = """\
You extract fluid product data from documents about data center coolants.

First decide: does this document describe one specific fluid product with real
property values (a product datasheet, SDS, or a comparison paper reporting
measured values)? Regulatory guides, standards documents, and general
background reading are NOT fluid products — set is_fluid_product to false and
leave profile empty.

If it is a fluid product, extract the profile. Rules:
- Only report property values the document actually states; convert units to
  the requested canonical unit when the conversion is trivial and lower the
  confidence accordingly.
- If the document compares several fluids, extract the single fluid the
  document is primarily about; if there is no primary fluid, pick the one
  with the most complete data.
- cas_number only if explicitly stated. pfas_free_claim reflects the
  document's own claim: true/false only if stated, otherwise null."""

SYSTEM_PROMPT += "\n" + DATA_NOT_COMMANDS


class ProfileDraft(BaseModel):
    """LLM-facing profile; source_document is stamped by code."""

    fluid_name: str
    manufacturer: str | None = None
    base_chemistry: BaseChemistry
    pfas_free_claim: bool | None = None
    cas_number: str | None = None
    properties: list[ExtractedProperty]


class ExtractionDecision(BaseModel):
    is_fluid_product: bool = Field(
        description="True only for datasheets/SDS/comparison data with real values for a specific fluid."
    )
    profile: ProfileDraft | None = Field(
        default=None, description="The extracted profile; null when is_fluid_product is false."
    )


def slugify(source: str) -> str:
    stem = Path(source).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")


def extract_profile(
    source: str, text: str, llm: BaseChatModel
) -> ExtractedFluidProfile | None:
    """One structured call; None when the document is not a fluid product."""
    decision: ExtractionDecision = llm.with_structured_output(ExtractionDecision).invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", f"DOCUMENT: {source}\n\n{wrap_reference_material(text[:MAX_DOC_CHARS])}"),
        ]
    )
    if not decision.is_fluid_product or decision.profile is None:
        return None
    return ExtractedFluidProfile(
        source_document=source, **decision.profile.model_dump()
    )


def run_extraction(
    docs: list[Document],
    out_dir: Path,
    llm: BaseChatModel | None = None,
) -> dict:
    """Extract profiles for every distinct source among docs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = out_dir / SKIP_REGISTRY
    skipped_registry: set[str] = (
        set(json.loads(registry_path.read_text())) if registry_path.exists() else set()
    )

    texts: dict[str, list[str]] = {}
    for doc in docs:
        texts.setdefault(doc.metadata.get("source", "unknown"), []).append(doc.page_content)

    summary: dict = {"extracted": {}, "skipped": [], "cached": []}
    for source, pages in texts.items():
        profile_path = out_dir / f"{slugify(source)}.json"
        if profile_path.exists() or source in skipped_registry:
            summary["cached"].append(source)
            continue

        llm = llm or ChatOpenAI(model=EXTRACTION_MODEL, temperature=0)
        profile = extract_profile(source, "\n\n".join(pages), llm)
        if profile is None:
            skipped_registry.add(source)
            summary["skipped"].append(source)
        else:
            profile_path.write_text(profile.model_dump_json(indent=2))
            summary["extracted"][source] = len(profile.properties)

    registry_path.write_text(json.dumps(sorted(skipped_registry), indent=2))
    return summary


def load_profiles(extracted_dir: str | Path) -> list[ExtractedFluidProfile]:
    """Load all extracted profiles; empty list when nothing has been extracted."""
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.is_dir():
        return []
    return [
        ExtractedFluidProfile.model_validate_json(path.read_text())
        for path in sorted(extracted_dir.glob("*.json"))
        if path.name != SKIP_REGISTRY
    ]
