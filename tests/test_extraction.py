import json
from pathlib import Path

from conftest import FakeStructuredLLM
from langchain_core.documents import Document

from coolant_copilot.extraction import (
    SKIP_REGISTRY,
    ExtractionDecision,
    ProfileDraft,
    load_profiles,
    run_extraction,
    slugify,
)
from coolant_copilot.schemas.extraction import (
    BaseChemistry,
    ExtractedFluidProfile,
    ExtractedProperty,
)
from coolant_copilot.state import PropertyName


def datasheet_decision() -> ExtractionDecision:
    return ExtractionDecision(
        is_fluid_product=True,
        profile=ProfileDraft(
            fluid_name="Shell S5 X",
            manufacturer="Shell",
            base_chemistry=BaseChemistry.SYNTHETIC_ESTER,
            pfas_free_claim=True,
            cas_number=None,
            properties=[
                ExtractedProperty(
                    property=PropertyName.THERMAL_CONDUCTIVITY,
                    value=0.14,
                    unit="W/m·K",
                    confidence=1.0,
                ),
                ExtractedProperty(
                    property=PropertyName.FLASH_POINT, value=270, unit="°C", confidence=1.0
                ),
            ],
        ),
    )


def not_a_product() -> ExtractionDecision:
    return ExtractionDecision(is_fluid_product=False, profile=None)


def docs() -> list[Document]:
    return [
        Document("page one of the datasheet", metadata={"source": "shell-s5x-datasheet.pdf", "page": 0}),
        Document("page two of the datasheet", metadata={"source": "shell-s5x-datasheet.pdf", "page": 1}),
        Document("regulatory background text", metadata={"source": "pfas-2026-regulatory-guide.md"}),
    ]


def test_slugify():
    assert slugify("shell-s5x-datasheet.pdf") == "shell-s5x-datasheet"
    assert slugify("PFAS 2026 Guide (final).md") == "pfas-2026-guide-final"


def test_extracts_products_and_skips_background(tmp_path: Path):
    llm = FakeStructuredLLM([datasheet_decision(), not_a_product()])

    summary = run_extraction(docs(), tmp_path, llm=llm)

    assert summary["extracted"] == {"shell-s5x-datasheet.pdf": 2}
    assert summary["skipped"] == ["pfas-2026-regulatory-guide.md"]
    assert summary["cached"] == []
    # One call per document, both pages of the datasheet in one prompt.
    assert llm.calls == 2
    assert llm.schemas == [ExtractionDecision, ExtractionDecision]
    first_prompt = llm.message_log[0][-1][1]
    assert "page one" in first_prompt and "page two" in first_prompt

    profile = ExtractedFluidProfile.model_validate_json(
        (tmp_path / "shell-s5x-datasheet.json").read_text()
    )
    # source_document is stamped by code, not the LLM.
    assert profile.source_document == "shell-s5x-datasheet.pdf"
    assert profile.fluid_name == "Shell S5 X"
    assert json.loads((tmp_path / SKIP_REGISTRY).read_text()) == [
        "pfas-2026-regulatory-guide.md"
    ]


def test_rerun_is_cached_and_makes_no_llm_calls(tmp_path: Path):
    first_llm = FakeStructuredLLM([datasheet_decision(), not_a_product()])
    run_extraction(docs(), tmp_path, llm=first_llm)

    second_llm = FakeStructuredLLM([datasheet_decision()])
    summary = run_extraction(docs(), tmp_path, llm=second_llm)

    assert second_llm.calls == 0
    assert summary["extracted"] == {} and summary["skipped"] == []
    assert sorted(summary["cached"]) == [
        "pfas-2026-regulatory-guide.md",
        "shell-s5x-datasheet.pdf",
    ]


def test_load_profiles(tmp_path: Path):
    llm = FakeStructuredLLM([datasheet_decision(), not_a_product()])
    run_extraction(docs(), tmp_path, llm=llm)

    profiles = load_profiles(tmp_path)

    assert len(profiles) == 1
    assert profiles[0].base_chemistry is BaseChemistry.SYNTHETIC_ESTER
    assert load_profiles(tmp_path / "missing") == []
