"""Schemas for structured extraction of real fluid data from source documents.

An ExtractedFluidProfile is ground truth pulled from a product datasheet or
comparison paper — real measured values, as opposed to the heuristic
estimates in tools/properties.py. The evaluator uses these as a
sanity-check reference for candidates of similar base chemistry.
"""

from enum import Enum

from pydantic import BaseModel, Field

from coolant_copilot.state import PropertyName


class BaseChemistry(str, Enum):
    SYNTHETIC_ESTER = "synthetic_ester"
    POLYALPHAOLEFIN = "polyalphaolefin"
    MINERAL_OIL = "mineral_oil"
    GLYCOL_WATER = "glycol_water"
    SILICONE = "silicone"
    FLUOROCARBON = "fluorocarbon"
    OTHER = "other"


class ExtractedProperty(BaseModel):
    property: PropertyName
    value: float
    unit: str
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How directly the document states this value: 1.0 = explicit "
            "number in the text, lower when inferred or unit-converted."
        ),
    )


class ExtractedFluidProfile(BaseModel):
    fluid_name: str
    manufacturer: str | None = None
    base_chemistry: BaseChemistry
    pfas_free_claim: bool | None = Field(
        default=None,
        description="The document's own claim; None when it does not address PFAS content.",
    )
    cas_number: str | None = Field(
        default=None, description="CAS number if the document states one; never guessed."
    )
    source_document: str
    properties: list[ExtractedProperty]
