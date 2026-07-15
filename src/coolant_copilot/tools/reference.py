"""Cross-check heuristic property estimates against extracted real-fluid profiles.

Deterministic, like every evaluator tool. A candidate's base fluid is mapped
to a BaseChemistry class by name; estimates are then compared against the
closest reference value among extracted profiles of the same chemistry.
Within tolerance → "validated", outside → "conflict", no reference data →
untouched.
"""

from langchain_core.runnables import chain

from coolant_copilot.schemas.extraction import BaseChemistry, ExtractedFluidProfile
from coolant_copilot.state import Candidate, PropertyEstimate, PropertyName
from coolant_copilot.streaming import emit_detail

# First match wins, so more specific patterns come before generic ones.
CHEMISTRY_PATTERNS: list[tuple[BaseChemistry, tuple[str, ...]]] = [
    (BaseChemistry.FLUOROCARBON, ("fluoro", "ptfe", "fc-")),
    (BaseChemistry.POLYALPHAOLEFIN, ("polyalphaolefin", "pao")),
    (BaseChemistry.SYNTHETIC_ESTER, ("oleate", "adipate", "sebacate", "stearate", "ester")),
    (BaseChemistry.MINERAL_OIL, ("mineral oil", "paraffinic", "naphthenic")),
    (BaseChemistry.GLYCOL_WATER, ("glycol",)),
    (BaseChemistry.SILICONE, ("silicone", "siloxane")),
]

# Relative tolerance for ratio-scale properties; absolute tolerance for the
# temperature properties, where ratios are meaningless (°C scale, can be < 0).
REL_TOLERANCE = 0.25
ABS_TOLERANCE: dict[PropertyName, float] = {
    PropertyName.FLASH_POINT: 30.0,
    PropertyName.POUR_POINT: 15.0,
}

# References the extractor itself was unsure about are not trustworthy anchors.
MIN_REFERENCE_CONFIDENCE = 0.5


def classify_base_chemistry(component_name: str) -> BaseChemistry:
    name = component_name.strip().lower()
    for chemistry, patterns in CHEMISTRY_PATTERNS:
        if any(p in name for p in patterns):
            return chemistry
    return BaseChemistry.OTHER


def _within_tolerance(prop: PropertyName, estimate: float, reference: float) -> bool:
    if prop in ABS_TOLERANCE:
        return abs(estimate - reference) <= ABS_TOLERANCE[prop]
    if reference == 0:
        return estimate == 0
    return abs(estimate - reference) / abs(reference) <= REL_TOLERANCE


def get_reference_fluid_properties(
    base_chemistry: str, profiles: list[ExtractedFluidProfile]
) -> list[dict]:
    """JSON-serializable summary of extracted reference fluids of one chemistry.

    Tolerant of LLM-supplied chemistry strings: unknown values return [].
    """
    try:
        chemistry = BaseChemistry(base_chemistry.strip().lower())
    except ValueError:
        return []
    return [
        {
            "fluid_name": p.fluid_name,
            "manufacturer": p.manufacturer,
            "pfas_free_claim": p.pfas_free_claim,
            "source_document": p.source_document,
            "properties": [
                {
                    "property": rp.property.value,
                    "value": rp.value,
                    "unit": rp.unit,
                    "confidence": rp.confidence,
                }
                for rp in p.properties
            ],
        }
        for p in profiles
        if p.base_chemistry is chemistry
    ]


@chain
def cross_check_estimates(inputs: dict) -> list[PropertyEstimate]:
    """Annotate estimates with validated/conflict against same-chemistry references.

    @chain makes this a Runnable so each cross-check appears as a named step
    in LangSmith traces. Call it with
    .invoke({"candidate": ..., "estimates": ..., "profiles": ...}).
    """
    candidate: Candidate = inputs["candidate"]
    estimates: list[PropertyEstimate] = inputs["estimates"]
    profiles: list[ExtractedFluidProfile] = inputs["profiles"]

    base = next(c for c in candidate.components if c.role == "base_fluid")
    chemistry = classify_base_chemistry(base.name)
    if chemistry is BaseChemistry.OTHER:
        return estimates
    references = [p for p in profiles if p.base_chemistry is chemistry]
    if not references:
        return estimates

    checked: list[PropertyEstimate] = []
    for est in estimates:
        candidates_for_prop = [
            (profile, rp)
            for profile in references
            for rp in profile.properties
            if rp.property is est.property and rp.confidence >= MIN_REFERENCE_CONFIDENCE
        ]
        if not candidates_for_prop:
            checked.append(est)
            continue
        profile, ref = min(candidates_for_prop, key=lambda pair: abs(pair[1].value - est.value))
        if _within_tolerance(est.property, est.value, ref.value):
            status, verb = "validated", "consistent with"
        else:
            status, verb = "conflict", "conflicts with"
        emit_detail(
            "property_estimator",
            "cross_check",
            {
                "candidate_id": est.candidate_id,
                "property": est.property.value,
                "estimate": est.value,
                "unit": est.unit,
                "reference_value": ref.value,
                "reference_source": profile.source_document,
                "status": status,
            },
        )
        note = (
            f"Estimate {est.value:g} {est.unit} {verb} {ref.value:g} {ref.unit} "
            f"reported for {profile.fluid_name} ({chemistry.value}, "
            f"{profile.source_document})"
        )
        checked.append(est.model_copy(update={"reference_check": status, "reference_note": note}))
    return checked
