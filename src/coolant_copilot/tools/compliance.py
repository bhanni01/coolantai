"""Deterministic PFAS / regulatory / spec-exclusion / material checks.

Screening only: a CAS seed set plus name patterns. Anything that cannot be
verified (no CAS number) gets an explicit needs_review flag rather than a
silent pass.
"""

from coolant_copilot.schemas.extraction import BaseChemistry
from coolant_copilot.state import Candidate, ComplianceFlag, Component, TargetSpec
from coolant_copilot.tools.reference import classify_base_chemistry

# Seed set of well-known PFAS CAS numbers (PFOA, PFOS, PFBS, PFHxA,
# methoxy-nonafluorobutane, PTFE). Screening aid, not an exhaustive list.
KNOWN_PFAS_CAS: set[str] = {
    "335-67-1",
    "1763-23-1",
    "375-73-5",
    "307-24-4",
    "163702-07-6",
    "9002-84-0",
}

PFAS_NAME_PATTERNS: tuple[str, ...] = ("fluoro", "fluorinated", "ptfe", "fc-")

# Region → PFAS rule identifier used in ComplianceFlag.regulation.
PFAS_REGULATIONS: dict[str, str] = {
    "EU": "EU-PFAS-2024",
    "US": "TSCA-8a7",
}

# Elastomer seal compatibility per base chemistry. Coarse literature
# heuristics: esters and hydrocarbons swell EPDM; NBR tolerates hydrocarbons
# but only conditionally tolerates polar esters; silicone fluids attack
# silicone rubber. Values: "compatible" | "conditional" | "incompatible".
DEFAULT_SEAL_MATERIALS: tuple[str, ...] = ("EPDM", "FKM", "NBR")
SEAL_COMPATIBILITY: dict[BaseChemistry, dict[str, str]] = {
    BaseChemistry.SYNTHETIC_ESTER: {"EPDM": "incompatible", "FKM": "compatible", "NBR": "conditional"},
    BaseChemistry.POLYALPHAOLEFIN: {"EPDM": "incompatible", "FKM": "compatible", "NBR": "compatible"},
    BaseChemistry.MINERAL_OIL: {"EPDM": "incompatible", "FKM": "compatible", "NBR": "compatible"},
    BaseChemistry.GLYCOL_WATER: {"EPDM": "compatible", "FKM": "compatible", "NBR": "conditional"},
    BaseChemistry.SILICONE: {"EPDM": "conditional", "FKM": "compatible", "NBR": "incompatible"},
}
_COMPAT_STATUS = {"compatible": "pass", "conditional": "needs_review", "incompatible": "fail"}


def is_pfas_suspect(component: Component) -> bool:
    if component.cas_number in KNOWN_PFAS_CAS:
        return True
    name = component.name.lower()
    return any(pattern in name for pattern in PFAS_NAME_PATTERNS)


def check_pfas_definition(candidate: Candidate, spec: TargetSpec) -> list[ComplianceFlag]:
    """Screen the composition against PFAS definitions per regulatory region."""
    if not spec.pfas_free_required:
        return []
    flags: list[ComplianceFlag] = []
    for region in spec.regulatory_regions:
        regulation = PFAS_REGULATIONS.get(region, f"PFAS-{region}")
        suspects = [c for c in candidate.components if is_pfas_suspect(c)]
        unverifiable = [
            c
            for c in candidate.components
            if c.cas_number is None and not is_pfas_suspect(c)
        ]
        for c in suspects:
            flags.append(
                ComplianceFlag(
                    candidate_id=candidate.id,
                    regulation=regulation,
                    status="fail",
                    component_name=c.name,
                    detail=f"{c.name} matches known PFAS CAS numbers or naming patterns.",
                )
            )
        for c in unverifiable:
            flags.append(
                ComplianceFlag(
                    candidate_id=candidate.id,
                    regulation=regulation,
                    status="needs_review",
                    component_name=c.name,
                    detail=f"{c.name} has no CAS number; cannot verify against CAS-based PFAS lists.",
                )
            )
        if not suspects and not unverifiable:
            flags.append(
                ComplianceFlag(
                    candidate_id=candidate.id,
                    regulation=regulation,
                    status="pass",
                    detail="No component matches known PFAS CAS numbers or naming patterns.",
                )
            )
    return flags


def check_excluded_substances(candidate: Candidate, spec: TargetSpec) -> list[ComplianceFlag]:
    """Flag components on the spec's excluded-substances list (name or CAS)."""
    if not spec.excluded_substances:
        return []
    excluded = {s.strip().lower() for s in spec.excluded_substances}
    flags: list[ComplianceFlag] = []
    for c in candidate.components:
        if c.name.strip().lower() in excluded or (
            c.cas_number is not None and c.cas_number in excluded
        ):
            flags.append(
                ComplianceFlag(
                    candidate_id=candidate.id,
                    regulation="SPEC-EXCLUDED",
                    status="fail",
                    component_name=c.name,
                    detail=f"{c.name} is on the spec's excluded-substances list.",
                )
            )
    return flags


def check_material_compatibility(
    candidate: Candidate, seal_materials: list[str] | None = None
) -> list[ComplianceFlag]:
    """Check the base fluid's chemistry against elastomer seal materials.

    One flag per seal material; unknown chemistries or seal materials get an
    explicit needs_review rather than a silent pass.
    """
    seals = [s.strip().upper() for s in (seal_materials or DEFAULT_SEAL_MATERIALS)]
    base = next(c for c in candidate.components if c.role == "base_fluid")
    chemistry = classify_base_chemistry(base.name)
    table = SEAL_COMPATIBILITY.get(chemistry)

    flags: list[ComplianceFlag] = []
    for seal in seals:
        rating = (table or {}).get(seal)
        if rating is None:
            flags.append(
                ComplianceFlag(
                    candidate_id=candidate.id,
                    regulation=f"MATERIAL-COMPAT-{seal}",
                    status="needs_review",
                    component_name=base.name,
                    detail=(
                        f"No compatibility data for {chemistry.value} base fluid "
                        f"with {seal} seals; requires lab verification."
                    ),
                )
            )
            continue
        flags.append(
            ComplianceFlag(
                candidate_id=candidate.id,
                regulation=f"MATERIAL-COMPAT-{seal}",
                status=_COMPAT_STATUS[rating],
                component_name=base.name,
                detail=(
                    f"{chemistry.value} base fluid is rated '{rating}' with "
                    f"{seal} elastomer seals."
                ),
            )
        )
    return flags


def check_compliance(candidate: Candidate, spec: TargetSpec) -> list[ComplianceFlag]:
    """PFAS screening plus spec exclusions (the classic evaluator bundle)."""
    return check_pfas_definition(candidate, spec) + check_excluded_substances(candidate, spec)
