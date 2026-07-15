"""Closed datacenter-profile input and its deterministic TargetSpec mapping.

The public form submits a DatacenterProfile — five closed dropdown selections,
no free text — and the server expands it into the full TargetSpec via
resolve_target_spec. A raw TargetSpec is never accepted from the form, so
every spec value the pipeline sees is decided here, deterministically.

optimization_priority deliberately does not touch the spec: it resolves to a
CriticWeights object (resolve_critic_weights) that the critic node applies
when ranking candidates.
"""

from typing import Literal, get_args

from langchain_core.runnables import chain
from pydantic import BaseModel, ConfigDict

from coolant_copilot.state import (
    CriticWeights,
    PropertyName,
    PropertyTarget,
    Region,
    TargetSpec,
)

CoolingMethod = Literal["single_phase_immersion", "direct_to_chip", "rear_door_heat_exchanger"]
RackDensity = Literal["standard", "high_density", "ultra_high_density"]
ClimateZone = Literal["temperate", "hot_humid", "cold"]
RegulatoryRegion = Literal["us", "eu", "apac", "global"]
OptimizationPriority = Literal["performance", "cost", "compliance", "lifespan"]


class DatacenterProfile(BaseModel):
    """What the public form submits: five closed selections, nothing free-text."""

    model_config = ConfigDict(extra="forbid")

    cooling_method: CoolingMethod
    rack_density: RackDensity
    climate_zone: ClimateZone
    regulatory_region: RegulatoryRegion
    optimization_priority: OptimizationPriority


# rack_density → minimum thermal conductivity (W/m·K).
THERMAL_CONDUCTIVITY_MIN: dict[RackDensity, float] = {
    "standard": 0.060,
    "high_density": 0.075,
    "ultra_high_density": 0.090,
}

BASE_FLASH_POINT_MIN = 150.0  # °C, before the density/climate adders

# climate_zone → maximum pour point (°C); hot_humid sets no pour-point target.
POUR_POINT_MAX: dict[ClimateZone, float] = {"temperate": -20.0, "cold": -40.0}

ComplianceRuleset = Literal["tsca", "reach", "regional_baseline", "strictest_combined"]

# regulatory_region → (ruleset, regions the compliance_checker runs). The
# checker selects PFAS rules per region: US → TSCA-8a7, EU → EU-PFAS-2024,
# and any other region → its PFAS-{region} baseline rule.
COMPLIANCE_RULESETS: dict[RegulatoryRegion, tuple[ComplianceRuleset, list[Region]]] = {
    "us": ("tsca", ["US"]),
    "eu": ("reach", ["EU"]),
    "apac": ("regional_baseline", ["JP", "CN", "KR"]),
    "global": ("strictest_combined", list(get_args(Region))),
}

# optimization_priority → critic ranking weights (the chosen axis dominates).
PRIORITY_WEIGHTS: dict[OptimizationPriority, CriticWeights] = {
    "performance": CriticWeights(performance=0.55, cost=0.15, compliance=0.15, lifespan=0.15),
    "cost": CriticWeights(performance=0.15, cost=0.55, compliance=0.15, lifespan=0.15),
    "compliance": CriticWeights(performance=0.15, cost=0.15, compliance=0.55, lifespan=0.15),
    "lifespan": CriticWeights(performance=0.15, cost=0.15, compliance=0.15, lifespan=0.55),
}


@chain
def resolve_target_spec(profile: DatacenterProfile) -> TargetSpec:
    """Deterministically expand a profile into the full TargetSpec (pure).

    @chain makes this a Runnable — call it with .invoke(profile) — so the
    resolution appears as a named step in LangSmith traces.
    """
    targets: list[PropertyTarget] = [
        PropertyTarget(
            property=PropertyName.THERMAL_CONDUCTIVITY,
            min_value=THERMAL_CONDUCTIVITY_MIN[profile.rack_density],
            unit="W/m·K",
            priority="must",
        )
    ]

    flash_point_min = BASE_FLASH_POINT_MIN
    if profile.rack_density == "ultra_high_density":
        flash_point_min += 15
    if profile.climate_zone == "hot_humid":
        flash_point_min += 10
    targets.append(
        PropertyTarget(
            property=PropertyName.FLASH_POINT,
            min_value=flash_point_min,
            unit="°C",
            priority="must",
        )
    )

    immersion = profile.cooling_method == "single_phase_immersion"
    if immersion:
        targets.append(
            PropertyTarget(
                property=PropertyName.DIELECTRIC_STRENGTH,
                min_value=35,
                unit="kV",
                priority="must",
            )
        )

    pour_point_max = POUR_POINT_MAX.get(profile.climate_zone)
    if pour_point_max is not None:
        targets.append(
            PropertyTarget(
                property=PropertyName.POUR_POINT,
                max_value=pour_point_max,
                unit="°C",
                priority="must",
            )
        )

    ruleset, regions = COMPLIANCE_RULESETS[profile.regulatory_region]
    return TargetSpec(
        name=f"DC-{profile.cooling_method}-{profile.rack_density}",
        application="single_phase_immersion" if immersion else "cold_plate",
        description=(
            f"PFAS-free coolant for {profile.cooling_method.replace('_', ' ')} cooling of "
            f"{profile.rack_density.replace('_', ' ')} racks in a "
            f"{profile.climate_zone.replace('_', ' ')} climate; "
            f"compliance ruleset: {ruleset}."
        ),
        property_targets=targets,
        regulatory_regions=regions,
        pfas_free_required=True,
        requires_dielectric=immersion,
        corrosion_inhibition_required=not immersion,
    )


def resolve_critic_weights(profile: DatacenterProfile) -> CriticWeights:
    """optimization_priority → critic ranking weights; never touches the spec."""
    return PRIORITY_WEIGHTS[profile.optimization_priority].model_copy()
