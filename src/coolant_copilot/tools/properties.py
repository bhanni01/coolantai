"""Deterministic property estimation for candidate formulations.

Estimates come from a small built-in table of literature values. When the
table has no data for a component, the estimate is omitted rather than
invented: a missing PropertyEstimate is the honest signal the critic reads
as a data gap.
"""

from coolant_copilot.state import Candidate, PropertyEstimate, PropertyName, PropertyTarget

# Blend properties that are reasonably approximated by a mass-weighted average.
MIXING_RULE_PROPERTIES = {
    PropertyName.THERMAL_CONDUCTIVITY,
    PropertyName.SPECIFIC_HEAT,
    PropertyName.DENSITY,
    PropertyName.GWP,
    PropertyName.COST,
}

# Properties dominated by the base fluid; a linear mixing rule would be
# physically dishonest for these, so they are read off the base fluid alone.
BASE_FLUID_PROPERTIES = {
    PropertyName.FLASH_POINT,
    PropertyName.POUR_POINT,
    PropertyName.KINEMATIC_VISCOSITY,
}

# Minimum mass fraction that must have table data for a mixing-rule estimate;
# smaller unknown fractions (trace additives) are renormalized out.
MIN_KNOWN_MASS = 0.9

UNITS: dict[PropertyName, str] = {
    PropertyName.THERMAL_CONDUCTIVITY: "W/m·K",
    PropertyName.KINEMATIC_VISCOSITY: "cSt",
    PropertyName.SPECIFIC_HEAT: "J/kg·K",
    PropertyName.DENSITY: "kg/m³",
    PropertyName.DIELECTRIC_STRENGTH: "kV",
    PropertyName.FLASH_POINT: "°C",
    PropertyName.POUR_POINT: "°C",
    PropertyName.GWP: "kg CO₂-eq/kg",
    PropertyName.COST: "USD/L",
}

# Literature-order-of-magnitude values keyed by lowercase component name.
# Kinematic viscosity at 40 °C. Additives carry only the properties we have
# data for; dielectric strength has no honest blend model, so it is absent
# and always surfaces as a gap for the lab to measure.
COMPONENT_PROPERTIES: dict[str, dict[PropertyName, float]] = {
    "pentaerythritol tetraoleate": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.15,
        PropertyName.SPECIFIC_HEAT: 1970.0,
        PropertyName.DENSITY: 915.0,
        PropertyName.KINEMATIC_VISCOSITY: 62.0,
        PropertyName.FLASH_POINT: 300.0,
        PropertyName.POUR_POINT: -27.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 8.0,
    },
    "trimethylolpropane trioleate": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.15,
        PropertyName.SPECIFIC_HEAT: 1960.0,
        PropertyName.DENSITY: 915.0,
        PropertyName.KINEMATIC_VISCOSITY: 48.0,
        PropertyName.FLASH_POINT: 310.0,
        PropertyName.POUR_POINT: -45.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 7.0,
    },
    "polyalphaolefin pao-2": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.137,
        PropertyName.SPECIFIC_HEAT: 2180.0,
        PropertyName.DENSITY: 798.0,
        PropertyName.KINEMATIC_VISCOSITY: 5.0,
        PropertyName.FLASH_POINT: 165.0,
        PropertyName.POUR_POINT: -66.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 5.0,
    },
    "polyalphaolefin pao-6": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.143,
        PropertyName.SPECIFIC_HEAT: 2150.0,
        PropertyName.DENSITY: 827.0,
        PropertyName.KINEMATIC_VISCOSITY: 31.0,
        PropertyName.FLASH_POINT: 235.0,
        PropertyName.POUR_POINT: -57.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 5.5,
    },
    "mineral oil": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.13,
        PropertyName.SPECIFIC_HEAT: 1900.0,
        PropertyName.DENSITY: 850.0,
        PropertyName.KINEMATIC_VISCOSITY: 20.0,
        PropertyName.FLASH_POINT: 210.0,
        PropertyName.POUR_POINT: -18.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 2.0,
    },
    "propylene glycol": {
        PropertyName.THERMAL_CONDUCTIVITY: 0.20,
        PropertyName.SPECIFIC_HEAT: 2480.0,
        PropertyName.DENSITY: 1036.0,
        PropertyName.KINEMATIC_VISCOSITY: 18.0,
        PropertyName.FLASH_POINT: 104.0,
        PropertyName.POUR_POINT: -59.0,
        PropertyName.GWP: 3.0,
        PropertyName.COST: 2.5,
    },
    "bht": {
        PropertyName.DENSITY: 1048.0,
        PropertyName.COST: 6.0,
    },
    "alkylated diphenylamine": {
        PropertyName.DENSITY: 980.0,
        PropertyName.COST: 9.0,
    },
    "tolyltriazole": {
        PropertyName.DENSITY: 1180.0,
        PropertyName.COST: 15.0,
    },
}


def lookup_component(name: str) -> dict[PropertyName, float] | None:
    return COMPONENT_PROPERTIES.get(name.strip().lower())


def _meets_target(
    value: float, prop: PropertyName, targets: list[PropertyTarget]
) -> bool | None:
    for target in targets:
        if target.property is not prop:
            continue
        if target.min_value is not None and value < target.min_value:
            return False
        if target.max_value is not None and value > target.max_value:
            return False
        return True
    return None


def estimate_properties(
    candidate: Candidate, targets: list[PropertyTarget]
) -> list[PropertyEstimate]:
    """Estimate blend properties from table data; omit what the table can't support."""
    estimates: list[PropertyEstimate] = []

    for prop in sorted(MIXING_RULE_PROPERTIES, key=lambda p: p.value):
        known = [
            (c.weight_fraction, data[prop])
            for c in candidate.components
            if (data := lookup_component(c.name)) is not None and prop in data
        ]
        known_mass = sum(w for w, _ in known)
        if known_mass < MIN_KNOWN_MASS:
            continue
        value = sum(w * v for w, v in known) / known_mass
        estimates.append(
            PropertyEstimate(
                candidate_id=candidate.id,
                property=prop,
                value=value,
                unit=UNITS[prop],
                method="linear_mixing_rule",
                meets_target=_meets_target(value, prop, targets),
            )
        )

    base = next(c for c in candidate.components if c.role == "base_fluid")
    base_data = lookup_component(base.name)
    if base_data is not None:
        for prop in sorted(BASE_FLUID_PROPERTIES, key=lambda p: p.value):
            if prop not in base_data:
                continue
            value = base_data[prop]
            estimates.append(
                PropertyEstimate(
                    candidate_id=candidate.id,
                    property=prop,
                    value=value,
                    unit=UNITS[prop],
                    method="table_lookup",
                    meets_target=_meets_target(value, prop, targets),
                )
            )

    return estimates
