import pytest

from coolant_copilot.state import Candidate, Component, PropertyName, PropertyTarget
from coolant_copilot.tools.properties import (
    COMPONENT_PROPERTIES,
    MIN_KNOWN_MASS,
    estimate_properties,
    lookup_component,
)


def make_blend(components: list[Component], candidate_id: str = "cand-0-rev0") -> Candidate:
    return Candidate(
        id=candidate_id,
        name="test blend",
        components=components,
        rationale="test",
    )


def targets(*specs: tuple[PropertyName, float | None, float | None]) -> list[PropertyTarget]:
    return [
        PropertyTarget(property=prop, min_value=lo, max_value=hi, unit="x", priority="must")
        for prop, lo, hi in specs
    ]


def test_lookup_is_case_insensitive():
    assert lookup_component("Pentaerythritol Tetraoleate") is not None
    assert lookup_component("unobtainium") is None


def test_linear_mixing_rule_exact_weighted_average():
    cand = make_blend(
        [
            Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=0.6),
            Component(name="Mineral oil", role="other", weight_fraction=0.4),
        ]
    )
    estimates = {e.property: e for e in estimate_properties(cand, [])}

    pe = COMPONENT_PROPERTIES["pentaerythritol tetraoleate"]
    mo = COMPONENT_PROPERTIES["mineral oil"]
    density = estimates[PropertyName.DENSITY]
    assert density.value == pytest.approx(
        0.6 * pe[PropertyName.DENSITY] + 0.4 * mo[PropertyName.DENSITY]
    )
    assert density.method == "linear_mixing_rule"
    assert density.candidate_id == "cand-0-rev0"


def test_base_fluid_properties_use_table_lookup_from_base_only():
    cand = make_blend(
        [
            Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=0.6),
            Component(name="Mineral oil", role="other", weight_fraction=0.4),
        ]
    )
    estimates = {e.property: e for e in estimate_properties(cand, [])}

    flash = estimates[PropertyName.FLASH_POINT]
    pe = COMPONENT_PROPERTIES["pentaerythritol tetraoleate"]
    assert flash.value == pe[PropertyName.FLASH_POINT]  # not blended with mineral oil
    assert flash.method == "table_lookup"


def test_minor_unknown_component_is_renormalized_out():
    # BHT has no thermal conductivity data; known mass 0.95 >= MIN_KNOWN_MASS.
    assert MIN_KNOWN_MASS <= 0.95
    cand = make_blend(
        [
            Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=0.95),
            Component(name="BHT", role="antioxidant", weight_fraction=0.05),
        ]
    )
    estimates = {e.property: e for e in estimate_properties(cand, [])}

    pe = COMPONENT_PROPERTIES["pentaerythritol tetraoleate"]
    assert estimates[PropertyName.THERMAL_CONDUCTIVITY].value == pytest.approx(
        pe[PropertyName.THERMAL_CONDUCTIVITY]
    )


def test_major_unknown_component_omits_estimates_instead_of_inventing():
    cand = make_blend(
        [Component(name="Unobtainium ester", role="base_fluid", weight_fraction=1.0)]
    )
    assert estimate_properties(cand, []) == []


def test_unknown_base_fluid_omits_base_fluid_properties():
    # 80% unknown base fluid: no flash point (base lookup fails) and no mixed
    # properties (known mass 0.2 < MIN_KNOWN_MASS).
    cand = make_blend(
        [
            Component(name="Unobtainium ester", role="base_fluid", weight_fraction=0.8),
            Component(name="Mineral oil", role="other", weight_fraction=0.2),
        ]
    )
    assert estimate_properties(cand, []) == []


def test_meets_target_true_false_and_none():
    cand = make_blend(
        [Component(name="Pentaerythritol tetraoleate", role="base_fluid", weight_fraction=1.0)]
    )
    spec_targets = targets(
        (PropertyName.THERMAL_CONDUCTIVITY, 0.13, None),  # PE at 0.15 -> True
        (PropertyName.DENSITY, None, 900.0),  # PE at 915 -> False
    )
    estimates = {e.property: e for e in estimate_properties(cand, spec_targets)}

    assert estimates[PropertyName.THERMAL_CONDUCTIVITY].meets_target is True
    assert estimates[PropertyName.DENSITY].meets_target is False
    assert estimates[PropertyName.FLASH_POINT].meets_target is None  # no target set
