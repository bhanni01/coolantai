"""Unit tests for the DatacenterProfile → TargetSpec/CriticWeights resolvers.

Pure deterministic mapping functions — known input/output pairs, no mocking.
"""

import pytest
from pydantic import ValidationError

from coolant_copilot.schemas.datacenter_profile import (
    DatacenterProfile,
    resolve_critic_weights,
    resolve_target_spec,
)
from coolant_copilot.state import PropertyName


def make_profile(**overrides) -> DatacenterProfile:
    defaults = dict(
        cooling_method="single_phase_immersion",
        rack_density="high_density",
        climate_zone="temperate",
        regulatory_region="us",
        optimization_priority="performance",
    )
    return DatacenterProfile(**{**defaults, **overrides})


def target_for(spec, prop: PropertyName):
    return next((t for t in spec.property_targets if t.property == prop), None)


class TestCoolingMethod:
    def test_immersion_requires_dielectric_with_strength_target(self):
        spec = resolve_target_spec(make_profile(cooling_method="single_phase_immersion"))
        assert spec.application == "single_phase_immersion"
        assert spec.requires_dielectric is True
        assert spec.corrosion_inhibition_required is False
        dielectric = target_for(spec, PropertyName.DIELECTRIC_STRENGTH)
        assert dielectric is not None
        assert dielectric.min_value == 35
        assert dielectric.priority == "must"

    @pytest.mark.parametrize("method", ["direct_to_chip", "rear_door_heat_exchanger"])
    def test_water_loop_methods_need_corrosion_inhibition_not_dielectric(self, method):
        spec = resolve_target_spec(make_profile(cooling_method=method))
        assert spec.application == "cold_plate"
        assert spec.requires_dielectric is False
        assert spec.corrosion_inhibition_required is True
        assert target_for(spec, PropertyName.DIELECTRIC_STRENGTH) is None


class TestRackDensity:
    @pytest.mark.parametrize(
        ("density", "tc_min"),
        [("standard", 0.060), ("high_density", 0.075), ("ultra_high_density", 0.090)],
    )
    def test_thermal_conductivity_minimum(self, density, tc_min):
        spec = resolve_target_spec(make_profile(rack_density=density))
        assert target_for(spec, PropertyName.THERMAL_CONDUCTIVITY).min_value == tc_min

    def test_ultra_high_density_raises_flash_point_by_15(self):
        base = resolve_target_spec(make_profile(rack_density="standard"))
        ultra = resolve_target_spec(make_profile(rack_density="ultra_high_density"))
        assert target_for(base, PropertyName.FLASH_POINT).min_value == 150
        assert target_for(ultra, PropertyName.FLASH_POINT).min_value == 165


class TestClimateZone:
    def test_temperate_pour_point(self):
        spec = resolve_target_spec(make_profile(climate_zone="temperate"))
        assert target_for(spec, PropertyName.POUR_POINT).max_value == -20

    def test_cold_pour_point(self):
        spec = resolve_target_spec(make_profile(climate_zone="cold"))
        assert target_for(spec, PropertyName.POUR_POINT).max_value == -40

    def test_hot_humid_raises_flash_point_and_sets_no_pour_point(self):
        spec = resolve_target_spec(make_profile(climate_zone="hot_humid"))
        assert target_for(spec, PropertyName.FLASH_POINT).min_value == 160
        assert target_for(spec, PropertyName.POUR_POINT) is None

    def test_flash_point_adders_stack(self):
        spec = resolve_target_spec(
            make_profile(rack_density="ultra_high_density", climate_zone="hot_humid")
        )
        assert target_for(spec, PropertyName.FLASH_POINT).min_value == 175


class TestRegulatoryRegion:
    @pytest.mark.parametrize(
        ("region", "expected"),
        [
            ("us", ["US"]),
            ("eu", ["EU"]),
            ("apac", ["JP", "CN", "KR"]),
            ("global", ["US", "EU", "UK", "JP", "CN", "KR"]),
        ],
    )
    def test_maps_to_compliance_regions(self, region, expected):
        spec = resolve_target_spec(make_profile(regulatory_region=region))
        assert spec.regulatory_regions == expected


class TestOptimizationPriority:
    @pytest.mark.parametrize("priority", ["performance", "cost", "compliance", "lifespan"])
    def test_chosen_axis_dominates_the_weights(self, priority):
        weights = resolve_critic_weights(make_profile(optimization_priority=priority))
        dumped = weights.model_dump()
        assert max(dumped, key=dumped.get) == priority

    def test_priority_never_touches_the_spec(self):
        specs = [
            resolve_target_spec(make_profile(optimization_priority=p))
            for p in ("performance", "cost", "compliance", "lifespan")
        ]
        assert all(s == specs[0] for s in specs)


def test_resolution_is_deterministic():
    assert resolve_target_spec(make_profile()) == resolve_target_spec(make_profile())


def test_profile_rejects_unknown_option_values_and_extra_fields():
    with pytest.raises(ValidationError):
        make_profile(cooling_method="two_phase_immersion")
    with pytest.raises(ValidationError):
        # A raw TargetSpec field must never sneak in through the public form.
        DatacenterProfile(**{**make_profile().model_dump(), "excluded_substances": []})
