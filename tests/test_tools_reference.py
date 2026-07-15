from conftest import make_candidate

from coolant_copilot.schemas.extraction import (
    BaseChemistry,
    ExtractedFluidProfile,
    ExtractedProperty,
)
from coolant_copilot.state import PropertyEstimate, PropertyName
from coolant_copilot.tools.reference import classify_base_chemistry, cross_check_estimates


def ester_profile(tc: float = 0.14, tc_confidence: float = 1.0) -> ExtractedFluidProfile:
    return ExtractedFluidProfile(
        fluid_name="Shell S5 X",
        manufacturer="Shell",
        base_chemistry=BaseChemistry.SYNTHETIC_ESTER,
        pfas_free_claim=True,
        source_document="shell-s5x-datasheet.pdf",
        properties=[
            ExtractedProperty(
                property=PropertyName.THERMAL_CONDUCTIVITY,
                value=tc,
                unit="W/m·K",
                confidence=tc_confidence,
            ),
            ExtractedProperty(
                property=PropertyName.FLASH_POINT, value=270, unit="°C", confidence=1.0
            ),
        ],
    )


def estimate(prop: PropertyName, value: float, unit: str) -> PropertyEstimate:
    return PropertyEstimate(
        candidate_id="cand-0-rev0", property=prop, value=value, unit=unit, method="test"
    )


class TestClassifyBaseChemistry:
    def test_known_chemistries(self):
        assert classify_base_chemistry("Pentaerythritol tetraoleate") is BaseChemistry.SYNTHETIC_ESTER
        assert classify_base_chemistry("Polyalphaolefin PAO-6") is BaseChemistry.POLYALPHAOLEFIN
        assert classify_base_chemistry("Mineral Oil") is BaseChemistry.MINERAL_OIL
        assert classify_base_chemistry("propylene glycol") is BaseChemistry.GLYCOL_WATER
        assert classify_base_chemistry("Perfluoropolyether") is BaseChemistry.FLUOROCARBON

    def test_unknown_is_other(self):
        assert classify_base_chemistry("Deionized water") is BaseChemistry.OTHER


class TestCrossCheck:
    def test_estimate_within_tolerance_is_validated(self):
        # Candidate base fluid is an oleate ester → matches the ester profile.
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.THERMAL_CONDUCTIVITY, 0.15, "W/m·K")], "profiles": [ester_profile(tc=0.14)]}
        )
        assert checked[0].reference_check == "validated"
        assert "Shell S5 X" in checked[0].reference_note
        assert "shell-s5x-datasheet.pdf" in checked[0].reference_note

    def test_estimate_far_off_is_conflict(self):
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.THERMAL_CONDUCTIVITY, 0.30, "W/m·K")], "profiles": [ester_profile(tc=0.14)]}
        )
        assert checked[0].reference_check == "conflict"

    def test_temperature_property_uses_absolute_tolerance(self):
        # 250 vs 270 °C is only within 30 °C absolute, not 25 % — must validate.
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.FLASH_POINT, 250, "°C")], "profiles": [ester_profile()]}
        )
        assert checked[0].reference_check == "validated"

    def test_no_matching_chemistry_leaves_estimates_untouched(self):
        pao_profile = ester_profile().model_copy(
            update={"base_chemistry": BaseChemistry.POLYALPHAOLEFIN}
        )
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.THERMAL_CONDUCTIVITY, 0.15, "W/m·K")], "profiles": [pao_profile]}
        )
        assert checked[0].reference_check is None

    def test_low_confidence_reference_is_ignored(self):
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.THERMAL_CONDUCTIVITY, 0.15, "W/m·K")], "profiles": [ester_profile(tc=0.14, tc_confidence=0.3)]}
        )
        assert checked[0].reference_check is None

    def test_closest_reference_wins(self):
        near = ester_profile(tc=0.15)
        far = ester_profile(tc=0.30).model_copy(update={"fluid_name": "Far Fluid"})
        checked = cross_check_estimates.invoke(
            {"candidate": make_candidate(), "estimates": [estimate(PropertyName.THERMAL_CONDUCTIVITY, 0.15, "W/m·K")], "profiles": [far, near]}
        )
        assert checked[0].reference_check == "validated"
        assert "Shell S5 X" in checked[0].reference_note
