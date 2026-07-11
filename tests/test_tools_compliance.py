from coolant_copilot.state import Candidate, Component, PropertyName, PropertyTarget, TargetSpec
from coolant_copilot.tools.compliance import check_compliance, is_pfas_suspect


def make_spec(**overrides) -> TargetSpec:
    defaults = dict(
        name="DC-Coolant-A",
        application="single_phase_immersion",
        description="PFAS-free immersion coolant.",
        property_targets=[
            PropertyTarget(
                property=PropertyName.THERMAL_CONDUCTIVITY,
                min_value=0.13,
                unit="W/m·K",
                priority="must",
            )
        ],
        regulatory_regions=["US", "EU"],
    )
    return TargetSpec(**{**defaults, **overrides})


def make_candidate(components: list[Component]) -> Candidate:
    return Candidate(id="cand-0-rev0", name="test", components=components, rationale="test")


CLEAN_ESTER = Component(
    name="Pentaerythritol tetraoleate",
    cas_number="19321-40-5",
    role="base_fluid",
    weight_fraction=1.0,
)


def test_pfas_cas_hit_fails_per_region():
    cand = make_candidate(
        [
            CLEAN_ESTER.model_copy(update={"weight_fraction": 0.9}),
            Component(
                name="Mystery additive",
                cas_number="1763-23-1",  # PFOS
                role="other",
                weight_fraction=0.1,
            ),
        ]
    )
    flags = check_compliance(cand, make_spec())

    fails = [f for f in flags if f.status == "fail"]
    assert {f.regulation for f in fails} == {"EU-PFAS-2024", "TSCA-8a7"}
    assert all(f.component_name == "Mystery additive" for f in fails)
    assert all(f.candidate_id == "cand-0-rev0" for f in flags)


def test_pfas_name_pattern_fails_without_cas():
    suspect = Component(name="Perfluoropolyether oil", role="base_fluid", weight_fraction=1.0)
    assert is_pfas_suspect(suspect)

    flags = check_compliance(make_candidate([suspect]), make_spec(regulatory_regions=["EU"]))
    fails = [f for f in flags if f.status == "fail"]
    assert len(fails) == 1
    assert fails[0].regulation == "EU-PFAS-2024"


def test_missing_cas_needs_review():
    cand = make_candidate(
        [
            CLEAN_ESTER.model_copy(update={"weight_fraction": 0.9}),
            Component(name="Proprietary inhibitor X", role="corrosion_inhibitor", weight_fraction=0.1),
        ]
    )
    flags = check_compliance(cand, make_spec(regulatory_regions=["US"]))

    reviews = [f for f in flags if f.status == "needs_review"]
    assert len(reviews) == 1
    assert reviews[0].component_name == "Proprietary inhibitor X"
    assert reviews[0].regulation == "TSCA-8a7"
    assert not [f for f in flags if f.status == "fail"]


def test_clean_candidate_gets_explicit_pass_per_region():
    flags = check_compliance(make_candidate([CLEAN_ESTER]), make_spec())

    assert {(f.regulation, f.status) for f in flags} == {
        ("EU-PFAS-2024", "pass"),
        ("TSCA-8a7", "pass"),
    }


def test_excluded_substance_by_name_and_cas():
    cand = make_candidate(
        [
            CLEAN_ESTER.model_copy(update={"weight_fraction": 0.9}),
            Component(name="BHT", cas_number="128-37-0", role="antioxidant", weight_fraction=0.1),
        ]
    )
    by_name = check_compliance(cand, make_spec(excluded_substances=["bht"]))
    by_cas = check_compliance(cand, make_spec(excluded_substances=["128-37-0"]))

    for flags in (by_name, by_cas):
        fails = [f for f in flags if f.regulation == "SPEC-EXCLUDED"]
        assert len(fails) == 1
        assert fails[0].status == "fail"
        assert fails[0].component_name == "BHT"


def test_pfas_not_required_skips_pfas_checks():
    suspect = Component(name="Perfluoropolyether oil", role="base_fluid", weight_fraction=1.0)
    flags = check_compliance(make_candidate([suspect]), make_spec(pfas_free_required=False))
    assert flags == []
