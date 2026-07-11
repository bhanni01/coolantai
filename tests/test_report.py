from conftest import make_candidate, make_finding, make_target_spec

from coolant_copilot.report import render_report
from coolant_copilot.state import (
    ComplianceFlag,
    CriticScore,
    DOEFactor,
    ExperimentPlan,
    ExperimentRun,
    GraphState,
    PropertyEstimate,
    PropertyName,
)


def make_full_state() -> GraphState:
    cand = make_candidate()
    return GraphState(
        target_spec=make_target_spec(),
        research_findings=[make_finding()],
        candidates=[cand],
        property_estimates=[
            PropertyEstimate(
                candidate_id=cand.id,
                property=PropertyName.THERMAL_CONDUCTIVITY,
                value=0.15,
                unit="W/m·K",
                method="linear_mixing_rule",
                meets_target=True,
            ),
            PropertyEstimate(
                candidate_id=cand.id,
                property=PropertyName.FLASH_POINT,
                value=120,
                unit="°C",
                method="table_lookup",
                meets_target=False,
            ),
        ],
        compliance_flags=[
            ComplianceFlag(
                candidate_id=cand.id,
                regulation="EU-PFAS-2024",
                status="needs_review",
                component_name="BHT",
                detail="Listed for review in one member state.",
            )
        ],
        critic_scores=[
            CriticScore(
                candidate_id=cand.id,
                performance=8,
                compliance_risk=9,
                practicality=7,
                overall=8,
                verdict="accept",
                feedback="",
            )
        ],
        revision_count=1,
        shortlist=[cand.id],
        experiment_plan=ExperimentPlan(
            objective="Validate thermal performance across the operating range.",
            design_type="full_factorial",
            factors=[DOEFactor(name="temperature", unit="°C", levels=[25, 45, 65])],
            runs=[
                ExperimentRun(
                    run_number=1,
                    candidate_id=cand.id,
                    factor_settings={"temperature": 25},
                    responses_to_measure=[PropertyName.THERMAL_CONDUCTIVITY],
                )
            ],
            safety_notes="Standard lab PPE.",
        ),
    )


def test_full_report_contains_all_sections():
    report = render_report(make_full_state())

    assert "# Formulation Report: DC-Coolant-A" in report
    assert "after 1 revision loop(s)" in report
    assert "## Target properties" in report
    assert "## Research findings" in report
    assert "esters.txt" in report
    assert "### 1. Synthetic ester blend (`cand-0-rev0`) — 8.0/10, accept" in report
    assert "Pentaerythritol tetraoleate" in report
    # Failed property estimate is highlighted, compliance flag is surfaced.
    assert "| flash_point | 120.0 | °C | table_lookup | **no** |" in report
    assert "EU-PFAS-2024: **needs_review** (BHT)" in report
    assert "## Lab validation plan (DOE)" in report
    assert "full factorial" in report


def test_report_handles_empty_run():
    report = render_report(GraphState(target_spec=make_target_spec()))

    assert "No candidates survived evaluation." in report
    assert "## Lab validation plan" not in report
