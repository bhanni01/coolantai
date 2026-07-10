"""Shared graph state schemas for the Coolant Formulation Copilot.

Every model here is either part of the LangGraph state (`GraphState`) or a
`with_structured_output` schema for an LLM node, so fields carry descriptions
and closed value sets use enums/Literals.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class PropertyName(str, Enum):
    THERMAL_CONDUCTIVITY = "thermal_conductivity"  # W/m·K
    KINEMATIC_VISCOSITY = "kinematic_viscosity"  # cSt @ 40°C
    SPECIFIC_HEAT = "specific_heat"  # J/kg·K
    DENSITY = "density"  # kg/m³
    DIELECTRIC_STRENGTH = "dielectric_strength"  # kV (breakdown)
    FLASH_POINT = "flash_point"  # °C
    POUR_POINT = "pour_point"  # °C
    GWP = "gwp"  # global warming potential
    COST = "cost"  # USD/L


# Broad sanity bounds: wide enough for any property in PropertyName (GWP in
# the thousands, °C values below zero), tight enough to reject absurd input.
_VALUE_BOUNDS = {"ge": -1e6, "le": 1e9}


class PropertyTarget(BaseModel):
    property: PropertyName
    min_value: float | None = Field(
        default=None, description="Lower acceptable bound, in `unit`.", **_VALUE_BOUNDS
    )
    max_value: float | None = Field(
        default=None, description="Upper acceptable bound, in `unit`.", **_VALUE_BOUNDS
    )
    unit: str = Field(max_length=20)
    priority: Literal["must", "should"] = Field(
        description="'must' targets are hard requirements; 'should' are preferences."
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> "PropertyTarget":
        if self.min_value is None and self.max_value is None:
            raise ValueError("property target needs at least one of min_value/max_value")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value")
        return self


Region = Literal["US", "EU", "UK", "JP", "CN", "KR"]


class TargetSpec(BaseModel):
    """Input to the graph: what the coolant must achieve.

    This is untrusted user input: free-text fields are length-capped here and
    are only interpolated into prompts through prompting.format_spec, which
    fences them in <user_input> tags.
    """

    name: str = Field(max_length=100)
    application: Literal["single_phase_immersion", "two_phase_immersion", "cold_plate"]
    description: str = Field(max_length=500)
    property_targets: list[PropertyTarget] = Field(max_length=len(PropertyName))
    regulatory_regions: list[Region] = Field(
        description="Markets to comply with, e.g. ['US', 'EU']."
    )
    pfas_free_required: bool = True
    excluded_substances: list[Annotated[str, Field(max_length=100)]] = Field(
        default_factory=list,
        max_length=50,
        description="Substance names or CAS numbers to never propose.",
    )
    requires_dielectric: bool = Field(
        default=False,
        description="Fluid contacts live electronics and must be electrically insulating.",
    )
    corrosion_inhibition_required: bool = Field(
        default=False,
        description="Closed-loop water-side cooling: formulation needs corrosion inhibitors.",
    )


class CriticWeights(BaseModel):
    """Ranking weights the critic applies in code to compute each overall score.

    Resolved server-side from a DatacenterProfile's optimization_priority —
    never part of the public form payload. The cost weight applies to the
    practicality subscore (sourcing, cost, handling).
    """

    performance: float = Field(default=0.25, ge=0, le=1)
    cost: float = Field(default=0.25, ge=0, le=1)
    compliance: float = Field(default=0.25, ge=0, le=1)
    lifespan: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def _check_total(self) -> "CriticWeights":
        if self.performance + self.cost + self.compliance + self.lifespan <= 0:
            raise ValueError("at least one critic weight must be positive")
        return self


class Component(BaseModel):
    name: str
    cas_number: str | None = Field(
        default=None, description="CAS registry number; None if not known — never guessed."
    )
    role: Literal[
        "base_fluid", "viscosity_modifier", "antioxidant", "corrosion_inhibitor", "other"
    ]
    weight_fraction: float = Field(ge=0.0, le=1.0)


class Candidate(BaseModel):
    """One proposed formulation (generator output)."""

    id: str = Field(description="Stable key referenced by estimates/flags/scores, e.g. 'cand-3-rev1'.")
    name: str
    components: list[Component]
    rationale: str
    source_refs: list[str] = Field(
        default_factory=list, description="ResearchFinding ids grounding this proposal."
    )
    revision: int = Field(default=0, description="Which critic loop produced it; 0 = initial.")

    @model_validator(mode="after")
    def _check_components(self) -> "Candidate":
        base_fluids = [c for c in self.components if c.role == "base_fluid"]
        if len(base_fluids) != 1:
            raise ValueError("candidate must have exactly one base_fluid component")
        total = sum(c.weight_fraction for c in self.components)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"component weight_fractions sum to {total}, expected 1.0")
        return self


class PropertyEstimate(BaseModel):
    """Deterministic property estimate for one candidate (evaluator tool output)."""

    candidate_id: str
    property: PropertyName
    value: float
    unit: str
    method: str = Field(description="Estimation method, e.g. 'linear_mixing_rule', 'table_lookup'.")
    uncertainty: float | None = Field(
        default=None, description="± in same unit, when the method provides it."
    )
    meets_target: bool | None = Field(
        default=None, description="None if the spec has no target for this property."
    )
    reference_check: Literal["validated", "conflict"] | None = Field(
        default=None,
        description=(
            "Comparison against a real extracted fluid of similar base "
            "chemistry; None when no reference data was available."
        ),
    )
    reference_note: str | None = Field(
        default=None, description="Human-readable detail of the reference comparison."
    )


class ComplianceFlag(BaseModel):
    """One regulatory check result for one candidate (evaluator tool output)."""

    candidate_id: str
    regulation: str = Field(description="Rule identifier, e.g. 'EU-PFAS-2024', 'TSCA-8a7'.")
    status: Literal["pass", "fail", "needs_review"]
    component_name: str | None = Field(
        default=None, description="Which ingredient triggered the flag, if any."
    )
    detail: str


class CriticScore(BaseModel):
    """Critic's assessment of one candidate (structured LLM output)."""

    candidate_id: str
    performance: float = Field(ge=0, le=10)
    compliance_risk: float = Field(ge=0, le=10, description="10 = no risk.")
    practicality: float = Field(ge=0, le=10, description="Sourcing, cost, handling.")
    lifespan: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Expected service life / thermal-oxidative stability; 5 = neutral.",
    )
    overall: float = Field(
        ge=0, le=10, description="Computed in code as the CriticWeights-weighted subscore average."
    )
    verdict: Literal["accept", "revise", "reject"]
    feedback: str = Field(description="Actionable guidance for the generator when verdict is 'revise'.")


class DOEFactor(BaseModel):
    name: str
    unit: str
    levels: list[float]


class ExperimentRun(BaseModel):
    run_number: int
    candidate_id: str
    factor_settings: dict[str, float] = Field(description="Factor name → level for this run.")
    responses_to_measure: list[PropertyName]


class ExperimentPlan(BaseModel):
    """DOE lab validation plan (structured LLM output)."""

    objective: str
    design_type: Literal[
        "full_factorial",
        "fractional_factorial",
        "box_behnken",
        "central_composite",
        "plackett_burman",
    ]
    factors: list[DOEFactor]
    runs: list[ExperimentRun]
    replicates: int = Field(default=1, ge=1)
    safety_notes: str


class ResearchFinding(BaseModel):
    """One relevant piece of retrieved literature (research node output)."""

    id: str
    source: str = Field(description="Chroma document id or citation.")
    summary: str
    relevance: str = Field(description="Why this matters for the current target spec.")


class ResearchBrief(BaseModel):
    """Distilled literature review (research node output)."""

    overview: str = Field(description="2-4 sentence synthesis of what the sources say about this spec.")
    findings: list[ResearchFinding]
    gaps: list[str] = Field(
        default_factory=list,
        description="Open questions the retrieved sources do not answer.",
    )


class GraphState(BaseModel):
    """The single state model passed through the LangGraph StateGraph.

    Nodes return partial dict updates with plain overwrite semantics: a node
    that appends to a list reads the current list and returns the full
    updated list.
    """

    target_spec: TargetSpec
    critic_weights: CriticWeights = Field(
        default_factory=CriticWeights,
        description="Ranking weights for the critic, from the profile's optimization_priority.",
    )
    research_findings: list[ResearchFinding] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    property_estimates: list[PropertyEstimate] = Field(default_factory=list)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)
    critic_scores: list[CriticScore] = Field(default_factory=list)
    revision_count: int = Field(
        default=0, description="Routing edge stops revising once this reaches 3."
    )
    revision_feedback: str | None = Field(
        default=None, description="Critic → generator channel for the revision loop."
    )
    shortlist: list[str] = Field(
        default_factory=list, description="Candidate ids ranked best-first."
    )
    experiment_plan: ExperimentPlan | None = None
