"""Render a final GraphState into a human-readable markdown report."""

from datetime import date

from coolant_copilot.state import GraphState


def _bounds(min_value: float | None, max_value: float | None) -> str:
    if min_value is not None and max_value is not None:
        return f"{min_value} – {max_value}"
    if min_value is not None:
        return f">= {min_value}"
    return f"<= {max_value}"


def render_report(state: GraphState) -> str:
    spec = state.target_spec
    lines = [
        f"# Formulation Report: {spec.name}",
        "",
        f"*Application: {spec.application.replace('_', ' ')} — generated {date.today().isoformat()}, "
        f"after {state.revision_count} revision loop(s).*",
        "",
        "## Target properties",
        "",
        "| Property | Target | Unit | Priority |",
        "|---|---|---|---|",
    ]
    for t in spec.property_targets:
        lines.append(
            f"| {t.property.value} | {_bounds(t.min_value, t.max_value)} | {t.unit} | {t.priority} |"
        )

    if state.research_findings:
        lines += ["", "## Research findings", ""]
        for f in state.research_findings:
            lines.append(f"- **{f.source}** ({f.id}): {f.summary} — {f.relevance}")

    candidates = {c.id: c for c in state.candidates}
    scores = {s.candidate_id: s for s in state.critic_scores}
    estimates_by_cand: dict[str, list] = {}
    for e in state.property_estimates:
        estimates_by_cand.setdefault(e.candidate_id, []).append(e)
    flags_by_cand: dict[str, list] = {}
    for fl in state.compliance_flags:
        flags_by_cand.setdefault(fl.candidate_id, []).append(fl)

    lines += ["", "## Ranked shortlist", ""]
    if not state.shortlist:
        lines.append("*No candidates survived evaluation.*")
    for rank, cand_id in enumerate(state.shortlist, start=1):
        cand = candidates[cand_id]
        score = scores.get(cand_id)
        header = f"### {rank}. {cand.name} (`{cand_id}`)"
        if score is not None:
            header += f" — {score.overall}/10, {score.verdict}"
        lines += [header, "", cand.rationale, "", "**Composition:**"]
        for comp in cand.components:
            cas = f", CAS {comp.cas_number}" if comp.cas_number else ""
            lines.append(
                f"- {comp.name} ({comp.role}{cas}) — {comp.weight_fraction:.1%}"
            )
        if cand_id in estimates_by_cand:
            lines += ["", "| Property | Estimate | Unit | Method | Meets target |", "|---|---|---|---|---|"]
            for e in estimates_by_cand[cand_id]:
                meets = "—" if e.meets_target is None else ("yes" if e.meets_target else "**no**")
                lines.append(
                    f"| {e.property.value} | {e.value} | {e.unit} | {e.method} | {meets} |"
                )
        reference_notes = [
            e for e in estimates_by_cand.get(cand_id, []) if e.reference_note
        ]
        if reference_notes:
            lines += ["", "**Reference fluid checks:**"]
            for e in reference_notes:
                marker = "✓" if e.reference_check == "validated" else "⚠"
                lines.append(f"- {marker} {e.property.value}: {e.reference_note}")
        failing = [
            fl for fl in flags_by_cand.get(cand_id, []) if fl.status != "pass"
        ]
        if failing:
            lines += ["", "**Compliance flags:**"]
            for fl in failing:
                comp = f" ({fl.component_name})" if fl.component_name else ""
                lines.append(f"- {fl.regulation}: **{fl.status}**{comp} — {fl.detail}")
        else:
            lines += ["", "All compliance checks passed."]
        lines.append("")

    plan = state.experiment_plan
    if plan is not None:
        lines += [
            "## Lab validation plan (DOE)",
            "",
            plan.objective,
            "",
            f"**Design:** {plan.design_type.replace('_', ' ')}, "
            f"{len(plan.runs)} run(s) × {plan.replicates} replicate(s).",
            "",
            "**Factors:**",
        ]
        for factor in plan.factors:
            levels = ", ".join(str(level) for level in factor.levels)
            lines.append(f"- {factor.name} ({factor.unit}): {levels}")
        lines += ["", "| Run | Candidate | Settings | Measure |", "|---|---|---|---|"]
        for run in plan.runs:
            settings = ", ".join(f"{k}={v}" for k, v in run.factor_settings.items())
            measures = ", ".join(p.value for p in run.responses_to_measure)
            lines.append(f"| {run.run_number} | {run.candidate_id} | {settings} | {measures} |")
        lines += ["", f"**Safety:** {plan.safety_notes}"]

    return "\n".join(lines) + "\n"
