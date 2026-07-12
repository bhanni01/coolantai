"""End-to-end API test: POST /run then stream GET /run/{id}/events.

POST /run accepts a DatacenterProfile (closed dropdown selections) which the
server resolves into the full TargetSpec + CriticWeights — a raw TargetSpec
body is rejected. The graph dependency is overridden with the same fake-wired
*real* graph used by test_graph, so this exercises actual LangGraph node
ordering (research → generator → property_estimator ∥ compliance_checker →
critic → experiment_planner) with no network or LLM calls.
"""

import json
import sys
from pathlib import Path

# Make the repo-root `api` package importable when running under pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from conftest import FakeStructuredLLM, FakeToolCallingLLM  # noqa: E402
from test_graph import StubVectorStore, make_fakes  # noqa: E402

from api.main import app  # noqa: E402
from api.pipeline import get_graph  # noqa: E402
from coolant_copilot.graph import build_graph  # noqa: E402
from coolant_copilot.nodes.critic import CriticDraft, ScoreDraft  # noqa: E402
from coolant_copilot.nodes.generator import CandidateDraft, GenerationDraft  # noqa: E402
from coolant_copilot.state import Component, GraphState  # noqa: E402

EXPECTED_ORDER = [
    "research",
    "generator",
    "property_estimator",
    "compliance_checker",
    "critic",
    "experiment_planner",
]

PROFILE = {
    "cooling_method": "single_phase_immersion",
    "rack_density": "high_density",
    "climate_zone": "temperate",
    "regulatory_region": "us",
    "optimization_priority": "performance",
}


def _fake_graph():
    """A real compiled graph wired with offline fakes (accept on first pass)."""
    research_llm, generator_llm, critic_llm, planner_llm = make_fakes("accept")
    return build_graph(
        StubVectorStore(),
        research_llm=research_llm,
        generator_llm=generator_llm,
        estimator_llm=FakeToolCallingLLM(),
        compliance_llm=FakeToolCallingLLM(),
        critic_llm=critic_llm,
        planner_llm=planner_llm,
    )


def _read_sse(client: TestClient, url: str) -> list[dict]:
    """Consume an SSE stream to completion, returning parsed data payloads."""
    events: list[dict] = []
    with client.stream("GET", url) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    return events


def test_post_run_then_stream_events_in_order_and_completes():
    app.dependency_overrides[get_graph] = _fake_graph
    try:
        with TestClient(app) as client:
            # POST returns immediately with a run_id (non-blocking).
            post = client.post("/run", json=PROFILE)
            assert post.status_code == 202
            run_id = post.json()["run_id"]
            assert run_id

            events = _read_sse(client, f"/run/{run_id}/events")

        # --- node events -------------------------------------------------
        node_events = [e for e in events if e["type"] == "node"]
        assert node_events, "no node events were streamed"

        # Every node event carries the required fields and a valid status.
        for e in node_events:
            assert set(e) >= {
                "node_name",
                "status",
                "output_summary",
                "timestamp",
                "loop_iteration",
            }
            assert e["status"] in {"active", "completed"}
            assert isinstance(e["loop_iteration"], int)

        # Each node emits exactly one 'active' (start) and one 'completed' event.
        active = [e["node_name"] for e in node_events if e["status"] == "active"]
        completed = [e["node_name"] for e in node_events if e["status"] == "completed"]
        assert sorted(active) == sorted(EXPECTED_ORDER), active
        assert sorted(completed) == sorted(EXPECTED_ORDER), completed
        # Completed events carry a human summary.
        assert all(
            e["output_summary"] for e in node_events if e["status"] == "completed"
        )

        pos = {name: completed.index(name) for name in EXPECTED_ORDER}
        # DAG ordering on completion (the two evaluators run in parallel, so no
        # order is asserted between them — only that both land after the
        # generator and before the critic).
        assert pos["research"] < pos["generator"]
        assert pos["generator"] < pos["property_estimator"]
        assert pos["generator"] < pos["compliance_checker"]
        assert pos["property_estimator"] < pos["critic"]
        assert pos["compliance_checker"] < pos["critic"]
        assert pos["critic"] < pos["experiment_planner"]
        assert completed[0] == "research"

        # Concurrency: both evaluators announce 'active' before EITHER finishes,
        # i.e. they are genuinely running at the same time (real fan-out, not
        # sequential). This is what the SSE-backed UI renders as both nodes lit.
        def idx(name: str, status: str) -> int:
            return next(
                i
                for i, e in enumerate(node_events)
                if e["node_name"] == name and e["status"] == status
            )

        pe_active, cc_active = idx("property_estimator", "active"), idx("compliance_checker", "active")
        pe_done, cc_done = idx("property_estimator", "completed"), idx("compliance_checker", "completed")
        assert max(pe_active, cc_active) < min(pe_done, cc_done), (
            "evaluators did not overlap: both should be active before either completes"
        )

        # Accept on the first pass → no revision loops.
        assert all(e["loop_iteration"] == 0 for e in node_events)

        # --- terminal complete event ------------------------------------
        assert events[-1]["type"] == "complete", events[-1]
        complete = events[-1]
        assert complete["status"] == "complete"
        result = complete["result"]

        # Ranked shortlist + candidates.
        assert result["shortlist"] == ["cand-0-rev0"]
        assert [c["id"] for c in result["ranked_candidates"]] == ["cand-0-rev0"]

        # DOE plan present.
        assert result["experiment_plan"] is not None
        assert result["experiment_plan"]["runs"][0]["candidate_id"] == "cand-0-rev0"

        # Token/cost summary shape (zeros under fakes, but well-formed).
        token_cost = result["token_cost"]
        assert set(token_cost) >= {"total_tokens", "total_cost_usd", "summary_text"}
        assert isinstance(token_cost["total_tokens"], int)

        # The embedded result reuses GraphState verbatim and re-validates.
        final = GraphState.model_validate(result["state"])
        assert final.shortlist == ["cand-0-rev0"]
        assert final.experiment_plan is not None
    finally:
        app.dependency_overrides.clear()


def test_events_for_unknown_run_id_returns_404():
    with TestClient(app) as client:
        assert client.get("/run/does-not-exist/events").status_code == 404


def test_raw_target_spec_body_is_rejected():
    """The public form must never smuggle a TargetSpec past the profile schema."""
    spec = json.loads((REPO_ROOT / "data" / "example_spec.json").read_text())
    with TestClient(app) as client:
        assert client.post("/run", json=spec).status_code == 422
        # Even a valid profile with spec fields bolted on is refused (extra=forbid).
        assert client.post("/run", json={**PROFILE, "excluded_substances": []}).status_code == 422


def _fake_graph_two_contrasting_candidates():
    """Fake-wired real graph: two candidates whose critic subscores mirror each
    other (strong performance/poor practicality vs the reverse), so the ranking
    depends entirely on the server-resolved CriticWeights."""
    research_llm, _, _, planner_llm = make_fakes("accept")

    def candidate(name: str, base: str, cas: str) -> CandidateDraft:
        return CandidateDraft(
            name=name,
            components=[
                Component(name=base, cas_number=cas, role="base_fluid", weight_fraction=1.0)
            ],
            rationale="High flash point.",
            source_ref_ids=["rf-0"],
        )

    generator_llm = FakeStructuredLLM(
        GenerationDraft(
            candidates=[
                candidate("Ester blend", "Pentaerythritol tetraoleate", "19321-40-5"),
                candidate("PAO blend", "Polyalphaolefin PAO-6", "68037-01-4"),
            ]
        )
    )
    critic_llm = FakeStructuredLLM(
        CriticDraft(
            scores=[
                ScoreDraft(candidate_index=0, performance=9, compliance_risk=6,
                           practicality=2, lifespan=5, verdict="accept", feedback=""),
                ScoreDraft(candidate_index=1, performance=2, compliance_risk=6,
                           practicality=9, lifespan=5, verdict="accept", feedback=""),
            ],
            feedback_for_generator="",
        )
    )
    return build_graph(
        StubVectorStore(),
        research_llm=research_llm,
        generator_llm=generator_llm,
        estimator_llm=FakeToolCallingLLM(),
        compliance_llm=FakeToolCallingLLM(),
        critic_llm=critic_llm,
        planner_llm=planner_llm,
    )


def test_optimization_priority_changes_the_critic_ranking():
    """Same profile POSTed twice with only optimization_priority changed: the
    resolved CriticWeights must actually flip the shortlist order."""
    app.dependency_overrides[get_graph] = _fake_graph_two_contrasting_candidates
    try:
        def run_shortlist(priority: str) -> list[str]:
            with TestClient(app) as client:
                post = client.post("/run", json={**PROFILE, "optimization_priority": priority})
                assert post.status_code == 202
                events = _read_sse(client, f"/run/{post.json()['run_id']}/events")
            assert events[-1]["type"] == "complete", events[-1]
            return events[-1]["result"]["shortlist"]

        assert run_shortlist("performance") == ["cand-0-rev0", "cand-1-rev0"]
        assert run_shortlist("cost") == ["cand-1-rev0", "cand-0-rev0"]
    finally:
        app.dependency_overrides.clear()
