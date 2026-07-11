#!/usr/bin/env python
"""Run the coolant formulation copilot end-to-end.

Usage:
    python main.py data/example_spec.json [--persist-dir ./chroma_db]
        [--collection coolant_sources] [--out reports/]

Reads OPENAI_API_KEY (required) and LangSmith settings from .env. When a
LangSmith API key is present, the full run — every node and LLM call — is
traced, and the trace URL is printed at the end.
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

LANGSMITH_PROJECT = "coolant-copilot"
OUTPUTS_DIR = Path("outputs")


def describe_update(node: str, update: dict) -> str:
    """One status line summarizing what a node just contributed."""
    if node == "research":
        return f"{len(update.get('research_findings', []))} research finding(s)"
    if node == "generator":
        candidates = update.get("candidates", [])
        revision = max((c.revision for c in candidates), default=0)
        label = f"revision {revision}" if revision else "initial pass"
        return f"{len(candidates)} candidate(s) so far ({label})"
    if node == "property_estimator":
        return f"{len(update.get('property_estimates', []))} property estimate(s)"
    if node == "compliance_checker":
        return f"{len(update.get('compliance_flags', []))} compliance flag(s)"
    if node == "critic":
        scores = update.get("critic_scores", [])
        if not scores:
            return "no new candidates to score"
        top = max(scores, key=lambda s: s.overall)
        return f"top candidate {top.candidate_id} scored {top.overall}/10 ({top.verdict})"
    if node == "experiment_planner":
        plan = update.get("experiment_plan")
        if plan is None:
            return "no plan produced"
        return f"{plan.design_type} DOE with {len(plan.runs)} run(s) × {plan.replicates} replicate(s)"
    return "done"


def setup_tracing() -> bool:
    """Enable LangSmith tracing if an API key is configured. Returns whether enabled."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return False
    # Current and legacy variable names, so any langchain version picks it up.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    return True


def trace_url(run_id: uuid.UUID) -> str | None:
    """Best-effort fetch of the LangSmith URL for this run."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        return Client().read_run(run_id).url
    except Exception:
        return None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Path to a TargetSpec JSON file.")
    parser.add_argument("--persist-dir", default=None, help="Chroma directory (default ./chroma_db).")
    parser.add_argument("--collection", default=None, help="Chroma collection name.")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Directory for saved reports.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set; add it to .env (see .env.example).")
    tracing = setup_tracing()
    if not tracing:
        print("Note: no LANGSMITH_API_KEY found — running without tracing.", file=sys.stderr)

    # Imports load langchain after tracing env vars are in place.
    from langchain_openai import OpenAIEmbeddings

    from coolant_copilot.extraction import load_profiles
    from coolant_copilot.graph import build_graph
    from coolant_copilot.ingestion import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, get_vectorstore
    from coolant_copilot.injection_audit import scan_target_spec
    from coolant_copilot.observability import TokenCostTracker, format_run_summary
    from coolant_copilot.report import render_report
    from coolant_copilot.state import GraphState, TargetSpec

    spec = TargetSpec.model_validate_json(args.spec.read_text())
    scan_target_spec(spec)  # visibility only: logs suspicious phrases, never blocks
    vectorstore = get_vectorstore(
        # Must match the embedding model used by scripts/ingest.py.
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_dir=args.persist_dir or DEFAULT_PERSIST_DIR,
        collection_name=args.collection or DEFAULT_COLLECTION,
    )

    profiles = load_profiles("data/extracted")
    if profiles:
        print(f"Loaded {len(profiles)} reference fluid profile(s) from data/extracted/.")

    graph = build_graph(vectorstore, reference_profiles=profiles)
    run_id = uuid.uuid4()
    # The tracker rides the callback manager, which LangGraph propagates to every
    # node and nested LLM call — so per-node token/cost is captured for the whole
    # run, mirroring what LangSmith records when tracing is on.
    tracker = TokenCostTracker()
    print(f"Running graph for spec '{spec.name}'...")
    # Stream node-by-node so progress is visible; updates carry full lists
    # (the state's overwrite semantics), so accumulating them yields the
    # final state.
    values = dict(GraphState(target_spec=spec))
    for chunk in graph.stream(
        GraphState(target_spec=spec),
        config={
            "run_name": f"coolant-copilot:{spec.name}",
            "run_id": run_id,
            "recursion_limit": 50,
            "callbacks": [tracker],
            "tags": ["coolant-copilot"],
        },
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node.startswith("__"):
                continue
            update = update or {}
            print(f"  ✓ {node}: {describe_update(node, update)}")
            values.update(update)
    final = GraphState.model_validate(values)

    report = render_report(final)
    print("\n" + report)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}-{datetime.now():%Y%m%d-%H%M%S}"
    report_path = args.out / f"{stem}.md"
    report_path.write_text(report)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUTS_DIR / f"run_{datetime.now():%Y%m%d-%H%M%S}.json"
    state_path.write_text(final.model_dump_json(indent=2))
    print(f"Report saved to {report_path}; full graph state saved to {state_path}.")

    print("\n" + format_run_summary(tracker, final.revision_count))

    if tracing:
        url = trace_url(run_id)
        if url:
            print(f"LangSmith trace: {url}")
        else:
            project = os.environ.get("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
            print(
                f"LangSmith trace logged to project '{project}' "
                f"(run id {run_id}) at https://smith.langchain.com"
            )


if __name__ == "__main__":
    main()
