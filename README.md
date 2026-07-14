# Coolant Formulation Copilot

A prototype system that takes a target specification for a PFAS-free data
center coolant and produces a ranked shortlist of candidate formulations,
along with a lab validation plan.

## How it works

The system uses a multi-agent pipeline built with LangGraph and LangChain.
Five specialized agents handle distinct parts of the formulation task.

A research agent retrieves relevant literature and datasheets from a Chroma
vector store built from real public source documents: product datasheets,
an industry specification, comparative technical papers, and a regulatory
guide. A generator agent proposes candidate formulations grounded in that
research. Two agents then run in parallel to evaluate each candidate: a
property estimator (thermal conductivity, viscosity, flash point) and a
compliance checker (PFAS regulatory status, material compatibility). Both
are implemented as tool calling agents, meaning the language model
explicitly invokes deterministic Python functions for the calculations and
rule checks rather than performing them itself. A critic agent scores
candidates against the specification and can send weak candidates back to
the generator for revision. Once a candidate is accepted, an experiment
planner agent generates a design of experiments plan for lab validation.

## Why the revision loop is capped at three iterations

The critic to generator revision loop is bounded at a maximum of three
iterations before the system accepts the best available candidate or
reports that none met the specification. This was a deliberate design
choice for three reasons.

First, in testing, accepted candidates converged within one to two
revisions. A candidate that has already received specific, actionable
feedback (for example a missing CAS number, or a property estimate that
conflicts with a known reference fluid) and still fails after three
attempts is unlikely to be salvageable through further language model
reasoning alone. At that point the limiting factor is usually missing
information, such as an incomplete source corpus or an underspecified
target, not something an additional revision cycle can fix. Continuing to
loop under those conditions mostly generates cost without generating
progress.

Second, an unbounded revision loop is a known failure mode in agentic
systems. Without a hard stop, a system can continue revising indefinitely
if the critic's threshold and the generator's outputs oscillate rather than
converge. A fixed cap guarantees termination and keeps runtime and API cost
predictable, which matters for a system meant to run live.

Third, a fixed cap keeps the system's behavior deterministic and auditable.
An adaptive stopping rule, for example one based on detecting a plateau in
the critic's score, would be harder to reason about and harder to explain
to a reviewer. Since this system is meant to support decisions in a
regulated manufacturing context, predictable and explainable behavior was
prioritized over a marginally more sample efficient stopping rule.

## Stack

Python, FastAPI, LangChain, LangGraph, Chroma, OpenAI, React, Tailwind CSS.

## Running locally

Copy `.env.example` to `.env` and fill in the required keys, then see the
setup instructions in the repository for installation and startup steps.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # pinned deps + the local package (src/ layout)
cp .env.example .env                 # then edit .env and add your OPENAI_API_KEY
```

`OPENAI_API_KEY` is the only required variable; `LANGSMITH_API_KEY` is optional
(enables tracing). See [.env.example](.env.example) for the full list.

## Running

```bash
python scripts/ingest.py             # build the Chroma store + extract fluid profiles
python main.py data/example_spec.json  # run end-to-end; prints + saves the report
```

## HTTP API

A FastAPI service in `api/` wraps the same pipeline (no pipeline code changed):

```bash
uvicorn api.main:app --reload        # from the repo root
```

- `POST /run` — body is a `TargetSpec` JSON; starts a run and returns a
  `run_id` immediately (does not block on completion).
- `GET /run/{run_id}/events` — Server-Sent Events, one per graph node as it
  finishes (`node_name`, `status`, `output_summary`, `timestamp`,
  `loop_iteration`), ending with a `complete` event carrying the ranked
  candidates, DOE plan, and token/cost summary.
- `GET /health` — returns 200; used as the Render health check (and handy for
  a free uptime pinger to avoid free-tier cold starts).

If `frontend/dist` exists, the same app serves the built dashboard from `/`,
so frontend and backend share one origin and no CORS setup is needed. During
development the Vite dev server (`npm run dev` in `frontend/`) proxies `/run`
and `/health` to `localhost:8000` instead.

## Deployment (Render)

`render.yaml` defines a single free-plan web service that builds the frontend
and serves it from the FastAPI app:

1. On [Render](https://render.com), create a **New → Blueprint** and connect
   this GitHub repo — it picks up `render.yaml` automatically. (Or create a
   web service manually and copy the build/start commands from the file.)
2. Select the **Free** plan.
3. Fill in the environment variables when prompted: `OPENAI_API_KEY`
   (required), plus `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, and
   `LANGSMITH_PROJECT` if you want tracing.
4. Deploy. On each deploy/restart the start command runs `scripts/ingest.py`
   first, so Chroma is repopulated before the server accepts requests (the
   free tier's disk is ephemeral).

Note the free tier spins the service down after ~15 minutes of inactivity; a
free uptime pinger hitting `GET /health` keeps it warm.

## Testing

```bash
pytest
```
