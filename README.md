# Coolant Formulation Copilot

Multi-agent (LangGraph) system that takes a target spec for a PFAS-free data
center coolant and produces (1) a ranked shortlist of candidate formulations and
(2) a DOE lab validation plan.

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
  candidates, DOE plan, and token/cost summary. CORS is enabled for
  `http://localhost:5173`.

## Testing

```bash
pytest
```
