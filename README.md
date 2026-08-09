# Restoration-Funding Prioritization Agent

Built for the [Mireye Build Challenge](https://www.mireye.com).

## What it does

Land trusts have limited restoration budgets and no systematic way to decide
where a dollar of funding does the most good. This agent takes a list of
candidate parcels/watersheds, decides for itself what data it needs, gathers
it from two real, independent sources, and produces a written, justified
funding-priority recommendation.

## What Mireye is combined with

**Erosion/deforestation risk** (via [mireye-risk-app](https://github.com/abhishekkankip663/mireye-risk-app),
a live RUSLE-based erosion model built on USDA/FEMA/USGS soil, slope, and
flood data, plus Global Forest Watch deforestation tracking) is combined with
**real county-level economic and demographic data** (via Mireye's own
`/v1/lookup` endpoint — population, growth, income, home price trend,
employment, Opportunity Zone status).

Most restoration-prioritization today is done on ecological severity alone,
or on relationships and politics. This agent forces an explicit trade-off:
*how bad is the erosion here* vs. *how much economic value would restoring
it actually protect* — and states which one it weighted more heavily, and
why, for every candidate.

## Who pays

Land trusts (1,000+ in the US, per the Land Trust Alliance) making real,
recurring restoration-funding allocation decisions with real, limited budgets.

## Why this is an agent, not a report

It isn't handed a fixed pipeline to run. It's given a **goal** and a
**toolbox** (`get_erosion_context`, `get_economic_context`) and decides for
itself, per candidate, what to check and when. Nothing is hardcoded about
which tool runs first or how many times — that's the model's decision, made
live, and the full tool-call trace is returned alongside the final answer so
you can see exactly what it did and why, not just what it concluded.

## Running it

```bash
pip install -r requirements.txt

export GROQ_API_KEY=...        # free, no card required: console.groq.com
export MIREYE_BEARER_TOKEN=... # from `uvx mireye-mcp login`, or your Mireye dashboard

# From the command line, against the included example:
python3 cli.py candidates_example.json --goal "We can only fund one project this quarter."

# Or as a live API + browser UI:
uvicorn server:app --reload
# open http://localhost:8000 for a simple form (enter parcels, see the
# agent's tool-call trace and recommendation render live)

# Or hit the API directly:
curl -X POST localhost:8000/prioritize -H "Content-Type: application/json" -d '{
  "candidates": [
    {"name": "Parcel A", "lat": 34.1478, "lng": -118.1445},
    {"name": "Parcel B", "lat": 47.267184, "lng": -106.995991}
  ]
}'
```

## Architecture

- `agent.py` — the agent itself: tool definitions, the reasoning loop, the
  system prompt that forces an explicit ecological-vs-economic trade-off
  statement per candidate. All reasoning/deciding/acting happens here.
- `cli.py` — run it directly, no server required.
- `server.py` — thin FastAPI wrapper exposing `POST /prioritize`.
- `frontend/index.html` — a minimal form (no framework) that calls
  `/prioritize` and renders the agent's own tool-call trace and final
  recommendation. It has no logic of its own — it's a window into the
  same agent, for people who'd rather not use a CLI (e.g. a GIS analyst
  at a land trust).

The erosion/deforestation tool calls a **live, publicly deployed API**
(`mireye-risk-app`), not shared source code — this repo has no dependency on
that project beyond its public HTTP contract, the same way it has no
dependency on Mireye's or Groq's internals either.

## Reasoning engine

[Groq](https://groq.com)'s free, OpenAI-compatible API (`openai/gpt-oss-120b`)
— zero cost, no credit card required.
