"""
Restoration-funding prioritization agent (Mireye Build Challenge).

An agent for land trusts with limited restoration budgets: given a list of
candidate parcels/watersheds, it decides for itself what data it needs,
gathers real erosion/deforestation risk and real county-level economic
context for each, reasons about the trade-off between ecological severity
and economic value protected, and produces a written, justified funding
priority recommendation -- not a static report for a human to interpret
themselves.

Two real, independent data sources, combined:
  1. Erosion/deforestation risk -- via a live risk-assessment API
     (github.com/<you>/mireye-risk-app), itself built on RUSLE erosion
     modeling, USDA/FEMA/USGS/GFW data, with honest fallback sourcing and
     confidence reporting.
  2. County-level economic/demographic context -- via Mireye's own
     /v1/lookup endpoint (population, growth, income, home price trend,
     employment, Opportunity Zone status).

Reasoning engine: Groq's free, OpenAI-compatible chat completions API.
"""

import json
import os
import time

import requests

# Free-tier hosts (this agent's own dependencies included) sleep after
# inactivity and can return a 502 for a few seconds while waking up --
# that's transient, not a real failure, so it's worth one short retry
# before giving up and reporting the gap to the model.
_RETRYABLE_STATUS = {502, 503, 504}


def _request_with_retry(fn, attempts=3, delay_seconds=6):
    last_exc = None
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            resp = fn()
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if is_last:
                raise
            time.sleep(delay_seconds)
            continue

        if resp.status_code in _RETRYABLE_STATUS and not is_last:
            time.sleep(delay_seconds)
            continue
        resp.raise_for_status()
        return resp
    raise last_exc

# The already-live, already-deployed risk-assessment service. The agent
# treats this exactly like any other external API -- no shared code, no
# import of that project's source, just its public HTTP contract.
RISK_APP_URL = os.environ.get("RISK_APP_URL", "https://mireye-risk-app.onrender.com")

MIREYE_BASE_URL = "https://api.mireye.com"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


def get_mireye_token() -> str:
    token = os.environ.get("MIREYE_BEARER_TOKEN")
    if token:
        return token
    raise RuntimeError("MIREYE_BEARER_TOKEN is not set.")


def get_erosion_context(lat: float, lng: float) -> dict:
    """Real erosion/deforestation risk for one point, via the live
    risk-assessment API -- a plain HTTP call, same as any other tool
    here. No dependency on that project's source code."""
    try:
        # RISK_APP_URL is a free-tier host that can take 40-60s+ to wake
        # from a cold start, returning 502 immediately (not a slow
        # timeout) while it boots -- the backoff has to span that whole
        # window, not just a few seconds.
        resp = _request_with_retry(
            lambda: requests.get(
                f"{RISK_APP_URL}/api/risk",
                params={"lat": lat, "lng": lng},
                timeout=60,
            ),
            attempts=4,
            delay_seconds=20,
        )
        data = resp.json()
        risk = data.get("risk") or {}
        rusle = risk.get("rusle_lite") or {}
        loss_years = data.get("tree_cover_loss_by_year") or []
        return {
            "composite_score": risk.get("score"),
            "level": risk.get("level"),
            "confidence": risk.get("confidence"),
            "rusle_relative_index": rusle.get("relative_index"),
            "annual_soil_loss_tons_per_acre": rusle.get("annual_soil_loss_tons_per_acre"),
            "recent_deforestation_ha_since_2018": sum(
                row.get("area_ha", 0) for row in loss_years
                if row.get("umd_tree_cover_loss__year", 0) >= 2018
            ),
            "top_factors": [
                f["label"] for f in (risk.get("factors") or [])
                if f.get("severity") in ("high", "moderate")
            ],
        }
    except Exception as e:
        return {"error": f"erosion context unavailable: {e}"}


def get_economic_context(lat: float, lng: float) -> dict:
    """Real county-level economic/demographic context for one point, via
    Mireye's /v1/lookup endpoint."""
    try:
        token = get_mireye_token()
        resp = _request_with_retry(lambda: requests.post(
            f"{MIREYE_BASE_URL}/v1/lookup",
            headers={"Authorization": f"Bearer {token}"},
            json={"input": f"{lat},{lng}", "include_parcel": False},
            timeout=30,
        ))
        data = resp.json()
        if data.get("disposition") != "resolved":
            return {"error": f"lookup did not resolve: {data.get('disposition')}"}
        market = data.get("county_market") or {}
        return {
            "county": data.get("county"),
            "state": data.get("state"),
            "cbsa_name": data.get("cbsa_name"),
            "population": market.get("population"),
            "population_growth_1yr_pct": market.get("population_growth_1yr_pct"),
            "median_household_income_usd": market.get("median_household_income_usd"),
            "home_price_index_yoy_pct": market.get("hpi_yoy_pct"),
            "employment_yoy_pct": market.get("employment_yoy_pct"),
            "in_opportunity_zone": data.get("in_opportunity_zone"),
        }
    except Exception as e:
        return {"error": f"economic context unavailable: {e}"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_erosion_context",
            "description": (
                "Get real erosion and deforestation risk data for one candidate "
                "parcel/location: composite risk score, RUSLE soil-loss estimate, "
                "and recent tree-cover loss."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["lat", "lng"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_economic_context",
            "description": (
                "Get real county-level economic and demographic context for one "
                "candidate parcel/location: population, population growth, median "
                "income, home price trend, employment trend, Opportunity Zone status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                },
                "required": ["lat", "lng"],
            },
        },
    },
]

TOOL_IMPLS = {
    "get_erosion_context": lambda args: get_erosion_context(args["lat"], args["lng"]),
    "get_economic_context": lambda args: get_economic_context(args["lat"], args["lng"]),
}

SYSTEM_PROMPT = (
    "You are a restoration-funding prioritization agent for a land trust with a "
    "limited budget. You will be given a list of candidate parcels/watersheds. For "
    "EACH one, call get_erosion_context and get_economic_context to gather real data "
    "before making any judgment -- never guess or assume values you haven't fetched.\n\n"
    "Two factors matter, and they can point in OPPOSITE directions -- your job is to "
    "name that trade-off explicitly, never to blur it:\n"
    "1. Ecological severity (erosion/deforestation risk): HIGHER severity is a reason "
    "TO restore a parcel -- it means there's more actual damage happening to fix.\n"
    "2. Economic value protected (population, growth, income, development pressure): "
    "HIGHER value protected is a SEPARATE reason to prioritize a parcel -- it means "
    "restoring it benefits more people/property, independent of how severe the erosion "
    "itself is.\n\n"
    "These are not the same axis and must never be conflated. Never write reasoning "
    "where 'lower risk' is used as a justification for priority -- if a parcel has "
    "lower ecological severity than another, that is a point AGAINST prioritizing it "
    "ecologically, and if you still recommend it, you must explicitly say you are "
    "trading lower ecological urgency for higher economic value protected (or vice "
    "versa) -- name the trade-off in one explicit sentence per candidate, don't just "
    "assert a conclusion.\n\n"
    "Produce a final ranked list. For each candidate: state its ecological severity, "
    "state its economic value protected, then state explicitly which one you weighted "
    "more heavily and why. Explicitly flag any candidate whose data was incomplete or "
    "low-confidence rather than treating it the same as a fully-verified result."
)


def _groq_chat(messages: list, model: str) -> dict:
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def run_prioritization_agent(candidates: list, goal: str = None, model: str = None) -> dict:
    """candidates: list of {"name": str, "lat": float, "lng": float}.
    Returns the agent's full tool-call trace plus its final written
    recommendation -- the trace is what proves it's actually reasoning
    across tools, not just formatting a canned answer."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set -- the agent has no reasoning engine to call.")

    model = model or GROQ_MODEL
    candidate_list = "\n".join(f"- {c['name']}: {c['lat']}, {c['lng']}" for c in candidates)
    user_prompt = (
        f"Goal: {goal or 'Prioritize these parcels for restoration funding.'}\n\n"
        f"Candidates:\n{candidate_list}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    trace = []
    for _ in range(12):  # hard cap so a confused model can't loop forever
        result = _groq_chat(messages, model)
        msg = result["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return {"recommendation": msg["content"], "trace": trace}

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"])
            tool_result = TOOL_IMPLS[fn_name](fn_args)
            trace.append({"tool": fn_name, "args": fn_args, "result": tool_result})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_result),
            })

    return {"recommendation": None, "trace": trace, "error": "agent did not converge within step limit"}
