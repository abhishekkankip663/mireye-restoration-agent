---
title: "Restoration-Funding Prioritization Agent"
author: "Abhishek Kankipati — Mireye Build Challenge"
geometry: margin=0.65in
fontsize: 10pt
mainfont: Helvetica
colorlinks: true
header-includes:
  - \setlength{\parskip}{0.5em}
  - \usepackage{titlesec}
  - \titlespacing*{\subsection}{0pt}{0.6em}{0.3em}
---

## The Problem

Soil erosion costs the US an estimated **\$37.6B a year** in lost productivity — and it's getting worse as deforestation compounds it. A land trust, watershed council, or NRCS field office responsible for hundreds of parcels across **3,144 US counties** can't send a geotechnical surveyor to each one — a single manual survey covers one point at a time.

Even where hazard data exists, there's no systematic way to decide *where limited restoration dollars actually do the most good*. Today, funding decisions get made on ecological severity alone, or on relationships — not on a real comparison of erosion risk against the economic value a restoration dollar would actually protect.

## What We Combine Mireye With

**Real-time erosion/deforestation risk** — a live RUSLE-based scoring API (A = R×K×LS×C×P), built on cited USDA soil, USGS slope, FEMA flood, and Global Forest Watch deforestation data, with honest handling of missing data (never fabricated, always disclosed) — combined with **real county-level economic context** via Mireye's own `/v1/lookup` endpoint: population, growth, median income, home-price trend, employment, and Opportunity Zone status.

## The Agent

Given a list of candidate parcels and a goal ("prioritize restoration funding"), the agent decides for itself — via live tool-calling, not a fixed script — which data to gather and in what order. For every candidate, it must call both `get_erosion_context` and `get_economic_context` before judging anything. It then reasons across two factors that can point in *opposite* directions — ecological severity (higher = more urgent to restore) and economic value protected (higher population/income = more benefit from restoring) — and is required to **state which one it weighted more heavily and why**, per candidate, rather than outputting an opaque ranking.

Tested behavior, verified directly: when erosion risk is tied between two candidates, it correctly falls back to economic value as the deciding factor and says so explicitly. When a data source fails outright, it never fabricates a number — it flags the gap, reasons with what it actually has, and tells you the ranking is provisional until real data is available.

## Who Pays

**Land trusts and watershed conservation nonprofits** — 1,000+ in the US (Land Trust Alliance) — making real, recurring, budget-constrained restoration-funding decisions today, with no systematic tool for weighing severity against value protected.

## Why This Is an Agent, Not a Report

Nothing here runs a fixed pipeline. The model decides which tools to call, for which candidate, in what order — a genuinely different decision each time depending on what it learns — and it produces a real, written, justified funding recommendation as its output, not a number for a human to interpret themselves. The full tool-call trace is returned alongside the recommendation, so every claim it makes is traceable back to a real, live data lookup.

## Links

**Agent repo (submission):** github.com/abhishekkankip663/mireye-restoration-agent
**Erosion/deforestation data source (live API, built prior to this challenge):** github.com/abhishekkankip663/mireye-risk-app
