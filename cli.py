#!/usr/bin/env python3
"""
Run the restoration-funding prioritization agent from the command line.

Usage:
    python3 cli.py candidates.json
    python3 cli.py candidates.json --goal "We can only fund one project this quarter."

candidates.json shape:
[
  {"name": "Parcel A", "lat": 34.1478, "lng": -118.1445},
  {"name": "Parcel B", "lat": 47.267184, "lng": -106.995991}
]
"""

import argparse
import json
import sys

from agent import run_prioritization_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates_file", type=str, help="Path to a JSON file of candidate parcels")
    parser.add_argument("--goal", type=str, default=None, help="Optional goal/prompt for the agent")
    parser.add_argument("--model", type=str, default=None, help="Override the Groq model")
    args = parser.parse_args()

    with open(args.candidates_file) as f:
        candidates = json.load(f)

    print(f"Running prioritization agent over {len(candidates)} candidates...\n", file=sys.stderr)
    try:
        result = run_prioritization_agent(candidates, goal=args.goal, model=args.model)
    except RuntimeError as e:
        print(f"\nCan't run the agent: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("AGENT REASONING TRACE (tool calls it made, and what it learned)")
    print("=" * 70)
    for step in result["trace"]:
        print(f"\n-> called {step['tool']}({step['args']})")
        print(json.dumps(step["result"], indent=2))

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print(result.get("recommendation") or f"[no recommendation -- {result.get('error')}]")


if __name__ == "__main__":
    main()
