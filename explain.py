"""
Narration layer: turns the swarm's structured output into a plain-English
"human truth". This is deliberately NOT where the intelligence lives — the model
only narrates figures the simulation already computed, and is instructed to
invent nothing. Requires OPENAI_API_KEY (loaded from .env).
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_client = OpenAI()  # reads OPENAI_API_KEY from the environment


SYSTEM = (
    "You are a sharp behavioural scientist briefing a retail buyer. You are given "
    "the numeric output of an agent-based simulation of shoppers choosing on a "
    "shelf. Explain what it means in plain English, tying each point to a named "
    "behavioural principle (loss aversion, social proof, habit, health goals, "
    "novelty-seeking, value). Be concrete and commercial. INVENT NO NUMBERS — use "
    "only the figures provided. Respond ONLY as compact JSON with keys: "
    "headline (one punchy sentence = the human truth), bullets (array of 3 short "
    "strings, each referencing a figure and a principle), recommendation (one "
    "sentence of what to do), caveat (one sentence naming a real limitation)."
)


def _facts(result: dict, scenario_name: str) -> dict:
    d = result["decomposition"]
    return {
        "scenario": scenario_name,
        "promoted_product": result["promoted"]["name"],
        "promo": result["promo"],
        "shoppers_simulated": result["n_agents"],
        "extra_units_sold_of_promoted": result["target_gain"],
        "of_which_new_to_category": d["new_to_category"],
        "of_which_cannibalised_own_lines": d["cannibalised"],
        "of_which_stolen_from_competitors": d["from_competitor"],
        "net_new_category_buyers": result["incremental_demand"],
        "say_do_gap_baseline_pct": result["say_do"]["baseline_pct"],
        "say_do_gap_with_promo_pct": result["say_do"]["promo_pct"],
        "say_do_headline": result["say_do"]["headline"],
        "top_segments_won": dict(sorted(result["segment_gain"].items(),
                                        key=lambda kv: kv[1], reverse=True)[:3]),
    }


def narrate(result: dict, scenario_name: str) -> dict:
    facts = _facts(result, scenario_name)
    resp = _client.chat.completions.create(
        model=_MODEL,
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(facts)},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    out["source"] = _MODEL
    out.setdefault("bullets", [])
    return out
