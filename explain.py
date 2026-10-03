"""
Narration layer: turns the swarm's numbers into a plain-English explanation.
The intelligence is the simulation; the model only narrates figures it is given
and is told to invent nothing. Requires OPENAI_API_KEY (loaded from .env).
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_client = OpenAI()


SYSTEM = (
    "You are a sharp retail analyst briefing a Tesco Price & Promotions team. You "
    "are given the numeric output of an agent-based simulation in which shoppers "
    "chose where to buy a product (Tesco vs competitor supermarkets) before and "
    "after a Tesco Clubcard promo. Explain what it means, tying points to "
    "behavioural drivers (loss aversion, store loyalty, value, social proof). Be "
    "concrete and commercial. The key judgement is incrementality: gain drawn from "
    "competitors or new demand is incremental to Tesco; gain cannibalised from "
    "Tesco's own substitute lines is not. INVENT NO NUMBERS - use only the figures "
    "provided. Respond ONLY as compact JSON with keys: headline (one punchy "
    "sentence), bullets (array of 3 short strings, each citing a figure and a "
    "driver), recommendation (one sentence), caveat (one sentence naming a real "
    "limitation)."
)


def _facts(result: dict) -> dict:
    d = result["decomposition"]
    return {
        "product": result["product"]["name"],
        "category": result["product"]["category"],
        "promo": result["promo"],
        "shoppers_simulated": result["n_agents"],
        "tesco_units_baseline": result["tesco_units"]["baseline"],
        "tesco_units_with_promo": result["tesco_units"]["promo"],
        "extra_units_of_promoted_line": result["tesco_units"]["gain"],
        "of_which_cannibalised_from_tesco_substitutes": d["cannibalised"],
        "of_which_drawn_from_competitors": d["from_competitor"],
        "drawn_from_competitors_total": d["from_competitor_total"],
        "of_which_new_demand": d["new_uplift"],
        "net_incremental_to_tesco": d["net_incremental"],
        "top_segments_won": dict(sorted(result["segment_gain"].items(),
                                        key=lambda kv: kv[1], reverse=True)[:3]),
    }


def narrate(result: dict) -> dict:
    facts = _facts(result)
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
