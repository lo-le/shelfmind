"""
Synthetic shopper population.

Each agent is a vector of behavioural weights (0-1 per dimension), a budget, and
a HOME RETAILER (the big-6 grocer they usually shop) sampled from their segment's
store affinity. The home retailer plus their habit_loyalty weight is what makes a
shopper stick with their usual store unless a deal is big enough to move them.

The segments and their store affinities are SYNTHETIC archetypes informed by
well-known grocery segmentation patterns, not a proprietary panel. This file and
choice.py are the single places to recalibrate against real loyalty-card data.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

_DATA = pathlib.Path(__file__).parent / "data"
_RET = json.loads((_DATA / "retailers.json").read_text(encoding="utf-8"))
RETAILERS = _RET["retailers"]
RETAILERS_BY_ID = {r["id"]: r for r in RETAILERS}
SEGMENT_AFFINITY = _RET["segment_affinity"]

DIMENSIONS = [
    "value",          # value-for-money / price sensitivity
    "deal_seeking",   # pull of discounts & anchored prices (loss aversion)
    "social_proof",   # susceptibility to what others are buying (herding)
    "habit_loyalty",  # stick with the usual store / brand
    "health",         # health motivation
    "novelty",        # openness to new & challenger products
    "taste",          # prioritise taste / quality
    "convenience",    # format, effort, store accessibility
]

SEGMENTS = [
    {"name": "Budget-Squeezed", "share": 0.26, "color": "#e4572e", "budget": 2.6,
     "blurb": "Watching every penny. Lives for the yellow sticker.",
     "means": {"value": 0.90, "deal_seeking": 0.85, "social_proof": 0.45, "habit_loyalty": 0.55, "health": 0.32, "novelty": 0.25, "taste": 0.50, "convenience": 0.55}},
    {"name": "Loyal Traditionalist", "share": 0.22, "color": "#4c6ef5", "budget": 3.6,
     "blurb": "Shops the same store every week. Trust beats novelty.",
     "means": {"value": 0.50, "deal_seeking": 0.45, "social_proof": 0.40, "habit_loyalty": 0.92, "health": 0.45, "novelty": 0.15, "taste": 0.62, "convenience": 0.55}},
    {"name": "Health-Conscious", "share": 0.18, "color": "#2fb380", "budget": 3.8,
     "blurb": "Reads the label. Wants better-for-you.",
     "means": {"value": 0.45, "deal_seeking": 0.40, "social_proof": 0.48, "habit_loyalty": 0.45, "health": 0.92, "novelty": 0.58, "taste": 0.55, "convenience": 0.45}},
    {"name": "Adventurous Foodie", "share": 0.14, "color": "#e8a33d", "budget": 4.4,
     "blurb": "Chasing the next great taste. Price is secondary.",
     "means": {"value": 0.35, "deal_seeking": 0.40, "social_proof": 0.55, "habit_loyalty": 0.30, "health": 0.55, "novelty": 0.90, "taste": 0.88, "convenience": 0.40}},
    {"name": "Trend-Follower", "share": 0.12, "color": "#b5559e", "budget": 3.6,
     "blurb": "If it's everywhere, they want it. Driven by the crowd.",
     "means": {"value": 0.45, "deal_seeking": 0.55, "social_proof": 0.92, "habit_loyalty": 0.35, "health": 0.50, "novelty": 0.80, "taste": 0.60, "convenience": 0.55}},
    {"name": "Convenience-Led", "share": 0.08, "color": "#6c8aa6", "budget": 3.6,
     "blurb": "Grab and go. Whatever is easiest wins.",
     "means": {"value": 0.55, "deal_seeking": 0.50, "social_proof": 0.50, "habit_loyalty": 0.55, "health": 0.40, "novelty": 0.45, "taste": 0.55, "convenience": 0.92}},
]


def _home_retailer(rng: np.random.Generator, segment_name: str) -> str:
    aff = SEGMENT_AFFINITY[segment_name]
    ids = list(aff.keys())
    w = np.array([aff[i] for i in ids], dtype=float)
    w = w / w.sum()
    return str(rng.choice(ids, p=w))


def generate_population(n: int = 1000, seed: int = 42) -> list[dict]:
    """Draw a reproducible population of `n` shopper agents."""
    rng = np.random.default_rng(seed)
    shares = np.array([s["share"] for s in SEGMENTS], dtype=float)
    shares = shares / shares.sum()
    seg_idx = rng.choice(len(SEGMENTS), size=n, p=shares)

    agents: list[dict] = []
    for i in range(n):
        seg = SEGMENTS[int(seg_idx[i])]
        weights = {d: float(np.clip(rng.normal(seg["means"][d], 0.12), 0.02, 0.99))
                   for d in DIMENSIONS}
        budget = float(np.clip(rng.normal(seg["budget"], 0.6), 1.0, 9.0))
        agents.append({
            "id": i,
            "segment": seg["name"],
            "color": seg["color"],
            "weights": weights,
            "budget": budget,
            "home_retailer": _home_retailer(rng, seg["name"]),
        })
    return agents


def segment_table() -> list[dict]:
    return [{"name": s["name"], "share": s["share"], "color": s["color"],
             "blurb": s["blurb"]} for s in SEGMENTS]


def retailer_table() -> list[dict]:
    return RETAILERS
