"""
Synthetic shopper population.

Each agent is a vector of behavioural weights (0-1 per dimension) plus a budget
and a "stated priority" (the thing they would SAY matters most to them). The
population is drawn from illustrative shopper segments whose shares and mean
weight vectors reflect well-known grocery segmentation patterns.

IMPORTANT (honesty): these segments are SYNTHETIC archetypes, not derived from
any proprietary loyalty-card panel. They are designed to be plausible and are
the single place to recalibrate the model against real panel data (see README).
"""
from __future__ import annotations

import numpy as np

# Behavioural dimensions the choice engine understands.
DIMENSIONS = [
    "value",          # value-for-money / price sensitivity
    "deal_seeking",   # pull of discounts & "was/now" framing (loss aversion)
    "social_proof",   # susceptibility to what others are buying (herding)
    "habit_loyalty",  # stick with the familiar / repeat purchase
    "health",         # health motivation
    "novelty",        # openness to new & challenger brands
    "taste",          # prioritise taste / quality / indulgence
    "convenience",    # format, effort, ease
]

# Dimensions an agent could plausibly *declare* as their top priority.
# Used to measure the say-do gap (what they claim vs what they buy).
DECLARABLE = ["value", "health", "taste", "novelty", "habit_loyalty"]

DECLARABLE_LABEL = {
    "value": "getting value for money",
    "health": "eating healthily",
    "taste": "taste and quality",
    "novelty": "trying new things",
    "habit_loyalty": "sticking with brands they trust",
}

SEGMENTS = [
    {
        "name": "Budget-Squeezed",
        "share": 0.26,
        "color": "#e4572e",
        "blurb": "Watching every penny. Lives for the yellow sticker.",
        "budget": 2.3,
        "means": {"value": 0.90, "deal_seeking": 0.85, "social_proof": 0.45,
                  "habit_loyalty": 0.55, "health": 0.32, "novelty": 0.25,
                  "taste": 0.50, "convenience": 0.55},
    },
    {
        "name": "Loyal Traditionalist",
        "share": 0.22,
        "color": "#4c6ef5",
        "blurb": "Buys what they always buy. Trust beats novelty.",
        "budget": 3.2,
        "means": {"value": 0.50, "deal_seeking": 0.45, "social_proof": 0.40,
                  "habit_loyalty": 0.92, "health": 0.45, "novelty": 0.15,
                  "taste": 0.62, "convenience": 0.55},
    },
    {
        "name": "Health-Conscious",
        "share": 0.18,
        "color": "#2fb380",
        "blurb": "Reads the label. Wants better-for-you.",
        "budget": 3.4,
        "means": {"value": 0.45, "deal_seeking": 0.40, "social_proof": 0.48,
                  "habit_loyalty": 0.45, "health": 0.92, "novelty": 0.58,
                  "taste": 0.55, "convenience": 0.45},
    },
    {
        "name": "Adventurous Foodie",
        "share": 0.14,
        "color": "#e8a33d",
        "blurb": "Chasing the next great taste. Price is secondary.",
        "budget": 4.1,
        "means": {"value": 0.35, "deal_seeking": 0.40, "social_proof": 0.55,
                  "habit_loyalty": 0.30, "health": 0.55, "novelty": 0.90,
                  "taste": 0.88, "convenience": 0.40},
    },
    {
        "name": "Trend-Follower",
        "share": 0.12,
        "color": "#b5559e",
        "blurb": "If it's everywhere, they want it. Driven by the crowd.",
        "budget": 3.3,
        "means": {"value": 0.45, "deal_seeking": 0.55, "social_proof": 0.92,
                  "habit_loyalty": 0.35, "health": 0.50, "novelty": 0.80,
                  "taste": 0.60, "convenience": 0.55},
    },
    {
        "name": "Convenience-Led",
        "share": 0.08,
        "color": "#6c8aa6",
        "blurb": "Grab and go. Whatever is easiest wins.",
        "budget": 3.3,
        "means": {"value": 0.55, "deal_seeking": 0.50, "social_proof": 0.50,
                  "habit_loyalty": 0.55, "health": 0.40, "novelty": 0.45,
                  "taste": 0.55, "convenience": 0.92},
    },
]


def _stated_priority(weights: dict) -> str:
    """The dimension the agent would declare as mattering most to them."""
    return max(DECLARABLE, key=lambda d: weights[d])


def generate_population(n: int = 600, seed: int = 42) -> list[dict]:
    """Draw a reproducible population of `n` shopper agents."""
    rng = np.random.default_rng(seed)
    shares = np.array([s["share"] for s in SEGMENTS], dtype=float)
    shares = shares / shares.sum()
    seg_idx = rng.choice(len(SEGMENTS), size=n, p=shares)

    agents: list[dict] = []
    for i in range(n):
        seg = SEGMENTS[int(seg_idx[i])]
        weights = {}
        for d in DIMENSIONS:
            w = rng.normal(seg["means"][d], 0.12)
            weights[d] = float(np.clip(w, 0.02, 0.99))
        budget = float(np.clip(rng.normal(seg["budget"], 0.6), 1.0, 7.0))
        agents.append({
            "id": i,
            "segment": seg["name"],
            "color": seg["color"],
            "weights": weights,
            "budget": budget,
            "stated_priority": _stated_priority(weights),
        })
    return agents


def segment_table() -> list[dict]:
    """Public description of the segments (for UI / transparency)."""
    return [
        {"name": s["name"], "share": s["share"], "color": s["color"],
         "blurb": s["blurb"]}
        for s in SEGMENTS
    ]
