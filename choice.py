"""
The choice engine — the maths the judges asked us to show.

Each shopper makes a discrete choice over the products on a shelf plus a
"walk away / no purchase" outside option, using a random-utility (multinomial
logit) model. For agent i and product j:

    U_ij = V_ij + g_ij

where g_ij is i.i.d. Gumbel noise (this is exactly what gives a logit choice
model) and the deterministic utility V_ij is a transparent weighted sum of
behavioural terms:

    V_ij =  Kv  * value        * (1 - price_norm)
          + Kd  * deal_seeking  * perceived_saving * saving_salience
          + Ks  * social_proof  * social_proof_j        <-- updated live by the swarm
          + Kh  * habit_loyalty * familiarity_j
          + Kl  * habit_loyalty * loyalty_bonus          <-- Clubcard-style price
          + Khe * health        * health_j
          + Kn  * novelty       * novelty_j
          + Kt  * taste         * taste_j
          + Kc  * convenience   * convenience_j
          - budget_penalty (if the shopper cannot afford it)

The shopper picks argmax over products and the outside option. Every coefficient
lives in PARAMS so the model is inspectable and tunable in one place.
"""
from __future__ import annotations

# Model coefficients (the "knobs"). Kept in one place on purpose.
PARAMS = {
    "Kv": 1.7,    # value for money
    "Kd": 1.9,    # deal / loss-aversion
    "Ks": 1.4,    # social proof
    "Kh": 1.4,    # habit / familiarity
    "Kl": 1.3,    # loyalty-price bonus
    "Khe": 1.25,  # health
    "Kn": 1.15,   # novelty
    "Kt": 1.45,   # taste / quality
    "Kc": 0.85,   # convenience
    "v_outside": 0.95,   # utility of walking away (the purchase bar)
    "tau": 0.55,         # Gumbel noise scale (choice stochasticity)
    "budget_penalty": 2.4,
}

# How each promo mechanic reshapes the offer. saving_salience amplifies the
# *perceived* saving (framing), loyalty_bonus is a members-only pull, and
# units_required drives pantry-loading / basket spend for multibuys.
PROMO_TYPES = {
    "none":        {"label": "No promotion",        "price_mult": 1.00, "saving_salience": 1.0, "loyalty_bonus": 0.0, "units_required": 1},
    "price_cut":   {"label": "Straight price cut",   "price_mult": None, "saving_salience": 1.0, "loyalty_bonus": 0.0, "units_required": 1},
    "was_now":     {"label": "'Was £x now £y'",      "price_mult": None, "saving_salience": 1.45, "loyalty_bonus": 0.0, "units_required": 1},
    "multibuy":    {"label": "Multibuy (2 for ...)", "price_mult": None, "saving_salience": 1.15, "loyalty_bonus": 0.0, "units_required": 2},
    "loyalty":     {"label": "Clubcard-style price", "price_mult": None, "saving_salience": 1.1, "loyalty_bonus": 1.0, "units_required": 1},
}


def apply_promo(product: dict, promo: dict | None) -> dict:
    """Return the product's effective offer features under a promo.

    promo = {"type": <PROMO_TYPES key>, "depth": 0..0.6}  (depth = fraction off)
    """
    spec = PROMO_TYPES[(promo or {}).get("type", "none")]
    depth = float((promo or {}).get("depth", 0.0)) if spec["price_mult"] is None else 0.0

    base_price = float(product["price"])
    ref_price = float(product.get("ref_price", base_price))
    if spec["price_mult"] is None:
        eff_price = base_price * (1.0 - depth)
    else:
        eff_price = base_price * spec["price_mult"]
        ref_price = base_price  # no promo -> reference is just shelf price

    perceived_saving = max(0.0, min(0.9, (ref_price - eff_price) / ref_price)) if ref_price > 0 else 0.0

    return {
        "id": product["id"],
        "name": product["name"],
        "brand": product["brand"],
        "owner": product.get("owner", product["brand"]),
        "is_incumbent": bool(product.get("is_incumbent", False)),
        "eff_price": eff_price,
        "ref_price": ref_price,
        "perceived_saving": perceived_saving,
        "saving_salience": spec["saving_salience"],
        "loyalty_bonus": spec["loyalty_bonus"] * (depth if depth else 0.0),
        "units_required": spec["units_required"],
        "taste": float(product["taste"]),
        "familiarity": float(product["familiarity"]),
        "health": float(product["health"]),
        "novelty": float(product["novelty"]),
        "convenience": float(product["convenience"]),
        "base_share": float(product.get("base_share", 0.0)),
        "promo_type": (promo or {}).get("type", "none"),
    }


def utility_terms(agent: dict, feat: dict, social_proof_j: float,
                  max_price: float, P: dict = PARAMS) -> dict:
    """Return the per-term contributions to V_ij (used for drill-down)."""
    w = agent["weights"]
    price_norm = min(1.0, feat["eff_price"] / max_price) if max_price > 0 else 1.0
    total_cost = feat["eff_price"] * feat["units_required"]
    over = max(0.0, total_cost - agent["budget"])
    budget_pen = -P["budget_penalty"] * (over / max(agent["budget"], 0.5)) if over > 0 else 0.0

    terms = {
        "value":        P["Kv"] * w["value"] * (1.0 - price_norm),
        "deal":         P["Kd"] * w["deal_seeking"] * feat["perceived_saving"] * feat["saving_salience"],
        "social_proof": P["Ks"] * w["social_proof"] * social_proof_j,
        "habit":        P["Kh"] * w["habit_loyalty"] * feat["familiarity"],
        "loyalty_deal": P["Kl"] * w["habit_loyalty"] * feat["loyalty_bonus"],
        "health":       P["Khe"] * w["health"] * feat["health"],
        "novelty":      P["Kn"] * w["novelty"] * feat["novelty"],
        "taste":        P["Kt"] * w["taste"] * feat["taste"],
        "convenience":  P["Kc"] * w["convenience"] * feat["convenience"],
        "budget":       budget_pen,
    }
    return terms


def deterministic_utility(agent: dict, feat: dict, social_proof_j: float,
                          max_price: float, P: dict = PARAMS) -> float:
    return sum(utility_terms(agent, feat, social_proof_j, max_price, P).values())
