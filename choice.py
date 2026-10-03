"""
The choice engine: where does a shopper buy this product?

Each shopper faces a set of OPTIONS for one product:
  - the product at Tesco (the price here is our lever),
  - the product at each competitor that stocks it (fixed prices from the db),
  - a few Tesco substitute lines (same need-state, for cannibalisation),
  - plus a "don't buy" outside option.

They pick the option with the highest utility (a random-utility / multinomial
logit model). For shopper i and option o:

    U_io = V_io + g_io            (g_io i.i.d. Gumbel noise -> logit choice)

    V_io =  Kprice * value        * (1 - price_norm_o)
          + Kdeal  * deal_seeking  * saving_o * salience_o   (any live promo, incl competitors')
          + Kret   * [ retailer_appeal_r + home_bonus*habit_loyalty if r is home ]
          + Kappeal* [ product_base + health/novelty/taste matched to the shopper ]
          + Ks     * social_proof  * social_proof_o          (updated live by the swarm)
          + Kconv  * convenience   * retailer_convenience_r
          - budget_penalty (if they cannot afford it)

Tesco's promo lowers the Tesco price (and adds an anchored saving), which pulls
price-sensitive, deal-seeking, low-loyalty shoppers across from competitors, and
some shoppers off their usual Tesco substitute (cannibalisation) or off "don't
buy" (new demand). All coefficients live in PARAMS.
"""
from __future__ import annotations

PARAMS = {
    "Kprice": 1.6,     # value for money
    "Kdeal": 1.8,      # deal / loss aversion
    "Kret": 1.5,       # retailer affinity / store loyalty
    "Kappeal": 1.4,    # product pull (brand + fit to the shopper)
    "Ks": 1.2,         # social proof
    "Kconv": 0.7,      # store convenience
    "home_bonus": 0.9, # extra pull of the shopper's usual store, scaled by loyalty
    "v_outside": 3.2,  # utility of not buying (the purchase bar for a niche product)
    "tau": 0.55,       # Gumbel noise scale
    "budget_penalty": 2.0,
}

# Tesco mechanic shaping (competitors' promos are already priced in the db).
PROMO_TYPES = {
    "none":      {"label": "No promotion",             "saving_salience": 1.0,  "units_required": 1},
    "price_cut": {"label": "Clubcard price cut",       "saving_salience": 1.4,  "units_required": 1},
    "multibuy":  {"label": "Clubcard multibuy (2 for)", "saving_salience": 1.15, "units_required": 2},
}


def perceived_saving(base: float, eff: float) -> float:
    return max(0.0, min(0.9, (base - eff) / base)) if base > 0 else 0.0


def utility_terms(agent: dict, opt: dict, social_proof_o: float,
                  price_max: float, retailers: dict, P: dict = PARAMS) -> dict:
    """Per-term contribution to V_io (also used for the drill-down)."""
    w = agent["weights"]
    r = retailers[opt["retailer"]]
    price_norm = min(1.0, opt["eff_price"] / price_max) if price_max > 0 else 1.0

    home = (opt["retailer"] == agent["home_retailer"])
    retailer_aff = r["appeal"] + (P["home_bonus"] * w["habit_loyalty"] if home else 0.0)

    appeal = opt["appeal_base"] + 0.5 * (
        w["health"] * opt["health"] + w["novelty"] * opt["novelty"] + w["taste"] * opt["taste"])

    total_cost = opt["eff_price"] * opt["units"]
    over = max(0.0, total_cost - agent["budget"])
    budget_pen = -P["budget_penalty"] * (over / max(agent["budget"], 0.5)) if over > 0 else 0.0

    return {
        "price":        P["Kprice"] * w["value"] * (1.0 - price_norm),
        "deal":         P["Kdeal"] * w["deal_seeking"] * opt["saving"] * opt["salience"],
        "retailer":     P["Kret"] * retailer_aff,
        "appeal":       P["Kappeal"] * appeal,
        "social_proof": P["Ks"] * w["social_proof"] * social_proof_o,
        "convenience":  P["Kconv"] * w["convenience"] * r["convenience"],
        "budget":       budget_pen,
    }


def deterministic_utility(agent: dict, opt: dict, social_proof_o: float,
                          price_max: float, retailers: dict, P: dict = PARAMS) -> float:
    return sum(utility_terms(agent, opt, social_proof_o, price_max, retailers, P).values())
