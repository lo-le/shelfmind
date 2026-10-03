"""
The interacting swarm (retailer choice).

For one product, every shopper chooses where to buy it (Tesco, a competitor that
stocks it, a Tesco substitute line) or not at all. They choose one at a time in a
seeded order; each choice updates that option's social proof, so later shoppers
can herd toward whatever is already popular.

We run it twice with the SAME shoppers and the SAME Gumbel noise (common random
numbers): once with Tesco at base price, once with Tesco's promo. The difference
is the causal effect of the Tesco promo, and because we can see each shopper's
before and after choice we can split the promoted line's gain into:

  - cannibalisation   : moved off a Tesco substitute (stays within Tesco)
  - drawn from competitors : moved off another retailer (incremental to Tesco)
  - new uplift        : was not going to buy at all (incremental demand)

Net incremental to Tesco = drawn from competitors + new uplift.
"""
from __future__ import annotations

import numpy as np

import choice as C
from personas import RETAILERS, RETAILERS_BY_ID

PRIOR_WEIGHT = 25.0
SUB_ATTR = {"health": 0.35, "novelty": 0.25, "taste": 0.55}  # generic substitute feel


def _competitor_eff(c: dict) -> tuple[float, float, float]:
    """Return (eff_price, saving, salience) for a competitor offer."""
    base = float(c["base"])
    promo = c.get("promo")
    if promo:
        eff = float(promo["price"])
        return eff, C.perceived_saving(base, eff), 1.3
    return base, 0.0, 1.0


def build_options(product: dict, tesco_eff: float, tesco_saving: float,
                  tesco_salience: float, tesco_units: int) -> list[dict]:
    ap = product["appeal"]
    opts: list[dict] = []
    # 0: Tesco (the lever)
    opts.append({
        "retailer": "tesco", "kind": "promoted", "bin": "tesco",
        "label": "Tesco", "eff_price": tesco_eff, "saving": tesco_saving,
        "salience": tesco_salience, "units": tesco_units,
        "appeal_base": ap["base"], "health": ap["health"], "novelty": ap["novelty"],
        "taste": ap["taste"], "prior": RETAILERS_BY_ID["tesco"]["market_share"],
    })
    # competitors that stock it
    for r in RETAILERS:
        if r["id"] == "tesco":
            continue
        c = product["competitors"].get(r["id"])
        if not c:
            continue
        eff, sav, sal = _competitor_eff(c)
        opts.append({
            "retailer": r["id"], "kind": "promoted", "bin": r["id"],
            "label": r["name"], "eff_price": eff, "saving": sav, "salience": sal,
            "units": 1, "appeal_base": ap["base"], "health": ap["health"],
            "novelty": ap["novelty"], "taste": ap["taste"], "prior": r["market_share"],
        })
    # Tesco substitutes (for cannibalisation)
    maxvol = max((s.get("volume", 100) for s in product["substitutes"]), default=100)
    for s in product["substitutes"]:
        opts.append({
            "retailer": "tesco", "kind": "substitute", "bin": "subs",
            "label": s["name"], "eff_price": float(s["price"]), "saving": 0.0,
            "salience": 1.0, "units": 1, "appeal_base": float(s["pull"]),
            "health": SUB_ATTR["health"], "novelty": SUB_ATTR["novelty"],
            "taste": SUB_ATTR["taste"], "prior": 0.15 * s.get("volume", 100) / maxvol,
        })
    return opts


def _tesco_offer(product: dict, promo: dict | None):
    """Return (eff_price, saving, salience, units) for Tesco under a promo."""
    base = float(product["tesco_base"])
    ptype = (promo or {}).get("type", "none")
    spec = C.PROMO_TYPES[ptype]
    if ptype == "none":
        return base, 0.0, 1.0, 1
    depth = float((promo or {}).get("depth", 0.0))
    eff = base * (1.0 - depth)
    return eff, C.perceived_saving(base, eff), spec["saving_salience"], spec["units_required"]


def _run_pass(agents, options, price_max, seed):
    n_opt = len(options)
    counts = np.zeros(n_opt, dtype=float)
    priors = np.array([o["prior"] for o in options], dtype=float)
    buyers = 0.0
    order = np.random.default_rng(seed).permutation(len(agents))
    choices = [None] * len(agents)
    for ai in order:
        agent = agents[int(ai)]
        sp = (counts + priors * PRIOR_WEIGHT) / (buyers + PRIOR_WEIGHT)
        g = np.random.default_rng(seed * 100003 + int(ai)).gumbel(0.0, 1.0, n_opt + 1)
        utils = np.empty(n_opt + 1)
        for j, o in enumerate(options):
            utils[j] = C.deterministic_utility(agent, o, sp[j], price_max, RETAILERS_BY_ID) + C.PARAMS["tau"] * g[j]
        utils[n_opt] = C.PARAMS["v_outside"] + C.PARAMS["tau"] * g[n_opt]
        pick = int(np.argmax(utils))
        if pick == n_opt:
            choices[int(ai)] = -1
        else:
            choices[int(ai)] = pick
            counts[pick] += 1.0
            buyers += 1.0
    return {"choices": choices, "counts": counts, "buyers": buyers}


def simulate(product: dict, agents: list[dict], promo: dict, seed: int = 7):
    n = len(agents)
    sub_prices = [float(s["price"]) for s in product["substitutes"]]
    comp_bases = [float(c["base"]) for c in product["competitors"].values() if c]
    price_max = max([float(product["tesco_base"])] + comp_bases + sub_prices)

    base_opts = build_options(product, float(product["tesco_base"]), 0.0, 1.0, 1)
    t_eff, t_sav, t_sal, t_units = _tesco_offer(product, promo)
    promo_opts = build_options(product, t_eff, t_sav, t_sal, t_units)

    base = _run_pass(agents, base_opts, price_max, seed)
    test = _run_pass(agents, promo_opts, price_max, seed)

    bins = ["tesco"] + [o["retailer"] for o in base_opts if o["bin"] not in ("tesco", "subs")] + ["subs", "__nobuy__"]
    opt_bin = {j: o["bin"] for j, o in enumerate(base_opts)}

    def bin_counts(run):
        bc = {b: 0 for b in bins}
        for ci in run["choices"]:
            bc[opt_bin[ci] if ci != -1 else "__nobuy__"] += 1
        return bc

    base_bins, promo_bins = bin_counts(base), bin_counts(test)

    # --- decomposition of Tesco's promoted line (option index 0) ----------
    cannibalised = 0
    new_uplift = 0
    defected = 0
    from_competitor = {}
    seg_gain = {}
    switchers = []
    for ai in range(n):
        b, t = base["choices"][ai], test["choices"][ai]
        if t == 0 and b != 0:
            if b == -1:
                new_uplift += 1
            elif base_opts[b]["kind"] == "substitute":
                cannibalised += 1
            else:
                rid = base_opts[b]["retailer"]
                from_competitor[rid] = from_competitor.get(rid, 0) + 1
            seg_gain[agents[ai]["segment"]] = seg_gain.get(agents[ai]["segment"], 0) + 1
            if len(switchers) < 40:
                switchers.append({"agent": ai, "from_bin": "did not buy" if b == -1 else base_opts[b]["label"]})
        elif b == 0 and t != 0:
            defected += 1

    tesco_base_units = int(base["counts"][0])
    tesco_promo_units = int(test["counts"][0])
    gain = tesco_promo_units - tesco_base_units
    comp_total = sum(from_competitor.values())
    net_incremental = comp_total + new_uplift

    # --- price table (static, from the db) --------------------------------
    price_table = [{
        "retailer": "Tesco", "id": "tesco", "stocked": True,
        "base": float(product["tesco_base"]),
        "promo": round(t_eff, 2) if promo and promo.get("type", "none") != "none" else None,
        "is_tesco": True,
    }]
    for r in RETAILERS:
        if r["id"] == "tesco":
            continue
        c = product["competitors"].get(r["id"])
        if not c:
            price_table.append({"retailer": r["name"], "id": r["id"], "stocked": False,
                                "base": None, "promo": None, "is_tesco": False})
        else:
            promo_price = float(c["promo"]["price"]) if c.get("promo") else None
            price_table.append({"retailer": r["name"], "id": r["id"], "stocked": True,
                                "base": float(c["base"]), "promo": promo_price, "is_tesco": False})

    # --- sample switchers with reasoning ----------------------------------
    samples = []
    for s in switchers[:6]:
        ai = s["agent"]
        agent = agents[ai]
        sp0 = (test["counts"][0] + promo_opts[0]["prior"] * PRIOR_WEIGHT) / (max(test["buyers"], 1.0) + PRIOR_WEIGHT)
        terms = C.utility_terms(agent, promo_opts[0], sp0, price_max, RETAILERS_BY_ID)
        top = sorted(((k, v) for k, v in terms.items() if k != "budget"),
                     key=lambda kv: kv[1], reverse=True)[:3]
        samples.append({
            "segment": agent["segment"], "color": agent["color"],
            "home": RETAILERS_BY_ID[agent["home_retailer"]]["name"],
            "came_from": s["from_bin"],
            "top_reasons": [{"factor": k, "weight": round(float(v), 2)} for k, v in top],
        })

    # --- viz: one dot per agent, bin = where they bought ------------------
    def _bin(ci):
        return opt_bin[ci] if ci != -1 else "__nobuy__"
    viz = {
        "colors": [agents[i]["color"] for i in range(n)],
        "base": [_bin(base["choices"][i]) for i in range(n)],
        "promo": [_bin(test["choices"][i]) for i in range(n)],
    }

    bin_labels = {"tesco": "Tesco", "subs": "Tesco substitutes", "__nobuy__": "Did not buy"}
    for r in RETAILERS:
        bin_labels.setdefault(r["id"], r["name"])

    return {
        "product": {"id": product["id"], "name": product["name"], "category": product["category"]},
        "promo": promo,
        "tesco_units": {"baseline": tesco_base_units, "promo": tesco_promo_units, "gain": gain},
        "decomposition": {
            "cannibalised": cannibalised,
            "from_competitor": from_competitor,
            "from_competitor_total": comp_total,
            "new_uplift": new_uplift,
            "defected": defected,
            "gross_inflow": cannibalised + comp_total + new_uplift,
            "net_incremental": net_incremental,
        },
        "price_table": price_table,
        "segment_gain": seg_gain,
        "before_after_bins": {"baseline": base_bins, "promo": promo_bins},
        "bins": bins,
        "bin_labels": bin_labels,
        "samples": samples,
        "viz": viz,
        "n_agents": n,
    }
