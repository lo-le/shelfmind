"""
The interacting swarm.

Agents choose one at a time in a seeded random order. After each choice the
product's *social proof* is updated, so later shoppers see what earlier shoppers
bought. Social-proof-susceptible agents herd toward popular products, which can
produce tipping points — a challenger brand that is ignored at low visibility
can run away once it crosses a threshold. This emergence is the whole point of
running a swarm rather than N independent draws.

We score each scenario twice — baseline (no promo) and with the promo — reusing
the SAME Gumbel noise per agent (the "common random numbers" variance-reduction
trick). So the difference between the two runs is the causal effect of the
promo, not noise, and we can trace exactly which shoppers switched and from
where (true incremental vs cannibalised vs stolen-from-competitor vs pantry).
"""
from __future__ import annotations

import numpy as np

import choice as C
from personas import DECLARABLE_LABEL

PRIOR_WEIGHT = 25.0      # strength of the starting social-proof prior
SNAPSHOTS = 24          # trajectory samples for the tipping-point chart


def _gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.gumbel(0.0, 1.0, size=size)


def _run_pass(agents, products, promo_target, promo, max_price, seed):
    """One swarm pass. Returns per-agent choices + a share trajectory."""
    feats = []
    for p in products:
        this_promo = promo if (p["id"] == promo_target) else {"type": "none", "depth": 0.0}
        feats.append(C.apply_promo(p, this_promo))
    n_products = len(feats)

    counts = np.zeros(n_products, dtype=float)
    priors = np.array([f["base_share"] for f in feats], dtype=float)
    buyers = 0.0

    order = np.random.default_rng(seed).permutation(len(agents))
    snap_every = max(1, len(agents) // SNAPSHOTS)
    trajectory = []

    choices = [None] * len(agents)  # product index or -1 for no-buy
    for step, ai in enumerate(order):
        agent = agents[int(ai)]
        # live social proof = blend of running share and the starting prior
        sp = (counts + priors * PRIOR_WEIGHT) / (buyers + PRIOR_WEIGHT)

        # paired Gumbel noise: same stream for this agent across baseline & promo
        g = _gumbel(np.random.default_rng(seed * 100003 + int(ai)), n_products + 1)
        utils = np.empty(n_products + 1)
        for j, f in enumerate(feats):
            utils[j] = C.deterministic_utility(agent, f, sp[j], max_price) + C.PARAMS["tau"] * g[j]
        utils[n_products] = C.PARAMS["v_outside"] + C.PARAMS["tau"] * g[n_products]

        pick = int(np.argmax(utils))
        if pick == n_products:
            choices[int(ai)] = -1
        else:
            choices[int(ai)] = pick
            counts[pick] += 1.0
            buyers += 1.0

        if step % snap_every == 0 or step == len(order) - 1:
            denom = max(buyers, 1.0)
            trajectory.append({
                "n": step + 1,
                "shares": {feats[j]["id"]: float(counts[j] / denom) for j in range(n_products)},
            })

    return {"feats": feats, "choices": choices, "counts": counts,
            "buyers": buyers, "trajectory": trajectory}


def simulate(agents, products, promo_target, promo, seed=7):
    """Run baseline + promo passes and build the full result payload."""
    max_price = max(float(p.get("ref_price", p["price"])) for p in products)
    pid = {i: p["id"] for i, p in enumerate(products)}
    owner = {i: p.get("owner", p["brand"]) for i, p in enumerate(products)}
    names = {i: p["name"] for i, p in enumerate(products)}

    base = _run_pass(agents, products, None, {"type": "none", "depth": 0.0}, max_price, seed)
    test = _run_pass(agents, products, promo_target, promo, max_price, seed)

    tgt = next(i for i, p in enumerate(products) if p["id"] == promo_target)

    n = len(agents)
    def units(run):
        u = {pid[j]: int(run["counts"][j]) for j in range(len(products))}
        u["__nobuy__"] = int(n - run["buyers"])
        return u

    base_u, test_u = units(base), units(test)

    # --- switching decomposition for the promoted product ----------------
    gained_from = {"new_to_category": 0, "cannibalised": 0, "from_competitor": 0}
    lost = 0
    switchers = []  # sample of shoppers who moved onto the promo
    for ai in range(n):
        b, t = base["choices"][ai], test["choices"][ai]
        if t == tgt and b != tgt:
            if b == -1:
                gained_from["new_to_category"] += 1
            elif owner[b] == owner[tgt]:
                gained_from["cannibalised"] += 1
            else:
                gained_from["from_competitor"] += 1
            if len(switchers) < 40:
                switchers.append({"agent": ai, "from": "walked away" if b == -1 else names[b]})
        elif b == tgt and t != tgt:
            lost += 1

    incremental = base_u["__nobuy__"] - test_u["__nobuy__"]  # category growth
    target_gain = test_u[pid[tgt]] - base_u[pid[tgt]]

    # --- say-do gap -------------------------------------------------------
    def say_do(run):
        contradictions, counts_by = 0, 0
        by_priority = {}
        cheapest = min(f["eff_price"] for f in run["feats"])
        for ai in range(n):
            c = run["choices"][ai]
            if c == -1:
                continue
            counts_by += 1
            sp = agents[ai]["stated_priority"]
            f = run["feats"][c]
            if sp == "value":
                # says price matters most, but paid >20% more than the cheapest option
                contradiction = f["eff_price"] > 1.2 * cheapest
            elif sp == "habit_loyalty":
                contradiction = f["familiarity"] < 0.45
            else:  # health / taste / novelty are product scores
                contradiction = f[sp] < 0.45
            by_priority.setdefault(sp, {"n": 0, "gap": 0})
            by_priority[sp]["n"] += 1
            if contradiction:
                contradictions += 1
                by_priority[sp]["gap"] += 1
        pct = (contradictions / counts_by * 100.0) if counts_by else 0.0
        return pct, by_priority

    base_gap, _ = say_do(base)
    test_gap, test_gap_by = say_do(test)

    # headline say-do example (largest contradicting group under the promo)
    say_do_headline = None
    best = None
    for sp, d in test_gap_by.items():
        if d["n"] >= 15 and (best is None or d["gap"] / d["n"] > best[1]):
            best = (sp, d["gap"] / d["n"], d["n"], d["gap"])
    if best:
        sp, rate, grp_n, grp_gap = best
        say_do_headline = {
            "stated": DECLARABLE_LABEL[sp],
            "pct": round(rate * 100.0),
            "group_n": grp_n,
        }

    # --- segment breakdown of who bought the promoted product -------------
    seg_gain = {}
    for ai in range(n):
        if test["choices"][ai] == tgt and base["choices"][ai] != tgt:
            s = agents[ai]["segment"]
            seg_gain[s] = seg_gain.get(s, 0) + 1

    # --- a few sample shoppers (with reasoning) for the drill-down --------
    samples = []
    for s in switchers[:6]:
        ai = s["agent"]
        agent = agents[ai]
        c = test["choices"][ai]
        f = test["feats"][c]
        sp_val = (test["counts"][c] + f["base_share"] * PRIOR_WEIGHT) / (max(test["buyers"], 1.0) + PRIOR_WEIGHT)
        terms = C.utility_terms(agent, f, sp_val, max_price)
        top = sorted(((k, v) for k, v in terms.items() if k != "budget"),
                     key=lambda kv: kv[1], reverse=True)[:3]
        samples.append({
            "segment": agent["segment"], "color": agent["color"],
            "stated_priority": DECLARABLE_LABEL[agent["stated_priority"]],
            "chose": f["name"], "came_from": s["from"],
            "top_reasons": [{"factor": k, "weight": round(float(v), 2)} for k, v in top],
        })

    # --- capped per-shopper choices for the swarm animation --------------
    disp = min(n, 420)
    def _pid(ci):
        return products[ci]["id"] if ci != -1 else "__nobuy__"
    viz = {
        "colors": [agents[i]["color"] for i in range(disp)],
        "segments": [agents[i]["segment"] for i in range(disp)],
        "base": [_pid(base["choices"][i]) for i in range(disp)],
        "promo": [_pid(test["choices"][i]) for i in range(disp)],
    }

    return {
        "promoted": {"id": products[tgt]["id"], "name": names[tgt],
                     "owner": owner[tgt]},
        "viz": viz,
        "promo": promo,
        "units": {"baseline": base_u, "promo": test_u},
        "target_gain": target_gain,
        "incremental_demand": incremental,         # net new buyers to the category
        "decomposition": {
            **gained_from,
            "defected": lost,
        },
        "say_do": {
            "baseline_pct": round(base_gap, 1),
            "promo_pct": round(test_gap, 1),
            "headline": say_do_headline,
        },
        "segment_gain": seg_gain,
        "trajectory": test["trajectory"],
        "baseline_trajectory": base["trajectory"],
        "samples": samples,
        "product_names": names,
        "n_agents": n,
    }
