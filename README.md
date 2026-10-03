# ShelfMind - Synthetic Shopper Swarm

**EAT_HACK 2026 · Human Truth track**

> Pre-test a Tesco promo against the big-6 before you spend a penny on it, and
> see where the volume really comes from.

ShelfMind simulates a population of behaviourally-distinct shopper agents
deciding **where to buy a product** - Tesco, or a competitor supermarket
(Sainsbury's, Asda, Morrisons, Aldi, Lidl), or a Tesco substitute line, or not at
all. You put a Tesco Clubcard promo on one product and watch the swarm move, then
read the only number that matters to a promotions analyst: **how much of the
uplift is actually incremental to Tesco** versus cannibalised from its own lines.

![Human Truth track](https://img.shields.io/badge/track-Human%20Truth-2fe6b0)

## Why this is more than a prompt wrapper

The intelligence is an **agent-based simulation with a transparent choice
model**, not an LLM. The language model only narrates figures the simulation has
already computed, and is told to invent nothing.

- **Heterogeneous agents.** Each shopper is a vector of behavioural weights
  (value, deal-seeking/loss-aversion, social proof, store loyalty, health,
  novelty, taste, convenience) drawn from illustrative grocery segments, plus a
  **home retailer** sampled from that segment's store affinity.
- **They interact.** Social-proof-driven shoppers re-weight toward whatever
  earlier shoppers already chose, so the swarm can herd.
- **Clean causal read-out.** Baseline (Tesco at base price) and promo runs reuse
  the same Gumbel noise per agent (common random numbers), so the difference is
  the promo's causal effect and we can trace every switcher's origin.

## The decomposition (what the job actually needs)

Competitors here are **retailers**, not brands. When Tesco's promo grows the
promoted line, ShelfMind splits that gain into:

- **Cannibalised** - moved off a Tesco substitute line. Stays within Tesco; a
  margin give-away, not incremental.
- **Won from competitors** - moved off another retailer (by retailer). Truly
  incremental to Tesco.
- **New demand** - was not going to buy at all. Incremental.

**Net incremental to Tesco = won from competitors + new demand.**

## The maths (the bit judges asked us to show)

Each shopper faces the set of options for one product (the product at each
retailer that stocks it, plus Tesco substitutes) and a "don't buy" outside
option, and picks the highest-utility one (a random-utility / multinomial logit
model). For shopper *i* and option *o*:

```
U_io = V_io + g_io            (g_io i.i.d. Gumbel noise -> logit choice)

V_io =  Kprice * value       * (1 - price_norm_o)
      + Kdeal  * deal_seeking * saving_o * salience_o      (any live promo, incl competitors')
      + Kret   * [ retailer_appeal_r + home_bonus*loyalty if r is the shopper's home store ]
      + Kappeal* [ product_pull + health/novelty/taste matched to the shopper ]
      + Ks     * social_proof * social_proof_o             (updated live by the swarm)
      + Kconv  * convenience  * retailer_convenience_r
      - budget_penalty (if they cannot afford it)
```

Pricing uses Tesco's terms: **base price** (no promo), **shelf price** (Clubcard
price cut), **best price** (per-unit multibuy). Every Tesco offer is modelled as a
Clubcard price, so the saving is anchored against the base price (a built-in
loss-aversion frame). All coefficients live in `choice.py -> PARAMS`.

## What it outputs

- **Extra Tesco units** of the promoted line (baseline to promo), split into
  cannibalised / won from competitors / new demand, with **net incremental**.
- **Competitor price table** - each big-6 retailer's current base and promo
  price for the product (and where it is not stocked).
- **Before to after by retailer** - who lost the shoppers Tesco gained.
- **Switchers by segment**, and sampled shoppers with their decision reasoning.
- A plain-English **AI explanation** tying the figures to behavioural drivers.

## Run it

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # then add your OPENAI_API_KEY
python app.py
# open http://127.0.0.1:8000
```

The swarm runs on deterministic maths and needs no API key; the key only powers
the AI explanation.

## Architecture

| File | Role |
| --- | --- |
| `personas.py` | Draws the population: behavioural weights + a home retailer per segment. |
| `choice.py` | The choice engine: random-utility maths and Tesco promo mechanics. |
| `swarm.py` | The interacting swarm, paired baseline/promo runs, and the cannibalised / competitor / new-demand decomposition. |
| `explain.py` | LLM narration layer (OpenAI), narrates numbers only. |
| `app.py` | FastAPI backend: `/api/simulate` and `/api/narrate`. |
| `templates/index.html` | Single-page UI: product picker, Tesco promo lever, swarm, decomposition, price table. |
| `data/products.json` | The RGC products: Tesco base price, competitor prices/promos, Tesco substitutes. |
| `data/retailers.json` | The big-6 retailers and each segment's store affinity. |

## Data & honesty

The product set is real (RGC's EAT_HACK brands), but the **prices and shopper
model are synthetic placeholders**: competitor prices in `data/products.json` are
plausible, not live-scraped, and the segments are illustrative archetypes, not a
loyalty-card panel. Results are **directionally useful, not a forecast**.

**To make it real:** load live competitor prices into `data/products.json`, and
calibrate the segment weights and `choice.py -> PARAMS` to real Clubcard /
Dunnhumby choice data, then validate predicted lifts against historical promos.

## Beyond the demo

- **Scale.** The choice model is pure maths; the agent loop vectorises to tens of
  thousands of agents. Social proof is the only sequential step.
- **Privacy.** No personal data: agents are synthetic. Calibration would use
  aggregate, anonymised panel statistics, not individual records.
- **Cost.** The swarm is free and instant; the LLM is one call per run, for
  narration only.
- **Limitations.** Synthetic prices and personas; competitors are static (no
  simultaneous competitor promo or reaction); a single shopping occasion; social
  proof is a simple herding rule.

## Built at EAT_HACK

Fresh build on the day. Stack: Python, FastAPI, NumPy, vanilla JS (hand-rolled
Canvas/SVG, no chart libraries), OpenAI for the narration layer only.
