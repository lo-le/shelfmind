# ShelfMind — Synthetic Shopper Swarm

**EAT_HACK 2026 · Human Truth track**

> Pre-test a promotion — or a new challenger brand — on a thousand synthetic
> shoppers *before you spend a penny on the real one*, and see the human truths
> behind the result.

A promotion is a behavioural lever. ShelfMind simulates a population of
behaviourally-distinct shopper agents choosing on a shelf, then shows not just
the predicted sales lift but **why** it happens: how much of the uplift is
genuinely incremental vs cannibalised from the brand's own lines vs stolen from
competitors, which shopper segments moved, and the **say-do gap** between what
shoppers claim to value and what they actually buy.

![Human Truth track](https://img.shields.io/badge/track-Human%20Truth-2fe6b0)

## Why this is more than a prompt wrapper

The intelligence is an **agent-based simulation with a transparent choice
model** — not an LLM. The language model is used *only* to narrate numbers the
simulation has already computed, and is instructed to invent nothing.

- **Heterogeneous agents.** Each shopper is a vector of behavioural weights
  (value, loss-aversion/deal-seeking, social proof, habit, health, novelty,
  taste, convenience) drawn from illustrative grocery segments.
- **They interact.** Social-proof-driven shoppers re-weight toward whatever
  earlier shoppers already chose, so the swarm can **tip** — a challenger brand
  ignored at low visibility can run away once it crosses a threshold. This
  emergence is why we run a *swarm*, not N independent draws.
- **Clean causal read-out.** Baseline (no promo) and promo runs reuse the same
  Gumbel noise per agent (*common random numbers*), so the difference between
  the two runs is the promo's causal effect and we can trace exactly which
  shoppers switched and from where.

## The maths (the bit judges asked us to show)

Each shopper makes a discrete choice over the products plus a "walk away"
outside option using a random-utility (multinomial logit) model. For shopper *i*
and product *j*:

```
U_ij = V_ij + g_ij            (g_ij = i.i.d. Gumbel noise  →  logit choice)

V_ij =  Kv ·value        ·(1 − price_norm)
      + Kd ·deal_seeking ·perceived_saving ·saving_salience   ← promo framing
      + Ks ·social_proof ·social_proof_j                       ← updated live by the swarm
      + Kh ·habit_loyalty·familiarity_j
      + Kl ·habit_loyalty·loyalty_bonus                        ← Clubcard-style price
      + Khe·health       ·health_j
      + Kn ·novelty      ·novelty_j
      + Kt ·taste        ·taste_j
      + Kc ·convenience  ·convenience_j
      − budget_penalty (if they can't afford it)
```

The shopper picks `argmax` over products and the outside option. Every
coefficient lives in one place (`choice.py → PARAMS`) so the model is
inspectable and tunable. Different promo mechanics reshape the offer in
behaviourally distinct ways (a "was/now" frame amplifies *perceived* saving more
than a plain cut of the same depth; a multibuy lowers unit price but loads the
basket; a loyalty price mainly pulls habitual shoppers).

## What it outputs

- **Extra units**, and the honest split: **truly incremental** (new to category)
  vs **cannibalised** (own lines) vs **stolen from competitors**.
- **Say-do gap** — e.g. the % of shoppers who say value/health matters most but
  bought against it, and how the promo widens or narrows that gap.
- **Which segments** the promo won, and a **tipping-dynamics** chart of the
  promoted product's share as the swarm grows (baseline vs promo).
- **Sample shoppers** with their actual decision reasoning.
- A plain-English **"human truth"** narration tying the figures to named
  behavioural principles.

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
the natural-language "human truth" narration.

## Architecture

| File | Role |
| --- | --- |
| `personas.py` | Draws a reproducible population from behavioural segments. |
| `choice.py` | The choice engine — the random-utility maths and promo mechanics. |
| `swarm.py` | The interacting swarm, paired baseline/promo runs, and the switching decomposition + say-do metrics. |
| `explain.py` | LLM narration layer (OpenAI) — narrates numbers only. |
| `app.py` | FastAPI backend: `/api/simulate` and `/api/narrate`. |
| `templates/index.html` | Single-page UI: animated swarm, metrics, charts, narrative. |
| `data/shelf.json` | Editable demo shelves (swap in real brands). |

## Data & honesty

There is **no proprietary data here**. The shopper segments are *synthetic
archetypes* informed by well-known grocery segmentation patterns — plausible,
but not calibrated to any loyalty-card panel. Results are **directionally
useful, not a forecast**.

**To make it real beyond the hack:** calibrate the segment weight vectors and
the model coefficients to real loyalty-card choice data (e.g. Dunnhumby /
Tesco Clubcard panels), then validate predicted lifts against historical
promotion outcomes. `personas.py` and `choice.py → PARAMS` are the single places
to recalibrate.

## Beyond the demo (scale, privacy, cost, limits)

- **Scale.** The choice model is pure maths; the agent loop vectorises to tens of
  thousands of agents. Social proof is the only sequential step and can be
  batched into rounds.
- **Privacy.** The simulation uses no personal data — agents are synthetic, so
  there is no PII to leak. Calibration would run on aggregate, anonymised panel
  statistics, not individual records.
- **Cost.** The swarm is free and instant. The LLM is called once per run for
  narration only, so cost is negligible and the core product works offline.
- **Limitations.** Un-calibrated personas; a single shopping occasion (no
  long-run habit formation or stockpiling across trips); within-category choice
  only; social proof is a simple herding rule, not a full network model.

## Built at EAT_HACK

Fresh build on the day. No substantial pre-existing work. Stack: Python,
FastAPI, NumPy, vanilla JS (hand-rolled SVG/Canvas, no chart libraries), OpenAI
for the narration layer only.
