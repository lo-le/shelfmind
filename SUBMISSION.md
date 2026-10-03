# EAT_HACK submission pack - ShelfMind

**Project name:** ShelfMind - Synthetic Shopper Swarm
**Track:** Human Truth
**Team:** solo
**Repo:** https://github.com/lo-le/shelfmind

---

## Project description (paste into the form - ~180 words)

ShelfMind is a synthetic shopper swarm: an agent-based simulator that lets a
Tesco promotions analyst pre-test a Clubcard promo before committing spend, and
see whether it actually wins business from competitors or just discounts shoppers
Tesco already had.

Pick one product. Up to 1,500 behaviourally-distinct shopper agents each decide
**where** to buy it - Tesco, a competitor supermarket (Sainsbury's, Asda,
Morrisons, Aldi, Lidl), a Tesco substitute line, or not at all - using price,
store loyalty, the deal, product appeal and social proof. Competitors hold their
current prices; the Tesco Clubcard promo is the only lever.

By running a baseline and the promo with common random numbers, ShelfMind splits
the uplift into what is **cannibalised** from Tesco's own lines, **won from
competitors** (by retailer), and **new demand** - so net incremental is explicit.
Most promos look like a win while mostly shuffling existing demand; ShelfMind
exposes that in seconds. The swarm is deterministic maths; an LLM only narrates
the numbers.

*Pre-existing work: none - built fresh at EAT_HACK.*

---

## 2-minute video script (shot list)

Record the screen at http://127.0.0.1:8000 with voiceover. Keep it under 2:00.

| Time | Say | Show |
| --- | --- | --- |
| 0:00-0:15 | "Most retail promotions look like a win, but a lot of them just discount shoppers you already had. I'm a Tesco Price & Promotions analyst, and proving whether a promo truly wins business from competitors normally takes a live trial and weeks of SQL." | Title/hero of ShelfMind. |
| 0:15-0:35 | "ShelfMind pre-tests it on a swarm of 1,000 synthetic shoppers. I pick a real challenger product, and each shopper decides where to buy it: Tesco, or Sainsbury's, Asda, Morrisons, Aldi, Lidl - or not at all." | Pick a product; point at the segment legend and the competitor price table. |
| 0:35-1:05 | "Our only lever is the Tesco Clubcard price. I'll cut this from base to shelf price and run it. Watch the swarm: shoppers flow into Tesco from the competitors." | Set the Clubcard price cut, Run, let the swarm animate baseline to promo. |
| 1:05-1:30 | "Plus 262 units - but here's the truth. Only 68% is incremental: 132 units won from competitors, mostly Asda who's already at a dollar; 45 genuinely new. 85 is cannibalised from Tesco's own lines." | The metric cards and the 'where the uplift came from' bar. |
| 1:30-1:50 | "You can see exactly who moved and from where - Asda and the substitute shelf shrink as Tesco grows - and which shopper segments switched." | Before-to-after by retailer, switchers by segment, a sample shopper. |
| 1:50-2:00 | "Under the hood it's a transparent logit choice model and an interacting swarm, not a chatbot. Load real competitor prices and calibrate to Clubcard data, and you can test any promo in seconds. That's ShelfMind." | The 'how the swarm chooses' panel and the AI explanation. |

**Note:** figures above are illustrative; use whatever the live run shows.

---

## Submission checklist (deadline 17:30, to the minute)

- [x] **Public GitHub repo**: https://github.com/lo-le/shelfmind
- [ ] **Video URL** (<=2 min) uploaded somewhere public (YouTube/Loom public, or Drive "anyone with link").
- [ ] **Project description** pasted (above).
- [ ] **Track:** Human Truth.
- [ ] **Best Brand vote:** your three favourite brands from The Shelf.
- [ ] Live product URL - optional, skip (no extra points).

### Repo description (paste into the GitHub About box)

```
Agent-based swarm that pre-tests a Tesco Clubcard promo against the big-6, splitting uplift into cannibalised, won-from-competitors and new demand.
```
