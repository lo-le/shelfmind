# EAT_HACK submission pack - ShelfMind

**Project name:** ShelfMind - Synthetic Shopper Swarm
**Track:** Human Truth
**Team:** (you) - solo

---

## Project description (paste into the form - ~175 words)

ShelfMind is a **synthetic shopper swarm**: an agent-based simulator that lets a
retailer or brand pre-test a promotion *before* spending real money on it.
Up to 1,500 behaviourally-distinct shopper agents (1,000 by default) each make a choice on a shelf
using a transparent random-utility model weighted by behavioural principles -
loss aversion, social proof, habit, health, novelty and value. The agents
interact: social-proof-driven shoppers herd, so the swarm can tip.

Crucially, ShelfMind doesn't just predict a sales lift - it explains **why** those
units moved, in behavioural terms. By running a baseline and a promo with common random
numbers, it decomposes the uplift into what is **truly incremental** vs
**cannibalised** from the brand's own lines vs **stolen from competitors**, shows
which segments moved, and surfaces the **say-do gap** between what shoppers claim
to value and what they actually buy.

This matters because most promotions look like a win while mostly shuffling
demand you already had - ShelfMind exposes that in seconds, not quarters. The
swarm is deterministic maths; an LLM only narrates the numbers.

*Pre-existing work: none - built fresh at EAT_HACK.*

---

## 2-minute video script (shot list)

Record the screen at http://127.0.0.1:8000 with voiceover. Keep it under 2:00.

| Time | Say | Show |
| --- | --- | --- |
| 0:00-0:15 | "Most retail promotions look like a win - but they just shuffle demand you already had. I'm a Tesco Price & Promotions analyst, and proving whether a promo is *actually* incremental normally takes a live trial and weeks of SQL." | Title/hero of ShelfMind. |
| 0:15-0:30 | "ShelfMind pre-tests a promotion on a swarm of synthetic shoppers - each one a different behavioural profile - before you spend a penny." | The shelf on the left; segment legend. |
| 0:30-1:05 | "I'll put a Clubcard price cut on this challenger bar, £1.60 down to £1.20, and run the swarm. Watch the shoppers choose, first with no promo, then with it." | Select PureBar → Clubcard price cut 25% → **Run**. Let the swarm animate baseline → promo. |
| 1:05-1:25 | "Plus 173 units - looks fantastic. But ShelfMind shows the truth: only **1% is genuinely new to the category**. 84% was stolen from competitors, and 14% cannibalised the brand's own sibling line." | The five metric cards. |
| 1:25-1:45 | "And here's what's really going on - a third of shoppers who *say* value or health matters most bought against it, because the deal's framing triggered loss aversion. These are the exact segments that moved, and why." | Say-do gap card, segment bars, a sample-shopper card, the AI explanation panel. |
| 1:45-2:00 | "Under the hood it's a transparent logit choice model and an interacting swarm that can tip on social proof - not a chatbot. Calibrate it to Clubcard data and you can test any promo, for any brand, in seconds. That's ShelfMind." | The tipping-dynamics chart; 'How the maths works' panel. |

**Tip:** before recording you can edit `data/shelf.json` to use the real
*The Shelf* brand names for extra relevance.

---

## Submission checklist (deadline 17:30, to the minute)

- [ ] **Public GitHub repo** pushed (see commands below) - must be public, no sign-in.
- [ ] **Video URL** (≤2 min) uploaded somewhere public (YouTube unlisted/public, Loom public, Drive "anyone with link").
- [ ] **Project description** pasted (above).
- [ ] **Track:** Human Truth.
- [ ] **Best Brand vote:** your three favourite brands from The Shelf.
- [ ] Live product URL - optional, skip (no extra points).

### Push the repo public

Create an empty **public** repo on GitHub, then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/shelfmind.git
git push -u origin main
```
