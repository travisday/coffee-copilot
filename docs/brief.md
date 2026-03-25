# Coffee Ops Copilot: Brief

## The Problem

An operations manager is responsible for keeping two coffee vending machines stocked. Today, that process is guesswork. Load machines based on yesterday's memory, a fixed routine, or gut feel. The cost of getting it wrong cuts both ways:

- **Overstock.** Machine capacity is wasted on products nobody orders in that window. Slots filled with low-demand items are slots unavailable for what actually sells.
- **Understock.** The machine runs out of Lattes at 10 a.m., the customer walks away, and the sale never shows up in the data.
- **Manual planning.** Interpreting raw numbers from a spreadsheet or model is time-consuming and difficult for non-technical users.

## The Solution

Coffee Ops Copilot is a natural-language AI assistant that answers one question: *"What should I stock, and when?"*

> **Manager:** "Plan tomorrow 7–10 a.m. for Machine 2."
>
> **Copilot:** "For Machine 2 tomorrow morning (06–09 and 09–12 windows), I'd suggest stocking **2 Lattes, 1 Americano, 1 Cappuccino, and 1 Americano with Milk.** This is a rough estimate. Machine 2 has limited history, so I'm using Machine 1's product mix as a starting point.
>
> Last Tuesday you bumped Lattes up by 1 for this window. Want me to keep that adjustment?"

The manager can confirm the recommendation, adjust quantities with inline sliders, or reset and start over. Every interaction takes seconds, not 10–15 minutes with a spreadsheet.

The interface addresses four AI-specific UX challenges:

- **Transparency.** Each recommendation shows its demand tier and the data behind it, including expected demand, mix proportion, and confidence level.
- **Uncertainty.** Sparse-data caveats and confidence labels ("Rough estimate") surface model limitations honestly.
- **Override.** Inline sliders let the manager adjust any product's stock level. An optional reason field captures *why*.
- **Feedback.** Every override is stored and surfaced in future sessions. The agent references past corrections conversationally.

### How It Works

A **Poisson GLM** forecasts total drink volume per 3-hour window per machine, using time bucket, machine identity, and a demand trend (growing 22%/year) as features. Machine 1's product-mix proportions then split that volume into a per-product stocking list. Machine 2 uses Machine 1's mix because its 260 transactions across 30 SKUs don't produce stable proportions. This is flagged transparently.

The agent turns a raw forecast into a stocking plan: it reasons that every product needs at least one unit, allocates proportionally across demand tiers, and presents an actionable recommendation. These are things a dashboard can't do.

### Why an Agent, Not a Dashboard

A spreadsheet with historical averages gets you 90% of the way to a forecast. The agent's value is the layer *above* the model: contextual reasoning that reduces cognitive load. The model predicts low single-digit drinks per 3-hour window (1.6 for Machine 1, 1.1 for Machine 2). A dashboard displays that number and leaves the manager to interpret it. But a vending machine doesn't stock "1.6 drinks," it stocks *products*. The agent bridges that gap:

- **From forecast to plan.** Volume predictions become per-product stocking levels with demand tiers, not just a number to interpret.
- **Conversation context.** The manager can ask follow-ups, drill into a specific machine, or change the time window without starting over.
- **Override memory.** Past corrections are surfaced conversationally so the agent learns from the manager's expertise.
- **External signals (next).** Weather, holidays, local events. Context an agent can reason over that a static forecast never will.

### Evaluation

| Strategy | MAE (drinks/window) | 90% Prediction Interval Coverage |
| --- | --- | --- |
| **AI Copilot (Poisson GLM + trend)** | **1.50** | **93.4%** |
| Global Average | 1.51 | 89.9% |
| Yesterday's Actual | 1.95 | 69.6% |

The copilot matches the global average on point accuracy and improves prediction interval calibration (93.4% vs. 89.9%). A flat historical average can't adapt to growing demand. Reactive stocking ("just repeat yesterday") is measurably worse. But the model is one piece. The real value is what the agent does with it: turning a number into a stocking list the manager can act on, with honest uncertainty, contextual reasoning, and memory of their past decisions.

### What the Data Already Tells Us

**Machine 2 is underperforming.** It averages roughly one drink per 3-hour window (about two-thirds of Machine 1's rate, but spread across 30 SKUs instead of 8). The MVP already scopes recommendations to Machine 1's proven 8-product set. The honest recommendation isn't just "stock fewer products." It's "consider whether this machine is in the right location."

**Product catalog is stale.** The system recommends from the existing product set but has no visibility into whether that set is the right one. If lattes are trending down and a new category is trending up, the manager has no way to know from internal data alone.

---

## What's Next

The forecasting model is the foundation. The roadmap is about making the *agent* smarter.

**Smarter stocking:**

- **Weather as a tool call.** Temperature drives hot/cold drink demand. Adjust recommendations on cold mornings or hot afternoons.
- **Holiday and event calendars.** Foot traffic signals the model can't see but the agent can reason over.
- **Recency-weighted product mix.** Catch shifting preferences instead of relying on all-time averages.
- **Feedback loop.** Use consistent overrides as training signal to update the model's baseline automatically.

**Strategic recommendations:**

- **Machine placement.** Compare per-machine performance against location benchmarks and flag underperformers with a recommendation to relocate, not just restock.
- **Product catalog intelligence.** Connect to external trend data so the agent can flag declining products and suggest replacements.
- **SKU rationalization.** Recommend pruning products that don't sell, freeing machine slots for new experiments.

**Infrastructure:**

- **Location context per machine.** Machine 2's weekend-heavy pattern suggests a different location type than Machine 1. Encoding that context lets the agent frame recommendations differently.
- **Containerization and CI/CD.** Package the application for reliable deployment with automated testing and delivery pipelines.
- **Model drift monitoring.** Track forecast accuracy over time and alert when the model needs retraining.
- **Frontend/backend split.** Separate the UI from the API layer for independent scaling and development.
