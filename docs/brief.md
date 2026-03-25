# Coffee Ops Copilot: Brief

## The Problem

An operations manager is responsible for keeping two coffee vending machines stocked. Today, that process is guesswork. Load machines based on yesterday's memory, a fixed routine, or gut feel. The cost of getting it wrong cuts both ways:

- **Overstock.** Perishable ingredients expire. Machine capacity is wasted on products nobody orders in that window.
- **Understock.** The machine runs out of Lattes at 10 a.m., the customer walks away, and the sale never shows up in the data.
- **Manual planning.** Every minute spent scanning spreadsheets or second-guessing inventory is a minute not spent on higher-value work.

## What the Copilot Does

Coffee Ops Copilot is a natural-language AI assistant that answers one question. *"What should I stock, and when?"* Here's what that looks like in practice:

> **Manager:** "Plan tomorrow 7–10 a.m. for Machine 2."
>
> **Copilot:** "For Machine 2 tomorrow morning (06–09 and 09–12 windows), I'd suggest stocking **2 Lattes, 1 Americano, 1 Cappuccino, and 1 Americano with Milk.** This is a rough estimate. Machine 2 has limited history, so I'm using Machine 1's product mix as a starting point. You know this machine better than the model does.
>
> Last Tuesday you bumped Lattes up by 1 for this window. Want me to keep that adjustment?"

The manager can accept the recommendation, adjust quantities with inline sliders, or reject it entirely. Every interaction takes seconds, not the 10–15 minutes of scanning a spreadsheet and doing the math by hand.

The interface addresses four AI-specific UX challenges directly:

- **Transparency.** Each recommendation shows its demand tier (high / moderate / keep-stocked) and the data behind it, including expected demand, mix proportion, and confidence level.
- **Uncertainty.** Machine 2's sparse-data caveat and confidence labels ("Rough estimate") surface model limitations without burying the manager in statistics.
- **Override.** Inline sliders let the manager adjust any product's stock level before confirming. An optional reason field captures *why*.
- **Feedback.** Every override is stored and surfaced in future sessions ("Last Tuesday you bumped Lattes to 3 for this window"), so the copilot adapts to the manager's real-world judgment over time.

---

## How It Works

### The Model (Two Stages)

A Poisson GLM forecasts total drink volume per 3-hour window per machine, using hour-of-day, machine identity, and a demand trend as features (demand is growing ~22%/year). Historical product-mix proportions then split that volume into a per-product stocking list. When Machine 2's sparse data (~250 transactions vs. Machine 1's ~3,600) makes its own mix unreliable, the system falls back to Machine 1's proportions and flags it transparently.


| What the Data Showed                                                               | What the Manager Gets                                                                                                      |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 96.5% of per-product time slots have zero sales, too sparse to model individually | A reliable "how busy will it be" number they can act on immediately, not noisy per-product guesses of 0 or 1               |
| Machine 2's product mix is unstable due to limited data                            | A useful recommendation for Machine 2 instead of "insufficient data," clearly flagged so they can layer in what they know |
| Demand is growing ~22%/year (statistically significant)                            | Forecasts that adapt to today's volume instead of averaging over stale history                                             |


The full EDA, model selection rationale, and assumption checks are documented in the analysis notebook (`notebooks/eda.ipynb`).

### Why an Agent, Not a Dashboard

A forecasting model is useful. But a spreadsheet with historical averages gets you 90% of the way there. Our baselines proved it. The agent's value is in the layer *above* the model, contextual reasoning that reduces cognitive load.

Consider the core challenge. The model predicts ~1.5 drinks per machine per 3-hour window. A dashboard would display that number and leave the manager to figure out what to do with it. But a vending machine doesn't stock "1.5 drinks," it stocks *products*. And even when total volume is low, any of the 8 products could be the one a customer walks up and orders. The agent bridges that gap. It takes a low-volume forecast, reasons that every product still needs at least one unit stocked, allocates proportionally across demand tiers, and presents a complete stocking plan the manager can act on immediately. A raw model output of "1.5" doesn't tell you to stock one of everything. The agent does.

What the agent combines that a dashboard cannot:

- **From forecast to plan.** Volume predictions become per-product stocking levels with demand tiers, not just a number to interpret.
- **Conversation context.** The manager can ask follow-up questions, drill into a specific machine, or change the time window without starting over.
- **Override history.** "You bumped Lattes up last Monday for this window." Past corrections inform future recommendations.
- **Machine-specific caveats.** Machine 2's sparse data gets different framing and confidence labels than Machine 1's steady traffic.
- **External signals (next).** Weather APIs, holiday calendars, local event feeds. These are the kinds of context an agent can reason over that a static forecast never will. With a weather tool wired in, the agent could say, *"Stock 2 extra hot chocolates tomorrow, it's going to be cold and Machine 2 usually sees a spike on weekend mornings."* Each new data source makes the agent smarter without changing the model.

---

## Evaluation and What's Next

### Does It Work?

The model was evaluated on a time-based holdout, trained on the first 80% of Machine 1's dates, tested on the remaining 20% (316 forecast windows).


| Strategy                             | MAE (drinks/window) | 90% CI Accuracy |
| ------------------------------------ | ------------------- | --------------- |
| **AI Copilot (Poisson GLM + trend)** | **1.50**            | **93.4%**       |
| Global Average                       | 1.51                | 89.9%           |
| Yesterday's Actual                   | 1.95                | 69.6%           |


The copilot matches the global average on point accuracy and meaningfully improves confidence interval calibration (93.4% vs. 89.9%). The win comes from one EDA insight. Demand is trending upward, and a flat historical average can't adapt to that. Reactive stocking ("just repeat yesterday") is measurably worse than either data-driven approach.

But the model is one piece. The real value is what the agent does with it, turning a number into a stocking list the manager can act on, with honest uncertainty, contextual reasoning, and memory of their past decisions.

### What the Data Already Tells Us

The model and sales tools surface patterns that go beyond daily stocking. A few stand out.

**Machine 2 is underperforming.** It averages less than one drink sold per 3-hour window. It carries 30 distinct SKUs but barely moves any of them. The MVP already scopes its recommendations to Machine 1's proven 8-product set because the remaining 22 products have almost no sales history. The honest recommendation here isn't just "stock fewer products," it's "consider whether this machine is in the right location." An agent could surface this proactively: *"Machine 2 sold 4 drinks last week. Machine 1 sold 47. Is this machine in the right spot?"*

**Product catalog is stale.** The current system recommends from the existing product set, but it has no visibility into whether that set is the right one. If lattes are trending down and a new category is trending up, the manager has no way to know from internal data alone. This is an opportunity for the agent to combine internal signals (declining product performance) with external ones (market trends, competitor launches) to flag when the catalog needs a refresh.

### What's Next: The Agent Gets Smarter

The forecasting model is the foundation. The roadmap is about making the *agent* more capable, both at stocking and at higher-level business decisions.

**Smarter stocking:**

- **Weather as a tool call.** Temperature drives hot/cold drink demand, and the agent can adjust recommendations on cold mornings or hot afternoons.
- **Holiday and event calendars.** Black Friday at a mall, a Saturday farmer's market, a local festival. These are foot traffic signals the model can't see but the agent can reason over. EDA tested both `is_weekend` and `is_holiday` with a generic calendar, but neither was significant with current data. Worth re-evaluating once the locale is known or more data is available.
- **Recency-weighted product mix.** Catch shifting preferences instead of relying on all-time averages.
- **Feedback loop.** Override history is already persisted and surfaced to the agent, so it can reference past corrections in conversation. The next step is closing the loop, using consistent overrides as training signal to update the model's baseline automatically.

**Strategic recommendations:**

- **Machine placement.** The agent already has per-machine volume data. A placement tool could compare performance against location benchmarks (foot traffic, nearby competitors, rent) and flag underperforming machines with a recommendation to relocate, not just restock.
- **Product catalog intelligence.** Connect the agent to external trend data (wholesale catalogs, social media trends, category-level market reports) so it can flag when a product is declining internally and something new is gaining traction externally. The manager asks "what should I change about Machine 2?" and gets back "drop the 22 products that never sell, consider adding X which is trending in your category."
- **SKU rationalization.** The tools already know which products sell and which don't. A dedicated analysis could recommend pruning the catalog to the products that actually move, freeing up machine slots for new experiments.

**Infrastructure:**

- **Location context per machine.** Machine 2's weekend-heavy pattern suggests a different location type than Machine 1. Encoding that context lets the agent frame recommendations differently.
- **Production hardening.** Containerization, CI/CD, model drift monitoring, and a proper frontend/backend split.
