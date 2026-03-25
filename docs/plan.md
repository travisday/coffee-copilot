# Coffee Vending Ops Copilot — Detailed Build Plan

**Focus:** Ops Copilot with light Pricing & Mix flavor
**Time Budget:** 4 hours
**Stack:** Streamlit + SQLite + LangGraph + statsmodels/scikit-learn + Claude API
**Deliverables:** Runnable repo, brief (2-3 pages), clickable prototype, README (≤1 page)

---

## North Star: The Ops Manager's Problem

This tool exists to help an operations manager make better stocking decisions. Today they're guessing — loading machines based on gut feel, yesterday's memory, or a fixed routine. The cost of getting it wrong goes both ways:

- **Overstock → waste.** Perishable ingredients expire. Machine capacity is wasted on products nobody orders in that window.
- **Understock → lost revenue.** Machine runs out of Lattes at 10am, customer walks away. No sale, no data, invisible loss.
- **Time → the manager's most expensive resource.** Every minute spent manually analyzing sales or second-guessing inventory is a minute not spent on higher-value work.

**What the tool gives them:**
1. A plain-English answer to "what should I stock and when?" — no spreadsheets, no dashboards to interpret
2. Honest uncertainty — "we're confident about Machine 1 mornings, less sure about Machine 2" so they can apply their own judgment where it matters
3. The ability to override and correct — the tool supports their decision, it doesn't make it for them
4. Revenue visibility — "you're underperforming in this window" so they can take action

**Every technical decision below should map to one of these outcomes.** If it doesn't help the ops manager stock better, save time, or make more money, it doesn't belong in the 4-hour build.

---

## Pre-Work

- [ ] Download the Kaggle Coffee Sales dataset (two CSV files: index1.csv and index2.csv — one per vending machine)
- [ ] Set up repo structure:
  ```
  coffee-copilot/
  ├── app.py                  # Streamlit entry point
  ├── agent/
  │   ├── graph.py            # LangGraph agent definition
  │   ├── tools.py            # Tool functions
  │   └── prompts.py          # System prompts
  ├── models/
  │   ├── forecaster.py       # Demand forecasting model
  │   └── analyzer.py         # Revenue/mix analysis (Track B flavor)
  ├── data/
  │   ├── index1.csv          # Machine 1 raw data (~3,600 rows)
  │   ├── index2.csv          # Machine 2 raw data (~250 rows)
  │   └── store.db            # SQLite (app state, logs, feedback)
  ├── notebooks/
  │   └── eda.ipynb           # EDA + model selection notebook
  ├── brief.md                # 2-3 page brief
  ├── README.md               # Setup + assumptions (≤1 page)
  └── requirements.txt
  ```
- [ ] Create `requirements.txt`: streamlit, langgraph, langchain-anthropic, statsmodels, scikit-learn, pandas, numpy, matplotlib, seaborn, plotly, sqlite3 (stdlib)
- [ ] Set up `.env` with `ANTHROPIC_API_KEY`
- [ ] Create the SQLite schema (tables for: tool_call_logs, user_overrides, user_feedback, conversation_history)

---

## Hour 1: EDA + Model (0:00 – 1:00)

### 1A. EDA Notebook (0:00 – 0:30)

Goal: Understand the data, validate model assumptions, produce charts you can reference in the brief.

**Dataset overview:**
- Two CSV files (index1.csv, index2.csv) = two vending machines
  - Machine 1: ~3,600 rows (primary, dense data)
  - Machine 2: ~250 rows (secondary, sparse data — great for demonstrating uncertainty)
- 6 columns: date, datetime, cash_type, card, money, coffee_name (no machine_id — inferred from file)
- 8 coffee products, March 2024 – ~present

**Data loading (add machine_id during ingestion):**
```python
df1 = pd.read_csv("data/index1.csv")
df1["machine_id"] = "machine_1"
df2 = pd.read_csv("data/index2.csv")
df2["machine_id"] = "machine_2"
df = pd.concat([df1, df2], ignore_index=True)
```

**Cell 1: General EDA (~20 min)**

Purpose: understand the data, produce 3-4 charts for the presentation, identify anything weird.

1. **Basic stats + machine comparison** — Row count per machine, date range, unique products, avg transactions/day per machine. Side-by-side bar of transactions + revenue per machine. **Why:** establishes the "two very different machines" story (Machine 1 ~15x busier) and justifies machine_id as a feature. Check avg ticket — if similar across machines, they sell at the same prices, just different volumes.

2. **Sales by hour** — Histogram of transaction hours, colored by machine. **Why:** validates your hour_bucket boundaries (06-09, 09-12, etc.) and reveals if you need an "outside hours" bucket. Also confirms whether both machines share similar peak hours or have different patterns. This chart goes in the presentation.

3. **Heatmaps: weekday × hour per machine** — One heatmap per machine. **Why:** this IS the core pattern your volume model needs to learn. Shows the interaction between day-of-week and time-of-day. Great presentation visual — immediately communicates "Monday mornings are busy, Sunday evenings are dead." If Machine 2's heatmap is mostly empty, that visually justifies why its predictions carry high uncertainty.

4. **Weekly trend over time** — Line chart of weekly sales count per machine. **Why:** determines if demand is growing, flat, or seasonal. If flat → day_of_week and hour are sufficient features. If there's a clear trend or seasonality → add month as a feature. Critical model design decision.

**Cut from general EDA (save time):**
- Product sales by machine → you'll build the product mix table in Stage 2 of the model. A quick `value_counts()` in a markdown cell is enough, no chart.
- Day of week bar chart → redundant with the heatmap, which shows day AND hour.
- Payment type distribution → one markdown sentence ("95% card") is enough for the brief. Not a model input.
- Revenue by daypart → one markdown sentence for the brief. The volume model doesn't use revenue.
- Customer repeat behavior → interesting but doesn't inform the model. One sentence in the brief if you have time.

**Cell 2: Model Assumption Checks (~10 min)**

Purpose: determine the right model and aggregation level. These charts/tables are your evidence for every model decision. Every chart here should directly support a decision you made.

**Decision: "Why aggregate to volume level?" → Charts 5, 6, 7 prove it**

5. **Aggregate to counts at product level** — Create (date, hour_bucket, coffee_name, machine_id) → count. Show head of table + total non-zero rows (~2,975). **Why:** establishes what the raw modeling target looks like. Narrator markdown: "Most non-zero counts are 1. This is too sparse to model."

6. **Zero-inflation check** — Table showing total_buckets, zero_buckets, zero_rate per machine. **Why:** the 96.5%/99.7% zero rates are the KEY finding that drives the two-stage design. This is the single most important number in your presentation. Narrator markdown: "96.5% of (date, hour, product, machine) slots have zero sales. Per-product modeling isn't viable at this granularity."

7. **Dispersion check at product level** — Mean vs variance scatter plot colored by machine, plus summary table of dispersion ratios per machine (median ~0.25). **Why:** shows underdispersion at product level — variance is much less than the mean. Confirms per-product counts are too regular/sparse for Poisson to be useful. Narrator markdown: "Dispersion well below 1.0 at product level — the data is sparser than Poisson assumes."

**Decision: "Why Poisson GLM?" → Chart 8 proves it**

8. **Aggregate to volume level + re-check dispersion** — Create (date, hour_bucket, machine_id) → total_count (collapse products). Two outputs:
   - **Histogram** of total_count distribution per machine. Shows counts of 1-5 with a tail to ~14. Narrator markdown: "At the volume level, we get meaningful counts to model."
   - **Dispersion table** by (machine, hour_bucket) showing ratios of 0.7-1.4 — right around 1.0. Narrator markdown: "Dispersion ratios cluster around 1.0 at the volume level. Poisson is well-suited here."
   - **Variance vs mean scatter** at volume level with diagonal reference line. Dots should cluster near the line. Narrator markdown: "Variance ≈ mean — textbook Poisson territory. Two-stage approach validated."

**Decision: "Why use Machine 1's product mix for both machines?" → Charts 9, 10 prove it**

9. **Product mix stability over time (Machine 1)** — Compute product proportions per hour_bucket for first half vs second half of data. Show table sorted by abs_diff. **Why:** validates that Machine 1's historical frequencies are reliable for Stage 2. Narrator markdown: "Top products shift by <9% between halves — mix is stable enough for historical proportions. Phase 2: recency-weighted mix."

10. **Machine 2 mix instability** — Same comparison for Machine 2. Show that products jump from 0% to 15%+ between halves (because sparse data creates random appearance/disappearance). **Why:** justifies the Machine 1 fallback decision. Narrator markdown: "Machine 2's mix is unstable — products appear and disappear between time periods due to ~250 total rows. Using Machine 1's mix as fallback."

**Decision: "Why a product threshold for recommendations?" → Chart 11 proves it**

11. **Product frequency by hour_bucket (Machine 1)** — Simple table showing product proportions per hour_bucket. Highlight products below 5% — these are the "rarely ordered" ones that won't appear in main recs. **Why:** justifies the 0.5-unit threshold. When the 90th percentile volume is 5 and a product is 3% of the mix, expected demand is 0.15 — not actionable. Narrator markdown: "Irish Whiskey, Tea, and Espresso are <5% of most windows. At expected volumes of 2-5 total drinks, these round to 0 — included as 'rarely ordered' footnote rather than main rec."

**Summary markdown cell at the end of Cell 2:**
```markdown
### EDA → Modeling Decisions → Business Outcomes

| Finding | Technical Decision | Business Outcome |
|---------|-------------------|-----------------|
| 96.5% zero rate at product level | Aggregate to total volume per slot | Gives the manager a reliable "how busy will it be" number instead of noisy per-product guesses |
| Dispersion ~1.0 at volume level | Poisson GLM is appropriate | Honest confidence intervals — manager knows when to trust the rec and when to apply judgment |
| Machine 1 mix stable (<9% drift) | Use historical frequencies for Stage 2 | Per-product stocking recs the manager can act on directly |
| Machine 2 mix unstable | Fall back to Machine 1's mix, flagged in UI | Manager still gets a useful rec for Machine 2 instead of "not enough data, sorry" |
| Several products <5% of mix | Threshold recs; "rarely ordered" footnote | Clean, actionable rec (4-5 products) instead of a noisy list of 12 where half say "stock 0" |
| Machine 2 has ~250 rows total | Flag uncertainty in UI | Manager knows to apply more personal judgment for Machine 2 — the tool supports, doesn't override |
```
This table goes straight into the brief and is a great presentation slide.

### 1B. Build the Forecasting Model (0:30 – 1:00)

**EDA findings that drive model design:**
- Data is **underdispersed** (variance < mean, dispersion ratios 0.25–0.55) — opposite of overdispersion
- **96.5% zero rate** for Machine 1, **99.7%** for Machine 2 at (date, hour_bucket, product, machine) granularity
- Most non-zero counts are 1–2 sales per bucket — too sparse to model per-product demand reliably
- This means: predicting "how many Lattes in 09-12 on Monday for Machine 1" is asking the model to predict 0 or 1. Not useful.

**Solution: Two-Stage Approach**

**Stage 1 — Aggregate Volume Model (Poisson GLM)**
Predict TOTAL drinks per (date, hour_bucket, machine_id) — collapse across products.
This dramatically reduces zeros and gives the model meaningful counts to work with.

```python
import statsmodels.api as sm

# Aggregate: total drinks per (date, hour_bucket, machine_id)
volume = df.groupby(['sale_date', 'hour_bucket', 'machine_id']).size().reset_index(name='total_count')
# Now counts are more like 3-8 per bucket instead of 0-1

# Features: day_of_week (one-hot), hour_bucket (one-hot), machine_id (one-hot)
# Optional: month, is_weekend
volume['day_of_week'] = pd.to_datetime(volume['sale_date']).dt.dayofweek

X = pd.get_dummies(volume[['day_of_week', 'hour_bucket', 'machine_id']], drop_first=True)
X = sm.add_constant(X)
y = volume['total_count']

model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# Check dispersion again at this aggregation level
print(f"Dispersion: {results.deviance / results.df_resid:.2f}")
# If still underdispersed (<1), Poisson is conservative (wider intervals than needed) — fine
# If overdispersed (>1.5), switch to NegativeBinomial
```

**Stage 2 — Product Mix Proportions (historical frequencies)**
For each (machine_id, hour_bucket), compute the historical product mix.
Machine 2 falls back to Machine 1's mix due to data sparsity.
No model needed — just SQL/pandas.

```python
# Product mix: what % of sales in each (machine_1, hour_bucket) is each product?
# Use Machine 1 only — it has ~3,600 rows and stable proportions.
# Machine 2 uses Machine 1's mix as a fallback (flagged in UI).
mix = (
    df.loc[df["machine_id"] == "machine_1"]
      .groupby(['hour_bucket', 'coffee_name']).size()
      .reset_index(name='count')
)
totals = (
    df.loc[df["machine_id"] == "machine_1"]
      .groupby(['hour_bucket']).size()
      .reset_index(name='total')
)
mix = mix.merge(totals, on=['hour_bucket'])
mix['proportion'] = mix['count'] / mix['total']

# Example output:
# 09-12, Latte → 0.28
# 09-12, Americano → 0.18
# 09-12, Americano with Milk → 0.15
# 09-12, Irish whiskey → 0.02  (below threshold — won't appear in main rec)
# ...
```

**Product recommendation threshold:**
Products with expected demand < 0.5 units don't appear in the main stocking rec.
Instead, they go in a "rarely ordered" footnote.
```python
REC_THRESHOLD = 0.5  # minimum expected units to include in main rec

for product, proportion in mix.items():
    expected = upper_total * proportion
    if expected >= REC_THRESHOLD:
        # Main rec: "Stock 3 Lattes"
        stock = math.ceil(expected)
    else:
        # Footnote: "Rarely ordered in this window: Irish Whiskey, Tea"
        pass
```
This keeps the UI clean — the manager sees 4-5 actionable products, not 12 rows where half say "stock 0."

**Machine 2 fallback logic:**
```python
def get_product_mix(self, machine_id, hour_bucket):
    # Always use Machine 1's mix proportions
    # Machine 2 has too few observations for stable per-product proportions
    mix = self._mix_table[(self._mix_table['hour_bucket'] == hour_bucket)]
    
    if machine_id == "machine_2":
        # Flag for UI transparency
        mix = mix.copy()
        mix['fallback'] = True  # "Product mix based on Machine 1 patterns"
    return mix
```

**How the two stages combine for a recommendation:**
1. Model predicts: "Machine 1, Monday, 09-12 → expected 2.8 total drinks (90th percentile: 5)"
2. Product mix says: 09-12 is 28% Latte, 18% Americano, 15% Americano w/ Milk...
3. Stocking rec (using 90th percentile): 5 × 28% = 1.4 → stock 2 Lattes, 5 × 18% = 0.9 → stock 1 Americano
4. Products below threshold: "Rarely ordered: Irish Whiskey, Tea, Espresso"
5. For Machine 2: same mix proportions, lower volume. UI shows: "⚠️ Product mix based on Machine 1 patterns (limited Machine 2 data)"

**Why this is the right design:**
- "At the per-product granularity, 96.5% of buckets are zero — there's not enough signal to model individual product demand. By aggregating to total volume, I get counts of 2-3 per slot that the Poisson GLM can actually learn from. EDA confirmed dispersion ratios of 0.7-1.4 at this level — right in the Poisson sweet spot."
- "Separating volume from mix is a clean decomposition. The volume model captures time-based patterns (when is it busy?). The mix captures preference patterns (what do people order?). These are two different questions with different data requirements."
- "Machine 1's product mix is stable across time (EDA showed <9% drift for top products). Machine 2's mix is unreliable — products appear and disappear between time periods because there's so little data. Using Machine 1's mix as a fallback is the pragmatic choice, and we're transparent about it in the UI."
- "The product mix is a simple frequency table, which is more robust than a model at this data scale. With more data, I'd promote it to a recency-weighted table or a multinomial model."

**Prediction intervals at the product level:**
Since you're splitting a Poisson prediction across products using proportions, the per-product uncertainty comes from two sources: (1) uncertainty in total volume, (2) uncertainty in the mix proportions. For the assessment, use the simple approach:
```python
from scipy.stats import poisson

lambda_total = model.predict(features)  # e.g., 2.8 total drinks
upper_total = poisson.ppf(0.90, lambda_total)  # e.g., 5

# Per-product stocking = upper_total × product_proportion, rounded up
for product, proportion in mix.items():
    expected = upper_total * proportion
    if expected >= REC_THRESHOLD:
        stock = math.ceil(expected)
```
Mention in the brief: "In production, I'd use a Dirichlet-Multinomial to properly propagate mix uncertainty into per-product intervals."

**Time range mapping (how the agent translates user queries to buckets):**
- User asks for a time range → agent maps to overlapping hour_bucket(s)
- "7-10am" → 06-09 + 09-12 buckets, sum predictions
- "3pm-4pm" → 12-15 bucket (whole bucket, not subdivided)
- "whole day" → sum all buckets
- Agent is transparent about this: "My forecasts operate at 3-hour windows. For the 09-12 window, I expect..."
- This honesty about granularity limits IS the transparency UX requirement in action

**Train/test split:** Time-based — train on first ~80% of dates, test on last ~20%.

**Evaluation metrics:**
- MAE on total counts per bucket (intuitive: "off by X drinks on average")
- Calibration of prediction intervals (do 90% intervals capture truth ~90% of the time?)
- Product mix stability: does the mix shift over time? If yes, note as limitation.
- Quick print of results → include in brief

**Wrap in a class:**
```python
class DemandForecaster:
    def __init__(self, db_path):
        # Load data, fit volume model, compute product mix proportions
        self.volume_model = None      # Poisson GLM on aggregate counts
        self.product_mix = None       # Dict of {(machine, hour_bucket): {product: proportion}}
    
    def predict_volume(self, date, hour_buckets, machine_id=None):
        # Returns: {machine_id: {hour_bucket: {total: int, lower: int, upper: int}}}
    
    def get_product_mix(self, machine_id, hour_bucket):
        # Returns: {product: proportion} from historical frequencies
    
    def get_stocking_recommendation(self, date, hour_start, hour_end, machine_id=None, safety_percentile=0.9):
        # 1. Map time range to hour_buckets
        # 2. Predict volume per bucket
        # 3. Apply product mix to get per-product recs
        # 4. Round up, apply guardrails (cap at historical max × 1.5)
        # Returns: {machine_id: {product: {recommended_stock: int, expected_demand: float, 
        #           confidence: str, reasoning: str}}}
    
    def get_coefficients_summary(self):
        # Returns human-readable coefficient interpretation
        # "Mondays see 1.3x the demand of Sundays"
        # "Machine 1 sees ~Xx the volume of Machine 2"
        # "09-12 is the busiest window at 1.8x the baseline"
```

**Track B flavor — Revenue Analyzer (quick, no model needed):**
```python
class RevenueAnalyzer:
    def __init__(self, db_path):
        # Connect to SQLite
    
    def daypart_performance(self, machine_id=None, date_range=None):
        # SQL query: revenue, count, avg price by daypart per machine
        # Flag: "Evening revenue is 40% below afternoon — consider promotions"
    
    def product_mix_insights(self, machine_id=None, date_range=None):
        # Top/bottom sellers by daypart, revenue concentration
        # "Latte dominates mornings (35%) but drops to 15% in evenings"
```

---

## Hour 2: Agent + Tools Layer (1:00 – 2:00)

### 2A. Define Tools (1:00 – 1:30)

Each tool is a Python function the LangGraph agent can call. Tools query SQLite and/or the model.

**Tool 1: `forecast_demand`**
- Input: target_date (str), start_hour (int), end_hour (int), machine_id (optional, default "both"), safety_level (optional, default "normal")
- Logic:
  1. Map safety_level to percentile ("conservative"=0.95, "normal"=0.9, "lean"=0.75)
  2. Call `DemandForecaster.get_stocking_recommendation()` — this handles bucket mapping, volume forecast, product mix split, rounding, and guardrails
- Output: Dict with per-machine total volume (mean + upper), per-product stocking recs with confidence, rarely ordered products, and which hour_buckets were used
- Example: "Tomorrow (Monday) 7-10am — Machine 1: 06-09 window ~2 drinks, 09-12 window ~3 drinks, total ~5. Breakdown: 2 Lattes, 1 Americano, 1 Americano w/ Milk, 1 Cappuccino. Rarely ordered: Irish Whiskey, Espresso. Machine 2: ~1-2 total drinks (low confidence). ⚠️ Product mix based on Machine 1 patterns."
- Transparency: include which hour_buckets were used, note if the time range doesn't align exactly
- Note: originally split into forecast_demand (mean/point estimate) and get_stocking_recommendation (upper bound). Consolidated because `get_stocking_recommendation()` already returns both mean and upper values — the LLM picks which to emphasize based on the user's question.

**Tool 2: `get_sales_summary`**
- Input: start_date (str), end_date (str), machine_id (optional, default "both"), group_by (optional: "product", "daypart", "day_of_week", "machine")
- Logic: SQL query on sales data
- Output: Summary stats — total sales, revenue, top products, busiest times, per machine or aggregate

**Tool 4: `get_revenue_insights`** (Track B flavor)
- Input: date_range (optional), focus (optional: "daypart", "product_mix", "payment_type")
- Logic: Call `RevenueAnalyzer` methods
- Output: Insights like "Evening revenue trails afternoon by 43%. Hot Chocolate dominates evenings — consider bundling with Espresso."

### 2B. Build LangGraph Agent (1:30 – 2:00)

```
graph structure:

[route_query] → decides which tool(s) to call
      ↓
[call_tools] → executes tool(s), gets results
      ↓
[synthesize] → LLM formats results into natural language response
      ↓
[respond] → return to user
```

**Agent system prompt (key framing):**
```
You are an AI operations copilot for a coffee vending machine business.
You support operations managers in making stocking and planning decisions.

Key principles:
- YOU SUPPORT, the manager DECIDES. Present recommendations, not commands.
- Lead with the action: "I'd suggest stocking X" not "The model predicts λ=2.8"
- Be honest about uncertainty: "I'm confident about this" vs "this is a rough estimate — you know this machine better than I do"
- Explain your reasoning when asked, but don't lead with the math
- Frame everything in business terms: waste reduction, stockout prevention, revenue opportunity
- When you're uncertain, say so. A confident wrong answer is worse than an honest "I don't have enough data to be sure"

Available tools: forecast_demand, get_sales_summary, get_revenue_insights
```

**Tool-call logging (nice-to-have, easy to add):**
```python
def log_tool_call(tool_name, inputs, outputs, latency_ms):
    # Insert into SQLite tool_call_logs table
    # Columns: id, timestamp, tool_name, inputs_json, outputs_json, latency_ms
```

**Conversation memory:**
- Store conversation history in SQLite `conversation_history` table
- Load last N messages on app start
- Simple but demonstrates the "memory/cached context" nice-to-have

---

## Hour 3: Streamlit UI (2:00 – 3:00)

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  ☕ Coffee Ops Copilot                          [Clear] │
├────────────────────┬────────────────────────────────────┤
│                    │                                    │
│   SIDEBAR          │   MAIN CHAT AREA                  │
│                    │                                    │
│   📊 Quick Stats   │   [Chat messages with AI]          │
│   - Today's sales  │                                    │
│   - Top product    │   Each response shows:             │
│   - Revenue        │   - Answer text                    │
│                    │   - 📋 Tool calls used (expandable)│
│   🔧 Settings      │   - 📊 Charts (if relevant)       │
│   - Safety level   │   - ⚙️ Override controls           │
│   - Default hours  │   - 👍👎 Feedback buttons          │
│                    │                                    │
│   📜 Recent Logs   │                                    │
│   - Last 5 tool    │                                    │
│     calls          │                                    │
│                    │                                    │
├────────────────────┴────────────────────────────────────┤
│  💬 Ask me anything... (e.g. "Plan tomorrow 7-10am")    │
└─────────────────────────────────────────────────────────┘
```

### UX Requirements (critical for real-world adoption)

**Core philosophy: the ops manager makes the decision. The tool makes them faster and better-informed.**

**1. Transparency — "User understands WHY the AI recommends something"**
- Business value: trust. If the manager doesn't understand why, they won't use it. If they don't use it, the tool has zero value.
- Every recommendation shows an expandable "How I got this" section
- Display: which hour windows were used, total volume forecast, product mix breakdown with percentages
- Example: "I recommended 2 Lattes because: Monday 09-12 typically sees ~3 total drinks. Lattes are 28% of morning sales. At a 90% safety buffer that's 5 × 28% = 1.4, rounded up to 2."
- When time ranges don't align: "Your 7-10am request spans two forecast windows (06-09 and 09-12). I summed both."
- For Machine 2: "⚠️ Product mix based on Machine 1 patterns — Machine 2 doesn't have enough history for its own. You know this machine better than the model does."
- The "How I got this" section is collapsed by default — it's there when the manager wants to sanity-check, not in the way when they trust the rec.

**2. Uncertainty — "Confidence levels communicated without overwhelming"**
- Business value: helps the manager know where to spend their judgment. High confidence → accept the rec and move on. Low confidence → apply your own knowledge.
- Primary display: plain English labels on each recommendation — "Confident" / "Less certain" / "Rough estimate"
- Secondary (expandable): actual range ("expect 2-5 drinks")
- For Machine 2, always flag: "Limited data — treat this as a rough guide, not a precise forecast"
- The goal is NOT to impress with statistical precision. It's to help the manager decide how much to trust each number.

**3. Override — "User can adjust or reject recommendations with minimal friction"**
- Business value: the manager has context the model doesn't — a conference next door, a broken machine part, a holiday. Overrides let them combine their knowledge with the model's patterns.
- After each stocking recommendation: inline sliders per product to adjust quantities up or down
- "Accept" / "Accept with Changes" / "Reject" buttons
- Overrides stored in SQLite with timestamp and optional reason
- This is THE key differentiator from a static dashboard. The model proposes, the manager disposes.

**4. Feedback — "User corrections can improve future recommendations"**
- Business value: the tool gets better over time. The manager's corrections are the most valuable training signal — they contain real-world knowledge the model can't access.
- 👍/👎 on every AI response
- Optional text: "What would have been better?"
- Stored in SQLite `user_feedback` table
- In the brief: explain how feedback would work in production — "If the manager consistently bumps Latte recs up by 1, the system learns to adjust its morning Latte proportion upward. Short-term: heuristic adjustment. Long-term: incorporate as training signal."

### Key Streamlit Components

```python
# Chat interface
st.chat_input("Ask me anything...")
st.chat_message("assistant")
st.chat_message("user")

# Expandable tool calls
with st.expander("🔧 How I got this"):
    st.json(tool_call_log)

# Override sliders
for product, rec in recommendation.items():
    adjusted = st.slider(f"{product}", 0, rec * 2, rec)

# Feedback
col1, col2 = st.columns(2)
col1.button("👍")
col2.button("👎")

# Charts (plotly for interactivity)
import plotly.express as px
fig = px.bar(forecast_df, x="product", y="forecast", error_y="upper_bound")
st.plotly_chart(fig)
```

### Quick Wins to Make It Feel Polished
- Add a few "suggested queries" as clickable buttons: "Plan tomorrow morning for both machines", "Show me this week's sales", "Where are we leaving money?", "Compare the two machines"
- Loading spinner while agent processes
- Conversation history persists across reruns (SQLite)
- Timestamp on each message

---

## Hour 4: Polish + Deliverables (3:00 – 4:00)

### 4A. Brief — 2-3 Pages (3:00 – 3:30)

**Page 1: The Problem & Who This Is For**
- The user: an operations manager responsible for stocking 2+ vending machines. Not technical. Time-constrained. Currently relies on gut feel and fixed routines.
- The cost of the status quo: overstock → waste, understock → lost revenue, manual planning → time that could be spent on higher-value work
- What we built: a natural language copilot that answers "what should I stock and when?" with honest, actionable recommendations the manager can accept, adjust, or override
- Design philosophy: the tool supports the manager's decision — it doesn't replace their judgment. It's most valuable where the data is strong, and honest about where the data is weak.

**Page 2: How It Works (Technical, But Framed Around Outcomes)**
- Two-stage model: (1) Poisson GLM forecasts total demand per time window per machine, (2) historical product mix breaks that into specific product recs
- Why two stages: EDA showed 96.5% of per-product time slots are empty — not enough data to forecast individual products reliably. Instead of giving the manager unreliable per-product numbers, we give them a reliable total with a trustworthy product breakdown. Honest > precise.
- Machine 2 fallback: Machine 2 has ~15x less data. Rather than showing "insufficient data," we use Machine 1's product mix as a starting point and flag it clearly — the manager probably knows Machine 2's quirks better than any model could with 250 data points.
- Agent + tools architecture: LangGraph routes natural language → tools → model → actionable response. Logged for observability.
- UX: transparency (why this rec?), uncertainty (how much to trust it?), override (adjust with your judgment), feedback (improve over time)

**Page 3: Impact & What's Next**
- Business impact framing:
  - Time savings: "Plan tomorrow morning" takes 10 seconds instead of 15 minutes of spreadsheet analysis
  - Waste reduction: safety-buffered recs (90th percentile) balance overstock risk against stockout risk — tunable per manager preference (conservative/normal/lean)
  - Revenue visibility: "You're leaving money on the table after 5pm" — Track B insights surface opportunities the manager might not notice in raw data
  - Machine 2 insight: the data sparsity itself is a business signal — is this machine underperforming because of location, visibility, or product selection?
- How I'd test in production:
  - A/B: AI-recommended stocking vs. manager's current routine over 4 weeks. Measure waste, stockouts, and manager time.
  - User testing: how long does "plan tomorrow morning" take? Does the manager trust the recs? Do they use the overrides?
- Phase 2: weather integration, recency-weighted product mix, more machines, feedback loop into retraining

### 4B. README — ≤1 Page (3:30 – 3:40)

```markdown
# Coffee Ops Copilot

## Quick Start
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
streamlit run app.py

## Architecture
[1-paragraph summary + small diagram]

## Assumptions
- Two vending machines identified by source file (index1.csv = Machine 1, index2.csv = Machine 2)
- Machine 2 has ~15x fewer transactions than Machine 1 — predictions for Machine 2 carry higher uncertainty
- Forecasts operate at 3-hour windows (hour_buckets); sub-bucket precision is not supported
- Product mix proportions are assumed stable over time (historical frequencies)
- Prices are fixed per product (no dynamic pricing)
- "Tomorrow" queries use day-of-week patterns, not calendar-specific events
- Cash transactions have no customer tracking (card field is null)

## Limitations
- Two-stage model: volume forecast is modeled, but product breakdown uses static proportions — if mix shifts seasonally, recs may drift
- Machine 2 has very sparse data (~250 rows) — volume forecasts carry high uncertainty
- Forecasts are at 3-hour granularity; can't distinguish 7am vs 9am demand within a bucket
- No external features (weather, holidays, events)
- No real-time data pipeline — model uses static CSV
- Override/feedback loop is demonstrated but not connected to model retraining

## What I'd Do Next
- Time-aware product mix (recency-weighted proportions, or multinomial model)
- Zero-Inflated Poisson for finer-grained per-product modeling with more data
- Prophet or Bayesian structural time-series for trend + seasonality
- Weather API integration (temperature → hot/cold drink demand)
- Hierarchical model: share strength across machines while learning machine-specific baselines
- Evaluation framework: automated backtesting pipeline
- Guardrails: safe ranges on recommendations
- Production: containerize, add CI/CD, monitoring dashboard
```

### 4C. Final Cleanup (3:40 – 4:00)
- [ ] Run the app end-to-end — make sure it starts clean
- [ ] Test 3-4 queries that might come up in the demo:
  - "Plan tomorrow 7-10am for both machines"
  - "What were our best sellers last week?"
  - "Where are we leaving money after 5pm?"
  - "How does Machine 2 compare to Machine 1?"
- [ ] Clean up notebook — make sure markdown cells tell a story
- [ ] Remove any debug prints or dead code
- [ ] Check that all files are in the repo and requirements.txt is complete
