# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_models.py -v

# Run a single test
python -m pytest tests/test_agent.py::test_forecast_demand_day_scope -v
```

## Environment

Requires `.env` with `OPENAI_API_KEY`. Optional: `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` for observability. See `.env.example`.

## Architecture

Coffee Ops Copilot is a natural-language AI assistant that helps coffee vending machine operators decide what to stock and when. Four layers:

**Agent (`agent/`)** — LangGraph ReAct agent with GPT-4o. `graph.py` builds the agent, `tools.py` defines 4 tools (forecast_demand, get_sales_summary, get_revenue_insights, get_model_insights), `prompts.py` holds the system prompt and behavioral principles. The agent reasons about which tools to call rather than following a fixed pipeline.

**Model (`models/`)** — Two-stage demand forecasting. `forecaster.py` runs a Poisson GLM (Stage 1: predict total drink volume per date/hour-bucket/machine) then applies historical product-mix proportions (Stage 2: split volume into per-product recommendations). Machine 2 uses Machine 1's product mix due to sparse data. `analyzer.py` handles revenue/daypart analytics via pandas aggregation.

**UI (`ui/`)** — Streamlit components. `forecast_explain.py` renders "Why this recommendation" expanders with model math. `overrides.py` provides inline sliders for adjusting recommendations with optional reason capture. `feedback.py` handles thumbs up/down with comment dialogs.

**Persistence (`db.py`)** — SQLite with three tables: conversation_history, user_overrides, user_feedback. Past overrides and negative feedback are injected into agent context on subsequent calls.

**Entry point (`app.py`)** — Streamlit app wiring everything together. Handles context injection (date, safety level, machine, feedback, overrides) into user messages, parses forecast data from agent responses, and manages session state across Streamlit reruns.

## Key Design Decisions

- **Context injection over tool proliferation:** Calendar info, feedback, and overrides are appended to user messages rather than requiring additional tool calls.
- **Scoped machine rendering:** The `machine_id` parameter in `forecast_demand` controls which machine's sliders the UI displays.
- **Poisson PPF for planning bounds:** Converts continuous rate predictions to discrete stocking numbers at configurable safety levels (conservative=95th, normal=90th, lean=75th percentile).
- **Machine 2 uses Machine 1's product mix:** Explicitly flagged in responses due to Machine 2's sparse data (~263 transactions vs ~3,637).

## Data

Source CSVs in `data/`: `index_1.csv` (Machine 1, dense) and `index_2.csv` (Machine 2, sparse). SQLite DB (`data/store.db`) created at runtime. Products are Machine 1's 8-item set. Time is bucketed into 3-hour windows (06-09, 09-12, 12-15, 15-18, 18-21).
