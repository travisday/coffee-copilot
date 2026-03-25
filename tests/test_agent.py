#!/usr/bin/env python3
"""Smoke tests for agent tools and graph compilation.

Run from the repo root:
    .venv/bin/python -m tests.test_agent
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {PASS}  {name}")
    else:
        failed += 1
        msg = f" — {detail}" if detail else ""
        print(f"  {FAIL}  {name}{msg}")


def main():
    global passed, failed

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    if not (data_dir / "index_1.csv").exists():
        print(f"Data not found at {data_dir}. Run from the repo root.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Tool imports
    # ------------------------------------------------------------------
    print("\n=== Tool Loading ===\n")

    from agent.tools import (
        forecast_demand,
        get_sales_summary,
        get_revenue_insights,
        get_model_insights,
        ALL_TOOLS,
    )

    check("tools module loads", True)
    check(f"ALL_TOOLS has 4 tools: {len(ALL_TOOLS)}", len(ALL_TOOLS) == 4)

    # ------------------------------------------------------------------
    # forecast_demand (window mode — explicit time range)
    # ------------------------------------------------------------------
    print("\n=== forecast_demand (window) ===\n")

    result = forecast_demand.invoke({
        "target_date": "2024-10-07",
        "plan_scope": "window",
        "start_hour": 7,
        "end_hour": 10,
    })
    check("returns both machines", "machine_1" in result and "machine_2" in result)

    m1 = result["machine_1"]
    check(f"buckets span 06-09 and 09-12: {m1['buckets_used']}",
          "06-09" in m1["buckets_used"] and "09-12" in m1["buckets_used"])
    check(f"plan_scope is window: {m1.get('plan_scope')}", m1.get("plan_scope") == "window")
    check(f"has recommendations: {len(m1['recommendations'])}",
          len(m1["recommendations"]) > 0)
    check(f"total_volume_mean > 0: {m1['total_volume_mean']}",
          m1["total_volume_mean"] > 0)
    check(f"total_volume_upper >= 0: {m1['total_volume_upper']}",
          m1["total_volume_upper"] >= 0)
    check(f"safety_percentile is 0.9: {m1['safety_percentile']}",
          m1["safety_percentile"] == 0.9)
    check("window_mode_cta present", bool(m1.get("window_mode_cta")))

    win_cap = m1.get("machine_slot_capacity")
    check("window machine_slot_capacity present", win_cap is not None and isinstance(win_cap, int))
    check("window capacity_basis is unique_products_sold",
          m1.get("capacity_basis") == "unique_products_sold")

    win_sum = sum(p["recommended_stock"] for p in m1["recommendations"])
    check(
        f"window allocations sum to capacity: {win_sum} vs {win_cap}",
        win_cap is not None and win_sum == win_cap,
    )

    check("window recommendations_by_tier is dict",
          isinstance(m1.get("recommendations_by_tier"), dict))
    check("window every rec has demand_tier",
          all("demand_tier" in r for r in m1["recommendations"]))

    for p in m1["recommendations"]:
        check(f"  {p['product']}: min {p['recommended_stock']}",
              p["recommended_stock"] >= 1)

    m2 = result["machine_2"]
    check("machine_2 mix_fallback is True", m2["mix_fallback"] is True)
    if m2["recommendations"]:
        check("machine_2 confidence says 'Rough estimate'",
              "Rough estimate" in m2["recommendations"][0]["confidence"])

    # safety level variants
    lean = forecast_demand.invoke({
        "target_date": "2024-10-07",
        "plan_scope": "window",
        "start_hour": 9,
        "end_hour": 12,
        "safety_level": "lean",
    })
    conservative = forecast_demand.invoke({
        "target_date": "2024-10-07",
        "plan_scope": "window",
        "start_hour": 9,
        "end_hour": 12,
        "safety_level": "conservative",
    })
    lean_upper = lean["machine_1"]["total_volume_upper"]
    cons_upper = conservative["machine_1"]["total_volume_upper"]
    check(f"conservative ({cons_upper}) >= lean ({lean_upper})",
          cons_upper >= lean_upper)

    # ------------------------------------------------------------------
    # forecast_demand (day mode — full-day minimum levels)
    # ------------------------------------------------------------------
    print("\n=== forecast_demand (day) ===\n")

    day = forecast_demand.invoke({
        "target_date": "2024-10-07",
        "plan_scope": "day",
    })
    d1 = day["machine_1"]
    d2 = day["machine_2"]
    check("plan_scope is day", d1.get("plan_scope") == "day")
    check("recommendations_by_tier present",
          isinstance(d1.get("recommendations_by_tier"), dict))
    check("every rec has demand_tier",
          all("demand_tier" in r for r in d1["recommendations"]))
    cap1 = d1.get("machine_slot_capacity")
    cap2 = d2.get("machine_slot_capacity")
    total_m1 = sum(r["recommended_stock"] for r in d1["recommendations"])
    total_m2 = sum(r["recommended_stock"] for r in d2["recommendations"])
    check(
        f"machine_1 sum {total_m1} == capacity {cap1}",
        cap1 is not None and total_m1 == cap1,
    )
    check(
        f"machine_2 sum {total_m2} == capacity {cap2}",
        cap2 is not None and total_m2 == cap2,
    )
    check("capacity_basis is unique_products_sold", d1.get("capacity_basis") == "unique_products_sold")
    check("per-machine capacity from data", isinstance(cap1, int) and isinstance(cap2, int))
    check("peak_demand_hints is a list", isinstance(d1.get("peak_demand_hints"), list))

    # ------------------------------------------------------------------
    # get_sales_summary
    # ------------------------------------------------------------------
    print("\n=== get_sales_summary ===\n")

    sales = get_sales_summary.invoke({
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
    })
    check(f"total_transactions > 0: {sales['total_transactions']}",
          sales["total_transactions"] > 0)
    check(f"total_revenue > 0: {sales['total_revenue']}",
          sales["total_revenue"] > 0)
    check(f"summary has items: {len(sales['summary'])}",
          len(sales["summary"]) > 0)
    check("grouped_by is 'product'", sales["grouped_by"] == "product")

    sales_dp = get_sales_summary.invoke({
        "start_date": "2024-06-01",
        "end_date": "2024-06-30",
        "group_by": "daypart",
    })
    check("daypart grouping works", sales_dp["grouped_by"] == "daypart")
    check(f"daypart has items: {len(sales_dp['summary'])}",
          len(sales_dp["summary"]) > 0)

    sales_empty = get_sales_summary.invoke({
        "start_date": "2020-01-01",
        "end_date": "2020-01-31",
    })
    check("empty range returns gracefully", sales_empty["total_transactions"] == 0)

    # ------------------------------------------------------------------
    # get_revenue_insights
    # ------------------------------------------------------------------
    print("\n=== get_revenue_insights ===\n")

    dp = get_revenue_insights.invoke({"focus": "daypart"})
    check("daypart has by_machine", len(dp["by_machine"]) > 0)
    check("daypart has insight", len(dp["insight"]) > 0)

    mix = get_revenue_insights.invoke({"focus": "product_mix"})
    check("product_mix has by_daypart", len(mix["by_daypart"]) > 0)
    check("product_mix has insight", len(mix["insight"]) > 0)

    # ------------------------------------------------------------------
    # get_model_insights
    # ------------------------------------------------------------------
    print("\n=== get_model_insights ===\n")

    insights = get_model_insights.invoke({})
    check("returns coefficients", isinstance(insights.get("coefficients"), list))
    check("coefficients non-empty", len(insights["coefficients"]) > 0)
    check("returns eval_metrics", isinstance(insights.get("eval_metrics"), dict))
    check("eval_metrics has mae", "mae" in insights["eval_metrics"])
    check("eval_metrics has coverage",
          "interval_coverage_90" in insights["eval_metrics"])
    check("eval_metrics has n_test", "n_test" in insights["eval_metrics"])

    # ------------------------------------------------------------------
    # Agent compilation (requires OPENAI_API_KEY)
    # ------------------------------------------------------------------
    print("\n=== Agent Compilation ===\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  SKIP  build_agent (OPENAI_API_KEY not set)")
    else:
        from agent.graph import build_agent
        agent = build_agent()
        check("agent compiles", agent is not None)
        check("agent has invoke method", hasattr(agent, "invoke"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
