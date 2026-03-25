"""Smoke tests for agent tools and graph compilation.

Run from the repo root:
    python -m pytest tests/test_agent.py -v
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


def _data_available() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / "data" / "index_1.csv").exists()


@unittest.skipUnless(_data_available(), "Data files not found")
class TestForecastDemandWindow(unittest.TestCase):
    """forecast_demand in window mode (explicit time range)."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.forecast_demand = forecast_demand
        cls.result = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "window",
            "start_hour": 7,
            "end_hour": 10,
        })
        cls.m1 = cls.result["machine_1"]
        cls.m2 = cls.result["machine_2"]

    def test_returns_both_machines(self):
        self.assertIn("machine_1", self.result)
        self.assertIn("machine_2", self.result)

    def test_buckets_span_expected(self):
        self.assertIn("06-09", self.m1["buckets_used"])
        self.assertIn("09-12", self.m1["buckets_used"])

    def test_plan_scope_is_window(self):
        self.assertEqual(self.m1.get("plan_scope"), "window")

    def test_has_recommendations(self):
        self.assertGreater(len(self.m1["recommendations"]), 0)

    def test_total_volume_mean_positive(self):
        self.assertGreater(self.m1["total_volume_mean"], 0)

    def test_total_volume_upper_non_negative(self):
        self.assertGreaterEqual(self.m1["total_volume_upper"], 0)

    def test_safety_percentile(self):
        self.assertEqual(self.m1["safety_percentile"], 0.9)

    def test_window_mode_cta_present(self):
        self.assertTrue(self.m1.get("window_mode_cta"))

    def test_allocations_sum_to_total_upper(self):
        upper = self.m1.get("total_volume_upper")
        total = sum(p["recommended_stock"] for p in self.m1["recommendations"])
        self.assertEqual(total, upper)

    def test_recommendations_by_tier_is_dict(self):
        self.assertIsInstance(self.m1.get("recommendations_by_tier"), dict)

    def test_every_rec_has_demand_tier(self):
        for r in self.m1["recommendations"]:
            self.assertIn("demand_tier", r)

    def test_every_rec_non_negative(self):
        for r in self.m1["recommendations"]:
            self.assertGreaterEqual(r["recommended_stock"], 0)

    def test_high_demand_products_get_stock(self):
        high = self.m1["recommendations_by_tier"].get("high", [])
        for r in high:
            self.assertGreaterEqual(r["recommended_stock"], 1)

    def test_machine_2_mix_fallback(self):
        self.assertTrue(self.m2["mix_fallback"])

    def test_machine_2_confidence_rough_estimate(self):
        if self.m2["recommendations"]:
            self.assertIn("Rough estimate", self.m2["recommendations"][0]["confidence"])


@unittest.skipUnless(_data_available(), "Data files not found")
class TestForecastDemandSafetyLevels(unittest.TestCase):
    """Conservative upper bound >= lean upper bound."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.lean = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "window",
            "start_hour": 9,
            "end_hour": 12,
            "safety_level": "lean",
        })
        cls.conservative = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "window",
            "start_hour": 9,
            "end_hour": 12,
            "safety_level": "conservative",
        })

    def test_conservative_ge_lean(self):
        lean_upper = self.lean["machine_1"]["total_volume_upper"]
        cons_upper = self.conservative["machine_1"]["total_volume_upper"]
        self.assertGreaterEqual(cons_upper, lean_upper)


@unittest.skipUnless(_data_available(), "Data files not found")
class TestForecastDemandDay(unittest.TestCase):
    """forecast_demand in day mode (full-day minimum levels)."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.day = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "day",
        })
        cls.d1 = cls.day["machine_1"]
        cls.d2 = cls.day["machine_2"]

    def test_plan_scope_is_day(self):
        self.assertEqual(self.d1.get("plan_scope"), "day")

    def test_recommendations_by_tier_present(self):
        self.assertIsInstance(self.d1.get("recommendations_by_tier"), dict)

    def test_every_rec_has_demand_tier(self):
        for r in self.d1["recommendations"]:
            self.assertIn("demand_tier", r)

    def test_machine_1_sum_equals_total_upper(self):
        upper = self.d1.get("total_volume_upper")
        total = sum(r["recommended_stock"] for r in self.d1["recommendations"])
        self.assertIsNotNone(upper)
        self.assertEqual(total, upper)

    def test_machine_2_sum_equals_total_upper(self):
        upper = self.d2.get("total_volume_upper")
        total = sum(r["recommended_stock"] for r in self.d2["recommendations"])
        self.assertIsNotNone(upper)
        self.assertEqual(total, upper)

    def test_peak_demand_hints_is_list(self):
        self.assertIsInstance(self.d1.get("peak_demand_hints"), list)


@unittest.skipUnless(_data_available(), "Data files not found")
class TestDayModeConfidence(unittest.TestCase):
    """Day mode should label Machine 1 as 'Confident'."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.day = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "day",
        })

    def test_day_mode_confidence_is_confident(self):
        recs = self.day["machine_1"]["recommendations"]
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertEqual(r["confidence"], "Confident")


@unittest.skipUnless(_data_available(), "Data files not found")
class TestWindowUsesTimeSpecificMix(unittest.TestCase):
    """Window proportions can differ from day (global) proportions."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.window = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "window",
            "start_hour": 7,
            "end_hour": 10,
        })
        cls.day = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "day",
        })

    def test_window_uses_time_specific_mix(self):
        win_props = {
            r["product"]: r["proportion"]
            for r in self.window["machine_1"]["recommendations"]
        }
        day_props = {
            r["product"]: r["proportion"]
            for r in self.day["machine_1"]["recommendations"]
        }
        # At least one product should have a different proportion
        diffs = [
            abs(win_props.get(p, 0) - day_props.get(p, 0))
            for p in set(win_props) | set(day_props)
        ]
        self.assertTrue(
            any(d > 0.005 for d in diffs),
            "Expected at least one product proportion to differ between window and day mode",
        )


@unittest.skipUnless(_data_available(), "Data files not found")
class TestToolLoading(unittest.TestCase):
    def test_all_tools_has_four(self):
        from agent.tools import ALL_TOOLS
        self.assertEqual(len(ALL_TOOLS), 4)


@unittest.skipUnless(_data_available(), "Data files not found")
class TestGetSalesSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.tools import get_sales_summary
        cls.get_sales_summary = get_sales_summary

    def test_basic_query(self):
        sales = self.get_sales_summary.invoke({
            "start_date": "2024-06-01",
            "end_date": "2024-06-30",
        })
        self.assertGreater(sales["total_transactions"], 0)
        self.assertGreater(sales["total_revenue"], 0)
        self.assertGreater(len(sales["summary"]), 0)
        self.assertEqual(sales["grouped_by"], "product")

    def test_daypart_grouping(self):
        sales = self.get_sales_summary.invoke({
            "start_date": "2024-06-01",
            "end_date": "2024-06-30",
            "group_by": "daypart",
        })
        self.assertEqual(sales["grouped_by"], "daypart")
        self.assertGreater(len(sales["summary"]), 0)

    def test_empty_range(self):
        sales = self.get_sales_summary.invoke({
            "start_date": "2020-01-01",
            "end_date": "2020-01-31",
        })
        self.assertEqual(sales["total_transactions"], 0)


@unittest.skipUnless(_data_available(), "Data files not found")
class TestGetRevenueInsights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.tools import get_revenue_insights
        cls.get_revenue_insights = get_revenue_insights

    def test_daypart(self):
        dp = self.get_revenue_insights.invoke({"focus": "daypart"})
        self.assertGreater(len(dp["by_machine"]), 0)
        self.assertGreater(len(dp["insight"]), 0)

    def test_product_mix(self):
        mix = self.get_revenue_insights.invoke({"focus": "product_mix"})
        self.assertGreater(len(mix["by_daypart"]), 0)
        self.assertGreater(len(mix["insight"]), 0)


@unittest.skipUnless(_data_available(), "Data files not found")
class TestGetModelInsights(unittest.TestCase):
    def test_returns_expected_structure(self):
        from agent.tools import get_model_insights
        insights = get_model_insights.invoke({})
        self.assertIsInstance(insights.get("coefficients"), list)
        self.assertGreater(len(insights["coefficients"]), 0)
        self.assertIsInstance(insights.get("eval_metrics"), dict)
        self.assertIn("mae", insights["eval_metrics"])
        self.assertIn("interval_coverage_90", insights["eval_metrics"])
        self.assertIn("n_test", insights["eval_metrics"])


@unittest.skipUnless(_data_available(), "Data files not found")
class TestFocusMachines(unittest.TestCase):
    """focus_machines key controls which machines get sliders in the UI."""

    @classmethod
    def setUpClass(cls):
        from db import init_db
        init_db()
        from agent.tools import forecast_demand
        cls.single = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "day",
            "machine_id": "machine_1",
        })
        cls.both = forecast_demand.invoke({
            "target_date": "2024-10-07",
            "plan_scope": "day",
        })

    def test_focus_machines_single(self):
        self.assertEqual(self.single["focus_machines"], ["machine_1"])
        # Both machine blocks are still present
        self.assertIn("machine_1", self.single)
        self.assertIn("machine_2", self.single)

    def test_focus_machines_both(self):
        self.assertEqual(self.both["focus_machines"], ["machine_1", "machine_2"])


@unittest.skipUnless(
    _data_available() and os.environ.get("OPENAI_API_KEY"),
    "Data files or OPENAI_API_KEY not available",
)
class TestAgentCompilation(unittest.TestCase):
    def test_agent_compiles(self):
        from agent.graph import build_agent
        agent = build_agent()
        self.assertIsNotNone(agent)
        self.assertTrue(hasattr(agent, "invoke"))


if __name__ == "__main__":
    unittest.main()
