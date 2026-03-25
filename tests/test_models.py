"""Smoke tests for DemandForecaster and RevenueAnalyzer.

Run from the repo root:
    python -m pytest tests/test_models.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _data_available() -> bool:
    return (DATA_DIR / "index_1.csv").exists()


@unittest.skipUnless(_data_available(), "Data files not found")
class TestDemandForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.forecaster import DemandForecaster, HOUR_BUCKETS
        cls.fc = DemandForecaster(DATA_DIR)
        cls.HOUR_BUCKETS = HOUR_BUCKETS

    # --- Eval metrics ---

    def test_eval_metrics_populated(self):
        self.assertIsNotNone(self.fc.eval_metrics["mae"])

    def test_mae_reasonable(self):
        self.assertLess(self.fc.eval_metrics["mae"], 3.0)

    def test_coverage_near_90(self):
        cov = self.fc.eval_metrics["interval_coverage_90"]
        self.assertGreaterEqual(cov, 0.75)
        self.assertLessEqual(cov, 0.98)

    def test_dispersion_near_one(self):
        d = self.fc.eval_metrics["dispersion"]
        self.assertGreaterEqual(d, 0.4)
        self.assertLessEqual(d, 1.6)

    def test_test_set_size(self):
        self.assertGreater(self.fc.eval_metrics["n_test"], 100)

    # --- Baseline metrics ---

    def test_baseline_global_average(self):
        bm = self.fc.baseline_metrics["global_average"]
        self.assertIsNotNone(bm["mae"])
        self.assertGreaterEqual(bm["interval_coverage_90"], 0.0)
        self.assertLessEqual(bm["interval_coverage_90"], 1.0)
        self.assertEqual(bm["n_test"], self.fc.eval_metrics["n_test"])

    def test_baseline_yesterday(self):
        bm = self.fc.baseline_metrics["yesterday"]
        self.assertIsNotNone(bm["mae"])
        self.assertGreaterEqual(bm["interval_coverage_90"], 0.0)
        self.assertLessEqual(bm["interval_coverage_90"], 1.0)
        self.assertEqual(bm["n_test"], self.fc.eval_metrics["n_test"])

    # --- predict_volume ---

    def test_predict_volume_single_machine(self):
        vol = self.fc.predict_volume("2024-10-07", ["09-12"], machine_id="machine_1")
        v = vol["machine_1"]["09-12"]
        self.assertIn("mean", v)
        self.assertGreater(v["mean"], 0)
        self.assertGreaterEqual(v["upper_90"], v["mean"])

    def test_machine_2_lower_than_machine_1(self):
        vol = self.fc.predict_volume("2024-10-07", ["09-12"])
        m1_mean = vol["machine_1"]["09-12"]["mean"]
        m2_mean = vol["machine_2"]["09-12"]["mean"]
        self.assertLess(m2_mean, m1_mean)

    # --- get_merged_mix ---

    def test_mix_has_proportions(self):
        mix = self.fc.get_merged_mix(["09-12"], "machine_1")
        self.assertGreater(len(mix["proportions"]), 0)
        total = sum(mix["proportions"].values())
        self.assertAlmostEqual(total, 1.0, delta=0.01)
        self.assertFalse(mix["fallback"])

    def test_machine_2_fallback(self):
        mix = self.fc.get_merged_mix(["09-12"], "machine_2")
        self.assertTrue(mix["fallback"])

    # --- get_coefficients_summary ---

    def test_coefficients_returned(self):
        coefs = self.fc.get_coefficients_summary()
        self.assertGreater(len(coefs), 0)

    def test_coefficients_include_machine_2(self):
        coefs = self.fc.get_coefficients_summary()
        self.assertTrue(any("Machine 2" in c["feature"] for c in coefs))

    def test_coefficients_include_hour_buckets(self):
        coefs = self.fc.get_coefficients_summary()
        self.assertTrue(any(c["feature"] in self.HOUR_BUCKETS for c in coefs))


@unittest.skipUnless(_data_available(), "Data files not found")
class TestRevenueAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.analyzer import RevenueAnalyzer
        cls.ra = RevenueAnalyzer(DATA_DIR)

    # --- daypart_performance ---

    def test_daypart_has_by_machine(self):
        dp = self.ra.daypart_performance()
        self.assertGreater(len(dp["by_machine"]), 0)

    def test_daypart_has_insight(self):
        dp = self.ra.daypart_performance()
        self.assertGreater(len(dp["insight"]), 0)

    def test_machine_1_dayparts(self):
        dp = self.ra.daypart_performance()
        m1_dp = dp["by_machine"]["machine_1"]
        self.assertGreaterEqual(len(m1_dp), 5)
        for row in m1_dp:
            self.assertGreater(row["transactions"], 0)
            self.assertGreater(row["revenue"], 0)

    def test_machine_2_filter(self):
        dp = self.ra.daypart_performance(machine_id="machine_2")
        self.assertIn("machine_2", dp["by_machine"])
        self.assertGreater(len(dp["insight"]), 0)

    # --- product_mix_insights ---

    def test_product_mix_has_by_daypart(self):
        pmi = self.ra.product_mix_insights()
        self.assertGreater(len(pmi["by_daypart"]), 0)

    def test_product_mix_has_insight(self):
        pmi = self.ra.product_mix_insights()
        self.assertGreater(len(pmi["insight"]), 0)

    def test_product_mix_top_3(self):
        pmi = self.ra.product_mix_insights()
        for hb in ["09-12", "15-18"]:
            d = pmi["by_daypart"].get(hb, {})
            top = d.get("top_3", [])
            self.assertEqual(len(top), 3)

    # --- date range filter ---

    def test_date_range_filter(self):
        dp = self.ra.daypart_performance(date_range=("2024-06-01", "2024-06-30"))
        self.assertGreater(len(dp["by_machine"]), 0)

    def test_empty_date_range(self):
        dp = self.ra.daypart_performance(date_range=("2020-01-01", "2020-01-31"))
        self.assertEqual(dp["by_machine"], {})


if __name__ == "__main__":
    unittest.main()
