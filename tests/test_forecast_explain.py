"""Unit tests for forecast_data normalization (no Streamlit runtime required for logic)."""

from __future__ import annotations

import unittest

from ui.forecast_explain import iter_forecast_machines, normalize_forecast_data


class TestNormalizeForecastData(unittest.TestCase):
    def test_canonical_keys(self) -> None:
        raw = {
            "Machine 1": {"recommendations": [{"product": "Latte"}], "total_volume_mean": 1.0},
            "m2": {"recommendations": [], "total_volume_mean": 0.5},
        }
        out = normalize_forecast_data(raw)
        assert out is not None
        self.assertIn("machine_1", out)
        self.assertIn("machine_2", out)
        self.assertEqual(out["machine_1"]["total_volume_mean"], 1.0)

    def test_unknown_blocks_fill_slots(self) -> None:
        raw = {
            "a": {"recommendations": [], "total_volume_mean": 2.0},
            "b": {"recommendations": [], "total_volume_mean": 1.0},
        }
        out = normalize_forecast_data(raw)
        assert out is not None
        self.assertIn("machine_1", out)
        self.assertIn("machine_2", out)

    def test_iter_order(self) -> None:
        fd = {
            "machine_2": {"recommendations": []},
            "machine_1": {"recommendations": []},
        }
        order = [m for m, _ in iter_forecast_machines(fd)]
        self.assertEqual(order, ["machine_1", "machine_2"])


if __name__ == "__main__":
    unittest.main()
