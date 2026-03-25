#!/usr/bin/env python3
"""Smoke-test runner for DemandForecaster and RevenueAnalyzer.

Run from the repo root:
    .venv/bin/python -m tests.test_models
"""

from __future__ import annotations

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

    # ---------------------------------------------------------------
    # DemandForecaster
    # ---------------------------------------------------------------
    print("\n=== DemandForecaster ===\n")

    from models.forecaster import DemandForecaster, HOUR_BUCKETS

    print("  Loading & fitting model...")
    fc = DemandForecaster(data_dir)
    print()

    # --- Eval metrics ---
    m = fc.eval_metrics
    check("eval_metrics populated", m["mae"] is not None)
    check(f"MAE reasonable (<3.0): {m['mae']}", m["mae"] < 3.0)
    check(
        f"90% coverage near 90%: {m['interval_coverage_90']}",
        0.75 <= m["interval_coverage_90"] <= 0.98,
    )
    check(
        f"dispersion near 1.0: {m['dispersion']}",
        0.4 <= m["dispersion"] <= 1.6,
    )
    check(f"test set size: {m['n_test']}", m["n_test"] > 100)

    # --- Baseline metrics ---
    bm = fc.baseline_metrics
    for strategy in ("global_average", "yesterday"):
        check(f"baseline '{strategy}' has mae", bm[strategy]["mae"] is not None)
        check(
            f"baseline '{strategy}' coverage in [0,1]: {bm[strategy]['interval_coverage_90']}",
            0.0 <= bm[strategy]["interval_coverage_90"] <= 1.0,
        )
        check(
            f"baseline '{strategy}' test size matches: {bm[strategy]['n_test']}",
            bm[strategy]["n_test"] == m["n_test"],
        )

    # --- predict_volume ---
    print()
    vol = fc.predict_volume("2024-10-07", ["09-12"], machine_id="machine_1")
    v = vol["machine_1"]["09-12"]
    check("predict_volume returns mean", "mean" in v)
    check(f"mean > 0: {v['mean']}", v["mean"] > 0)
    check(f"upper_90 >= mean: {v['upper_90']} >= {v['mean']}", v["upper_90"] >= v["mean"])

    vol_both = fc.predict_volume("2024-10-07", ["09-12"])
    m1_mean = vol_both["machine_1"]["09-12"]["mean"]
    m2_mean = vol_both["machine_2"]["09-12"]["mean"]
    check(
        f"Machine 2 lower than Machine 1: {m2_mean} < {m1_mean}",
        m2_mean < m1_mean,
    )

    # --- get_merged_mix ---
    print()
    mix = fc.get_merged_mix(["09-12"], "machine_1")
    check("mix has proportions", len(mix["proportions"]) > 0)
    total_prop = sum(mix["proportions"].values())
    check(f"proportions sum to ~1.0: {total_prop:.4f}", abs(total_prop - 1.0) < 0.01)
    check("machine_1 not flagged as fallback", mix["fallback"] is False)

    mix2 = fc.get_merged_mix(["09-12"], "machine_2")
    check("machine_2 flagged as fallback", mix2["fallback"] is True)

    # --- get_coefficients_summary ---
    print()
    coefs = fc.get_coefficients_summary()
    check(f"coefficients returned: {len(coefs)} insights", len(coefs) > 0)
    has_machine = any("Machine 2" in c["feature"] for c in coefs)
    check("includes Machine 2 coefficient", has_machine)
    has_bucket = any(c["feature"] in HOUR_BUCKETS for c in coefs)
    check("includes hour_bucket coefficients", has_bucket)

    for c in coefs:
        print(f"    {c['interpretation']}")

    # ---------------------------------------------------------------
    # RevenueAnalyzer
    # ---------------------------------------------------------------
    print("\n=== RevenueAnalyzer ===\n")

    from models.analyzer import RevenueAnalyzer

    ra = RevenueAnalyzer(data_dir)

    # --- daypart_performance ---
    dp = ra.daypart_performance()
    check("daypart has by_machine", len(dp["by_machine"]) > 0)
    check("daypart has insight", len(dp["insight"]) > 0)

    m1_dp = dp["by_machine"]["machine_1"]
    check(f"machine_1 has {len(m1_dp)} dayparts", len(m1_dp) >= 5)
    for row in m1_dp:
        check(
            f"  {row['hour_bucket']}: {row['transactions']} txns, rev={row['revenue']}",
            row["transactions"] > 0 and row["revenue"] > 0,
        )

    dp_m2 = ra.daypart_performance(machine_id="machine_2")
    check("machine_2 filter works", "machine_2" in dp_m2["by_machine"])
    check("machine_2 insight populated", len(dp_m2["insight"]) > 0)

    # --- product_mix_insights ---
    print()
    pmi = ra.product_mix_insights()
    check("product_mix has by_daypart", len(pmi["by_daypart"]) > 0)
    check("product_mix has insight", len(pmi["insight"]) > 0)
    print(f"    Insight: {pmi['insight']}")

    for hb in ["09-12", "15-18"]:
        d = pmi["by_daypart"].get(hb, {})
        top = d.get("top_3", [])
        check(f"  {hb} top 3: {[t['coffee_name'] for t in top]}", len(top) == 3)

    # --- date range filter ---
    dp_range = ra.daypart_performance(date_range=("2024-06-01", "2024-06-30"))
    check("date range filter works", len(dp_range["by_machine"]) > 0)

    dp_empty = ra.daypart_performance(date_range=("2020-01-01", "2020-01-31"))
    check("empty range returns gracefully", dp_empty["by_machine"] == {})

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
