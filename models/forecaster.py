"""Two-stage demand forecaster for coffee vending machines.

Stage 1: Poisson GLM predicts *total* drinks per (date, hour_bucket, machine).
Stage 2: Historical product-mix proportions split the total into per-product
          stocking recommendations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUR_BUCKETS = ["06-09", "09-12", "12-15", "15-18", "18-21", "outside"]

# Business hours only — used for full-day minimum-inventory planning (excludes "outside").
BUSINESS_HOUR_BUCKETS = ["06-09", "09-12", "12-15", "15-18", "18-21"]

BUCKET_BOUNDS: dict[str, tuple[int, int]] = {
    "06-09": (6, 9),
    "09-12": (9, 12),
    "12-15": (12, 15),
    "15-18": (15, 18),
    "18-21": (18, 21),
}

TRAIN_FRACTION = 0.80        # time-based split on Machine 1 dates


def bucket_label(hb: str) -> str:
    """Convert a bucket key like ``'06-09'`` to ``'06:00–09:00'``."""
    parts = hb.split("-")
    if len(parts) != 2:
        return hb
    return f"{parts[0]}:00–{parts[1]}:00"


def _map_hour_bucket(hour: int) -> str:
    if 6 <= hour < 9:
        return "06-09"
    if 9 <= hour < 12:
        return "09-12"
    if 12 <= hour < 15:
        return "12-15"
    if 15 <= hour < 18:
        return "15-18"
    if 18 <= hour < 21:
        return "18-21"
    return "outside"


def overlapping_buckets(hour_start: int, hour_end: int) -> list[str]:
    """Return the hour_bucket labels that overlap [hour_start, hour_end)."""
    buckets: list[str] = []
    for label, (lo, hi) in BUCKET_BOUNDS.items():
        if hour_start < hi and hour_end > lo:
            buckets.append(label)
    if not buckets:
        buckets.append("outside")
    return buckets


# ---------------------------------------------------------------------------
# DemandForecaster
# ---------------------------------------------------------------------------

class DemandForecaster:
    """Fits at construction time; exposes prediction + recommendation methods."""

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        self._raw = self._load(data_dir)
        self._volume = self._aggregate_volume(self._raw)
        self._feature_cols: list[str] = []

        self._fit_result: sm.genmod.generalized_linear_model.GLMResultsWrapper | None = None
        self._train_idx: np.ndarray | None = None
        self._test_idx: np.ndarray | None = None

        self._fit_model()

        self._product_mix = self._build_product_mix(self._raw)
        self._historical_max = self._compute_historical_max(self._raw)
        self.eval_metrics = self._evaluate()
        self.baseline_metrics = self._evaluate_baselines()

    # ------------------------------------------------------------------
    # Public read-only access to internal data
    # ------------------------------------------------------------------

    @property
    def raw_data(self) -> pd.DataFrame:
        return self._raw

    @property
    def product_mix(self) -> pd.DataFrame:
        return self._product_mix

    @property
    def historical_max(self) -> pd.DataFrame:
        return self._historical_max

    # ------------------------------------------------------------------
    # Data loading (mirrors EDA notebook patterns)
    # ------------------------------------------------------------------

    @staticmethod
    def _load(data_dir: Path) -> pd.DataFrame:
        df1 = pd.read_csv(data_dir / "index_1.csv")
        df2 = pd.read_csv(data_dir / "index_2.csv")
        if "card" not in df2.columns:
            df2["card"] = pd.NA
        df1["machine_id"] = "machine_1"
        df2["machine_id"] = "machine_2"
        df = pd.concat([df1, df2], ignore_index=True)

        for col in ["date", "datetime"]:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

        df["money"] = pd.to_numeric(df["money"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.day_name()
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon … 6=Sun
        df["sale_date"] = df["datetime"].dt.date
        df["hour_bucket"] = df["hour"].map(_map_hour_bucket)
        return df

    @staticmethod
    def _aggregate_volume(df: pd.DataFrame) -> pd.DataFrame:
        vol = (
            df.groupby(["sale_date", "hour_bucket", "machine_id"], as_index=False)
            .size()
            .rename(columns={"size": "total_count"})
        )
        vol["sale_date"] = pd.to_datetime(vol["sale_date"])
        vol["day_of_week"] = vol["sale_date"].dt.dayofweek
        return vol

    # ------------------------------------------------------------------
    # Stage 1: Poisson GLM
    # ------------------------------------------------------------------

    def _build_features(self, vol: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(
            vol[["hour_bucket", "machine_id"]],
            columns=["hour_bucket", "machine_id"],
            drop_first=True,
            dtype=float,
        )
        X["trend"] = vol["trend"].values
        X = sm.add_constant(X)
        self._feature_cols = list(X.columns)
        return X

    def _features_for(self, vol: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix aligned to an already-fitted model."""
        X = pd.get_dummies(
            vol[["hour_bucket", "machine_id"]],
            columns=["hour_bucket", "machine_id"],
            drop_first=True,
            dtype=float,
        )
        X["trend"] = vol["trend"].values
        X = sm.add_constant(X)
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[self._feature_cols]

    def _fit_model(self) -> None:
        vol = self._volume

        self._min_date = vol["sale_date"].min()
        vol["trend"] = (vol["sale_date"] - self._min_date).dt.days / 365.0

        # Global cutoff: 80th percentile of Machine 1's unique dates.
        # Machine 2's data (Feb-Mar 2025) falls entirely after this cutoff,
        # so we include ALL Machine 2 rows in training to learn the machine_id
        # coefficient.  Evaluation uses Machine 1 post-cutoff data only.
        m1_dates = sorted(vol.loc[vol["machine_id"] == "machine_1", "sale_date"].unique())
        cutoff_idx = int(len(m1_dates) * TRAIN_FRACTION)
        cutoff_date = m1_dates[cutoff_idx]

        is_m1 = vol["machine_id"] == "machine_1"
        train_mask = (~is_m1) | (vol["sale_date"] < cutoff_date)
        test_mask = is_m1 & (vol["sale_date"] >= cutoff_date)

        self._train_idx = np.where(train_mask)[0]
        self._test_idx = np.where(test_mask)[0]
        self._cutoff_date = cutoff_date

        train = vol.loc[train_mask]
        X_train = self._build_features(train)
        y_train = train["total_count"]

        model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        self._fit_result = model.fit()

    def _evaluate(self) -> dict[str, Any]:
        """Evaluate on Machine 1 test data only (Machine 2 test set too small)."""
        vol = self._volume
        test = vol.iloc[self._test_idx].copy()
        m1_test = test.loc[test["machine_id"] == "machine_1"].copy()

        if m1_test.empty:
            return {"mae": None, "interval_coverage_90": None, "n_test": 0}

        X_test = self._features_for(m1_test)
        mu = self._fit_result.predict(X_test)

        m1_test = m1_test.copy()
        m1_test["predicted"] = mu.values
        m1_test["lower_90"] = poisson.ppf(0.05, mu.values)
        m1_test["upper_90"] = poisson.ppf(0.95, mu.values)

        mae = float(np.mean(np.abs(m1_test["total_count"] - m1_test["predicted"])))
        in_interval = (
            (m1_test["total_count"] >= m1_test["lower_90"])
            & (m1_test["total_count"] <= m1_test["upper_90"])
        )
        coverage = float(in_interval.mean())

        return {
            "mae": round(mae, 3),
            "interval_coverage_90": round(coverage, 3),
            "n_test": len(m1_test),
            "cutoff_date": str(self._cutoff_date.date()),
            "dispersion": round(
                self._fit_result.deviance / self._fit_result.df_resid, 3
            ),
        }

    def _evaluate_baselines(self) -> dict[str, dict[str, Any]]:
        """Compute MAE and 90% interval coverage for two naive baselines.

        Uses the same Machine 1 test set as ``_evaluate`` so the numbers are
        directly comparable to the Poisson GLM results.
        """
        vol = self._volume
        m1 = vol[vol["machine_id"] == "machine_1"]
        train_m1 = m1[m1["sale_date"] < self._cutoff_date]
        test_m1 = vol.iloc[self._test_idx].copy()
        test_m1 = test_m1[test_m1["machine_id"] == "machine_1"].copy()

        if test_m1.empty:
            empty = {"mae": None, "interval_coverage_90": None, "n_test": 0}
            return {"global_average": empty, "yesterday": empty}

        actual = test_m1["total_count"].values

        # -- Strategy 1: Global Average --------------------------------
        global_means = (
            train_m1
            .groupby(["hour_bucket", "machine_id"], as_index=False)["total_count"]
            .mean()
            .rename(columns={"total_count": "pred_global_avg"})
        )
        ga = test_m1.merge(global_means, on=["hour_bucket", "machine_id"], how="left")
        ga["pred_global_avg"] = ga["pred_global_avg"].fillna(0.0)
        ga_mu = ga["pred_global_avg"].values

        ga_mae = float(np.mean(np.abs(actual - ga_mu)))
        ga_lower = poisson.ppf(0.05, ga_mu)
        ga_upper = poisson.ppf(0.95, ga_mu)
        ga_in = (actual >= ga_lower) & (actual <= ga_upper)
        ga_coverage = float(ga_in.mean())

        # -- Strategy 2: Yesterday's Actual ----------------------------
        vol_lookup = (
            m1.set_index(["sale_date", "hour_bucket"])["total_count"]
            .to_dict()
        )
        yesterday_preds = np.array([
            vol_lookup.get(
                (row["sale_date"] - pd.Timedelta(days=1), row["hour_bucket"]),
                0,
            )
            for _, row in test_m1.iterrows()
        ], dtype=float)

        yd_mae = float(np.mean(np.abs(actual - yesterday_preds)))
        yd_mu = np.maximum(yesterday_preds, 0.0)
        yd_lower = poisson.ppf(0.05, yd_mu)
        yd_upper = poisson.ppf(0.95, yd_mu)
        yd_in = (actual >= yd_lower) & (actual <= yd_upper)
        yd_coverage = float(yd_in.mean())

        return {
            "global_average": {
                "mae": round(ga_mae, 3),
                "interval_coverage_90": round(ga_coverage, 3),
                "n_test": len(test_m1),
            },
            "yesterday": {
                "mae": round(yd_mae, 3),
                "interval_coverage_90": round(yd_coverage, 3),
                "n_test": len(test_m1),
            },
        }

    # ------------------------------------------------------------------
    # Stage 2: Product mix proportions (Machine 1 only)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_product_mix(df: pd.DataFrame) -> pd.DataFrame:
        m1 = df.loc[df["machine_id"] == "machine_1"]
        counts = (
            m1.groupby(["hour_bucket", "coffee_name"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        totals = (
            m1.groupby("hour_bucket", as_index=False)
            .size()
            .rename(columns={"size": "total"})
        )
        mix = counts.merge(totals, on="hour_bucket")
        mix["proportion"] = mix["count"] / mix["total"]
        return mix

    @staticmethod
    def _compute_historical_max(df: pd.DataFrame) -> pd.DataFrame:
        """Max total drinks observed per (hour_bucket, machine_id) day."""
        return (
            df.groupby(["sale_date", "hour_bucket", "machine_id"], as_index=False)
            .size()
            .rename(columns={"size": "total"})
            .groupby(["hour_bucket", "machine_id"], as_index=False)["total"]
            .max()
            .rename(columns={"total": "hist_max"})
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_volume(
        self,
        date: str,
        hour_buckets: list[str] | None = None,
        machine_id: str | None = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Predict total drinks per (machine, hour_bucket).

        Returns
        -------
        {machine_id: {hour_bucket: {"mean": …, "lower": …, "upper": …}}}
        """
        dt = pd.Timestamp(date)
        trend = (dt - self._min_date).days / 365.0
        machines = [machine_id] if machine_id else ["machine_1", "machine_2"]
        if hour_buckets is None:
            hour_buckets = HOUR_BUCKETS

        rows = []
        for mid in machines:
            for hb in hour_buckets:
                rows.append({
                    "hour_bucket": hb,
                    "machine_id": mid,
                    "trend": trend,
                })
        pred_df = pd.DataFrame(rows)
        X = self._features_for(pred_df)
        mu = self._fit_result.predict(X).values

        result: dict[str, dict[str, dict[str, float]]] = {}
        for i, row in pred_df.iterrows():
            mid = row["machine_id"]
            hb = row["hour_bucket"]
            lam = float(mu[i])
            result.setdefault(mid, {})[hb] = {
                "mean": round(lam, 2),
                "lower_90": int(poisson.ppf(0.05, lam)),
                "upper_90": int(poisson.ppf(0.95, lam)),
            }
        return result

    def get_coefficients_summary(self) -> list[dict[str, str]]:
        """Human-readable interpretation of GLM coefficients as multipliers."""
        params = self._fit_result.params
        insights: list[dict[str, str]] = []

        if "trend" in params.index:
            multiplier = round(np.exp(params["trend"]), 2)
            direction = "growing" if multiplier > 1 else "declining"
            insights.append({
                "feature": "Annual trend",
                "vs": "baseline",
                "multiplier": multiplier,
                "interpretation": (
                    f"Demand is {direction} at {multiplier}x per year"
                ),
            })

        baseline_bucket = "06-09"
        for hb in HOUR_BUCKETS[1:]:
            col = f"hour_bucket_{hb}"
            if col in params.index:
                multiplier = round(np.exp(params[col]), 2)
                direction = "busier" if multiplier > 1 else "quieter"
                insights.append({
                    "feature": hb,
                    "vs": baseline_bucket,
                    "multiplier": multiplier,
                    "interpretation": (
                        f"{hb} is {multiplier}x vs {baseline_bucket} ({direction})"
                    ),
                })

        col = "machine_id_machine_2"
        if col in params.index:
            multiplier = round(np.exp(params[col]), 2)
            insights.append({
                "feature": "Machine 2",
                "vs": "Machine 1",
                "multiplier": multiplier,
                "interpretation": (
                    f"Machine 2 sees {multiplier}x the volume of Machine 1"
                ),
            })

        return insights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_merged_mix(
        self, buckets: list[str], machine_id: str
    ) -> dict[str, Any]:
        """Weighted-average product mix across multiple hour buckets."""
        mix = self._product_mix
        subset = mix.loc[mix["hour_bucket"].isin(buckets)]
        if subset.empty:
            return {"proportions": {}, "fallback": machine_id == "machine_2"}

        agg = (
            subset.groupby("coffee_name", as_index=False)["count"]
            .sum()
        )
        total = agg["count"].sum()
        proportions = dict(
            zip(agg["coffee_name"], (agg["count"] / total).round(4))
        )
        return {
            "proportions": proportions,
            "fallback": machine_id == "machine_2",
        }

    @staticmethod
    def confidence_label(machine_id: str, n_buckets: int, plan_scope: str = "window") -> str:
        if machine_id == "machine_2":
            return "Rough estimate (limited Machine 2 data)"
        if plan_scope == "day":
            return "Confident"
        if n_buckets > 2:
            return "Less certain (summed across many windows)"
        return "Confident"

    def machine_1_product_names(self) -> list[str]:
        """All products with at least one sale on Machine 1, stable sort by name."""
        m1 = self._raw.loc[self._raw["machine_id"] == "machine_1"]
        names = sorted(m1["coffee_name"].dropna().unique().tolist())
        return names

    def unique_product_count(self, machine_id: str) -> int:
        """How many distinct drink SKUs appear in historical sales for this machine."""
        s = self._raw.loc[self._raw["machine_id"] == machine_id, "coffee_name"]
        return int(s.dropna().nunique())

    def slot_capacity_for_machine(self, machine_id: str) -> int:
        """Use as vending slot budget: one floor per distinct SKU ever sold (min 1)."""
        return max(self.unique_product_count(machine_id), 1)

    def global_mix_proportions_machine_1(self) -> dict[str, float]:
        """Overall Machine 1 share of each product (all business buckets combined)."""
        m1 = self._raw.loc[
            (self._raw["machine_id"] == "machine_1")
            & (self._raw["hour_bucket"].isin(BUSINESS_HOUR_BUCKETS))
        ]
        if m1.empty:
            return {}
        vc = m1["coffee_name"].value_counts()
        total = int(vc.sum())
        return {str(k): float(v) / total for k, v in vc.items()}

    def peak_hour_bucket_per_product(self) -> dict[str, str]:
        """For each product, the business bucket with the most Machine 1 transactions."""
        m1 = self._raw.loc[
            (self._raw["machine_id"] == "machine_1")
            & (self._raw["hour_bucket"].isin(BUSINESS_HOUR_BUCKETS))
        ]
        if m1.empty:
            return {}
        g = (
            m1.groupby(["coffee_name", "hour_bucket"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
        out: dict[str, str] = {}
        for name, sub in g.groupby("coffee_name"):
            row = sub.loc[sub["n"].idxmax()]
            out[str(name)] = str(row["hour_bucket"])
        return out


def integer_allocation_largest_remainder(
    total_units: int,
    weights: dict[str, float],
) -> dict[str, int]:
    """Split ``total_units`` across keys proportionally to non-negative weights (sums exactly)."""
    if total_units <= 0 or not weights:
        return {k: 0 for k in weights}
    keys = list(weights.keys())
    w = [max(0.0, float(weights[k])) for k in keys]
    s = sum(w)
    if s <= 0:
        return {k: 0 for k in keys}
    raw = [total_units * wi / s for wi in w]
    floors = [int(math.floor(x)) for x in raw]
    rem = total_units - sum(floors)
    fracs = sorted(
        range(len(keys)),
        key=lambda i: raw[i] - floors[i],
        reverse=True,
    )
    for i in range(rem):
        floors[fracs[i % len(fracs)]] += 1
    return {keys[i]: floors[i] for i in range(len(keys))}


def allocate_daily_minimum_levels(
    products: list[str],
    expected_by_product: dict[str, float],
    machine_slot_capacity: int,
    min_floor: int = 1,
) -> tuple[dict[str, int], str | None]:
    """Assign integer minimum levels per SKU, sum ≤ capacity, extras by expected weight.

    If ``min_floor * n_products`` exceeds ``machine_slot_capacity``, levels are
    scaled down in priority order (highest expected first) until the sum fits.
    """
    n = len(products)
    if n == 0:
        return {}, None

    cap = max(machine_slot_capacity, 1)
    weights = {p: max(expected_by_product.get(p, 0.0), 1e-9) for p in products}

    if min_floor * n <= cap:
        base = {p: min_floor for p in products}
        rem = cap - min_floor * n
        if rem <= 0:
            return base, None
        extras = integer_allocation_largest_remainder(
            rem, {p: weights[p] for p in products},
        )
        levels = {p: base[p] + extras.get(p, 0) for p in products}
        return levels, None

    # Not enough capacity for min_floor on every SKU — reduce from lowest-demand first.
    note = (
        f"Total minimum floors ({min_floor}×{n} SKUs) exceed machine capacity "
        f"({cap}); levels were reduced starting from lowest expected demand."
    )
    order = sorted(products, key=lambda p: weights[p])  # ascending — drop first
    levels = {p: min_floor for p in products}
    total = sum(levels.values())
    while total > cap:
        # decrease one unit from lowest-priority product still above 0
        for p in order:
            if levels[p] > 0:
                levels[p] -= 1
                total -= 1
                break
        if total <= cap:
            break
    return levels, note


def assign_demand_tiers(
    products: list[str],
    expected_by_product: dict[str, float],
) -> dict[str, str]:
    """Assign each product to high / moderate / keep_stocked by expected demand tertiles."""
    if not products:
        return {}
    scored = sorted(
        products,
        key=lambda p: expected_by_product.get(p, 0.0),
        reverse=True,
    )
    n = len(scored)
    t1 = max(1, math.ceil(n / 3))
    t2 = max(1, math.ceil(2 * n / 3))
    out: dict[str, str] = {}
    for i, p in enumerate(scored):
        if i < t1:
            out[p] = "high"
        elif i < t2:
            out[p] = "moderate"
        else:
            out[p] = "keep_stocked"
    return out
