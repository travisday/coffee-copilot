"""Revenue and product-mix analytics for coffee vending machines.

No model fitting — pure pandas aggregation over the raw transaction data.
Provides the insights the agent surfaces for Track B (Pricing & Mix) queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .forecaster import _map_hour_bucket


class RevenueAnalyzer:
    """Aggregation-based revenue and product-mix insights."""

    def __init__(self, data_dir_or_df: str | Path | pd.DataFrame) -> None:
        if isinstance(data_dir_or_df, pd.DataFrame):
            self._df = data_dir_or_df
        else:
            self._df = self._load(Path(data_dir_or_df))

    # ------------------------------------------------------------------
    # Data loading (same pipeline as DemandForecaster)
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
        df["sale_date"] = df["datetime"].dt.date
        df["hour_bucket"] = df["hour"].map(_map_hour_bucket)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter(
        self,
        machine_id: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        df = self._df
        if machine_id:
            df = df.loc[df["machine_id"] == machine_id]
        if date_range:
            start = pd.Timestamp(date_range[0]).date()
            end = pd.Timestamp(date_range[1]).date()
            df = df.loc[(df["sale_date"] >= start) & (df["sale_date"] <= end)]
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def daypart_performance(
        self,
        machine_id: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Revenue, transaction count, and avg price by daypart per machine.

        Flags the weakest daypart with a suggestion.
        """
        df = self._filter(machine_id, date_range)
        if df.empty:
            return {"by_machine": {}, "insight": "No data for the selected filters."}

        grouped = (
            df.groupby(["machine_id", "hour_bucket"], as_index=False)
            .agg(
                transactions=("money", "size"),
                revenue=("money", "sum"),
                avg_price=("money", "mean"),
            )
        )
        grouped["revenue"] = grouped["revenue"].round(2)
        grouped["avg_price"] = grouped["avg_price"].round(2)

        by_machine: dict[str, list[dict[str, Any]]] = {}
        for mid, grp in grouped.groupby("machine_id"):
            grp_sorted = grp.sort_values("hour_bucket")
            by_machine[str(mid)] = grp_sorted.drop(columns="machine_id").to_dict(
                orient="records"
            )

        insight = self._daypart_insight(grouped)
        return {"by_machine": by_machine, "insight": insight}

    def product_mix_insights(
        self,
        machine_id: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Top/bottom sellers by daypart, revenue concentration."""
        df = self._filter(machine_id, date_range)
        if df.empty:
            return {"by_daypart": {}, "insight": "No data for the selected filters."}

        product_dp = (
            df.groupby(["hour_bucket", "coffee_name"], as_index=False)
            .agg(
                transactions=("money", "size"),
                revenue=("money", "sum"),
            )
        )
        product_dp["revenue"] = product_dp["revenue"].round(2)

        by_daypart: dict[str, dict[str, Any]] = {}
        for hb, grp in product_dp.groupby("hour_bucket"):
            grp_sorted = grp.sort_values("revenue", ascending=False)
            total_rev = grp_sorted["revenue"].sum()
            grp_sorted["share"] = (grp_sorted["revenue"] / total_rev).round(4)
            by_daypart[str(hb)] = {
                "top_3": (
                    grp_sorted.head(3)[["coffee_name", "revenue", "share"]]
                    .to_dict(orient="records")
                ),
                "bottom_3": (
                    grp_sorted.tail(3)[["coffee_name", "revenue", "share"]]
                    .to_dict(orient="records")
                ),
                "total_revenue": round(float(total_rev), 2),
            }

        insight = self._mix_insight(product_dp)
        return {"by_daypart": by_daypart, "insight": insight}

    # ------------------------------------------------------------------
    # Insight generators
    # ------------------------------------------------------------------

    @staticmethod
    def _daypart_insight(grouped: pd.DataFrame) -> str:
        # Use machine_1 if present, otherwise whatever machine is available
        if "machine_1" in grouped["machine_id"].values:
            subset = grouped.loc[grouped["machine_id"] == "machine_1"]
        else:
            first_mid = grouped["machine_id"].iloc[0]
            subset = grouped.loc[grouped["machine_id"] == first_mid]
        if subset.empty:
            return ""
        best = subset.loc[subset["revenue"].idxmax()]
        worst = subset.loc[subset["revenue"].idxmin()]
        if best["revenue"] == 0:
            return "No revenue recorded in the selected range."
        pct_gap = round((1 - worst["revenue"] / best["revenue"]) * 100)
        return (
            f"{worst['hour_bucket']} revenue is {pct_gap}% below "
            f"{best['hour_bucket']} (the strongest window). "
            f"Consider promotions or product changes in the "
            f"{worst['hour_bucket']} window to close the gap."
        )

    @staticmethod
    def _mix_insight(product_dp: pd.DataFrame) -> str:
        overall = (
            product_dp.groupby("coffee_name", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
        )
        if overall.empty:
            return ""
        total = overall["revenue"].sum()
        top = overall.iloc[0]
        share = round(top["revenue"] / total * 100)
        return (
            f"{top['coffee_name']} dominates overall revenue at {share}% share. "
            f"The top 3 products account for "
            f"{round(overall.head(3)['revenue'].sum() / total * 100)}% of all revenue."
        )
