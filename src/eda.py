"""
Module 3: Exploratory Data Analysis (EDA)
Analyzes:
  - Daily, weekly, monthly, and yearly sales trends
  - Product-wise and category-wise sales
  - Store-wise and region-wise sales
  - Promotion and holiday impact
  - Seasonal demand patterns
  - Top-selling and lowest-selling products
Generates publication-quality charts into reports/figures/
"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ExploratoryDataAnalysis:
    """
    Executes deep exploratory data analysis on the cleaned sales dataset
    and produces high-resolution visual reports.
    """

    def __init__(self, data_path: str = "dataset/processed/cleaned_sales.csv") -> None:
        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        # Ensure Revenue feature is available for analysis
        if "Revenue" not in self.df.columns:
            self.df["Revenue"] = (
                self.df["Units_Sold"] * self.df["Unit_Price"] * (1 - self.df["Discount"] / 100.0)
            )

        os.makedirs("reports/figures", exist_ok=True)
        # Configure seaborn styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    def run_all_analysis(self) -> Dict[str, Any]:
        """Runs all analytical segments and exports corresponding visual charts."""
        print("[EDA] Starting Exploratory Data Analysis...")
        summary = {}

        summary["temporal_trends"] = self.plot_temporal_trends()
        summary["product_category"] = self.plot_product_category_analysis()
        summary["store_region"] = self.plot_store_region_analysis()
        summary["promotions_holidays"] = self.plot_promotion_holiday_impact()
        summary["top_low_products"] = self.plot_top_and_low_products()

        print("[EDA] Completed all analyses. Charts saved in 'reports/figures/'.")
        return summary

    def plot_temporal_trends(self) -> Dict[str, Any]:
        """Analyzes Daily, Weekly, Monthly, and Yearly sales trends."""
        daily_sales = self.df.groupby("Date")["Units_Sold"].sum().reset_index()

        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["YearMonth"] = self.df["Date"].dt.to_period("M").astype(str)
        self.df["DayOfWeek"] = self.df["Date"].dt.day_name()

        monthly_sales = self.df.groupby("YearMonth")["Units_Sold"].sum().reset_index()
        yearly_sales = self.df.groupby("Year")["Units_Sold"].sum().to_dict()

        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_sales = self.df.groupby("DayOfWeek")["Units_Sold"].mean().reindex(dow_order)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Daily sales timeline
        axes[0, 0].plot(daily_sales["Date"], daily_sales["Units_Sold"], color="#1f77b4", alpha=0.7, lw=1)
        axes[0, 0].set_title("Overall Daily Units Sold Trend", fontsize=13, fontweight="bold")
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Total Units Sold")

        # 2. Monthly sales trend
        axes[0, 1].plot(monthly_sales["YearMonth"], monthly_sales["Units_Sold"], marker="o", color="#ff7f0e", lw=2)
        axes[0, 1].set_title("Monthly Total Units Sold", fontsize=13, fontweight="bold")
        axes[0, 1].set_xlabel("Year-Month")
        axes[0, 1].set_ylabel("Total Units Sold")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Day of week average sales
        sns.barplot(x=dow_sales.index, y=dow_sales.values, ax=axes[1, 0], palette="Blues_d")
        axes[1, 0].set_title("Average Units Sold by Day of Week (Weekend Lift)", fontsize=13, fontweight="bold")
        axes[1, 0].set_xlabel("Day of Week")
        axes[1, 0].set_ylabel("Mean Units Sold per Store-Product")
        axes[1, 0].tick_params(axis="x", rotation=30)

        # 4. Yearly sales comparison
        years = list(yearly_sales.keys())
        totals = list(yearly_sales.values())
        sns.barplot(x=years, y=totals, ax=axes[1, 1], palette="Greens_d")
        axes[1, 1].set_title("Yearly Total Sales Volume", fontsize=13, fontweight="bold")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Total Units Sold")

        plt.savefig("reports/figures/01_temporal_sales_trends.png", dpi=200)
        plt.close()
        return {"yearly_sales": yearly_sales, "total_daily_records": len(daily_sales)}

    def plot_product_category_analysis(self) -> Dict[str, Any]:
        """Analyzes Category-wise and Product-wise sales performance."""
        cat_agg = self.df.groupby("Product_Category").agg(
            Total_Units=("Units_Sold", "sum"),
            Total_Revenue=("Revenue", "sum")
        ).sort_values("Total_Units", ascending=False).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=cat_agg, x="Product_Category", y="Total_Units", ax=axes[0], palette="viridis")
        axes[0].set_title("Total Units Sold by Product Category", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Category")
        axes[0].set_ylabel("Total Units Sold")
        axes[0].tick_params(axis="x", rotation=25)

        sns.barplot(data=cat_agg, x="Product_Category", y="Total_Revenue", ax=axes[1], palette="magma")
        axes[1].set_title("Total Revenue by Product Category ($)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Revenue ($)")
        axes[1].tick_params(axis="x", rotation=25)

        plt.savefig("reports/figures/02_category_sales_analysis.png", dpi=200)
        plt.close()
        return cat_agg.to_dict(orient="records")

    def plot_store_region_analysis(self) -> Dict[str, Any]:
        """Analyzes Store-wise and Region-wise sales distributions."""
        store_agg = self.df.groupby(["Store_ID", "Region"]).agg(
            Total_Units=("Units_Sold", "sum"),
            Avg_Daily_Units=("Units_Sold", "mean"),
            Total_Revenue=("Revenue", "sum")
        ).reset_index().sort_values("Total_Units", ascending=False)

        region_agg = self.df.groupby("Region")["Units_Sold"].sum().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.barplot(data=store_agg, x="Store_ID", y="Total_Units", hue="Region", ax=axes[0], dodge=False)
        axes[0].set_title("Sales Volume by Store ID and Region", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Store ID")
        axes[0].set_ylabel("Units Sold")

        axes[1].pie(
            region_agg["Units_Sold"],
            labels=region_agg["Region"],
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel")
        )
        axes[1].set_title("Regional Sales Distribution (%)", fontsize=13, fontweight="bold")

        plt.savefig("reports/figures/03_store_region_analysis.png", dpi=200)
        plt.close()
        return store_agg.to_dict(orient="records")

    def plot_promotion_holiday_impact(self) -> Dict[str, Any]:
        """Evaluates promotional uplift and holiday effects on sales demand."""
        promo_impact = self.df.groupby("Promotion")["Units_Sold"].agg(["mean", "std", "count"]).rename(
            index={0: "No Promotion", 1: "Active Promotion"}
        )

        holiday_impact = self.df.groupby("Holiday")["Units_Sold"].agg(["mean", "std", "count"]).rename(
            index={0: "Non-Holiday", 1: "Holiday"}
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(data=self.df, x="Promotion", y="Units_Sold", ax=axes[0], palette="Set2")
        axes[0].set_xticklabels(["Regular Days", "Promotional Days"])
        axes[0].set_title("Promotion Impact on Units Sold", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Units Sold")

        sns.boxplot(data=self.df, x="Holiday", y="Units_Sold", ax=axes[1], palette="Set1")
        axes[1].set_xticklabels(["Non-Holiday Days", "Holiday Days"])
        axes[1].set_title("Holiday Impact on Units Sold", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Units Sold")

        plt.savefig("reports/figures/04_promotion_holiday_impact.png", dpi=200)
        plt.close()

        promo_lift_pct = round(
            ((promo_impact.loc["Active Promotion", "mean"] - promo_impact.loc["No Promotion", "mean"])
             / promo_impact.loc["No Promotion", "mean"]) * 100, 2
        )
        holiday_lift_pct = round(
            ((holiday_impact.loc["Holiday", "mean"] - holiday_impact.loc["Non-Holiday", "mean"])
             / holiday_impact.loc["Non-Holiday", "mean"]) * 100, 2
        )
        return {"promo_lift_pct": promo_lift_pct, "holiday_lift_pct": holiday_lift_pct}

    def plot_top_and_low_products(self) -> Dict[str, Any]:
        """Identifies top-selling and lowest-selling products."""
        prod_agg = self.df.groupby(["Product_ID", "Product_Category"]).agg(
            Total_Units=("Units_Sold", "sum"),
            Avg_Price=("Unit_Price", "mean"),
            Total_Revenue=("Revenue", "sum")
        ).reset_index().sort_values("Total_Units", ascending=False)

        top_products = prod_agg.head(5)
        low_products = prod_agg.tail(5)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=top_products, x="Product_ID", y="Total_Units", hue="Product_Category", ax=axes[0], dodge=False)
        axes[0].set_title("Top 5 Selling Products (Volume)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Product ID")
        axes[0].set_ylabel("Total Units Sold")

        sns.barplot(data=low_products, x="Product_ID", y="Total_Units", hue="Product_Category", ax=axes[1], dodge=False)
        axes[1].set_title("Lowest 5 Selling Products (Volume)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Product ID")
        axes[1].set_ylabel("Total Units Sold")

        plt.savefig("reports/figures/05_top_low_products.png", dpi=200)
        plt.close()

        return {
            "top_products": top_products.to_dict(orient="records"),
            "low_products": low_products.to_dict(orient="records"),
        }


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis("dataset/processed/cleaned_sales.csv")
    results = eda.run_all_analysis()
    print("Promotional Lift:", results["promotions_holidays"]["promo_lift_pct"], "%")
    print("Holiday Lift:", results["promotions_holidays"]["holiday_lift_pct"], "%")
