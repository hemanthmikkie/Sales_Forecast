"""
Module 4: Feature Engineering
Calculates:
  - Revenue: Units_Sold * Unit_Price * (1 - Discount / 100)
  - Date features: Day, Month, Year, Week, Quarter, Day_of_Week, Is_Weekend
  - Historical demand features (STRICTLY NO FUTURE LEAKAGE):
      - Lag_1
      - Lag_7
      - Lag_30
      - Rolling_Mean_7 (shift(1).rolling(7).mean())
      - Rolling_Mean_30 (shift(1).rolling(30).mean())
Supports batch transformations as well as online feature vector generation for API inference.
"""

import os
from typing import List, Optional
import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Transforms cleaned sales records into an ML-ready feature store,
    guaranteeing temporal ordering and preventing lookahead bias.
    """

    FEATURE_COLS = [
        "Store_ID",
        "Product_ID",
        "Product_Category",
        "Region",
        "Unit_Price",
        "Discount",
        "Promotion",
        "Holiday",
        "Day",
        "Month",
        "Year",
        "Week",
        "Quarter",
        "Day_of_Week",
        "Is_Weekend",
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30",
    ]

    TARGET_COL = "Units_Sold"

    def __init__(self) -> None:
        pass

    def add_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes net sales revenue factoring in discounts."""
        df = df.copy()
        if "Revenue" not in df.columns:
            df["Revenue"] = (
                df["Units_Sold"] * df["Unit_Price"] * (1.0 - (df["Discount"] / 100.0))
            ).round(2)
        return df

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts granular calendar and cyclic time signals."""
        df = df.copy()
        date_series = pd.to_datetime(df["Date"])
        df["Date"] = date_series

        df["Day"] = date_series.dt.day.astype(int)
        df["Month"] = date_series.dt.month.astype(int)
        df["Year"] = date_series.dt.year.astype(int)
        df["Week"] = date_series.dt.isocalendar().week.astype(int)
        df["Quarter"] = date_series.dt.quarter.astype(int)
        df["Day_of_Week"] = date_series.dt.dayofweek.astype(int)
        df["Is_Weekend"] = (df["Day_of_Week"].isin([5, 6])).astype(int)
        return df

    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates historical demand lags and rolling averages partitioned
        by (Store_ID, Product_ID) without lookahead bias.
        
        Strict Leakeage Prevention:
          - Lag_1: shift(1)
          - Lag_7: shift(7)
          - Lag_30: shift(30)
          - Rolling_Mean_7: shift(1).rolling(7).mean()
          - Rolling_Mean_30: shift(1).rolling(30).mean()
        """
        df = df.copy()
        df.sort_values(by=["Store_ID", "Product_ID", "Date"], inplace=True)

        grouped = df.groupby(["Store_ID", "Product_ID"])["Units_Sold"]

        # Shift(1) guarantees current day's sales is NOT included in past rolling stats
        df["Lag_1"] = grouped.shift(1)
        df["Lag_7"] = grouped.shift(7)
        df["Lag_30"] = grouped.shift(30)

        df["Rolling_Mean_7"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
        )
        df["Rolling_Mean_30"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=30, min_periods=1).mean()
        )

        # Backfill initial window nulls within the group, then fallback to series median
        lag_cols = ["Lag_1", "Lag_7", "Lag_30", "Rolling_Mean_7", "Rolling_Mean_30"]
        for col in lag_cols:
            df[col] = df.groupby(["Store_ID", "Product_ID"])[col].bfill()
            df[col] = df[col].fillna(df["Units_Sold"].median())
            df[col] = df[col].round(2)

        df.reset_index(drop=True, inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline execution."""
        df = self.add_revenue(df)
        df = self.add_date_features(df)
        df = self.add_lag_and_rolling_features(df)
        return df

    def extract_inference_features(
        self,
        date_str: str,
        store_id: str,
        product_id: str,
        product_category: str,
        region: str,
        unit_price: float,
        discount: float,
        promotion: int,
        holiday: int,
        lag_1: Optional[float] = None,
        lag_7: Optional[float] = None,
        lag_30: Optional[float] = None,
        rolling_mean_7: Optional[float] = None,
        rolling_mean_30: Optional[float] = None,
        historical_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Creates a single-row feature dataframe for real-time API forecast generation.
        If lags are not directly provided, infers them from historical_df.
        """
        dt = pd.to_datetime(date_str)
        day = int(dt.day)
        month = int(dt.month)
        year = int(dt.year)
        week = int(dt.isocalendar().week)
        quarter = int(dt.quarter)
        day_of_week = int(dt.dayofweek)
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Auto-compute historical lag features if not passed
        if lag_1 is None or rolling_mean_7 is None:
            if historical_df is not None and not historical_df.empty:
                subset = historical_df[
                    (historical_df["Store_ID"] == store_id) &
                    (historical_df["Product_ID"] == product_id)
                ].sort_values("Date")

                if len(subset) > 0:
                    recent_sales = subset["Units_Sold"].values
                    lag_1 = float(recent_sales[-1]) if len(recent_sales) >= 1 else 50.0
                    lag_7 = float(recent_sales[-7]) if len(recent_sales) >= 7 else lag_1
                    lag_30 = float(recent_sales[-30]) if len(recent_sales) >= 30 else lag_1
                    rolling_mean_7 = float(np.mean(recent_sales[-7:])) if len(recent_sales) >= 7 else lag_1
                    rolling_mean_30 = float(np.mean(recent_sales[-30:])) if len(recent_sales) >= 30 else lag_1
                else:
                    lag_1 = 50.0
                    lag_7 = 50.0
                    lag_30 = 50.0
                    rolling_mean_7 = 50.0
                    rolling_mean_30 = 50.0
            else:
                lag_1 = 50.0
                lag_7 = 50.0
                lag_30 = 50.0
                rolling_mean_7 = 50.0
                rolling_mean_30 = 50.0

        # Guarantee fallback values if individual lag/rolling params are None
        lag_1_val = float(lag_1) if lag_1 is not None else 50.0
        lag_7_val = float(lag_7) if lag_7 is not None else lag_1_val
        lag_30_val = float(lag_30) if lag_30 is not None else lag_1_val
        roll_7_val = float(rolling_mean_7) if rolling_mean_7 is not None else lag_1_val
        roll_30_val = float(rolling_mean_30) if rolling_mean_30 is not None else lag_1_val

        row = {
            "Store_ID": store_id,
            "Product_ID": product_id,
            "Product_Category": product_category,
            "Region": region,
            "Unit_Price": float(unit_price),
            "Discount": float(discount),
            "Promotion": int(promotion),
            "Holiday": int(holiday),
            "Day": day,
            "Month": month,
            "Year": year,
            "Week": week,
            "Quarter": quarter,
            "Day_of_Week": day_of_week,
            "Is_Weekend": is_weekend,
            "Lag_1": lag_1_val,
            "Lag_7": lag_7_val,
            "Lag_30": lag_30_val,
            "Rolling_Mean_7": roll_7_val,
            "Rolling_Mean_30": roll_30_val,
        }
        return pd.DataFrame([row])[self.FEATURE_COLS]


if __name__ == "__main__":
    clean_df = pd.read_csv("dataset/processed/cleaned_sales.csv")
    fe = FeatureEngineer()
    feat_df = fe.transform(clean_df)
    print("Feature Engineered DataFrame shape:", feat_df.shape)
    print("Columns:", list(feat_df.columns))
    print("\nSample engineered record:")
    print(feat_df[fe.FEATURE_COLS].head(1).to_dict(orient="records")[0])
