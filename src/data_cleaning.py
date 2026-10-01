"""
Module 2: Data Cleaning
Handles:
  - Missing value imputation
  - Duplicate detection and removal
  - Datetime conversion & chronological sorting
  - Domain validations (Units_Sold >= 0, Unit_Price > 0, Inventory_Level >= 0)
  - Product ID & Store ID regex validation
  - Statistical outlier identification
  - Data type normalization
  - Saving cleaned dataset separately to 'dataset/processed/cleaned_sales.csv'
"""

import os
import re
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class DataCleaner:
    """
    DataCleaner provides automated, configurable data hygiene routines
    to prepare raw retail transaction logs for modeling.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.raw_df = df.copy()
        self.df = df.copy()
        self.cleaning_audit: Dict[str, Any] = {}

    def clean(self) -> pd.DataFrame:
        """
        Executes the full end-to-end cleaning pipeline.
        """
        initial_rows = len(self.df)

        # 1. Remove duplicate rows
        duplicates_count = int(self.df.duplicated().sum())
        self.df.drop_duplicates(inplace=True)

        # 2. Date conversion & validation
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        invalid_dates = self.df["Date"].isnull().sum()
        if invalid_dates > 0:
            self.df = self.df.dropna(subset=["Date"])

        # 3. Store ID and Product ID pattern validation
        valid_store_pattern = re.compile(r"^S\d{2,}$")
        valid_prod_pattern = re.compile(r"^P\d{2,}$")

        valid_stores = self.df["Store_ID"].astype(str).str.match(valid_store_pattern)
        valid_prods = self.df["Product_ID"].astype(str).str.match(valid_prod_pattern)
        id_filtered = ~(valid_stores & valid_prods)
        invalid_ids_count = int(id_filtered.sum())
        if invalid_ids_count > 0:
            self.df = self.df[valid_stores & valid_prods]

        # 4. Domain Validations (Sales, Prices, Inventory)
        # Invalid sales (< 0)
        invalid_sales_count = int((self.df["Units_Sold"] < 0).sum())
        self.df = self.df[self.df["Units_Sold"] >= 0]

        # Invalid prices (<= 0 or missing before imputation)
        invalid_price_count = int((self.df["Unit_Price"] <= 0).sum(skipna=True))
        self.df = self.df[(self.df["Unit_Price"].isna()) | (self.df["Unit_Price"] > 0)]

        # 5. Missing Value Imputation
        missing_before = self.df.isnull().sum().to_dict()

        # Discount: if missing, default to 0.0
        self.df["Discount"] = self.df["Discount"].fillna(0.0)

        # Unit_Price: impute with median price per Product_ID
        product_median_prices = self.df.groupby("Product_ID")["Unit_Price"].transform("median")
        self.df["Unit_Price"] = self.df["Unit_Price"].fillna(product_median_prices)

        # Inventory_Level: impute with median per (Store_ID, Product_ID)
        store_prod_median_inv = self.df.groupby(["Store_ID", "Product_ID"])["Inventory_Level"].transform("median")
        self.df["Inventory_Level"] = self.df["Inventory_Level"].fillna(store_prod_median_inv)
        # Any residual missing fallback to global median
        self.df["Inventory_Level"] = self.df["Inventory_Level"].fillna(self.df["Inventory_Level"].median())
        self.df["Inventory_Level"] = self.df["Inventory_Level"].clip(lower=0)

        # 6. Outlier Analysis on Target (Units_Sold)
        # Use IQR method for diagnostic reporting
        q25 = self.df["Units_Sold"].quantile(0.25)
        q75 = self.df["Units_Sold"].quantile(0.75)
        iqr = q75 - q25
        upper_bound = q75 + 3.0 * iqr  # 3*IQR flags extreme statistical anomalies
        outliers_detected = int((self.df["Units_Sold"] > upper_bound).sum())

        # 7. Type Enforcement
        self.df["Units_Sold"] = self.df["Units_Sold"].astype(int)
        self.df["Unit_Price"] = self.df["Unit_Price"].astype(float).round(2)
        self.df["Discount"] = self.df["Discount"].astype(float).round(2)
        self.df["Promotion"] = self.df["Promotion"].astype(int)
        self.df["Holiday"] = self.df["Holiday"].astype(int)
        self.df["Inventory_Level"] = self.df["Inventory_Level"].astype(int)

        # Sort chronologically by Store, Product, and Date for consistent time-series processing
        self.df.sort_values(by=["Store_ID", "Product_ID", "Date"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Record audit log
        self.cleaning_audit = {
            "initial_rows": initial_rows,
            "final_rows": len(self.df),
            "rows_removed": initial_rows - len(self.df),
            "duplicates_removed": duplicates_count,
            "invalid_dates_dropped": int(invalid_dates),
            "invalid_ids_dropped": invalid_ids_count,
            "negative_sales_dropped": invalid_sales_count,
            "negative_prices_dropped": invalid_price_count,
            "missing_values_handled": missing_before,
            "extreme_outliers_detected": outliers_detected,
            "outlier_upper_bound_3iqr": float(upper_bound),
        }
        return self.df

    def display_audit_report(self) -> None:
        """Prints a detailed audit report of cleaning results."""
        if not self.cleaning_audit:
            print("Cleaning has not been run yet.")
            return

        print("=" * 60)
        print(" DATA CLEANING AUDIT REPORT")
        print("=" * 60)
        print(f"Initial Records:       {self.cleaning_audit['initial_rows']:,}")
        print(f"Cleaned Records:       {self.cleaning_audit['final_rows']:,}")
        print(f"Total Rows Filtered:   {self.cleaning_audit['rows_removed']:,}")
        print(f"Duplicates Dropped:    {self.cleaning_audit['duplicates_removed']:,}")
        print(f"Negative Sales Dropped:{self.cleaning_audit['negative_sales_dropped']:,}")
        print(f"Invalid Prices Dropped:{self.cleaning_audit['negative_prices_dropped']:,}")
        print(f"Extreme Outliers (>3*IQR): {self.cleaning_audit['extreme_outliers_detected']:,}")
        print("\nMissing Values Imputed Successfully:")
        for col, cnt in self.cleaning_audit["missing_values_handled"].items():
            if cnt > 0:
                print(f"  - {col}: {cnt} null values imputed")
        print("=" * 60)

    def save_cleaned_data(self, output_path: str = "dataset/processed/cleaned_sales.csv") -> None:
        """Persists the sanitized dataset separately from raw data."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"[DataCleaner] Cleaned dataset saved separately at: '{output_path}'")


def run_cleaning_pipeline(
    raw_path: str = "dataset/raw/sales_data.csv",
    processed_path: str = "dataset/processed/cleaned_sales.csv"
) -> pd.DataFrame:
    """Convenience pipeline function to load raw data, clean, and save processed data."""
    raw_df = pd.read_csv(raw_path)
    cleaner = DataCleaner(raw_df)
    clean_df = cleaner.clean()
    cleaner.display_audit_report()
    cleaner.save_cleaned_data(processed_path)
    return clean_df


if __name__ == "__main__":
    run_cleaning_pipeline()
