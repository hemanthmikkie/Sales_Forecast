"""
Module 1: Data Loading
Responsible for loading raw sales datasets, conducting structural audits
(shapes, dtypes, null values, duplicates, summary statistics), and providing a clean interface.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class DataLoader:
    """
    DataLoader encapsulates operations to ingest and inspect tabular sales data.
    """

    def __init__(self, file_path: str = "dataset/raw/sales_data.csv") -> None:
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV dataset from disk with robust error handling.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Source file not found at: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path)
            print(f"[DataLoader] Successfully loaded {len(self.data):,} records from '{self.file_path}'")
            return self.data
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}") from e

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Performs structural and statistical audit on the loaded dataset.
        Returns a dictionary containing:
          - shape: (rows, cols)
          - columns: list of column names
          - data_types: mapping of col to dtype string
          - missing_values: count of missing values per column
          - duplicate_records: count of duplicated rows
          - numerical_summary: describe() dictionary for numerical features
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        report = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "data_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_records": int(self.data.duplicated().sum()),
            "numerical_summary": self.data.describe().to_dict(),
        }
        return report

    def display_inspection(self) -> None:
        """
        Prints a formatted executive overview of the dataset audit to standard output.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        report = self.get_summary_report()
        print("=" * 60)
        print(" DATA AUDIT & INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Dataset Shape: {report['shape'][0]:,} rows x {report['shape'][1]} columns\n")

        print("Columns & Data Types:")
        for col, dt in report["data_types"].items():
            null_count = report["missing_values"][col]
            print(f"  - {col:<20} | Type: {dt:<10} | Missing: {null_count}")

        print(f"\nDuplicate Records: {report['duplicate_records']:,}")
        print("\nFive-Point Numerical Summary:")
        print(self.data.describe().round(2).to_string())
        print("=" * 60)

    def save_snapshot(self, output_path: str) -> None:
        """
        Saves a snapshot of the dataframe to the designated location.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        print(f"[DataLoader] Dataset snapshot saved to '{output_path}'")


if __name__ == "__main__":
    loader = DataLoader("dataset/raw/sales_data.csv")
    df = loader.load_data()
    loader.display_inspection()
