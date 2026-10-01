"""
Unit and Integration Tests for Module 1 (Data Loading) & Module 2 (Data Cleaning).
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner


@pytest.fixture
def sample_dirty_data():
    """Generates a small controlled dirty dataset for deterministic testing."""
    records = [
        {"Date": "2025-01-01", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 50, "Unit_Price": 120.0, "Discount": 10.0, "Promotion": 1, "Holiday": 1, "Inventory_Level": 300, "Region": "North"},
        {"Date": "2025-01-02", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 45, "Unit_Price": 120.0, "Discount": 0.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 280, "Region": "North"},
        # Duplicate row
        {"Date": "2025-01-02", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 45, "Unit_Price": 120.0, "Discount": 0.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 280, "Region": "North"},
        # Missing discount
        {"Date": "2025-01-03", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 40, "Unit_Price": 120.0, "Discount": np.nan, "Promotion": 0, "Holiday": 0, "Inventory_Level": 260, "Region": "North"},
        # Missing price
        {"Date": "2025-01-04", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 60, "Unit_Price": np.nan, "Discount": 5.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 240, "Region": "North"},
        # Negative units sold (invalid)
        {"Date": "2025-01-05", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": -10, "Unit_Price": 120.0, "Discount": 0.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 200, "Region": "North"},
        # Invalid price (<= 0)
        {"Date": "2025-01-06", "Store_ID": "S01", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 30, "Unit_Price": -5.0, "Discount": 0.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 190, "Region": "North"},
        # Invalid store ID
        {"Date": "2025-01-07", "Store_ID": "INVALID", "Product_ID": "P101", "Product_Category": "Electronics", "Units_Sold": 35, "Unit_Price": 120.0, "Discount": 0.0, "Promotion": 0, "Holiday": 0, "Inventory_Level": 180, "Region": "North"},
    ]
    return pd.DataFrame(records)


def test_data_loader_inspection(tmp_path, sample_dirty_data):
    """Verifies DataLoader file ingestion and audit report generation."""
    csv_file = tmp_path / "test_raw.csv"
    sample_dirty_data.to_csv(csv_file, index=False)

    loader = DataLoader(str(csv_file))
    df = loader.load_data()
    assert len(df) == 8
    assert "Units_Sold" in df.columns

    report = loader.get_summary_report()
    assert report["shape"] == (8, 11)
    assert report["duplicate_records"] == 1
    assert report["missing_values"]["Discount"] == 1
    assert report["missing_values"]["Unit_Price"] == 1


def test_data_cleaning_pipeline(sample_dirty_data):
    """Verifies duplicate drops, missing imputation, invalid sales/price filtering."""
    cleaner = DataCleaner(sample_dirty_data)
    clean_df = cleaner.clean()

    # 1. Duplicates removed (1 duplicate row dropped)
    # 2. Negative sales dropped (1 row)
    # 3. Negative price dropped (1 row)
    # 4. Invalid store ID dropped (1 row)
    # Expected remaining: 8 - 4 = 4 rows
    assert len(clean_df) == 4
    assert (clean_df["Units_Sold"] >= 0).all()
    assert (clean_df["Unit_Price"] > 0).all()
    assert clean_df["Discount"].isnull().sum() == 0
    assert clean_df["Unit_Price"].isnull().sum() == 0
    assert clean_df["Inventory_Level"].isnull().sum() == 0
    assert clean_df["Store_ID"].str.startswith("S").all()


def test_data_loader_missing_file_error():
    """Verifies graceful FileNotFoundError on non-existent paths."""
    loader = DataLoader("dataset/non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()
