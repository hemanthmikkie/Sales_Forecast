"""
Unit and Integration Tests for Module 4 (Feature Engineering), Module 5 (Forecasting), and Module 6 (Evaluation).
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer
from src.forecasting import DemandForecaster, BaselineNaiveForecaster, BaselineMovingAverageForecaster
from src.evaluation import ModelEvaluator


@pytest.fixture
def sample_timeseries_df():
    """Generates a small clean daily time series across 40 days."""
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    records = []
    for d in dates:
        records.append({
            "Date": d.strftime("%Y-%m-%d"),
            "Store_ID": "S01",
            "Product_ID": "P101",
            "Product_Category": "Electronics",
            "Units_Sold": 50 + int(d.day % 10),
            "Unit_Price": 100.0,
            "Discount": 10.0 if d.day % 7 == 0 else 0.0,
            "Promotion": 1 if d.day % 7 == 0 else 0,
            "Holiday": 1 if d.day == 1 else 0,
            "Inventory_Level": 300,
            "Region": "North",
        })
    return pd.DataFrame(records)


def test_feature_engineering_calculations(sample_timeseries_df):
    """Verifies revenue, calendar extraction, and non-lookahead lag/rolling calculations."""
    fe = FeatureEngineer()
    feat_df = fe.transform(sample_timeseries_df)

    # 1. Revenue
    # units = 50, price = 100, discount = 0 -> revenue = 5000.0
    first_row = feat_df.iloc[0]
    expected_rev = round(first_row["Units_Sold"] * first_row["Unit_Price"] * (1 - first_row["Discount"] / 100.0), 2)
    assert first_row["Revenue"] == expected_rev

    # 2. Date features
    assert "Day" in feat_df.columns
    assert "Month" in feat_df.columns
    assert "Year" in feat_df.columns
    assert "Week" in feat_df.columns
    assert "Quarter" in feat_df.columns
    assert "Day_of_Week" in feat_df.columns
    assert "Is_Weekend" in feat_df.columns

    # 3. Historical demand features
    assert "Lag_1" in feat_df.columns
    assert "Lag_7" in feat_df.columns
    assert "Lag_30" in feat_df.columns
    assert "Rolling_Mean_7" in feat_df.columns
    assert "Rolling_Mean_30" in feat_df.columns

    # Verify no nulls in engineered features
    for col in fe.FEATURE_COLS:
        assert feat_df[col].isnull().sum() == 0, f"Column {col} has nulls!"

    # Verify Lag_1 of day index 5 equals Units_Sold of day index 4 (strict non-lookahead)
    assert feat_df.loc[5, "Lag_1"] == feat_df.loc[4, "Units_Sold"]


def test_chronological_split(sample_timeseries_df, tmp_path):
    """Guarantees chronological separation between training and test sets."""
    csv_file = tmp_path / "ts_data.csv"
    sample_timeseries_df.to_csv(csv_file, index=False)

    forecaster = DemandForecaster(str(csv_file))
    X_tr, X_te, y_tr, y_te = forecaster.prepare_data(test_split_ratio=0.25)

    assert len(X_tr) > 0
    assert len(X_te) > 0
    assert len(X_tr) + len(X_te) == len(sample_timeseries_df)

    # Chronological integrity: Train date boundaries strictly precede test dates
    train_dates = forecaster.df.iloc[:len(X_tr)]["Date"]
    test_dates = forecaster.df.iloc[len(X_tr):]["Date"]
    assert train_dates.max() < test_dates.min()


def test_model_evaluator_metrics():
    """Verifies MAE, RMSE, and MAPE calculations."""
    evaluator = ModelEvaluator()
    y_true = np.array([100.0, 150.0, 200.0, 250.0])
    y_pred = np.array([110.0, 140.0, 210.0, 240.0])

    metrics = evaluator.evaluate_model("TestModel", y_true, y_pred)
    assert metrics["Model Name"] == "TestModel"
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert metrics["MAPE (%)"] > 0.0

    # Compare models leaderboard
    preds = {
        "Model A": y_pred,
        "Model B": y_pred + 20.0,
    }
    comp_df = evaluator.compare_models(preds, y_true)
    assert len(comp_df) == 2
    assert comp_df.iloc[0]["Model Name"] == "Model A"  # Model A has lower error


def test_inference_feature_extractor():
    """Verifies single-row inference feature vector extraction for real-time forecasting."""
    fe = FeatureEngineer()
    df_inf = fe.extract_inference_features(
        date_str="2026-10-15",
        store_id="S01",
        product_id="P101",
        product_category="Electronics",
        region="North",
        unit_price=120.0,
        discount=10.0,
        promotion=1,
        holiday=0,
        lag_1=55.0,
        rolling_mean_7=52.0
    )
    assert len(df_inf) == 1
    assert df_inf["Day"].iloc[0] == 15
    assert df_inf["Month"].iloc[0] == 10
    assert df_inf["Year"].iloc[0] == 2026
    assert df_inf["Lag_1"].iloc[0] == 55.0
