"""
Module 5: Demand Forecasting & Training
Implements:
  - Naive Forecast Baseline
  - Moving Average Baseline
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Chronological time-series split (Strictly train on past, test on future)
  - Scikit-learn Preprocessing Pipeline (One-Hot Encoders & Scalers)
  - Joblib serialization for best model & preprocessor
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.feature_engineering import FeatureEngineer


class BaselineNaiveForecaster:
    """
    Predicts the immediate previous day demand (Lag_1) as a naive baseline.
    """
    def __init__(self) -> None:
        self.name = "Naive Forecast (Lag-1)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineNaiveForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "Lag_1" in X.columns:
            return np.maximum(0, X["Lag_1"].values)
        return np.zeros(len(X))


class BaselineMovingAverageForecaster:
    """
    Predicts the 7-day rolling moving average demand as a time-series moving average baseline.
    """
    def __init__(self) -> None:
        self.name = "Moving Average (7-Day)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineMovingAverageForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "Rolling_Mean_7" in X.columns:
            return np.maximum(0, X["Rolling_Mean_7"].values)
        elif "Lag_1" in X.columns:
            return np.maximum(0, X["Lag_1"].values)
        return np.zeros(len(X))


class DemandForecaster:
    """
    Orchestrates time-series split, pipeline preprocessing, multi-model training,
    and model persistence.
    """

    CATEGORICAL_FEATURES = ["Store_ID", "Product_ID", "Product_Category", "Region"]
    NUMERICAL_FEATURES = [
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

    def __init__(self, data_path: str = "dataset/processed/cleaned_sales.csv") -> None:
        self.data_path = data_path
        self.fe = FeatureEngineer()
        self.df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None

    def prepare_data(self, test_split_ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads cleaned data, engineers features, and executes a strictly chronological
        train/test split. (Zero shuffling / zero lookahead).
        """
        raw_clean_df = pd.read_csv(self.data_path)
        self.df = self.fe.transform(raw_clean_df)

        # Sort strictly by date
        self.df.sort_values("Date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        unique_dates = self.df["Date"].drop_duplicates().sort_values().values
        split_idx = int(len(unique_dates) * (1.0 - test_split_ratio))
        split_date = unique_dates[split_idx]

        train_mask = self.df["Date"] < split_date
        test_mask = self.df["Date"] >= split_date

        train_df = self.df[train_mask].copy()
        test_df = self.df[test_mask].copy()

        feature_cols = self.CATEGORICAL_FEATURES + self.NUMERICAL_FEATURES
        X_train = train_df[feature_cols]
        y_train = train_df[self.fe.TARGET_COL]

        X_test = test_df[feature_cols]
        y_test = test_df[self.fe.TARGET_COL]

        print(f"[Forecaster] Chronological split at date: {split_date}")
        print(f"  Train set: {len(X_train):,} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
        print(f"  Test set:  {len(X_test):,} rows ({test_df['Date'].min()} to {test_df['Date'].max()})")

        return X_train, X_test, y_train, y_test

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs an sklearn ColumnTransformer fitting one-hot & scaling pipelines."""
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
                ("num", numerical_transformer, self.NUMERICAL_FEATURES),
            ]
        )
        return self.preprocessor

    def train_and_evaluate_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, np.ndarray]:
        """
        Trains Naive, Moving Average, Linear Regression, Random Forest, and XGBoost models.
        Fits preprocessor ONLY on training data to avoid data leakage.
        """
        self.build_preprocessor()

        print("[Forecaster] Fitting preprocessor exclusively on training partition...")
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # 1. Baseline: Naive Forecast (Lag-1)
        naive = BaselineNaiveForecaster()
        naive.fit(X_train, y_train)
        self.models[naive.name] = naive
        self.predictions[naive.name] = naive.predict(X_test)

        # 2. Baseline: Moving Average (7-Day)
        ma = BaselineMovingAverageForecaster()
        ma.fit(X_train, y_train)
        self.models[ma.name] = ma
        self.predictions[ma.name] = ma.predict(X_test)

        # 3. Linear Regression
        print("[Forecaster] Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train_proc, y_train)
        self.models["Linear Regression"] = lr
        self.predictions["Linear Regression"] = np.maximum(0, lr.predict(X_test_proc))

        # 4. Random Forest Regressor
        print("[Forecaster] Training Random Forest Regressor (n_estimators=100)...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train_proc, y_train)
        self.models["Random Forest Regressor"] = rf
        self.predictions["Random Forest Regressor"] = np.maximum(0, rf.predict(X_test_proc))

        # 5. XGBoost Regressor
        print("[Forecaster] Training XGBoost Regressor...")
        xgb = XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train_proc, y_train)
        self.models["XGBoost Regressor"] = xgb
        self.predictions["XGBoost Regressor"] = np.maximum(0, xgb.predict(X_test_proc))

        print("[Forecaster] All 5 models trained successfully.")
        return self.predictions

    def save_artifacts(
        self,
        best_model_name: str,
        model_path: str = "models/demand_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl"
    ) -> None:
        """Serializes the best performing model and preprocessing pipeline with Joblib."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        metadata = {
            "model_name": best_model_name,
            "categorical_features": self.CATEGORICAL_FEATURES,
            "numerical_features": self.NUMERICAL_FEATURES,
            "model": self.best_model,
        }
        joblib.dump(metadata, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"[Forecaster] Best model '{best_model_name}' saved to '{model_path}'")
        print(f"[Forecaster] Preprocessor saved to '{preprocessor_path}'")


if __name__ == "__main__":
    forecaster = DemandForecaster("dataset/processed/cleaned_sales.csv")
    X_tr, X_te, y_tr, y_te = forecaster.prepare_data(test_split_ratio=0.20)
    preds = forecaster.train_and_evaluate_all(X_tr, X_te, y_tr, y_te)
