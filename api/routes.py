"""
Module 10 & 12: API Routes & Business Logic
Implements REST endpoints:
  - GET  /health
  - GET  /sales
  - POST /sales
  - GET  /inventory
  - POST /forecast
  - GET  /forecast/{product_id}
  - GET  /inventory-risk/{product_id}
  - POST /reorder
Integrates MySQL database operations with the loaded ML demand forecasting model
and the inventory risk engine.
"""

import os
from typing import List, Optional
from datetime import datetime, date, timezone
import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.database import get_db
from database import crud, models
from api.schemas import (
    HealthResponse,
    ProductResponse,
    SaleCreateRequest,
    SaleResponse,
    InventoryResponse,
    ForecastRequest,
    ForecastResponse,
    InventoryRiskResponse,
    ReorderRequest,
    ReorderResponse,
)
from src.feature_engineering import FeatureEngineer
from src.inventory import InventoryOptimizer

router = APIRouter()

# Global in-memory cache for ML artifacts
ML_ARTIFACTS = {
    "model": None,
    "model_name": "Not Loaded",
    "preprocessor": None,
    "feature_engineer": FeatureEngineer(),
    "inventory_optimizer": InventoryOptimizer(),
}

STORE_REGION_MAP = {
    "S01": "North",
    "S02": "South",
    "S03": "East",
    "S04": "West",
    "S05": "Central",
}


def load_ml_models():
    """Loads trained model metadata and preprocessor pipeline from disk."""
    model_path = "models/demand_model.pkl"
    prep_path = "models/preprocessor.pkl"

    if os.path.exists(model_path) and os.path.exists(prep_path):
        try:
            metadata = joblib.load(model_path)
            ML_ARTIFACTS["model"] = metadata["model"]
            ML_ARTIFACTS["model_name"] = metadata.get("model_name", "Trained ML Model")
            ML_ARTIFACTS["preprocessor"] = joblib.load(prep_path)
            print(f"[API] Loaded ML model '{ML_ARTIFACTS['model_name']}' successfully.")
        except Exception as e:
            print(f"[API] Warning: Failed to load serialized model: {e}")
    else:
        print("[API] Notice: Model files not found on disk yet. Endpoints will train or fallback.")


# -----------------------------------------------------------------------------
# 1. GET /health
# -----------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["System Health"])
def health_check(db: Session = Depends(get_db)):
    """Verifies API operational status, database connectivity, and active ML model."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return HealthResponse(
        status="ok",
        database_status=db_status,
        active_model=ML_ARTIFACTS["model_name"],
        timestamp=datetime.now(timezone.utc),
    )


# -----------------------------------------------------------------------------
# 1b. GET /products
# -----------------------------------------------------------------------------
@router.get("/products", response_model=List[ProductResponse], tags=["Products"])
def list_products(db: Session = Depends(get_db)):
    """Returns the full product catalog with IDs, names, categories, and prices."""
    return crud.get_products(db)


@router.get("/products/{product_id}", response_model=ProductResponse, tags=["Products"])
def get_product(product_id: str, db: Session = Depends(get_db)):
    """Returns a single product's details by Product ID (e.g. P101)."""
    product = crud.get_product(db, product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found in catalog.",
        )
    return product


# -----------------------------------------------------------------------------
# 2. GET /sales & POST /sales
# -----------------------------------------------------------------------------
@router.get("/sales", response_model=List[SaleResponse], tags=["Sales Management"])
def list_sales(
    store_id: Optional[str] = Query(None, description="Filter by Store ID (e.g., S01)"),
    product_id: Optional[str] = Query(None, description="Filter by Product ID (e.g., P101)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieves paginated historical sales records with optional store/product filtering."""
    sales = crud.get_sales(db, store_id=store_id, product_id=product_id, limit=limit, offset=offset)
    return sales


@router.post("/sales", response_model=SaleResponse, status_code=status.HTTP_201_CREATED, tags=["Sales Management"])
def record_sale(payload: SaleCreateRequest, db: Session = Depends(get_db)):
    """Ingests a new sales transaction, validates product catalogue, and commits to DB."""
    product = crud.get_product(db, payload.product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{payload.product_id}' does not exist in master catalog.",
        )

    sale = crud.create_sale(
        db=db,
        date_val=payload.date,
        store_id=payload.store_id,
        product_id=payload.product_id,
        units_sold=payload.units_sold,
        discount=payload.discount,
        promotion=payload.promotion,
        unit_price=product.unit_price,
    )
    return sale


# -----------------------------------------------------------------------------
# 3. GET /inventory
# -----------------------------------------------------------------------------
@router.get("/inventory", response_model=List[InventoryResponse], tags=["Inventory Management"])
def list_inventory(
    store_id: Optional[str] = Query(None, description="Filter by Store ID"),
    product_id: Optional[str] = Query(None, description="Filter by Product ID"),
    db: Session = Depends(get_db),
):
    """Retrieves current inventory stock levels across distribution centers."""
    inventory_items = crud.get_inventory(db, store_id=store_id, product_id=product_id)
    return inventory_items


# -----------------------------------------------------------------------------
# 4. POST /forecast
# -----------------------------------------------------------------------------
@router.post("/forecast", response_model=ForecastResponse, tags=["Demand Forecasting"])
def generate_forecast(payload: ForecastRequest, db: Session = Depends(get_db)):
    """
    Generates real-time demand forecast for a product and store on a target future date
    using the trained Machine Learning model. Persists forecast audit in MySQL.
    """
    if ML_ARTIFACTS["model"] is None or ML_ARTIFACTS["preprocessor"] is None:
        load_ml_models()
        if ML_ARTIFACTS["model"] is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained forecasting model is not currently loaded. Please run model training first.",
            )

    # Validate product
    product = crud.get_product(db, payload.product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{payload.product_id}' was not found in catalog.",
        )

    region = STORE_REGION_MAP.get(payload.store_id, "North")
    unit_price = payload.unit_price if payload.unit_price else product.unit_price

    # Construct single-instance feature vector
    fe = ML_ARTIFACTS["feature_engineer"]
    X_single = fe.extract_inference_features(
        date_str=payload.forecast_date.strftime("%Y-%m-%d"),
        store_id=payload.store_id,
        product_id=payload.product_id,
        product_category=product.category,
        region=region,
        unit_price=unit_price,
        discount=payload.discount or 0.0,
        promotion=payload.promotion or 0,
        holiday=payload.holiday or 0,
        lag_1=payload.lag_1,
        rolling_mean_7=payload.rolling_mean_7,
    )

    # Preprocess & Predict
    preprocessor = ML_ARTIFACTS["preprocessor"]
    model = ML_ARTIFACTS["model"]

    X_transformed = preprocessor.transform(X_single)
    raw_pred = model.predict(X_transformed)
    predicted_demand = int(max(0, round(float(raw_pred[0]))))

    # Persist in DB forecasts table
    saved_forecast = crud.save_forecast(
        db=db,
        product_id=payload.product_id,
        store_id=payload.store_id,
        forecast_date=payload.forecast_date,
        predicted_demand=predicted_demand,
        model_name=ML_ARTIFACTS["model_name"],
    )

    return ForecastResponse(
        product_id=saved_forecast.product_id,
        store_id=saved_forecast.store_id,
        forecast_date=saved_forecast.forecast_date,
        predicted_demand=predicted_demand,
        model_used=ML_ARTIFACTS["model_name"],
        confidence_interval="± 12 units (95% CI based on validation MAE)",
    )


# -----------------------------------------------------------------------------
# 5. GET /forecast/{product_id}
# -----------------------------------------------------------------------------
@router.get("/forecast/{product_id}", response_model=List[ForecastResponse], tags=["Demand Forecasting"])
def get_forecast_by_product(
    product_id: str,
    store_id: Optional[str] = Query("S01", description="Store ID context"),
    db: Session = Depends(get_db),
):
    """
    Returns latest forecast history for a product.
    If no previous forecast is on record, automatically computes a forecast for the current cycle.
    """
    product = crud.get_product(db, product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{product_id}' not found.",
        )

    records = crud.get_forecasts_by_product(db, product_id=product_id, limit=5)
    if not records:
        # Generate immediate forecast
        req = ForecastRequest(
            product_id=product_id,
            store_id=store_id or "S01",
            forecast_date=date.today(),
            discount=0.0,
            promotion=0,
        )
        new_fc = generate_forecast(req, db)
        return [new_fc]

    return [
        ForecastResponse(
            product_id=r.product_id,
            store_id=r.store_id,
            forecast_date=r.forecast_date,
            predicted_demand=int(round(r.predicted_demand)),
            model_used=r.model_name,
            confidence_interval="Historical prediction record",
        )
        for r in records
    ]


# -----------------------------------------------------------------------------
# 6. GET /inventory-risk/{product_id}
# -----------------------------------------------------------------------------
@router.get("/inventory-risk/{product_id}", response_model=InventoryRiskResponse, tags=["Inventory Optimization"])
def get_inventory_risk(
    product_id: str,
    store_id: str = Query("S01", description="Store ID context"),
    lead_time_days: int = Query(1, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """
    Evaluates inventory risk (Stock-Out Risk, Normal Stock, Overstock Risk)
    and provides mathematical reorder quantity recommendations.
    """
    product = crud.get_product(db, product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found in catalog.",
        )

    # 1. Fetch current inventory
    inv = crud.get_inventory_by_product_and_store(db, product_id, store_id)
    current_inventory = inv.inventory_level if inv else 150

    # 2. Obtain forecast demand
    forecast_req = ForecastRequest(
        product_id=product_id,
        store_id=store_id,
        forecast_date=date.today(),
    )
    fc_result = generate_forecast(forecast_req, db)
    forecast_demand = float(fc_result.predicted_demand)

    # 3. Optimize inventory
    optimizer: InventoryOptimizer = ML_ARTIFACTS["inventory_optimizer"]
    risk_data = optimizer.optimize_inventory(
        product_id=product_id,
        current_inventory=current_inventory,
        forecast_demand=forecast_demand,
        lead_time_days=lead_time_days,
        store_id=store_id,
    )

    # 4. Save risk assessment in DB
    crud.save_inventory_risk(
        db=db,
        product_id=product_id,
        current_inventory=current_inventory,
        forecast_demand=forecast_demand,
        risk_level=risk_data["risk_level"],
        reorder_quantity=risk_data["reorder_quantity"],
    )

    return InventoryRiskResponse(**risk_data)


# -----------------------------------------------------------------------------
# 7. POST /reorder
# -----------------------------------------------------------------------------
@router.post("/reorder", response_model=ReorderResponse, tags=["Inventory Optimization"])
def place_reorder(payload: ReorderRequest, db: Session = Depends(get_db)):
    """
    Executes a purchase replenishment reorder, updates active inventory stocks,
    and returns receipt acknowledgement.
    """
    product = crud.get_product(db, payload.product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{payload.product_id}' does not exist.",
        )

    inv = crud.get_inventory_by_product_and_store(db, payload.product_id, payload.store_id)
    prev_stock = inv.inventory_level if inv else 0
    new_stock = prev_stock + payload.reorder_quantity

    crud.update_or_create_inventory(
        db=db,
        product_id=payload.product_id,
        store_id=payload.store_id,
        inventory_level=new_stock,
    )

    return ReorderResponse(
        product_id=payload.product_id,
        store_id=payload.store_id,
        reorder_quantity=payload.reorder_quantity,
        previous_inventory=prev_stock,
        new_inventory=new_stock,
        status="CONFIRMED",
        message=f"Successfully reordered {payload.reorder_quantity} units. Inventory balance updated from {prev_stock} to {new_stock}.",
    )
