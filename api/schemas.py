"""
Module 10: Pydantic Validation Schemas
Request and response contract definitions for FastAPI endpoints.
"""

from typing import Optional, List
from datetime import date as PyDate, datetime as PyDateTime
from pydantic import BaseModel, Field, ConfigDict


# --- Health Schema ---
class HealthResponse(BaseModel):
    status: str
    database_status: str
    active_model: str
    timestamp: PyDateTime


# --- Product Schemas ---
class ProductBase(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "P101"})
    product_name: str = Field(..., json_schema_extra={"example": "Wireless Noise-Canceling Headphones"})
    category: str = Field(..., json_schema_extra={"example": "Electronics"})
    unit_price: float = Field(..., gt=0, json_schema_extra={"example": 120.0})


class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)


# --- Sales Schemas ---
class SaleCreateRequest(BaseModel):
    date: PyDate = Field(..., json_schema_extra={"example": "2026-10-01"})
    store_id: str = Field(..., json_schema_extra={"example": "S01"})
    product_id: str = Field(..., json_schema_extra={"example": "P101"})
    units_sold: int = Field(..., ge=0, json_schema_extra={"example": 65})
    discount: float = Field(0.0, ge=0.0, le=100.0, json_schema_extra={"example": 10.0})
    promotion: int = Field(0, ge=0, le=1, json_schema_extra={"example": 1})


class SaleResponse(BaseModel):
    sale_id: int
    date: PyDate
    store_id: str
    product_id: str
    units_sold: int
    discount: float
    promotion: int
    revenue: float

    model_config = ConfigDict(from_attributes=True)


# --- Inventory Schemas ---
class InventoryResponse(BaseModel):
    inventory_id: int
    product_id: str
    store_id: str
    inventory_level: int
    updated_at: Optional[PyDateTime] = None

    model_config = ConfigDict(from_attributes=True)


class InventoryUpdateRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "P101"})
    store_id: str = Field(..., json_schema_extra={"example": "S01"})
    inventory_level: int = Field(..., ge=0, json_schema_extra={"example": 300})


# --- Forecast Schemas ---
class ForecastRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "P102"})
    store_id: str = Field(..., json_schema_extra={"example": "S01"})
    forecast_date: PyDate = Field(..., json_schema_extra={"example": "2026-10-05"})
    unit_price: Optional[float] = Field(None, gt=0, json_schema_extra={"example": 85.0})
    discount: Optional[float] = Field(0.0, ge=0.0, le=100.0, json_schema_extra={"example": 15.0})
    promotion: Optional[int] = Field(0, ge=0, le=1, json_schema_extra={"example": 1})
    holiday: Optional[int] = Field(0, ge=0, le=1, json_schema_extra={"example": 0})
    lag_1: Optional[float] = Field(None, ge=0, json_schema_extra={"example": 52.0})
    rolling_mean_7: Optional[float] = Field(None, ge=0, json_schema_extra={"example": 58.5})


class ForecastResponse(BaseModel):
    product_id: str
    store_id: str
    forecast_date: PyDate
    predicted_demand: int
    model_used: str
    confidence_interval: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# --- Inventory Risk & Optimization Schemas ---
class InventoryRiskResponse(BaseModel):
    product_id: str
    store_id: str
    current_inventory: int
    forecast_demand: float
    safety_stock: float
    lead_time_days: int
    required_stock: float
    reorder_quantity: int
    risk_level: str
    recommendation: str


# --- Reorder Schemas ---
class ReorderRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "P102"})
    store_id: str = Field(..., json_schema_extra={"example": "S01"})
    reorder_quantity: int = Field(..., gt=0, json_schema_extra={"example": 150})
    supplier_notes: Optional[str] = Field(None, json_schema_extra={"example": "Expedited delivery requested"})


class ReorderResponse(BaseModel):
    product_id: str
    store_id: str
    reorder_quantity: int
    previous_inventory: int
    new_inventory: int
    status: str
    message: str
