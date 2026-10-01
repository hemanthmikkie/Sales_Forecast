"""
Unit and Integration Tests for Module 8 (Inventory Optimization) and Module 9 (Inventory Risk Engine).
"""

import pytest
from src.inventory import InventoryOptimizer


@pytest.fixture
def optimizer():
    return InventoryOptimizer(
        default_lead_time_days=1,
        default_safety_stock_ratio=0.20,
        stockout_threshold_ratio=0.75,
        overstock_threshold_multiplier=2.0
    )


def test_safety_stock_calculation(optimizer):
    """Verifies proportional safety stock calculation and minimum bounds."""
    # 20% of 100 is 20.0
    ss = optimizer.calculate_safety_stock(forecast_demand=100.0)
    assert ss == 20.0

    # Very small demand triggers minimum floor of 5 units
    ss_small = optimizer.calculate_safety_stock(forecast_demand=10.0)
    assert ss_small == 5.0

    # Explicit safety stock override
    ss_override = optimizer.calculate_safety_stock(forecast_demand=100.0, safety_stock=35.0)
    assert ss_override == 35.0


def test_reorder_quantity_deficit(optimizer):
    """Verifies Reorder Quantity = max(Required Stock - Current Inventory, 0)."""
    # Demand = 100, Lead Time = 1, Safety Stock = 20 -> Required Stock = 120
    # Current Inventory = 30 -> Deficit = 120 - 30 = 90
    result = optimizer.optimize_inventory(
        product_id="P101",
        current_inventory=30,
        forecast_demand=100.0,
        safety_stock=20.0,
        lead_time_days=1
    )
    assert result["required_stock"] == 120.0
    assert result["reorder_quantity"] == 90
    assert result["risk_level"] == "Stock-Out Risk"


def test_reorder_quantity_zero_when_sufficient(optimizer):
    """Verifies Reorder Quantity is zero when inventory already meets or exceeds required stock."""
    # Required Stock = 120, Current Inventory = 150 -> Reorder = 0
    result = optimizer.optimize_inventory(
        product_id="P101",
        current_inventory=150,
        forecast_demand=100.0,
        safety_stock=20.0,
        lead_time_days=1
    )
    assert result["reorder_quantity"] == 0
    assert result["risk_level"] == "Normal Stock"


def test_overstock_risk_detection(optimizer):
    """Verifies Overstock Risk is assigned when stock exceeds 2x required stock."""
    # Required Stock = 120, Overstock threshold (2x) = 240
    # Current Inventory = 350 -> Overstock Risk
    result = optimizer.optimize_inventory(
        product_id="P101",
        current_inventory=350,
        forecast_demand=100.0,
        safety_stock=20.0,
        lead_time_days=1
    )
    assert result["risk_level"] == "Overstock Risk"
    assert result["reorder_quantity"] == 0
    assert "Excess inventory" in result["recommendation"]


def test_negative_input_validation(optimizer):
    """Verifies ValueError is raised for negative inventory or negative forecast demand."""
    with pytest.raises(ValueError):
        optimizer.optimize_inventory(product_id="P101", current_inventory=-10, forecast_demand=50.0)

    with pytest.raises(ValueError):
        optimizer.optimize_inventory(product_id="P101", current_inventory=50, forecast_demand=-20.0)
