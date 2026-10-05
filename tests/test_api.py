"""
Integration Tests for FastAPI Endpoints (Modules 10 & 12).
Tests all required routes, HTTP status codes, validation rules,
ML model predictions, and DB persistence.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_get_health():
    """Verifies GET /health endpoint returns 200 and operational metadata."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "database_status" in data
        assert "active_model" in data


def test_get_sales():
    """Verifies GET /sales endpoint returns list of sales records."""
    with TestClient(app) as test_client:
        response = test_client.get("/sales?limit=5")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_post_sales_valid():
    """Verifies POST /sales records new transaction successfully."""
    with TestClient(app) as test_client:
        payload = {
            "date": "2026-10-01",
            "store_id": "S01",
            "product_id": "P101",
            "units_sold": 45,
            "discount": 10.0,
            "promotion": 1
        }
        response = test_client.post("/sales", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["product_id"] == "P101"
        assert data["units_sold"] == 45
        assert data["revenue"] > 0


def test_post_sales_invalid_product():
    """Verifies POST /sales returns 404 for non-existent product catalog ID."""
    with TestClient(app) as test_client:
        payload = {
            "date": "2026-10-01",
            "store_id": "S01",
            "product_id": "P9999_DOES_NOT_EXIST",
            "units_sold": 10,
            "discount": 0.0,
            "promotion": 0
        }
        response = test_client.post("/sales", json=payload)
        assert response.status_code == 404


def test_post_sales_invalid_negative_units():
    """Verifies POST /sales rejects negative sales units with 422 Unprocessable Entity."""
    with TestClient(app) as test_client:
        payload = {
            "date": "2026-10-01",
            "store_id": "S01",
            "product_id": "P101",
            "units_sold": -25,
            "discount": 0.0,
            "promotion": 0
        }
        response = test_client.post("/sales", json=payload)
        assert response.status_code == 422


def test_get_inventory():
    """Verifies GET /inventory returns active stock positions."""
    with TestClient(app) as test_client:
        response = test_client.get("/inventory")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_post_forecast_ml_prediction():
    """Verifies POST /forecast executes model inference and returns predicted demand."""
    with TestClient(app) as test_client:
        payload = {
            "product_id": "P102",
            "store_id": "S01",
            "forecast_date": "2026-10-15",
            "unit_price": 85.0,
            "discount": 15.0,
            "promotion": 1,
            "holiday": 0,
            "lag_1": 55.0,
            "rolling_mean_7": 60.0
        }
        response = test_client.post("/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "P102"
        assert data["store_id"] == "S01"
        assert "predicted_demand" in data
        assert isinstance(data["predicted_demand"], int)
        assert data["predicted_demand"] >= 0
        assert "model_used" in data


def test_get_forecast_by_product():
    """Verifies GET /forecast/{product_id} returns forecasts."""
    with TestClient(app) as test_client:
        response = test_client.get("/forecast/P102?store_id=S01")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["product_id"] == "P102"


def test_get_inventory_risk_evaluation():
    """Verifies GET /inventory-risk/{product_id} classifies risk and gives reorder quantity."""
    with TestClient(app) as test_client:
        response = test_client.get("/inventory-risk/P101?store_id=S01&lead_time_days=2")
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "P101"
        assert data["risk_level"] in ["Stock-Out Risk", "Normal Stock", "Overstock Risk"]
        assert "reorder_quantity" in data
        assert data["reorder_quantity"] >= 0
        assert "recommendation" in data


def test_post_reorder_replenishment():
    """Verifies POST /reorder updates inventory stocks and returns confirmation."""
    with TestClient(app) as test_client:
        payload = {
            "product_id": "P101",
            "store_id": "S01",
            "reorder_quantity": 75,
            "supplier_notes": "Urgent order for weekend sales"
        }
        response = test_client.post("/reorder", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "CONFIRMED"
        assert data["reorder_quantity"] == 75
        assert data["new_inventory"] == data["previous_inventory"] + 75


def test_get_products():
    """Verifies GET /products returns all catalog products."""
    with TestClient(app) as test_client:
        response = test_client.get("/products")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "product_id" in data[0]
        assert "product_name" in data[0]
        assert "category" in data[0]
        assert "unit_price" in data[0]


def test_get_product_by_id():
    """Verifies GET /products/{product_id} returns correct product."""
    with TestClient(app) as test_client:
        response = test_client.get("/products/P101")
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "P101"
        assert data["category"] == "Electronics"


def test_get_product_not_found():
    """Verifies GET /products/{product_id} returns 404 for unknown ID."""
    with TestClient(app) as test_client:
        response = test_client.get("/products/P9999_INVALID")
        assert response.status_code == 404
