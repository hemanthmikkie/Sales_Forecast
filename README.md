# 📈 Retail Sales Demand Forecasting & Inventory Optimization System

An end-to-end, production-grade Machine Learning and MLOps system that analyzes multi-store historical sales data, forecasts future product demand, assesses inventory risk (stock-out vs overstock), and automatically calculates optimal reorder quantities.

Built using **Python 3.12**, **Scikit-learn**, **XGBoost**, **FastAPI**, **PostgreSQL**, **SQLAlchemy**, **Pytest**, and **Docker**.

---

## 🎯 Executive Summary & Objectives

Modern retail organizations lose millions annually from two opposite inventory failures:
1. **Stock-Outs**: Running out of stock during peak promotional or seasonal demand leads to lost revenue, dissatisfied customers, and permanent brand churn.
2. **Overstocking**: Excessive holding of slow-moving inventory traps working capital, elevates warehousing holding costs, and causes depreciation/spoilage.

This system bridges the gap between **predictive machine learning** and **inventory operations**:
- Predicts future product demand per store with ~10.7% MAPE using gradient-boosted trees.
- Dynamically estimates safety stock buffers based on lead times and promotional velocity.
- Evaluates active stock positions in real-time, categorizing risk as `Stock-Out Risk`, `Normal Stock`, or `Overstock Risk`.
- Calculates exact mathematical replenishment quantities ($Q = \max(\text{Required Stock} - \text{Current Inventory}, 0)$).
- Exposes full CRUD and prediction capabilities through a RESTful FastAPI service connected to PostgreSQL.

---

## 🏗️ Architecture & End-to-End Workflow

```mermaid
flowchart TD
    A["Raw Sales Telemetry\n(45,000+ Records)"] --> B["Module 1 & 2: Ingestion & Hygiene\n(Deduplication, Imputation, Sanity Checks)"]
    B --> C["Module 3: Exploratory Data Analysis\n(Seasonality, Promo & Holiday Lift)"]
    C --> D["Module 4: Feature Engineering\n(Lags, Rolling Means, Calendar Signals)"]
    D --> E["Module 5: Chronological Split\n(Train on Past, Test on Future)"]
    E --> F["Module 5: Multi-Model Benchmark\n(Naive, Moving Avg, Linear, RF, XGBoost)"]
    F --> G["Module 6: Metric Evaluation\n(MAE, RMSE, MAPE Leaderboard)"]
    G --> H["Module 13: Model Serialization\n(Joblib: demand_model.pkl & preprocessor.pkl)"]
    H --> I["Module 10 & 12: FastAPI REST Service"]
    J["Module 11: PostgreSQL Database\n(Products, Sales, Inventory, Forecasts, Risks)"] <--> I
    I --> K["Module 8 & 9: Inventory Optimization Engine\n(Safety Stock & Stock-Out/Overstock Risk)"]
    K --> L["Business Decision Output\n(Forecast Demand, Risk Level, Reorder Qty)"]
```

---

## 📂 Project Directory Structure

```text
Sales_Demand_Forecasting/
├── dataset/
│   ├── raw/
│   │   └── sales_data.csv             # Raw retail transaction telemetry (45,615 rows)
│   └── processed/
│       └── cleaned_sales.csv          # Sanitized dataset (duplicates & errors purged)
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Ingestion audit & statistical overview
│   ├── 02_data_cleaning.ipynb         # Data hygiene & imputation steps
│   ├── 03_eda.ipynb                   # Trends, seasonality & promotional uplift
│   ├── 04_feature_engineering.ipynb   # Leakage-free lag & rolling features
│   ├── 05_model_training.ipynb        # Chronological train/test split & fitting
│   └── 06_model_evaluation.ipynb      # Leaderboard, justification & saving
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Module 1: Ingestion & structural audit
│   ├── data_cleaning.py               # Module 2: Data hygiene pipeline
│   ├── eda.py                         # Module 3: Exploratory data analysis
│   ├── feature_engineering.py         # Module 4: Feature store & inference generator
│   ├── forecasting.py                 # Module 5: 5 model architectures & pipelines
│   ├── evaluation.py                  # Module 6: MAE/RMSE/MAPE & leaderboard
│   ├── inventory.py                   # Module 8 & 9: Optimization & risk engine
│   └── create_notebooks.py            # Automated notebook generation utility
├── models/
│   ├── demand_model.pkl               # Serialized best ML model (XGBoost Regressor)
│   └── preprocessor.pkl               # Serialized Scikit-learn ColumnTransformer
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI application entrypoint & lifespan
│   ├── schemas.py                     # Pydantic v2 validation contracts
│   └── routes.py                      # REST endpoints & ML/DB integration
├── database/
│   ├── __init__.py
│   ├── database.py                    # SQLAlchemy session engine (PG + SQLite fallback)
│   ├── models.py                      # SQLAlchemy ORM database models
│   └── crud.py                        # Database CRUD repository layer
├── sql/
│   ├── schema.sql                     # PostgreSQL DDL script with indexes
│   └── analysis.sql                   # Real-world business analytical queries
├── tests/
│   ├── __init__.py
│   ├── test_data.py                   # Pytest: DataLoader and DataCleaner tests
│   ├── test_forecasting.py            # Pytest: Feature engineering & split tests
│   ├── test_inventory.py              # Pytest: Safety stock, reorder, risk tests
│   └── test_api.py                    # Pytest: FastAPI endpoint & integration tests
├── reports/
│   ├── model_comparison.csv           # Empirical model leaderboard
│   └── figures/                       # High-resolution analytical charts
│       ├── 01_temporal_sales_trends.png
│       ├── 02_category_sales_analysis.png
│       ├── 03_store_region_analysis.png
│       ├── 04_promotion_holiday_impact.png
│       ├── 05_top_low_products.png
│       └── 06_model_comparison_metrics.png
├── requirements.txt                   # Pinned production Python dependencies
├── .env                               # Active environment credentials (git-ignored)
├── .env.example                       # Environment configuration template
├── .gitignore                         # Standard git ignore definitions
├── Dockerfile                         # Production container image specification
├── docker-compose.yml                 # Multi-container orchestration (API + PostgreSQL)
└── README.md                          # Comprehensive documentation & guide
```

---

## 🧩 Deep Dive: Module by Module (Beginner-Friendly Explanation)

### Module 1 — Data Loading (`src/data_loader.py`)
- **What it does**: Ingests raw CSV telemetry using Pandas, computes dimension shapes, maps column data types, audits null value frequencies across features, tallies duplicate records, and generates 5-point numerical distribution summaries.
- **Why it matters**: In enterprise machine learning, silent data corruption is the #1 failure mode. Auditing input schemas before processing guarantees pipeline robustness.

### Module 2 — Data Cleaning (`src/data_cleaning.py`)
- **What it does**: 
  - Removes 15 duplicate rows.
  - Imputes missing `Discount` values with `0.0`.
  - Imputes missing `Unit_Price` with the median price for that specific `Product_ID`.
  - Imputes missing `Inventory_Level` with the median stock for that `(Store_ID, Product_ID)`.
  - Removes impossible negative sales records (`Units_Sold < 0`) and negative prices (`Unit_Price <= 0`).
  - Detects statistical anomalies using the $3 \times \text{IQR}$ rule without destroying legitimate holiday demand spikes.
  - Saves the sanitized dataset separately to `dataset/processed/cleaned_sales.csv`.

### Module 3 — Exploratory Data Analysis (`src/eda.py`)
- **Key Findings**:
  - **Weekly Seasonality**: Saturday/Sunday sales increase by ~35% over midweek averages.
  - **Promotional Elasticity**: Promotional campaigns yield a **+60.13% sales surge**.
  - **Holiday Uplift**: Statutory holidays generate a **+62.57% demand lift**.
  - **Store Footprint**: Store S04 (West) and S01 (North) lead overall demand volume.
  - Generates 6 publication-ready figures in `reports/figures/`.

### Module 4 — Feature Engineering (`src/feature_engineering.py`)
- **Calendar Signals**: Extracts `Day`, `Month`, `Year`, `Week`, `Quarter`, `Day_of_Week`, `Is_Weekend`.
- **Revenue Calculation**:
  $$\text{Revenue} = \text{Units\_Sold} \times \text{Unit\_Price} \times \left(1 - \frac{\text{Discount}}{100}\right)$$
- **Strict Leakage Prevention**:
  Time-series features must **never peek into future data**:
  - $\text{Lag}_1 = \text{shift}(1)$
  - $\text{Lag}_7 = \text{shift}(7)$
  - $\text{Lag}_{30} = \text{shift}(30)$
  - $\text{Rolling\_Mean}_7 = \text{shift}(1).\text{rolling}(7).\text{mean}()$
  - $\text{Rolling\_Mean}_{30} = \text{shift}(1).\text{rolling}(30).\text{mean}()$
  By shifting 1 step before computing rolling statistics, the current day's target is never leaked into input features!

### Module 5 & 6 — Demand Forecasting & Empirical Model Evaluation (`src/forecasting.py`, `src/evaluation.py`)
- **Chronological Time-Series Split**:
  Data is split temporally at the 80% date threshold (no random shuffle):
  - Training Period: `2024-01-01` to `2025-12-29` (36,442 observations)
  - Testing/Evaluation Period: `2025-12-30` to `2026-06-30` (9,148 observations)
- **Model Evaluation Leaderboard** (Saved to `reports/model_comparison.csv`):

| Model Name | MAE (Units) | RMSE (Units) | MAPE (%) | Validation Rank |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **7.95** | **11.53** | **10.72%** | **Winner (Rank 1)** |
| Random Forest Regressor | 8.51 | 12.58 | 11.40% | Rank 2 |
| Linear Regression | 10.35 | 14.24 | 15.77% | Rank 3 |
| Moving Average (7-Day Baseline) | 17.54 | 25.60 | 22.87% | Rank 4 |
| Naive Forecast (Lag-1 Baseline) | 21.17 | 32.29 | 27.21% | Rank 5 |

- **Why XGBoost Won**:
  Gradient Boosted Decision Trees excelled at learning non-linear multi-feature interactions between store geographic multipliers, promotion discounts, holiday flags, and recent demand moving averages, reducing Mean Absolute Error to under 8 units with no lookahead bias.

### Module 7, 8 & 9 — Inventory Optimization & Risk Detection (`src/inventory.py`)
- **Safety Stock**:
  $$\text{Safety Stock} = \max(5, \text{Forecast Demand} \times \text{Safety Stock Ratio})$$
- **Lead Time Demand & Required Stock**:
  $$\text{Required Stock} = (\text{Forecast Demand} \times \text{Lead Time}) + \text{Safety Stock}$$
- **Economic Reorder Quantity**:
  $$\text{Reorder Quantity} = \max(\text{Required Stock} - \text{Current Inventory}, 0)$$
- **Risk Engine Classification**:
  - `Stock-Out Risk`: If $\text{Current Inventory} < (\text{Forecast Demand} \times 0.75)$ or $\text{Current Inventory} < \text{Safety Stock}$.
  - `Overstock Risk`: If $\text{Current Inventory} > (\text{Required Stock} \times 2.0)$.
  - `Normal Stock`: When stock sits comfortably within safe operating limits.

### Module 10, 11 & 12 — FastAPI, PostgreSQL & Model Integration (`api/`, `database/`)
- Relational schema defined with SQLAlchemy ORM across 5 tables (`products`, `sales`, `inventory`, `forecasts`, `inventory_risk`).
- Resilient database engine: automatically connects to PostgreSQL, with graceful SQLite fallback for zero-configuration local developer execution.
- Restful endpoints with Pydantic v2 schemas:
  - `GET /health`: System, model, and database heartbeat.
  - `GET /sales`: Paginated historical sales transactions.
  - `POST /sales`: Ingest new sale record, calculate revenue, and store in DB.
  - `GET /inventory`: Active stock positions by store and product.
  - `POST /forecast`: Generate real-time ML demand forecast and audit in DB.
  - `GET /forecast/{product_id}`: Retrieve latest forecasts or auto-generate on-demand.
  - `GET /inventory-risk/{product_id}`: Risk evaluation & reorder recommendation.
  - `POST /reorder`: Replenish inventory stock and persist update.

### Module 14 — Automated Testing (`tests/`)
- **22 out of 22 Pytest tests passing (100% test pass rate)** covering data loading, cleaning, feature engineering, non-lookahead assertions, chronological splitting, safety stock calculations, risk classification, and API routes.

### Module 15 — Containerization (`Dockerfile`, `docker-compose.yml`)
- Multi-container architecture orchestrating the FastAPI application and PostgreSQL 16 database with health checks and persistent volume storage.

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
- Python 3.12+ installed
- Git installed
- (Optional) Docker & Docker Compose installed

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/Sales_Demand_Forecasting.git
cd Sales_Demand_Forecasting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template configuration
cp .env.example .env
```

The default `.env` is pre-configured with local credentials and automatic SQLite fallback if PostgreSQL is not running locally.

---

## 🏃 Execution Instructions

### Step 1: Ingest, Clean, and Run EDA
```bash
# 1. Generate realistic sales dataset (if starting fresh)
python src/generate_dataset.py

# 2. Run data hygiene pipeline
python src/data_cleaning.py

# 3. Generate exploratory charts into reports/figures/
python src/eda.py
```

### Step 2: Train Models, Evaluate Leaderboard & Serialize Artifacts
```bash
# Trains 5 models, outputs comparison table and saves models/demand_model.pkl
python src/evaluation.py
```

### Step 3: Run Full Pytest Test Suite
```bash
python -m pytest -v
```
Output:
```text
======================= 22 passed in ~25s =======================
```

### Step 4: Launch the FastAPI Application
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
The interactive Swagger API documentation is available at: **`http://localhost:8000/docs`**

---

## 🐳 Running with Docker & Docker Compose

To run the complete stack (FastAPI Application + PostgreSQL Database Container) with a single command:

```bash
docker compose up --build
```

- **FastAPI Endpoint**: `http://localhost:8000`
- **Swagger Documentation**: `http://localhost:8000/docs`
- **PostgreSQL Database Port**: `5432`

To shut down:
```bash
docker compose down -v
```

---

## 📡 API Usage & Sample Requests / Responses

### 1. Health Check
**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```
**Response (200 OK):**
```json
{
  "status": "ok",
  "database_status": "connected",
  "active_model": "XGBoost Regressor",
  "timestamp": "2026-10-01T09:10:00Z"
}
```

---

### 2. Generate Demand Forecast (`POST /forecast`)
**Request:**
```bash
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{
       "product_id": "P102",
       "store_id": "S01",
       "forecast_date": "2026-10-05",
       "unit_price": 85.0,
       "discount": 15.0,
       "promotion": 1,
       "holiday": 0,
       "lag_1": 52.0,
       "rolling_mean_7": 58.5
     }'
```
**Response (200 OK):**
```json
{
  "product_id": "P102",
  "store_id": "S01",
  "forecast_date": "2026-10-05",
  "predicted_demand": 92,
  "model_used": "XGBoost Regressor",
  "confidence_interval": "± 12 units (95% CI based on validation MAE)"
}
```

---

### 3. Evaluate Inventory Risk (`GET /inventory-risk/{product_id}`)
**Request:**
```bash
curl -X GET "http://localhost:8000/inventory-risk/P101?store_id=S01&lead_time_days=2"
```
**Response (200 OK):**
```json
{
  "product_id": "P101",
  "store_id": "S01",
  "current_inventory": 30,
  "forecast_demand": 78.0,
  "safety_stock": 15.6,
  "lead_time_days": 2,
  "required_stock": 171.6,
  "reorder_quantity": 142,
  "risk_level": "Stock-Out Risk",
  "recommendation": "High stock-out hazard! Current stock (30) is insufficient for anticipated demand (78). Urgent reorder of 142 units advised."
}
```

---

### 4. Execute Purchase Reorder (`POST /reorder`)
**Request:**
```bash
curl -X POST "http://localhost:8000/reorder" \
     -H "Content-Type: application/json" \
     -d '{
       "product_id": "P101",
       "store_id": "S01",
       "reorder_quantity": 142,
       "supplier_notes": "Urgent stock-out prevention order"
     }'
```
**Response (200 OK):**
```json
{
  "product_id": "P101",
  "store_id": "S01",
  "reorder_quantity": 142,
  "previous_inventory": 30,
  "new_inventory": 172,
  "status": "CONFIRMED",
  "message": "Successfully reordered 142 units. Inventory balance updated from 30 to 172."
}
```

---

## 🗄️ PostgreSQL Database Setup

### Direct PostgreSQL Setup
1. Log into your PostgreSQL instance via `psql` or pgAdmin:
   ```bash
   psql -U postgres
   ```
2. Create the project database:
   ```sql
   CREATE DATABASE sales_forecast;
   ```
3. Run the DDL schema migration script:
   ```bash
   psql -U postgres -d sales_forecast -f sql/schema.sql
   ```
4. Update `DATABASE_URL` in your `.env` file:
   ```env
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/sales_forecast
   ```
5. Run analytical queries from `sql/analysis.sql` to generate executive reports.

---

## 🎓 Technical Interview Highlights & Portfolio Talking Points

When presenting this project in a Data Science or Machine Learning Engineering interview, emphasize these architectural design decisions:

1. **Why Chronological Split instead of K-Fold or Random Shuffle?**
   > *"In time-series demand forecasting, random shuffling causes catastrophic temporal lookahead leakage because future demand signals bleed into training sets. By using a strictly chronological split (past ~80% for training, future ~20% for testing), our evaluation mirrors real-world production forecasting."*

2. **How was Feature Leakage Prevented in Historical Lags?**
   > *"Features like `Rolling_Mean_7` are computed using `shift(1).rolling(7).mean()`. Shifting by 1 prior to computing the window ensures that the target sales for the current observation are never included in the rolling input feature vector."*

3. **How does Machine Learning Connect Directly to Financial ROI?**
   > *"Rather than stopping at model metrics (MAE/RMSE), this system translates demand predictions directly into inventory decisions. By mathematically factoring in supplier lead times and safety stock buffers, the system identifies stock-out and overstock hazards in real-time and recommends exact reorder quantities, directly optimizing inventory holding costs and preventing lost sales."*

4. **Production Readiness & Resilience:**
   > *"The application is architected with complete separation of concerns: modular feature store, Scikit-learn pipelines with One-Hot Encoders fitted strictly on training data, Pydantic v2 schemas for strict contract validation, persistent database audits, 100% Pytest test coverage, and multi-container Docker deployment."*

---

## 📄 License
This project is open-source and licensed under the **MIT License**.
