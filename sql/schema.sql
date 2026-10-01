-- ====================================================================
-- SALES DEMAND FORECASTING & INVENTORY OPTIMIZATION DATABASE SCHEMA
-- PostgreSQL DDL Script
-- ====================================================================

-- 1. Products Master Table
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL CHECK (unit_price > 0)
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- 2. Sales Transactions Table
CREATE TABLE IF NOT EXISTS sales (
    sale_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    units_sold INT NOT NULL CHECK (units_sold >= 0),
    discount NUMERIC(5, 2) DEFAULT 0.0 CHECK (discount >= 0 AND discount <= 100),
    promotion INT DEFAULT 0 CHECK (promotion IN (0, 1)),
    revenue NUMERIC(12, 2) NOT NULL CHECK (revenue >= 0),
    CONSTRAINT fk_sales_product FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_sales_store_id ON sales(store_id);
CREATE INDEX IF NOT EXISTS idx_sales_product_id ON sales(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_store_prod_date ON sales(store_id, product_id, date);

-- 3. Inventory Status Table
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    inventory_level INT NOT NULL CHECK (inventory_level >= 0),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_inventory_product FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT uq_inventory_product_store UNIQUE (product_id, store_id)
);

CREATE INDEX IF NOT EXISTS idx_inventory_store_prod ON inventory(store_id, product_id);

-- 4. Forecasts Audit Table
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    forecast_date DATE NOT NULL,
    predicted_demand NUMERIC(10, 2) NOT NULL CHECK (predicted_demand >= 0),
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_forecasts_product FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_forecast_prod_store_date ON forecasts(product_id, store_id, forecast_date);

-- 5. Inventory Risk Assessments Table
CREATE TABLE IF NOT EXISTS inventory_risk (
    risk_id SERIAL PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    current_inventory INT NOT NULL CHECK (current_inventory >= 0),
    forecast_demand NUMERIC(10, 2) NOT NULL CHECK (forecast_demand >= 0),
    risk_level VARCHAR(50) NOT NULL CHECK (risk_level IN ('Stock-Out Risk', 'Normal Stock', 'Overstock Risk')),
    reorder_quantity INT NOT NULL CHECK (reorder_quantity >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_risk_product FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_risk_product ON inventory_risk(product_id);
CREATE INDEX IF NOT EXISTS idx_risk_level ON inventory_risk(risk_level);
