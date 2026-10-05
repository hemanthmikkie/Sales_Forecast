-- ====================================================================
-- SALES DEMAND FORECASTING & INVENTORY OPTIMIZATION DATABASE SCHEMA
-- MySQL 8.0 DDL Script
-- Run: mysql -u root -p sales_forecast < sql/schema_mysql.sql
-- ====================================================================

CREATE DATABASE IF NOT EXISTS sales_forecast CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sales_forecast;

-- 1. Products Master Table
CREATE TABLE IF NOT EXISTS products (
    product_id   VARCHAR(50)  NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category     VARCHAR(100) NOT NULL,
    unit_price   DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (product_id),
    CONSTRAINT chk_products_price CHECK (unit_price > 0),
    INDEX idx_products_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Sales Transactions Table
CREATE TABLE IF NOT EXISTS sales (
    sale_id    INT            NOT NULL AUTO_INCREMENT,
    date       DATE           NOT NULL,
    store_id   VARCHAR(50)    NOT NULL,
    product_id VARCHAR(50)    NOT NULL,
    units_sold INT            NOT NULL,
    discount   DECIMAL(5, 2)  NOT NULL DEFAULT 0.0,
    promotion  TINYINT(1)     NOT NULL DEFAULT 0,
    revenue    DECIMAL(12, 2) NOT NULL,
    PRIMARY KEY (sale_id),
    CONSTRAINT fk_sales_product FOREIGN KEY (product_id)
        REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT chk_sales_units   CHECK (units_sold >= 0),
    CONSTRAINT chk_sales_discount CHECK (discount >= 0 AND discount <= 100),
    CONSTRAINT chk_sales_promotion CHECK (promotion IN (0, 1)),
    CONSTRAINT chk_sales_revenue  CHECK (revenue >= 0),
    INDEX idx_sales_date          (date),
    INDEX idx_sales_store_id      (store_id),
    INDEX idx_sales_product_id    (product_id),
    INDEX idx_sales_store_prod_date (store_id, product_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Inventory Status Table
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id    INT         NOT NULL AUTO_INCREMENT,
    product_id      VARCHAR(50) NOT NULL,
    store_id        VARCHAR(50) NOT NULL,
    inventory_level INT         NOT NULL,
    updated_at      DATETIME    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (inventory_id),
    CONSTRAINT fk_inventory_product FOREIGN KEY (product_id)
        REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT uq_inventory_product_store UNIQUE (product_id, store_id),
    CONSTRAINT chk_inventory_level CHECK (inventory_level >= 0),
    INDEX idx_inv_store_prod (store_id, product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Forecasts Audit Table
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id      INT            NOT NULL AUTO_INCREMENT,
    product_id       VARCHAR(50)    NOT NULL,
    store_id         VARCHAR(50)    NOT NULL,
    forecast_date    DATE           NOT NULL,
    predicted_demand DECIMAL(10, 2) NOT NULL,
    model_name       VARCHAR(100)   NOT NULL,
    created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (forecast_id),
    CONSTRAINT fk_forecasts_product FOREIGN KEY (product_id)
        REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT chk_forecasts_demand CHECK (predicted_demand >= 0),
    INDEX idx_forecast_prod_store_date (product_id, store_id, forecast_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Inventory Risk Assessments Table
CREATE TABLE IF NOT EXISTS inventory_risk (
    risk_id           INT            NOT NULL AUTO_INCREMENT,
    product_id        VARCHAR(50)    NOT NULL,
    current_inventory INT            NOT NULL,
    forecast_demand   DECIMAL(10, 2) NOT NULL,
    risk_level        VARCHAR(50)    NOT NULL,
    reorder_quantity  INT            NOT NULL,
    created_at        DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (risk_id),
    CONSTRAINT fk_risk_product FOREIGN KEY (product_id)
        REFERENCES products(product_id) ON DELETE CASCADE,
    CONSTRAINT chk_risk_inventory CHECK (current_inventory >= 0),
    CONSTRAINT chk_risk_demand    CHECK (forecast_demand >= 0),
    CONSTRAINT chk_risk_reorder   CHECK (reorder_quantity >= 0),
    INDEX idx_risk_product (product_id),
    INDEX idx_risk_level   (risk_level),
    INDEX idx_risk_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
