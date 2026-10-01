-- ====================================================================
-- BUSINESS ANALYTICAL & INVENTORY OPTIMIZATION SQL QUERIES
-- ====================================================================

-- 1. Monthly Revenue and Volume Growth by Product Category
SELECT 
    TO_CHAR(s.date, 'YYYY-MM') AS sales_month,
    p.category,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(AVG(s.discount), 2) AS avg_discount_pct
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY 1, 2
ORDER BY 1 DESC, 4 DESC;

-- 2. Top 3 Revenue-Generating Products Per Store (Window Function)
WITH ranked_store_products AS (
    SELECT 
        s.store_id,
        p.product_id,
        p.product_name,
        p.category,
        SUM(s.revenue) AS store_revenue,
        DENSE_RANK() OVER(PARTITION BY s.store_id ORDER BY SUM(s.revenue) DESC) AS rank_in_store
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY s.store_id, p.product_id, p.product_name, p.category
)
SELECT 
    store_id,
    rank_in_store,
    product_id,
    product_name,
    category,
    ROUND(store_revenue, 2) AS total_revenue
FROM ranked_store_products
WHERE rank_in_store <= 3
ORDER BY store_id, rank_in_store;

-- 3. Inventory Stock-Out Risk Detection (Current Stock vs. 7-Day Velocity)
WITH recent_sales_velocity AS (
    SELECT 
        store_id,
        product_id,
        ROUND(AVG(units_sold), 2) AS avg_daily_sales_last_7d
    FROM sales
    WHERE date >= (SELECT MAX(date) - INTERVAL '7 days' FROM sales)
    GROUP BY store_id, product_id
)
SELECT 
    i.store_id,
    i.product_id,
    p.product_name,
    i.inventory_level AS current_inventory,
    COALESCE(rsv.avg_daily_sales_last_7d, 0) AS daily_run_rate,
    ROUND(i.inventory_level / NULLIF(rsv.avg_daily_sales_last_7d, 0), 1) AS estimated_days_of_stock,
    CASE 
        WHEN i.inventory_level < (COALESCE(rsv.avg_daily_sales_last_7d, 0) * 3) THEN 'URGENT: Stock-Out Hazard'
        WHEN i.inventory_level > (COALESCE(rsv.avg_daily_sales_last_7d, 0) * 14) THEN 'WARNING: Overstock Capital Trapped'
        ELSE 'OPTIMAL: Balanced Stock'
    END AS inventory_health_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id
LEFT JOIN recent_sales_velocity rsv ON i.store_id = rsv.store_id AND i.product_id = rsv.product_id
ORDER BY estimated_days_of_stock ASC;

-- 4. Promotional Elasticity Analysis (Sales Lift with vs without Promotion)
SELECT 
    p.category,
    ROUND(AVG(CASE WHEN s.promotion = 1 THEN s.units_sold END), 2) AS avg_units_on_promo,
    ROUND(AVG(CASE WHEN s.promotion = 0 THEN s.units_sold END), 2) AS avg_units_no_promo,
    ROUND(
        ((AVG(CASE WHEN s.promotion = 1 THEN s.units_sold END) - 
          AVG(CASE WHEN s.promotion = 0 THEN s.units_sold END)) / 
          NULLIF(AVG(CASE WHEN s.promotion = 0 THEN s.units_sold END), 0)) * 100, 
        2
    ) AS promotion_sales_lift_pct
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY promotion_sales_lift_pct DESC;

-- 5. Forecast Accuracy Audit (Predicted Demand vs Actual Sales)
SELECT 
    f.product_id,
    f.store_id,
    f.forecast_date,
    f.model_name,
    f.predicted_demand,
    s.units_sold AS actual_units_sold,
    ROUND(ABS(f.predicted_demand - s.units_sold), 2) AS absolute_error,
    ROUND((ABS(f.predicted_demand - s.units_sold) / NULLIF(s.units_sold, 0)) * 100, 2) AS error_percentage
FROM forecasts f
JOIN sales s ON f.product_id = s.product_id 
            AND f.store_id = s.store_id 
            AND f.forecast_date = s.date
ORDER BY f.forecast_date DESC
LIMIT 50;
