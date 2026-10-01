"""
Dataset Generator for Sales Demand Forecasting & Inventory Optimization System.
Generates realistic retail sales data with seasonality, promotions, holidays,
store/product hierarchies, and intentional minor anomalies for data cleaning demonstrations.
"""

import numpy as np
import pandas as pd
import os

def generate_sales_data(
    start_date: str = "2024-01-01",
    end_date: str = "2026-06-30",
    random_seed: int = 42,
    output_path: str = "dataset/raw/sales_data.csv"
) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # 1. Product Master Definition
    products = [
        {"Product_ID": "P101", "Product_Name": "Wireless Noise-Canceling Headphones", "Product_Category": "Electronics", "Base_Price": 120.0, "Base_Demand": 45},
        {"Product_ID": "P102", "Product_Name": "Smart Fitness Watch", "Product_Category": "Electronics", "Base_Price": 85.0, "Base_Demand": 55},
        {"Product_ID": "P103", "Product_Name": "Organic Arabica Coffee Beans 1kg", "Product_Category": "Groceries", "Base_Price": 22.0, "Base_Demand": 90},
        {"Product_ID": "P104", "Product_Name": "Extra Virgin Olive Oil 1L", "Product_Category": "Groceries", "Base_Price": 18.0, "Base_Demand": 75},
        {"Product_ID": "P105", "Product_Name": "Classic Denim Jacket", "Product_Category": "Apparel", "Base_Price": 65.0, "Base_Demand": 35},
        {"Product_ID": "P106", "Product_Name": "Performance Running Shoes", "Product_Category": "Apparel", "Base_Price": 95.0, "Base_Demand": 40},
        {"Product_ID": "P107", "Product_Name": "Stainless Steel Cookware Set", "Product_Category": "Home & Kitchen", "Base_Price": 150.0, "Base_Demand": 25},
        {"Product_ID": "P108", "Product_Name": "Ergonomic Memory Foam Pillow", "Product_Category": "Home & Kitchen", "Base_Price": 38.0, "Base_Demand": 60},
        {"Product_ID": "P109", "Product_Name": "Hydrating Facial Serum 50ml", "Product_Category": "Health & Beauty", "Base_Price": 28.0, "Base_Demand": 65},
        {"Product_ID": "P110", "Product_Name": "Plant-Based Multivitamin 90ct", "Product_Category": "Health & Beauty", "Base_Price": 24.0, "Base_Demand": 70},
    ]
    
    # 2. Store Master Definition
    stores = [
        {"Store_ID": "S01", "Region": "North", "Store_Multiplier": 1.25},
        {"Store_ID": "S02", "Region": "South", "Store_Multiplier": 0.90},
        {"Store_ID": "S03", "Region": "East", "Store_Multiplier": 1.10},
        {"Store_ID": "S04", "Region": "West", "Store_Multiplier": 1.35},
        {"Store_ID": "S05", "Region": "Central", "Store_Multiplier": 0.85},
    ]
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Major US/Retail Holiday approximate dates
    holiday_dates = set(pd.to_datetime([
        "2024-01-01", "2024-02-14", "2024-07-04", "2024-09-02", "2024-11-28", "2024-11-29", "2024-12-24", "2024-12-25", "2024-12-31",
        "2025-01-01", "2025-02-14", "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31",
        "2026-01-01", "2026-02-14", "2026-05-25"
    ]))
    
    records = []
    
    for date in dates:
        is_holiday = 1 if date in holiday_dates else 0
        day_of_week = date.dayofweek # 0=Mon, 6=Sun
        is_weekend = 1 if day_of_week in [5, 6] else 0
        month = date.month
        
        # Seasonality factors
        weekend_mult = 1.35 if is_weekend else 1.0
        holiday_mult = 1.55 if is_holiday else 1.0
        # Q4 boost (Oct, Nov, Dec)
        month_mult = 1.30 if month in [11, 12] else (1.15 if month in [6, 7] else 1.0)
        
        for store in stores:
            s_id = store["Store_ID"]
            region = store["Region"]
            s_mult = store["Store_Multiplier"]
            
            for prod in products:
                p_id = prod["Product_ID"]
                category = prod["Product_Category"]
                base_price = prod["Base_Price"]
                base_demand = prod["Base_Demand"]
                
                # Random promotion probability: ~12% chance on weekdays, ~25% on weekends/holidays
                promo_prob = 0.25 if (is_weekend or is_holiday) else 0.12
                has_promo = 1 if np.random.rand() < promo_prob else 0
                
                # Discount selection
                if has_promo:
                    discount = float(np.random.choice([10, 15, 20, 25, 30]))
                    promo_mult = 1.0 + (discount / 40.0) # bigger discount brings more sales
                else:
                    discount = 0.0
                    promo_mult = 1.0
                
                # Price variation slightly (±2%)
                unit_price = round(base_price * (1.0 + np.random.uniform(-0.02, 0.02)), 2)
                
                # Calculate expected demand with noise
                expected_demand = base_demand * s_mult * weekend_mult * holiday_mult * month_mult * promo_mult
                noise = np.random.normal(loc=0, scale=0.12 * expected_demand)
                units_sold = int(max(0, round(expected_demand + noise)))
                
                # Dynamic inventory level (typically 3 to 10 days of average demand)
                inv_base = int(base_demand * s_mult * 7)
                inv_fluctuation = int(np.random.uniform(-0.35, 0.65) * inv_base)
                inventory_level = max(10, inv_base + inv_fluctuation)
                
                records.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Store_ID": s_id,
                    "Product_ID": p_id,
                    "Product_Category": category,
                    "Units_Sold": units_sold,
                    "Unit_Price": unit_price,
                    "Discount": discount,
                    "Promotion": has_promo,
                    "Holiday": is_holiday,
                    "Inventory_Level": inventory_level,
                    "Region": region
                })
                
    df = pd.DataFrame(records)
    
    # Inject a few realistic dirty data instances for demonstration of Module 1 & 2 cleaning:
    # 1. Duplicate rows (15 rows)
    dup_indices = np.random.choice(df.index, size=15, replace=False)
    duplicates = df.loc[dup_indices].copy()
    
    # 2. Missing values in Discount, Inventory_Level, Unit_Price
    nan_discount_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[nan_discount_idx, "Discount"] = np.nan
    
    nan_inventory_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[nan_inventory_idx, "Inventory_Level"] = np.nan
    
    nan_price_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[nan_price_idx, "Unit_Price"] = np.nan
    
    # 3. Invalid negative units or negative price (e.g. 5 erroneous records)
    neg_units_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[neg_units_idx, "Units_Sold"] = -5
    
    neg_price_idx = np.random.choice(df.index, size=4, replace=False)
    df.loc[neg_price_idx, "Unit_Price"] = -10.0
    
    # Append duplicates and shuffle
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} rows and {len(df.columns)} columns saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_sales_data()
