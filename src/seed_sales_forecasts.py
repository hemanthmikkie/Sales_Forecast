"""
Seed Script: Realistic Sales, Forecasts & Inventory Risk Records
---------------------------------------------------------------
Inserts:
  - 1,250 Sales records  (50 products × 5 stores × 5 recent months)
  - 250   Forecast records (50 products × 5 stores → next 30-day demand)
  - 250   InventoryRisk records (50 products × 5 stores)

Run from project root:
    python src/seed_sales_forecasts.py
"""

import random
import sys
import os
from datetime import date, datetime, timedelta, timezone

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.database import SessionLocal
from database.models import Product, Inventory, Sale, Forecast, InventoryRisk

random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
STORES = ["S01", "S02", "S03", "S04", "S05"]
MODEL_USED = "XGBoost Regressor"
TODAY = date.today()

# Base daily demand by category (units/day per store)
CATEGORY_DEMAND = {
    "Electronics":     {"min": 2,  "max": 12},
    "Groceries":       {"min": 15, "max": 60},
    "Apparel":         {"min": 5,  "max": 25},
    "Home & Kitchen":  {"min": 3,  "max": 18},
    "Health & Beauty": {"min": 8,  "max": 35},
}

# Risk thresholds (units)
RISK_THRESHOLDS = {
    "Electronics":     {"low": 50,  "med": 25},
    "Groceries":       {"low": 120, "med": 60},
    "Apparel":         {"low": 80,  "med": 40},
    "Home & Kitchen":  {"low": 60,  "med": 30},
    "Health & Beauty": {"low": 90,  "med": 45},
}


def daily_units(category: str, promotion: int, holiday: int) -> int:
    """Generate realistic daily units sold with promo/holiday uplift."""
    d = CATEGORY_DEMAND[category]
    base = random.randint(d["min"], d["max"])
    if promotion:
        base = int(base * random.uniform(1.3, 1.8))
    if holiday:
        base = int(base * random.uniform(1.2, 1.5))
    return max(1, base)


def risk_level(inventory: int, forecast: float, category: str) -> str:
    t = RISK_THRESHOLDS[category]
    if inventory <= t["med"]:
        return "HIGH"
    elif inventory <= t["low"]:
        return "MEDIUM"
    elif inventory > forecast * 2.5:
        return "OVERSTOCK"
    else:
        return "LOW"


def reorder_qty(inventory: int, forecast: float) -> int:
    needed = max(0, forecast * 1.5 - inventory)
    return int(round(needed / 10) * 10) if needed > 0 else 0


def main():
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        if not products:
            print("[ERROR] No products found. Run seed_50_items.py first.")
            return

        product_map = {p.product_id: p for p in products}
        inventory_map = {}
        for inv in db.query(Inventory).all():
            inventory_map[(inv.product_id, inv.store_id)] = inv.inventory_level

        print("=" * 65)
        print("  SEEDING SALES, FORECASTS & INVENTORY RISKS INTO MySQL")
        print("=" * 65)

        # ── 1. SALES (last 5 months, one record per product/store/month) ──────
        print("\n[1/3] Inserting Sales records ...")
        sales_added = 0
        for month_offset in range(5, 0, -1):          # months: -5 → -1
            # first day of that month
            ref = TODAY.replace(day=1) - timedelta(days=30 * month_offset)
            # pick a mid-month sale date for each product/store
            for product in products:
                for store in STORES:
                    # vary sale date within the month
                    sale_day = ref + timedelta(days=random.randint(0, 25))
                    promotion = random.randint(0, 1)
                    holiday   = random.randint(0, 1)
                    units     = daily_units(product.category, promotion, holiday)
                    discount  = round(random.choice([0, 5, 10, 15, 20]), 1)
                    revenue   = round(
                        units * product.unit_price * (1 - discount / 100), 2
                    )
                    sale = Sale(
                        date=sale_day,
                        store_id=store,
                        product_id=product.product_id,
                        units_sold=units,
                        discount=discount,
                        promotion=promotion,
                        revenue=revenue,
                    )
                    db.add(sale)
                    sales_added += 1

        db.flush()
        print(f"    -> {sales_added} sale records staged.")

        # ── 2. FORECASTS (next 30 days, one per product/store) ───────────────
        print("\n[2/3] Inserting Forecast records ...")
        forecast_date = TODAY + timedelta(days=30)
        forecasts_added = 0
        for product in products:
            for store in STORES:
                d = CATEGORY_DEMAND[product.category]
                # 30-day predicted demand = avg daily × 30 with slight noise
                avg_daily = (d["min"] + d["max"]) / 2
                noise     = random.uniform(0.85, 1.15)
                predicted  = round(avg_daily * 30 * noise, 2)

                fc = Forecast(
                    product_id=product.product_id,
                    store_id=store,
                    forecast_date=forecast_date,
                    predicted_demand=predicted,
                    model_name=MODEL_USED,
                )
                db.add(fc)
                forecasts_added += 1

        db.flush()
        print(f"    -> {forecasts_added} forecast records staged.")

        # ── 3. INVENTORY RISKS (one per product/store) ───────────────────────
        print("\n[3/3] Inserting Inventory Risk records ...")
        risks_added = 0
        for product in products:
            for store in STORES:
                inv_level = inventory_map.get(
                    (product.product_id, store),
                    random.randint(20, 200),
                )
                d = CATEGORY_DEMAND[product.category]
                avg_daily  = (d["min"] + d["max"]) / 2
                fc_demand  = round(avg_daily * 30 * random.uniform(0.85, 1.15), 2)
                r_level    = risk_level(inv_level, fc_demand, product.category)
                r_qty      = reorder_qty(inv_level, fc_demand)

                risk = InventoryRisk(
                    product_id=product.product_id,
                    current_inventory=inv_level,
                    forecast_demand=fc_demand,
                    risk_level=r_level,
                    reorder_quantity=r_qty,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(risk)
                risks_added += 1

        db.commit()
        print(f"    -> {risks_added} inventory risk records staged.")

        # ── VERIFICATION ─────────────────────────────────────────────────────
        total_sales    = db.query(Sale).count()
        total_fc       = db.query(Forecast).count()
        total_risks    = db.query(InventoryRisk).count()

        print()
        print("=" * 65)
        print("  DATABASE VERIFICATION")
        print("=" * 65)
        print(f"  Products        : {db.query(Product).count()}")
        print(f"  Inventory       : {db.query(Inventory).count()}")
        print(f"  Sales           : {total_sales}  (new: {sales_added})")
        print(f"  Forecasts       : {total_fc}  (new: {forecasts_added})")
        print(f"  Inventory Risks : {total_risks}  (new: {risks_added})")
        print("=" * 65)

        # Risk breakdown
        print("\n  Risk Level Breakdown:")
        for level in ["HIGH", "MEDIUM", "LOW", "OVERSTOCK"]:
            count = db.query(InventoryRisk).filter(
                InventoryRisk.risk_level == level
            ).count()
            bar = "#" * (count // 5)
            print(f"    {level:<12}: {count:>4}  {bar}")

        print()
        print("  Successfully seeded Sales, Forecasts & Inventory Risks!")
        print("=" * 65)

    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Rollback triggered: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
