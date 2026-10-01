"""
Module 11: Database CRUD Utilities
Provides database access functions for products, sales, inventory levels,
model forecasts, and inventory risk evaluations.
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import Product, Sale, Inventory, Forecast, InventoryRisk


def seed_master_data(db: Session) -> None:
    """Populates standard products if the products table is empty."""
    existing_count = db.query(Product).count()
    if existing_count > 0:
        return

    sample_products = [
        Product(product_id="P101", product_name="Wireless Noise-Canceling Headphones", category="Electronics", unit_price=120.0),
        Product(product_id="P102", product_name="Smart Fitness Watch", category="Electronics", unit_price=85.0),
        Product(product_id="P103", product_name="Organic Arabica Coffee Beans 1kg", category="Groceries", unit_price=22.0),
        Product(product_id="P104", product_name="Extra Virgin Olive Oil 1L", category="Groceries", unit_price=18.0),
        Product(product_id="P105", product_name="Classic Denim Jacket", category="Apparel", unit_price=65.0),
        Product(product_id="P106", product_name="Performance Running Shoes", category="Apparel", unit_price=95.0),
        Product(product_id="P107", product_name="Stainless Steel Cookware Set", category="Home & Kitchen", unit_price=150.0),
        Product(product_id="P108", product_name="Ergonomic Memory Foam Pillow", category="Home & Kitchen", unit_price=38.0),
        Product(product_id="P109", product_name="Hydrating Facial Serum 50ml", category="Health & Beauty", unit_price=28.0),
        Product(product_id="P110", product_name="Plant-Based Multivitamin 90ct", category="Health & Beauty", unit_price=24.0),
    ]
    db.add_all(sample_products)
    db.commit()
    print(f"[CRUD] Seeded {len(sample_products)} catalog products.")

    # Seed baseline inventory
    stores = ["S01", "S02", "S03", "S04", "S05"]
    inv_records = []
    for p in sample_products:
        for s in stores:
            inv_records.append(
                Inventory(
                    product_id=p.product_id,
                    store_id=s,
                    inventory_level=250,
                    updated_at=datetime.now(timezone.utc)
                )
            )
    db.add_all(inv_records)
    db.commit()
    print(f"[CRUD] Seeded {len(inv_records)} baseline inventory positions.")


def get_products(db: Session) -> List[Product]:
    return db.query(Product).all()


def get_product(db: Session, product_id: str) -> Optional[Product]:
    return db.query(Product).filter(Product.product_id == product_id).first()


def get_sales(
    db: Session,
    store_id: Optional[str] = None,
    product_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Sale]:
    query = db.query(Sale)
    if store_id:
        query = query.filter(Sale.store_id == store_id)
    if product_id:
        query = query.filter(Sale.product_id == product_id)
    return query.order_by(desc(Sale.date)).offset(offset).limit(limit).all()


def create_sale(
    db: Session,
    date_val: date,
    store_id: str,
    product_id: str,
    units_sold: int,
    discount: float,
    promotion: int,
    unit_price: float
) -> Sale:
    revenue = round(units_sold * unit_price * (1.0 - discount / 100.0), 2)
    sale = Sale(
        date=date_val,
        store_id=store_id,
        product_id=product_id,
        units_sold=units_sold,
        discount=discount,
        promotion=promotion,
        revenue=revenue
    )
    db.add(sale)
    db.commit()
    db.refresh(sale)
    return sale


def get_inventory(db: Session, store_id: Optional[str] = None, product_id: Optional[str] = None) -> List[Inventory]:
    query = db.query(Inventory)
    if store_id:
        query = query.filter(Inventory.store_id == store_id)
    if product_id:
        query = query.filter(Inventory.product_id == product_id)
    return query.all()


def get_inventory_by_product_and_store(db: Session, product_id: str, store_id: str) -> Optional[Inventory]:
    return db.query(Inventory).filter(
        Inventory.product_id == product_id,
        Inventory.store_id == store_id
    ).first()


def update_or_create_inventory(db: Session, product_id: str, store_id: str, inventory_level: int) -> Inventory:
    inv = get_inventory_by_product_and_store(db, product_id, store_id)
    if inv:
        inv.inventory_level = inventory_level
        inv.updated_at = datetime.now(timezone.utc)
    else:
        inv = Inventory(
            product_id=product_id,
            store_id=store_id,
            inventory_level=inventory_level,
            updated_at=datetime.now(timezone.utc)
        )
        db.add(inv)
    db.commit()
    db.refresh(inv)
    return inv


def save_forecast(
    db: Session,
    product_id: str,
    store_id: str,
    forecast_date: date,
    predicted_demand: float,
    model_name: str
) -> Forecast:
    forecast = Forecast(
        product_id=product_id,
        store_id=store_id,
        forecast_date=forecast_date,
        predicted_demand=predicted_demand,
        model_name=model_name
    )
    db.add(forecast)
    db.commit()
    db.refresh(forecast)
    return forecast


def get_forecasts_by_product(db: Session, product_id: str, limit: int = 10) -> List[Forecast]:
    return (
        db.query(Forecast)
        .filter(Forecast.product_id == product_id)
        .order_by(desc(Forecast.forecast_id))
        .limit(limit)
        .all()
    )


def save_inventory_risk(
    db: Session,
    product_id: str,
    current_inventory: int,
    forecast_demand: float,
    risk_level: str,
    reorder_quantity: int
) -> InventoryRisk:
    risk_rec = InventoryRisk(
        product_id=product_id,
        current_inventory=current_inventory,
        forecast_demand=forecast_demand,
        risk_level=risk_level,
        reorder_quantity=reorder_quantity,
        created_at=datetime.now(timezone.utc)
    )
    db.add(risk_rec)
    db.commit()
    db.refresh(risk_rec)
    return risk_rec


def get_latest_inventory_risk(db: Session, product_id: str) -> Optional[InventoryRisk]:
    return (
        db.query(InventoryRisk)
        .filter(InventoryRisk.product_id == product_id)
        .order_by(desc(InventoryRisk.created_at))
        .first()
    )
