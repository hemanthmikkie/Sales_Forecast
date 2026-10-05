"""
Database Seeder: 50 Retail Products & Baseline Inventory
Populates the live database (MySQL) with 50 diverse retail catalog products
across 5 major categories (Electronics, Groceries, Apparel, Home & Kitchen, Health & Beauty),
and populates baseline inventory across all 5 regional distribution stores (S01-S05).
"""

import sys
import os
from datetime import datetime, timezone, date
import random

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import engine, SessionLocal, Base
from database.models import Product, Inventory, Sale

# 50 Realistic Retail Catalog Products
PRODUCTS_50 = [
    # --- Category 1: Electronics (10 Products) ---
    ("P101", "Wireless Noise-Canceling Headphones", "Electronics", 129.99),
    ("P102", "Smart Fitness Watch Series 5", "Electronics", 89.50),
    ("P103", "Ultra-Slim 4K OLED TV 55-inch", "Electronics", 649.00),
    ("P104", "Bluetooth Waterproof Speaker", "Electronics", 45.00),
    ("P105", "Ergonomic Mechanical Keyboard", "Electronics", 75.00),
    ("P106", "Wireless Gaming Optical Mouse", "Electronics", 39.99),
    ("P107", "Fast Wireless Charging Pad 15W", "Electronics", 24.99),
    ("P108", "Portable USB-C External SSD 1TB", "Electronics", 109.99),
    ("P109", "True Wireless Earbuds with ANC", "Electronics", 59.95),
    ("P110", "Smart Home Security Camera 1080p", "Electronics", 49.99),

    # --- Category 2: Groceries & Gourmet (10 Products) ---
    ("P111", "Organic Arabica Coffee Beans 1kg", "Groceries", 22.50),
    ("P112", "Cold-Pressed Extra Virgin Olive Oil 1L", "Groceries", 18.75),
    ("P113", "Pure Wildflower Raw Honey 500g", "Groceries", 12.99),
    ("P114", "Almond Butter Gluten-Free 400g", "Groceries", 9.50),
    ("P115", "Organic Quinoa Grain 1kg", "Groceries", 7.99),
    ("P116", "Artisanal Dark Chocolate 85% 100g", "Groceries", 4.50),
    ("P117", "Japanese Ceremonial Matcha Green Tea 50g", "Groceries", 24.00),
    ("P118", "Gluten-Free Rolled Oats 1.2kg", "Groceries", 6.80),
    ("P119", "Sparkling Spring Water 12-Pack", "Groceries", 14.50),
    ("P120", "Himalayan Pink Rock Salt Grinder 380g", "Groceries", 5.25),

    # --- Category 3: Apparel & Fashion (10 Products) ---
    ("P121", "Classic Denim Trucker Jacket", "Apparel", 68.00),
    ("P122", "Performance Running Shoes Breathable", "Apparel", 95.00),
    ("P123", "Organic Cotton Crewneck T-Shirt", "Apparel", 22.00),
    ("P124", "Slim-Fit Chino Stretch Trousers", "Apparel", 48.00),
    ("P125", "Merino Wool Pullover Sweater", "Apparel", 79.99),
    ("P126", "Water-Resistant Hooded Windbreaker", "Apparel", 58.50),
    ("P127", "Thermal Fleece Running Leggings", "Apparel", 36.00),
    ("P128", "Polarized Aviator Sunglasses UV400", "Apparel", 32.00),
    ("P129", "Genuine Leather Casual Belt", "Apparel", 28.00),
    ("P130", "Cushioned Athletic Socks 6-Pack", "Apparel", 16.50),

    # --- Category 4: Home & Kitchen (10 Products) ---
    ("P131", "Stainless Steel Multi-Ply Cookware Set", "Home & Kitchen", 189.00),
    ("P132", "Ergonomic Memory Foam Contour Pillow", "Home & Kitchen", 39.50),
    ("P133", "Programmable Drip Coffee Maker 12-Cup", "Home & Kitchen", 64.99),
    ("P134", "High-Speed Countertop Blender 1200W", "Home & Kitchen", 89.00),
    ("P135", "Non-Stick Cast Iron Skillet 12-inch", "Home & Kitchen", 34.50),
    ("P136", "Aromatherapy Ultrasonic Essential Oil Diffuser", "Home & Kitchen", 27.99),
    ("P137", "Microfiber Luxury Queen Sheet Set", "Home & Kitchen", 44.00),
    ("P138", "Cordless Stick Vacuum Cleaner Rechargeable", "Home & Kitchen", 139.00),
    ("P139", "Glass Meal Prep Containers 10-Piece Set", "Home & Kitchen", 29.99),
    ("P140", "Digital Kitchen Food Scale 5kg", "Home & Kitchen", 18.50),

    # --- Category 5: Health & Beauty (10 Products) ---
    ("P141", "Hydrating Hyaluronic Acid Serum 50ml", "Health & Beauty", 28.00),
    ("P142", "Plant-Based Daily Multivitamin 90ct", "Health & Beauty", 23.50),
    ("P143", "Broad-Spectrum SPF 50 Mineral Sunscreen 100ml", "Health & Beauty", 19.99),
    ("P144", "Sonic Electric Rechargeable Toothbrush", "Health & Beauty", 42.00),
    ("P145", "Deep Cleansing Foaming Facial Cleanser 200ml", "Health & Beauty", 16.50),
    ("P146", "Organic Cold-Pressed Argan Hair Oil 100ml", "Health & Beauty", 21.00),
    ("P147", "Whey Protein Isolate Powder Vanilla 1kg", "Health & Beauty", 38.00),
    ("P148", "Lavender Sea Salt Exfoliating Body Scrub 300g", "Health & Beauty", 17.50),
    ("P149", "Collagen Peptide Powder Hydrolyzed 450g", "Health & Beauty", 32.00),
    ("P150", "Deep Tissue Percussion Muscle Massage Gun", "Health & Beauty", 79.00),
]

STORES = ["S01", "S02", "S03", "S04", "S05"]


def seed_50_items() -> None:
    """Inserts 50 items and updates their store inventory levels in MySQL."""
    print("=" * 60)
    print(" SEEDING 50 RETAIL ITEMS INTO DATABASE")
    print("=" * 60)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        inserted_prods = 0
        updated_prods = 0

        for p_id, p_name, cat, price in PRODUCTS_50:
            existing = db.query(Product).filter(Product.product_id == p_id).first()
            if not existing:
                prod = Product(
                    product_id=p_id,
                    product_name=p_name,
                    category=cat,
                    unit_price=float(price)
                )
                db.add(prod)
                inserted_prods += 1
            else:
                existing.product_name = p_name
                existing.category = cat
                existing.unit_price = float(price)
                updated_prods += 1

        db.commit()
        print(f"Products: {inserted_prods} newly inserted, {updated_prods} updated/verified.")

        # Seed/Update Inventory for all 50 products across all 5 stores
        random.seed(42)
        inv_count = 0
        for p_id, _, _, _ in PRODUCTS_50:
            for s_id in STORES:
                existing_inv = db.query(Inventory).filter(
                    Inventory.product_id == p_id,
                    Inventory.store_id == s_id
                ).first()

                # Generate realistic random inventory level (50 to 450 units)
                inv_stock = random.randint(60, 420)

                if existing_inv:
                    existing_inv.inventory_level = inv_stock
                    existing_inv.updated_at = datetime.now(timezone.utc)
                else:
                    new_inv = Inventory(
                        product_id=p_id,
                        store_id=s_id,
                        inventory_level=inv_stock,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(new_inv)
                inv_count += 1

        db.commit()
        print(f"Inventory: Seeded/Updated {inv_count} stock positions across stores S01-S05.")

        # Total verification
        total_products = db.query(Product).count()
        total_inventory = db.query(Inventory).count()

        print("-" * 60)
        print(f"DATABASE VERIFICATION:")
        print(f"  - Total Products in Database:  {total_products}")
        print(f"  - Total Inventory Records:    {total_inventory}")
        print("=" * 60)
        print("Successfully seeded 50 products and inventory into MySQL!")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_50_items()
