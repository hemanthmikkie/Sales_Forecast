"""
Module 11: SQLAlchemy Database Models
Defines schema for:
  - products
  - sales
  - inventory
  - forecasts
  - inventory_risk
with primary keys, foreign keys, and performant indexes.
"""

from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship
from database.database import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String(50), primary_key=True, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    unit_price = Column(Float, nullable=False)

    # Relationships
    sales = relationship("Sale", back_populates="product", cascade="all, delete-orphan")
    inventory_records = relationship("Inventory", back_populates="product", cascade="all, delete-orphan")
    forecasts = relationship("Forecast", back_populates="product", cascade="all, delete-orphan")
    risks = relationship("InventoryRisk", back_populates="product", cascade="all, delete-orphan")


class Sale(Base):
    __tablename__ = "sales"

    sale_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    date = Column(Date, nullable=False, index=True)
    store_id = Column(String(50), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False, index=True)
    units_sold = Column(Integer, nullable=False)
    discount = Column(Float, default=0.0)
    promotion = Column(Integer, default=0)
    revenue = Column(Float, nullable=False)

    # Relationship
    product = relationship("Product", back_populates="sales")

    __table_args__ = (
        Index("idx_sales_store_prod_date", "store_id", "product_id", "date"),
    )


class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False, index=True)
    store_id = Column(String(50), nullable=False, index=True)
    inventory_level = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    product = relationship("Product", back_populates="inventory_records")

    __table_args__ = (
        Index("idx_inv_store_prod", "store_id", "product_id"),
    )


class Forecast(Base):
    __tablename__ = "forecasts"

    forecast_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False, index=True)
    store_id = Column(String(50), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    predicted_demand = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)

    # Relationship
    product = relationship("Product", back_populates="forecasts")

    __table_args__ = (
        Index("idx_forecast_prod_store_date", "product_id", "store_id", "forecast_date"),
    )


class InventoryRisk(Base):
    __tablename__ = "inventory_risk"

    risk_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False, index=True)
    current_inventory = Column(Integer, nullable=False)
    forecast_demand = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False, index=True)
    reorder_quantity = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationship
    product = relationship("Product", back_populates="risks")
