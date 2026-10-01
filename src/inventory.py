"""
Module 8 & 9: Inventory Optimization & Risk Detection Engine
Implements:
  - Required Stock Calculation: Forecast Demand * Lead Time + Safety Stock
  - Dynamic & Configurable Safety Stock & Lead Time
  - Economic Reorder Quantity: max(Required Stock - Current Inventory, 0)
  - Risk Classification:
      - Stock-Out Risk: Current Inventory < Expected Required Stock * threshold
      - Overstock Risk: Current Inventory > Expected Required Stock * overstock_multiplier
      - Normal Stock: Inventory levels balanced with predicted demand
"""

from typing import Dict, Any, Optional
import math


class InventoryOptimizer:
    """
    Computes stock requirements, recommended reorder quantities,
    and classifies inventory risk levels.
    """

    def __init__(
        self,
        default_lead_time_days: int = 1,
        default_safety_stock_ratio: float = 0.20,
        stockout_threshold_ratio: float = 0.75,
        overstock_threshold_multiplier: float = 2.0,
    ) -> None:
        self.default_lead_time_days = default_lead_time_days
        self.default_safety_stock_ratio = default_safety_stock_ratio
        self.stockout_threshold_ratio = stockout_threshold_ratio
        self.overstock_threshold_multiplier = overstock_threshold_multiplier

    def calculate_safety_stock(
        self,
        forecast_demand: float,
        safety_stock: Optional[float] = None,
        safety_stock_ratio: Optional[float] = None
    ) -> float:
        """
        Calculates buffer safety stock. If a direct safety_stock value is given, uses it.
        Otherwise applies a configurable ratio of forecast demand.
        """
        if safety_stock is not None and safety_stock >= 0:
            return float(safety_stock)

        ratio = safety_stock_ratio if safety_stock_ratio is not None else self.default_safety_stock_ratio
        # Safety stock proportional to forecast volume (minimum 5 units)
        computed = max(5.0, round(forecast_demand * ratio, 2))
        return float(computed)

    def optimize_inventory(
        self,
        product_id: str,
        current_inventory: int,
        forecast_demand: float,
        safety_stock: Optional[float] = None,
        lead_time_days: Optional[int] = None,
        store_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates Required Stock, Reorder Quantity, and Inventory Risk Level.
        
        Logic:
          - Lead Time Demand = Forecast Demand * Lead Time Days
          - Required Stock = Lead Time Demand + Safety Stock
          - Reorder Quantity = max(Required Stock - Current Inventory, 0)
          - Risk Classification:
              If Current Inventory < (Forecast Demand * stockout_threshold_ratio):
                  Risk = 'Stock-Out Risk'
              Else if Current Inventory > (Required Stock * overstock_threshold_multiplier):
                  Risk = 'Overstock Risk'
              Else:
                  Risk = 'Normal Stock'
        """
        if current_inventory < 0:
            raise ValueError(f"Current inventory cannot be negative: {current_inventory}")
        if forecast_demand < 0:
            raise ValueError(f"Forecast demand cannot be negative: {forecast_demand}")

        lt = lead_time_days if lead_time_days is not None else self.default_lead_time_days
        lt = max(1, lt)

        effective_safety_stock = self.calculate_safety_stock(forecast_demand, safety_stock)
        lead_time_demand = round(forecast_demand * lt, 2)
        required_stock = round(lead_time_demand + effective_safety_stock, 2)

        # Reorder Quantity
        deficit = required_stock - current_inventory
        reorder_quantity = int(max(0, math.ceil(deficit)))

        # Risk Detection Logic
        stockout_limit = forecast_demand * self.stockout_threshold_ratio
        overstock_limit = required_stock * self.overstock_threshold_multiplier

        if current_inventory < stockout_limit or current_inventory < effective_safety_stock:
            risk_level = "Stock-Out Risk"
            recommendation = (
                f"High stock-out hazard! Current stock ({current_inventory}) is insufficient "
                f"for anticipated demand ({forecast_demand:.0f}). Urgent reorder of {reorder_quantity} units advised."
            )
        elif current_inventory > overstock_limit:
            risk_level = "Overstock Risk"
            recommendation = (
                f"Excess inventory detected! Current stock ({current_inventory}) exceeds 2x "
                f"required buffer ({required_stock:.0f}). Suspend replenishment to reduce holding costs."
            )
        else:
            risk_level = "Normal Stock"
            if reorder_quantity > 0:
                recommendation = (
                    f"Inventory is operating within healthy parameters. Routine top-up of "
                    f"{reorder_quantity} units recommended to sustain target safety buffer."
                )
            else:
                recommendation = "Stock levels are fully sufficient to satisfy target demand cycle."

        return {
            "product_id": product_id,
            "store_id": store_id or "N/A",
            "current_inventory": int(current_inventory),
            "forecast_demand": float(round(forecast_demand, 2)),
            "safety_stock": float(round(effective_safety_stock, 2)),
            "lead_time_days": int(lt),
            "required_stock": float(round(required_stock, 2)),
            "reorder_quantity": int(reorder_quantity),
            "risk_level": risk_level,
            "recommendation": recommendation,
        }


if __name__ == "__main__":
    optimizer = InventoryOptimizer()

    # Case 1: Low stock (Stock-out risk)
    res1 = optimizer.optimize_inventory(
        product_id="P101",
        current_inventory=20,
        forecast_demand=120.0,
        store_id="S01"
    )
    print("Test Case 1 (Depleted Stock):", res1["risk_level"], "| Reorder:", res1["reorder_quantity"])

    # Case 2: Healthy stock (Normal)
    res2 = optimizer.optimize_inventory(
        product_id="P102",
        current_inventory=140,
        forecast_demand=100.0,
        store_id="S01"
    )
    print("Test Case 2 (Healthy Stock):", res2["risk_level"], "| Reorder:", res2["reorder_quantity"])

    # Case 3: Overstocked (Overstock risk)
    res3 = optimizer.optimize_inventory(
        product_id="P103",
        current_inventory=600,
        forecast_demand=80.0,
        store_id="S01"
    )
    print("Test Case 3 (Excessive Stock):", res3["risk_level"], "| Reorder:", res3["reorder_quantity"])
