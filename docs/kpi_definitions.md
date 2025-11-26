# KPI Definitions

## Average Order Value (AOV)

- AOV = SUM(UnitPrice _ Quantity _ (1 - Discount)) / COUNT(DISTINCT OrderID)

## Gross Margin

- GM = SUM((UnitPrice - CostOfGoods) _ Quantity _ (1 - Discount))
- If cost is missing, approximate with category-level average (document your
  approach).
