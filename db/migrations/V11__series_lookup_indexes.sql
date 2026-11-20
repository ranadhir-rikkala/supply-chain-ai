-- The series endpoint looks up one sku and warehouse; without these it scans every partition.
CREATE INDEX IF NOT EXISTS fact_sales_daily_series ON core.fact_sales_daily (sku, warehouse_id, feed_date);
CREATE INDEX IF NOT EXISTS fact_inventory_daily_series ON core.fact_inventory_daily (sku, warehouse_id, feed_date);
