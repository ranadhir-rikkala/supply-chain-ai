-- Read-only role for Grafana: only the ETL and model tables its panels query.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_reader') THEN
        CREATE ROLE grafana_reader LOGIN;
    END IF;
END
$$;
ALTER ROLE grafana_reader WITH LOGIN PASSWORD '${grafana_db_password}';
-- Even if a grant were widened by mistake, this role cannot write.
ALTER ROLE grafana_reader SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE ${flyway:database} TO grafana_reader;
GRANT USAGE ON SCHEMA ops, ml TO grafana_reader;
GRANT SELECT ON ops.etl_run, ops.etl_file TO grafana_reader;
GRANT SELECT ON ml.model_version, ml.backtest_metric, ml.forecast_run TO grafana_reader;
