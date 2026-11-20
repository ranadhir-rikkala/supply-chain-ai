-- Least-privilege role for forecast-api and forecast-worker: read core, read and write ml.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'forecast_service') THEN
        CREATE ROLE forecast_service LOGIN;
    END IF;
END
$$;
-- The password comes from a Flyway placeholder, so it never lives in git.
ALTER ROLE forecast_service WITH LOGIN PASSWORD '${forecast_db_password}';
GRANT CONNECT ON DATABASE ${flyway:database} TO forecast_service;
GRANT USAGE ON SCHEMA core TO forecast_service;
GRANT SELECT ON ALL TABLES IN SCHEMA core TO forecast_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT SELECT ON TABLES TO forecast_service;
GRANT USAGE ON SCHEMA ml TO forecast_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ml TO forecast_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ml TO forecast_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO forecast_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA ml GRANT USAGE, SELECT ON SEQUENCES TO forecast_service;
