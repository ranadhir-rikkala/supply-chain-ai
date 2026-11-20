"""Internal forecast service called by the platform API."""

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from http import HTTPStatus
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, Path, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from pipeline.forecast.jobs import FORECAST_JOB, TRAINING_JOB, JobPublishError, RabbitJobPublisher
from pipeline.forecast.model import HORIZON_DAYS
from pipeline.forecast.observability import REQUEST_ID_HEADER, request_id_var, resolve_request_id
from pipeline.forecast.registry import (
    ModelNotFound,
    QualityGateFailed,
    activate_model,
    active_model,
    get_model,
    list_models,
)
from pipeline.forecast.runs import NoActiveModel, RunNotFound, create_run, get_run, mark_failed
from pipeline.forecast.series import SeriesNotFound, fetch_series

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_BASE = "https://supplychain.example/problems/"
ACTOR_PATTERN = r"^[A-Za-z0-9._@-]{1,100}$"
VERSION_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
SKU_PATTERN = r"^[A-Za-z0-9-]{1,32}$"
WAREHOUSE_PATTERN = r"^[A-Za-z0-9-]{1,32}$"
UNAUTHENTICATED_PATHS = {"/health/live", "/health/ready", "/metrics"}
DATABASE_RETRY_SECONDS = "5"


@dataclass(frozen=True)
class Settings:
    """Service configuration read from the environment."""

    database_url: str
    service_token: str
    pool_min_size: int = 1
    pool_max_size: int = 8
    pool_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings; refuses to start without a service token of at least 16 characters."""
        token = os.getenv("FORECAST_SERVICE_TOKEN", "")
        if len(token) < 16:
            raise RuntimeError("FORECAST_SERVICE_TOKEN must be set to at least 16 characters")
        return cls(
            database_url=os.environ["DATABASE_URL"],
            service_token=token,
            pool_max_size=int(os.getenv("DB_POOL_MAX_SIZE", "8")),
        )


class ApiProblem(Exception):
    """An error that is returned as RFC 9457 problem+json."""

    def __init__(self, status: int, slug: str, title: str, detail: str, **extra: Any):
        super().__init__(detail)
        self.status, self.slug, self.title, self.detail, self.extra = (
            status,
            slug,
            title,
            detail,
            extra,
        )


class StrictBody(BaseModel):
    """Request body base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ActivationRequest(StrictBody):
    """Body of POST /v1/models/{version}/activation."""

    requested_by: str = Field(pattern=ACTOR_PATTERN)
    reason: str = Field(min_length=3, max_length=500)


class TrainingRequest(StrictBody):
    """Body of POST /v1/training-runs."""

    requested_by: str = Field(pattern=ACTOR_PATTERN)


class ForecastRunRequest(StrictBody):
    """Body of POST /v1/forecast-runs."""

    requested_by: str = Field(pattern=ACTOR_PATTERN)
    origin_date: date | None = None
    horizon_days: int = Field(default=HORIZON_DAYS, ge=1, le=HORIZON_DAYS)


def problem_response(
    status: int,
    slug: str,
    title: str,
    detail: str | None,
    request: Request,
    headers: dict | None = None,
    **extra: Any,
) -> JSONResponse:
    """Build an application/problem+json response that includes the request id."""
    body = {
        "type": f"{PROBLEM_BASE}{slug}" if slug != "about:blank" else "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "requestId": getattr(request.state, "request_id", None),
        **extra,
    }
    return JSONResponse(
        jsonable_encoder({k: v for k, v in body.items() if v is not None}),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def _summary(model: dict) -> dict:
    weekly = (model.get("metrics") or {}).get("weekly", {})
    daily = (model.get("metrics") or {}).get("daily", {})
    return {
        "model_version": model["model_version"],
        "status": model["status"],
        "passed_gate": model["passed_gate"],
        "weekly_wape": weekly.get("wape"),
        "baseline_weekly_wape": weekly.get("baseline_wape"),
        "daily_wape": daily.get("wape"),
        "baseline_daily_wape": daily.get("baseline_wape"),
        "train_start": model["train_start"],
        "train_end": model["train_end"],
        "test_start": model["test_start"],
        "test_end": model["test_end"],
        "created_at": model["created_at"],
        "activated_at": model["activated_at"],
    }


def _detail(model: dict) -> dict:
    metrics = model.get("metrics") or {}
    segments: dict[str, dict[str, list]] = {"daily": {}, "weekly": {}}
    for row in model["segment_metrics"]:
        grain = row["grain"].lower()
        segments[grain].setdefault(row["segment_type"].lower(), []).append(
            {key: row[key] for key in row if key not in ("grain", "segment_type")}
        )
    return {
        **_summary(model),
        "data_fingerprint": model["data_fingerprint"],
        "artifact_key": model["artifact_key"],
        "params": model["params"],
        "gate": metrics.get("gate"),
        "metrics": {
            "weekly": {
                k: v for k, v in metrics.get("weekly", {}).items() if not k.startswith("by_")
            },
            "daily": {k: v for k, v in metrics.get("daily", {}).items() if not k.startswith("by_")},
            "by_category": {
                "weekly": segments["weekly"].get("category", []),
                "daily": segments["daily"].get("category", []),
            },
            "by_warehouse": {
                "weekly": segments["weekly"].get("warehouse", []),
                "daily": segments["daily"].get("warehouse", []),
            },
        },
        "status_history": model["status_history"],
    }


def _run(run: dict) -> dict:
    return {key: run[key] for key in run}


def create_app(
    settings: Settings | None = None,
    pool: ConnectionPool | None = None,
    publisher=None,
) -> FastAPI:
    """Build the FastAPI app with its connection pool, job publisher and metrics."""
    settings = settings or Settings.from_env()
    pool = pool or ConnectionPool(
        settings.database_url,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        timeout=settings.pool_timeout_seconds,
        # Each pooled connection is tested before use, so a database restart does not surface as a 500.
        check=ConnectionPool.check_connection,
        open=False,
        name="forecast-api",
    )
    publisher = publisher or RabbitJobPublisher()
    registry = CollectorRegistry()
    requests_total = Counter(
        "forecast_api_requests_total",
        "HTTP requests handled by the forecast API",
        ["method", "route", "status"],
        registry=registry,
    )
    request_seconds = Histogram(
        "forecast_api_request_duration_seconds",
        "HTTP request latency of the forecast API",
        ["method", "route"],
        registry=registry,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    jobs_published = Counter(
        "forecast_api_jobs_published_total",
        "Jobs published to RabbitMQ",
        ["job_type", "outcome"],
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pool.closed:
            # Startup does not wait for the database; readiness reports DOWN until it is reachable.
            pool.open(wait=False)
        yield
        pool.close()

    app = FastAPI(
        title="Forecast service",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.pool = pool

    def require_token(request: Request) -> None:
        """Reject requests without the service bearer token."""
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        # Constant-time comparison, so response timing does not reveal the token.
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode(), settings.service_token.encode()
        ):
            raise ApiProblem(
                401,
                "unauthorized",
                "Unauthorized",
                "a valid service bearer token is required",
            )

    Authorized = Depends(require_token)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error")
            response = problem_response(
                500, "about:blank", "Internal Server Error", "unexpected error", request
            )
        finally:
            request_id_var.reset(token)
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        elapsed = time.perf_counter() - started
        requests_total.labels(request.method, route_path, str(response.status_code)).inc()
        request_seconds.labels(request.method, route_path).observe(elapsed)
        response.headers[REQUEST_ID_HEADER] = request_id
        if route_path not in UNAUTHENTICATED_PATHS:
            logger.info(
                "request handled",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": route_path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
        return response

    @app.exception_handler(ApiProblem)
    async def handle_problem(request: Request, error: ApiProblem):
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return problem_response(
            error.status, error.slug, error.title, error.detail, request, headers, **error.extra
        )

    @app.exception_handler(HTTPException)
    async def handle_http(request: Request, error: HTTPException):
        return problem_response(
            error.status_code,
            "about:blank",
            HTTPStatus(error.status_code).phrase,
            str(error.detail) if error.detail else None,
            request,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, error: RequestValidationError):
        errors = [
            {"location": list(item["loc"]), "message": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return problem_response(
            422,
            "validation-error",
            "Invalid request",
            "the request is invalid",
            request,
            errors=errors,
        )

    @app.exception_handler(psycopg.OperationalError)
    @app.exception_handler(PoolTimeout)
    async def handle_database(request: Request, error: Exception):
        logger.warning("database unavailable", extra={"error": str(error)})
        return problem_response(
            503,
            "database-unavailable",
            "Database unavailable",
            "the database is not reachable; retry shortly",
            request,
            {"Retry-After": DATABASE_RETRY_SECONDS},
        )

    @app.exception_handler(JobPublishError)
    async def handle_publish(request: Request, error: JobPublishError):
        return problem_response(
            503,
            "queue-unavailable",
            "Job queue unavailable",
            "the job could not be queued; retry shortly",
            request,
            {"Retry-After": DATABASE_RETRY_SECONDS},
        )

    @app.get("/health/live")
    def live() -> dict:
        """Liveness: the process is up."""
        return {"status": "UP"}

    @app.get("/health/ready")
    def ready(request: Request):
        """Readiness: the database answers; also reports the ACTIVE model."""
        try:
            with pool.connection(timeout=2) as conn:
                model = active_model(conn)
        except (psycopg.OperationalError, PoolTimeout):
            return problem_response(
                503,
                "not-ready",
                "Service not ready",
                "the database is not reachable",
                request,
                {"Retry-After": DATABASE_RETRY_SECONDS},
                checks={"database": "DOWN"},
            )
        return {
            "status": "UP",
            "checks": {"database": "UP"},
            "activeModel": model["model_version"] if model else None,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus metrics."""
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", dependencies=[Authorized])
    def models() -> dict:
        """All model versions with their weekly and daily WAPE."""
        with pool.connection() as conn:
            return {"items": [_summary(model) for model in list_models(conn)]}

    @app.get("/v1/models/active", dependencies=[Authorized])
    def models_active() -> dict:
        """The ACTIVE model with metrics by category and warehouse."""
        with pool.connection() as conn:
            model = active_model(conn)
            if model is None:
                raise ApiProblem(404, "no-active-model", "No active model", "no model is ACTIVE")
            return _detail(get_model(conn, model["model_version"]))

    @app.get("/v1/models/{version}", dependencies=[Authorized])
    def model_detail(version: Annotated[str, Path(pattern=VERSION_PATTERN)]) -> dict:
        """One model version with segment metrics and status history."""
        try:
            with pool.connection() as conn:
                return _detail(get_model(conn, version))
        except ModelNotFound as error:
            raise ApiProblem(
                404, "model-not-found", "Model not found", f"model {version} does not exist"
            ) from error

    @app.post("/v1/models/{version}/activation", dependencies=[Authorized])
    def activation(
        version: Annotated[str, Path(pattern=VERSION_PATTERN)], body: ActivationRequest
    ) -> dict:
        """Promote a gate-passing model or roll back to an older one; 409 if it failed the gate."""
        with pool.connection() as conn:
            return _activate(conn, version, body)

    def _activate(conn, version: str, body: ActivationRequest) -> dict:
        try:
            result = activate_model(conn, version, body.requested_by, body.reason)
        except ModelNotFound as error:
            raise ApiProblem(
                404, "model-not-found", "Model not found", f"model {version} does not exist"
            ) from error
        except QualityGateFailed as error:
            raise ApiProblem(
                409,
                "quality-gate-failed",
                "Model failed the quality gate",
                str(error),
                modelVersion=version,
            ) from error
        logger.info("model activated", extra={**result, "requested_by": body.requested_by})
        return {**_detail(get_model(conn, version)), "previous_model_version": result["previous"]}

    @app.post("/v1/training-runs", status_code=202, dependencies=[Authorized])
    def training_runs(request: Request, body: TrainingRequest) -> dict:
        """Queue a training job."""
        job_id = str(uuid.uuid4())
        message = {"job_type": TRAINING_JOB, "job_id": job_id, "requested_by": body.requested_by}
        try:
            publisher.publish(message, request.state.request_id)
        except JobPublishError:
            jobs_published.labels(TRAINING_JOB, "failed").inc()
            raise
        jobs_published.labels(TRAINING_JOB, "queued").inc()
        return {"jobId": job_id, "status": "QUEUED"}

    @app.post("/v1/forecast-runs", status_code=202, dependencies=[Authorized])
    def forecast_runs(request: Request, response: Response, body: ForecastRunRequest) -> dict:
        """Queue a forecast run for the ACTIVE model."""
        with pool.connection() as conn:
            return _queue_forecast(conn, request, response, body)

    def _queue_forecast(conn, request: Request, response: Response, body: ForecastRunRequest):
        try:
            run = create_run(conn, body.requested_by, body.origin_date, body.horizon_days)
        except NoActiveModel as error:
            raise ApiProblem(409, "no-active-model", "No active model", str(error)) from error
        message = {
            "job_type": FORECAST_JOB,
            "job_id": str(run["run_id"]),
            "run_id": str(run["run_id"]),
            "requested_by": body.requested_by,
        }
        try:
            publisher.publish(message, request.state.request_id)
        except JobPublishError:
            jobs_published.labels(FORECAST_JOB, "failed").inc()
            # The run row already exists, so mark it FAILED rather than leave it QUEUED forever.
            mark_failed(conn, run["run_id"], "could not enqueue the forecast job")
            raise
        jobs_published.labels(FORECAST_JOB, "queued").inc()
        response.headers["Location"] = f"/v1/forecast-runs/{run['run_id']}"
        return _run(run)

    @app.get("/v1/forecast-runs/{run_id}", dependencies=[Authorized])
    def forecast_run(run_id: uuid.UUID) -> dict:
        """Status of a forecast run."""
        try:
            with pool.connection() as conn:
                return _run(get_run(conn, run_id))
        except RunNotFound as error:
            raise ApiProblem(
                404, "run-not-found", "Forecast run not found", f"run {run_id} does not exist"
            ) from error

    @app.get("/v1/series/{sku}/{warehouse_id}", dependencies=[Authorized])
    def series(
        sku: Annotated[str, Path(pattern=SKU_PATTERN)],
        warehouse_id: Annotated[str, Path(pattern=WAREHOUSE_PATTERN)],
        history_days: Annotated[int, Query(ge=1, le=1096)] = 120,
    ) -> dict:
        """History, backtest and forecast of one series."""
        try:
            with pool.connection() as conn:
                return fetch_series(conn, sku, warehouse_id, history_days)
        except SeriesNotFound as error:
            raise ApiProblem(
                404, "series-not-found", "Series not found", f"no series {sku} at {warehouse_id}"
            ) from error

    return app
