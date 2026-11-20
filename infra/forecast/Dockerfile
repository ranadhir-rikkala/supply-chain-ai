# Build wheels in a separate stage so compilers never reach the runtime image.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml ./
COPY pipeline ./pipeline
# xgboost-cpu avoids the CUDA libraries of the full xgboost wheel.
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[service]" "xgboost-cpu>=2.1"

FROM python:3.12-slim
RUN apt-get update \
    # XGBoost needs the OpenMP runtime, and the service runs as a non-root user.
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 forecast \
    && mkdir -p /data \
    && chown forecast:forecast /data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl \
    && rm -rf /wheels
USER forecast
WORKDIR /home/forecast
ENV PYTHONUNBUFFERED=1 LOCAL_STORE_ROOT=/data STORE=local
EXPOSE 8000 9100
CMD ["scforecast", "serve"]
