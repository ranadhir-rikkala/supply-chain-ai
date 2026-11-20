.PHONY: venv up down generate load test lint
PYTHON ?= python3.12
VENV ?= .venv
PIP = $(VENV)/bin/pip
SCGEN = $(VENV)/bin/scgen
SCETL = $(VENV)/bin/scetl

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e '.[dev,forecast,service]'

up:
	docker compose up -d postgres rabbitmq flyway && docker compose up -d --scale etl-worker=2 etl-worker

down:
	docker compose down

generate:
	$(SCGEN) --output $${DATA_ROOT:-data}

load:
	$(SCETL) load-master --root $${DATA_ROOT:-data}

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check . && $(VENV)/bin/ruff format --check .
