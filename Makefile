# sorakAi developer ergonomics. Linux-only (per the engineering principles).
# Pass VENV=<path> to point at a different virtualenv; defaults to the venv
# that lives next to the repo.

VENV       ?= ../sorakaienv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
RUFF       := $(VENV)/bin/ruff
MYPY       := $(VENV)/bin/mypy
PYTEST     := $(VENV)/bin/pytest
PIP_COMPILE := $(VENV)/bin/pip-compile
PRECOMMIT  := $(VENV)/bin/pre-commit

PKG        := sorakai
TEST_DIR   := tests

.DEFAULT_GOAL := help

.PHONY: help venv install install-dev install-ui lock lint lint-fix format \
        typecheck test test-cov openapi openapi-check dev up down clean eval ui

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

venv: ## Create the virtualenv at $$VENV (defaults to ../sorakaienv)
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: ## Install runtime deps from requirements.txt
	$(PIP) install -r requirements.txt

install-dev: ## Install runtime + dev deps from requirements-dev.txt
	$(PIP) install -r requirements-dev.txt
	$(PRECOMMIT) install --install-hooks

lock: ## Recompile requirements*.txt from requirements*.in (never emits private index URLs)
	$(PIP_COMPILE) --resolver=backtracking --strip-extras --no-emit-index-url --no-emit-trusted-host \
		--output-file=requirements.txt requirements.in
	$(PIP_COMPILE) --resolver=backtracking --strip-extras --no-emit-index-url --no-emit-trusted-host \
		--output-file=requirements-dev.txt requirements-dev.in
	$(PIP_COMPILE) --resolver=backtracking --strip-extras --no-emit-index-url --no-emit-trusted-host \
		--output-file=requirements-ui.txt requirements-ui.in

install-ui: ## Install the Streamlit UI deps (heavy; not needed by services)
	$(PIP) install -r requirements-ui.txt

ui: ## Launch the Streamlit chat UI against the gateway
	$(VENV)/bin/streamlit run ui/streamlit_app.py

lint: ## Run ruff (must be green on PR)
	$(RUFF) check $(PKG) $(TEST_DIR) scripts ui
	$(RUFF) format --check $(PKG) $(TEST_DIR) scripts ui

lint-fix: ## Apply ruff autofixes
	$(RUFF) check --fix $(PKG) $(TEST_DIR) scripts ui
	$(RUFF) format $(PKG) $(TEST_DIR) scripts ui

format: lint-fix ## Alias for lint-fix

typecheck: ## Run mypy --strict (must be green on PR)
	$(MYPY) $(PKG) $(TEST_DIR) scripts ui

test: ## Run the test suite
	$(PYTEST) $(TEST_DIR) --no-cov

test-cov: ## Run the test suite with coverage
	$(PYTEST) $(TEST_DIR) --cov=$(PKG) --cov-report=term --cov-report=xml

openapi: ## Regenerate openapi/*.openapi.{json,yaml} from the live apps
	$(PY) scripts/export_openapi.py --yaml --output openapi

openapi-check: ## Fail if committed openapi/*.json drifts from the code
	$(PY) scripts/export_openapi.py --check --output openapi

dev: ## Run all three services with --reload
	@echo "Start gateway, ingest, rag in three terminals. Example:"
	@echo "  $(PY) -m uvicorn sorakai.ingest.app:app  --reload --port 8001"
	@echo "  $(PY) -m uvicorn sorakai.rag.app:app     --reload --port 8002"
	@echo "  $(PY) -m uvicorn sorakai.gateway.app:app --reload --port 8000"

up: ## docker compose up --build -d
	docker compose up --build -d

down: ## docker compose down -v
	docker compose down -v

clean: ## Remove caches and coverage artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov

eval: ## Placeholder until Wave 9 adds scripts/eval.py
	@echo "Eval harness lands in Wave 9 of the overhaul plan."
