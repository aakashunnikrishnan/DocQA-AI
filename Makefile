# ============================================================================
# DocQA AI - Makefile
# ============================================================================
# Usage: make [target]
#
# Available targets:
#   help            - Show this help message
#   install         - Install project dependencies
#   install-dev     - Install development dependencies
#   update          - Update dependencies
#   clean           - Clean build artifacts and cache
#   clean-all       - Clean everything including data
#   test            - Run tests
#   test-coverage   - Run tests with coverage report
#   test-integration - Run integration tests
#   lint            - Run linters
#   format          - Format code
#   type-check      - Run type checking
#   pre-commit      - Run all pre-commit checks
#   run             - Run the API server
#   run-dev         - Run API server in development mode
#   ingest          - Ingest documents
#   build-docker    - Build Docker image
#   docker-up       - Start Docker services
#   docker-down     - Stop Docker services
#   docker-logs     - View Docker logs
#   docs            - Build documentation
#   docs-serve      - Serve documentation locally
#   bench           - Run benchmarks
#   security        - Run security checks
#   build           - Build distribution package
#   release         - Create release
# ============================================================================

# Environment variables
PYTHON := python3
PIP := pip
POETRY := poetry
PYTEST := pytest
BLACK := black
ISORT := isort
RUFF := ruff
MYPY := mypy
FLAKE8 := flake8

# Project variables
PROJECT_NAME := docqa-ai
VERSION := $(shell python -c "import sys; sys.path.insert(0, '.'); from src.utils.config import get_config; print(get_config().version if hasattr(get_config(), 'version') else '1.0.0')" 2>/dev/null || echo "1.0.0")
PYTHON_FILES := src/ api/ tests/ scripts/
DOCKER_IMAGE := docqa-ai
DOCKER_TAG := latest
ENV_FILE := .env

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# Help
# ============================================================================
.PHONY: help
help:
	@echo "$(BLUE)DocQA AI - Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  $(BLUE)install$(NC)         - Install project dependencies"
	@echo "  $(BLUE)install-dev$(NC)     - Install development dependencies"
	@echo "  $(BLUE)update$(NC)          - Update dependencies"
	@echo "  $(BLUE)clean$(NC)           - Clean build artifacts and cache"
	@echo "  $(BLUE)clean-all$(NC)       - Clean everything including data"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  $(BLUE)test$(NC)            - Run tests"
	@echo "  $(BLUE)test-coverage$(NC)   - Run tests with coverage report"
	@echo "  $(BLUE)test-integration$(NC)- Run integration tests"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@echo "  $(BLUE)lint$(NC)            - Run linters"
	@echo "  $(BLUE)format$(NC)          - Format code"
	@echo "  $(BLUE)type-check$(NC)      - Run type checking"
	@echo "  $(BLUE)pre-commit$(NC)      - Run all pre-commit checks"
	@echo ""
	@echo "$(YELLOW)Running:$(NC)"
	@echo "  $(BLUE)run$(NC)             - Run the API server"
	@echo "  $(BLUE)run-dev$(NC)         - Run API server in development mode"
	@echo "  $(BLUE)ingest$(NC)          - Ingest documents"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  $(BLUE)build-docker$(NC)    - Build Docker image"
	@echo "  $(BLUE)docker-up$(NC)       - Start Docker services"
	@echo "  $(BLUE)docker-down$(NC)     - Stop Docker services"
	@echo "  $(BLUE)docker-logs$(NC)     - View Docker logs"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  $(BLUE)docs$(NC)            - Build documentation"
	@echo "  $(BLUE)docs-serve$(NC)      - Serve documentation locally"
	@echo ""
	@echo "$(YELLOW)Other:$(NC)"
	@echo "  $(BLUE)bench$(NC)           - Run benchmarks"
	@echo "  $(BLUE)security$(NC)        - Run security checks"
	@echo "  $(BLUE)build$(NC)           - Build distribution package"
	@echo "  $(BLUE)release$(NC)         - Create release"

# ============================================================================
# Installation
# ============================================================================
.PHONY: install
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

.PHONY: install-dev
install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -r requirements-dev.txt
	@$(PIP) install -e .
	@pre-commit install
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

.PHONY: update
update:
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@$(PIP) install --upgrade -r requirements.txt
	@$(PIP) install --upgrade -r requirements-dev.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

# ============================================================================
# Cleaning
# ============================================================================
.PHONY: clean
clean:
	@echo "$(BLUE)Cleaning...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ htmlcov/ site/ .coverage coverage.xml 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned build artifacts$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE)Cleaning everything...$(NC)"
	@rm -rf data/embeddings/ data/vector_store/ data/intermediate/ logs/ cache/ 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned everything$(NC)"

# ============================================================================
# Testing
# ============================================================================
.PHONY: test
test:
	@echo "$(BLUE)Running tests...$(NC)"
	@$(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)✓ Tests passed$(NC)"

.PHONY: test-coverage
test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) tests/ --cov=src --cov=api --cov-report=html --cov-report=term --cov-report=xml -v
	@echo "$(GREEN)✓ Coverage report generated$(NC)"
	@echo "$(YELLOW)Open htmlcov/index.html to view coverage report$(NC)"

.PHONY: test-integration
test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(PYTEST) tests/test_integration.py -v
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

.PHONY: test-all
test-all: test test-integration test-coverage
	@echo "$(GREEN)✓ All tests passed$(NC)"

# ============================================================================
# Code Quality
# ============================================================================
.PHONY: lint
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "$(YELLOW)Black...$(NC)"
	@$(BLACK) --check $(PYTHON_FILES) || true
	@echo "$(YELLOW)isort...$(NC)"
	@$(ISORT) --check-only $(PYTHON_FILES) || true
	@echo "$(YELLOW)Ruff...$(NC)"
	@$(RUFF) check $(PYTHON_FILES)
	@echo "$(YELLOW)Flake8...$(NC)"
	@$(FLAKE8) $(PYTHON_FILES) --count --max-complexity=10 --max-line-length=127 --statistics || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

.PHONY: format
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(BLACK) $(PYTHON_FILES)
	@$(ISORT) $(PYTHON_FILES)
	@echo "$(GREEN)✓ Code formatted$(NC)"

.PHONY: type-check
type-check:
	@echo "$(BLUE)Running type checking...$(NC)"
	@$(MYPY) src/ api/ --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)✓ Type checking passed$(NC)"

.PHONY: pre-commit
pre-commit: format lint type-check test
	@echo "$(GREEN)✓ All pre-commit checks passed$(NC)"

# ============================================================================
# Running
# ============================================================================
.PHONY: run
run:
	@echo "$(BLUE)Starting API server...$(NC)"
	@$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

.PHONY: run-dev
run-dev:
	@echo "$(BLUE)Starting API server in development mode...$(NC)"
	@$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload --log-level debug

.PHONY: ingest
ingest:
	@echo "$(BLUE)Ingesting documents...$(NC)"
	@$(PYTHON) scripts/ingest_documents.py $(INPUT)

.PHONY: ingest-sample
ingest-sample:
	@echo "$(BLUE)Ingesting sample documents...$(NC)"
	@$(PYTHON) scripts/ingest_documents.py data/raw/sample_docs/ -o data/vector_store

.PHONY: ingest-watch
ingest-watch:
	@echo "$(BLUE)Watching for document changes...$(NC)"
	@$(PYTHON) scripts/ingest_documents.py --watch $(INPUT)

# ============================================================================
# Docker
# ============================================================================
.PHONY: build-docker
build-docker:
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f docker/Dockerfile .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

.PHONY: docker-up
docker-up:
	@echo "$(BLUE)Starting Docker services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Docker services started$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:3000 (if configured)$(NC)"

.PHONY: docker-up-full
docker-up-full:
	@echo "$(BLUE)Starting all Docker services...$(NC)"
	@docker-compose --profile full up -d
	@echo "$(GREEN)✓ All Docker services started$(NC)"

.PHONY: docker-down
docker-down:
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✓ Docker services stopped$(NC)"

.PHONY: docker-down-clean
docker-down-clean:
	@echo "$(BLUE)Stopping Docker services and removing volumes...$(NC)"
	@docker-compose down -v
	@echo "$(GREEN)✓ Docker services stopped and volumes removed$(NC)"

.PHONY: docker-logs
docker-logs:
	@docker-compose logs -f --tail=100

.PHONY: docker-build-prod
docker-build-prod:
	@echo "$(BLUE)Building production Docker image...$(NC)"
	@docker build -t $(DOCKER_IMAGE):production -f docker/Dockerfile --target production .
	@echo "$(GREEN)✓ Production Docker image built$(NC)"

# ============================================================================
# Documentation
# ============================================================================
.PHONY: docs
docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	@mkdocs build --clean
	@echo "$(GREEN)✓ Documentation built in site/$(NC)"

.PHONY: docs-serve
docs-serve:
	@echo "$(BLUE)Serving documentation...$(NC)"
	@mkdocs serve

# ============================================================================
# Benchmarks
# ============================================================================
.PHONY: bench
bench:
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@$(PYTEST) tests/ --benchmark-only --benchmark-json=benchmarks.json
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

# ============================================================================
# Security
# ============================================================================
.PHONY: security
security:
	@echo "$(BLUE)Running security checks...$(NC)"
	@echo "$(YELLOW)Safety...$(NC)"
	@$(PYTHON) -m safety check -r requirements.txt
	@echo "$(YELLOW)Bandit...$(NC)"
	@bandit -r src/ api/ -ll
	@echo "$(YELLOW)Trivy...$(NC)"
	@trivy fs . --severity HIGH,CRITICAL --exit-code 1 || true
	@echo "$(GREEN)✓ Security checks complete$(NC)"

# ============================================================================
# Build
# ============================================================================
.PHONY: build
build: clean
	@echo "$(BLUE)Building distribution package...$(NC)"
	@$(PYTHON) -m pip install --upgrade build
	@$(PYTHON) -m build
	@echo "$(GREEN)✓ Package built in dist/$(NC)"

.PHONY: release
release: build
	@echo "$(BLUE)Creating release...$(NC)"
	@git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@git push origin v$(VERSION)
	@echo "$(GREEN)✓ Release v$(VERSION) created$(NC)"

# ============================================================================
# Environment
# ============================================================================
.PHONY: env
env:
	@echo "$(BLUE)Setting up environment...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env file from .env.example$(NC)"; \
		echo "$(YELLOW)Please edit .env with your configuration$(NC)"; \
	else \
		echo "$(GREEN)✓ .env file already exists$(NC)"; \
	fi

.PHONY: env-check
env-check:
	@echo "$(BLUE)Checking environment...$(NC)"
	@echo "$(YELLOW)Python version:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Pip version:$(NC) $$($(PIP) --version)"
	@echo "$(YELLOW)OpenAI API key:$(NC) $$(if [ -n "$$OPENAI_API_KEY" ]; then echo "$(GREEN)✓ Set$(NC)"; else echo "$(RED)✗ Not set$(NC)"; fi)"
	@echo "$(YELLOW)Environment:$(NC) $$(python -c "import os; print(os.getenv('ENVIRONMENT', 'development'))")"
	@echo "$(YELLOW)Log level:$(NC) $$(python -c "import os; print(os.getenv('LOG_LEVEL', 'INFO'))")"

# ============================================================================
# Development Utilities
# ============================================================================
.PHONY: shell
shell:
	@$(PYTHON) -c "import IPython; IPython.terminal.ipapp.launch_new_instance()"

.PHONY: notebook
notebook:
	@$(PYTHON) -m jupyter lab notebooks/

.PHONY: update-schema
update-schema:
	@echo "$(BLUE)Updating API schema...$(NC)"
	@$(PYTHON) -c "from api.app import app; import json; print(json.dumps(app.openapi(), indent=2))" > api/openapi.json
	@echo "$(GREEN)✓ API schema updated$(NC)"

# ============================================================================
# Database
# ============================================================================
.PHONY: db-init
db-init:
	@echo "$(BLUE)Initializing database...$(NC)"
	@$(PYTHON) -c "from src.utils.database import init_db; init_db()"
	@echo "$(GREEN)✓ Database initialized$(NC)"

.PHONY: db-migrate
db-migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	@alembic upgrade head
	@echo "$(GREEN)✓ Database migrations applied$(NC)"

.PHONY: db-rollback
db-rollback:
	@echo "$(BLUE)Rolling back database...$(NC)"
	@alembic downgrade -1
	@echo "$(GREEN)✓ Database rollback complete$(NC)"

# ============================================================================
# Production
# ============================================================================
.PHONY: deploy
deploy:
	@echo "$(BLUE)Deploying to production...$(NC)"
	@echo "$(YELLOW)This will deploy to production. Are you sure? [y/N]$(NC)" && read ans && [ $${ans:-N} = y ]
	@docker-compose -f docker/docker-compose.prod.yml up -d
	@echo "$(GREEN)✓ Deployment complete$(NC)"

.PHONY: health
health:
	@echo "$(BLUE)Checking health...$(NC)"
	@curl -s http://localhost:8000/health | $(PYTHON) -m json.tool || echo "$(RED)✗ Health check failed$(NC)"

# ============================================================================
# Default target
# ============================================================================
.DEFAULT_GOAL := help

# ============================================================================
# Phony targets (always run, even if file exists)
# ============================================================================
.PHONY: help install install-dev update clean clean-all test test-coverage \
	test-integration test-all lint format type-check pre-commit run run-dev \
	ingest ingest-sample ingest-watch build-docker docker-up docker-up-full \
	docker-down docker-down-clean docker-logs docker-build-prod docs docs-serve \
	bench security build release env env-check shell notebook update-schema \
	db-init db-migrate db-rollback deploy health
