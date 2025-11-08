.PHONY: help install test lint format clean dvc-setup dvc-repro dvc-push dvc-pull

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ML Pipeline Commands
install: ## Install all dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	cd CommentAnaysisExtension/backend && pip install -r requirements.txt

dvc-setup: ## Initialize DVC (if not already initialized)
	@echo "Setting up DVC..."
	@dvc --version || (echo "DVC not installed. Install with: pip install dvc" && exit 1)
	@dvc remote list || dvc remote add -d storage s3://your-bucket/path

dvc-repro: ## Run DVC pipeline
	dvc repro

dvc-push: ## Push data and models to DVC remote
	dvc push

dvc-pull: ## Pull data and models from DVC remote
	dvc pull

run-pipeline: ## Run the complete ML pipeline
	python main.py

# Code Quality
test: ## Run all tests
	pytest CommentAnaysisExtension/backend/tests/ -v
	pytest tests/ -v || true

lint: ## Run linters
	flake8 src/ --max-line-length=120
	flake8 CommentAnaysisExtension/backend/ --max-line-length=120
	black --check --line-length 120 .

format: ## Format code
	black --line-length 120 .
	isort .

clean: ## Clean cache and temporary files
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist build

# Docker Commands
docker-build-backend: ## Build backend Docker image
	cd CommentAnaysisExtension/backend && docker build -t comment-analysis-backend:latest .

docker-run-backend: ## Run backend Docker container
	cd CommentAnaysisExtension/backend && docker run -p 8000:8000 --env-file .env comment-analysis-backend:latest

# Development
dev-backend: ## Run backend in development mode
	cd CommentAnaysisExtension/backend && python run_local.py

# Documentation
docs: ## Generate documentation (if using sphinx/mkdocs)
	@echo "Documentation generation not configured yet"

# Data Management
download-data: ## Download data using DVC
	dvc pull artifacts/data_ingestion/reddit.csv.dvc

# Model Management
register-model: ## Register model in MLflow
	python -m src.CommentAnalysis.pipeline.register_model

# CI/CD Helpers
check-all: lint test ## Run all checks (lint + test)

