.PHONY: install test lint clean run pipeline pipeline-sample dashboard \
       docker-build docker-pipeline docker-dashboard docker-up docker-down

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/customer360.duckdb

pipeline:
	python -m src.main

pipeline-sample:
	python -m src.main --sample

dashboard:
	streamlit run src/dashboard/app.py

run: pipeline

docker-build:
	docker build --target pipeline -t customer-360-pipeline .
	docker build --target dashboard -t customer-360-dashboard .

docker-pipeline:
	docker run --rm -v customer360-data:/app/data customer-360-pipeline

docker-dashboard:
	docker run --rm -p 8501:8501 -v customer360-data:/app/data customer-360-dashboard

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
