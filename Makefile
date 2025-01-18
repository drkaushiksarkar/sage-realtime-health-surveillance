# sage-realtime-health-surveillance
.PHONY: install test lint train serve

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/

train:
	python -m src.training.train --config configs/default.yaml

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
