.PHONY: build up down train-isolation-forest train-autoencoder train health logs clean

DATASET ?= data/UNSW_NB15_training-set.csv

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

train-isolation-forest:
	python3 -m agents.isolation_forest.train --dataset $(DATASET)

train-autoencoder:
	python3 -m agents.autoencoder.train --dataset $(DATASET)

train: train-isolation-forest train-autoencoder

health:
	@echo "Checking agent health..."
	@curl -s http://localhost:8081/a2a/health || echo "isolation_forest: not responding"
	@curl -s http://localhost:8082/a2a/health || echo "autoencoder: not responding"
	@curl -s http://localhost:8084/a2a/health || echo "llm: not responding"

logs:
	docker compose logs -f

clean:
	rm -rf artifacts/replay/*.jsonl artifacts/replay/*.json artifacts/replay/*.yaml artifacts/state/*.sqlite
