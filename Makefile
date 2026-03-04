.PHONY: build up down train-ocsvm train-lstm-autoencoder train-wgan-gp train health logs clean benchmark

DATASET ?= data/UNSW_NB15_training-set.csv

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

train-ocsvm:
	python3 -m agents.ocsvm.train --dataset $(DATASET)

train-lstm-autoencoder:
	python3 -m agents.lstm_autoencoder.train --dataset $(DATASET)

train-wgan-gp:
	python3 -m agents.wgan_gp.train --dataset $(DATASET)

train: train-ocsvm train-lstm-autoencoder train-wgan-gp

health:
	@echo "Checking agent health..."
	@curl -s http://localhost:8081/a2a/health || echo "ocsvm: not responding"
	@curl -s http://localhost:8082/a2a/health || echo "lstm_autoencoder: not responding"
	@curl -s http://localhost:8084/a2a/health || echo "wgan_gp: not responding"

logs:
	docker compose logs -f

benchmark:
	python3 benchmark.py --mode all --dataset data/UNSW_NB15_testing-set.csv --config config/orchestrator_config.utility.yaml

clean:
	rm -rf artifacts/runs
