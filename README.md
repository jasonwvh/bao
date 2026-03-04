# BAO (Lean Bayesian Agent Orchestrator)

Lean multi-agent orchestrator for UNSW-NB15 with:
- Bayesian posterior updates
- Uncertainty-gated escalation
- Approximate VOI query gating
- Cost-aware final decision (`accept|reject|defer`)

## Architecture
- Agents are independent HTTP services with A2A-style endpoints:
  - `GET /a2a/health`
  - `GET /a2a/capabilities`
  - `POST /a2a/infer`
- Orchestrator runtime lives in:
  - `orchestrator/runtime.py`
  - `orchestrator/belief.py`
  - `orchestrator/decisioning.py`
  - `orchestrator/a2a.py`
  - `orchestrator/state.py`
  - `orchestrator/data.py`
  - `orchestrator/config.py`
- Single benchmark entrypoint: `benchmark.py`

## Quickstart

### 1) Train models
```bash
make train
```

### 2) Start agents
```bash
make build
make up
make health
```

### 3) Run benchmarks (single entrypoint)

BAO only:
```bash
python3 benchmark.py \
  --mode bao \
  --dataset data/UNSW_NB15_testing-set.csv \
  --config config/orchestrator_config.utility.yaml
```

Single agent:
```bash
python3 benchmark.py \
  --mode agent \
  --agent ocsvm \
  --dataset data/UNSW_NB15_testing-set.csv \
  --config config/orchestrator_config.utility.yaml
```

All approaches in one run:
```bash
python3 benchmark.py \
  --mode all \
  --dataset data/UNSW_NB15_testing-set.csv \
  --config config/orchestrator_config.utility.yaml
```

### 4) Output contract (per run)
Each run creates `artifacts/runs/<run_id>/` with only:
- `benchmark.json`
- `replay_results.json`
- `run_manifest.json`
- `state.sqlite`

No `flows.jsonl` and no `replay_results.jsonl` are generated.
If `--run-id` already exists, the runner appends a numeric suffix (for example `_01`) so artifacts and sqlite are never reused.

## Configs
- `config/orchestrator_config.utility.yaml` (utility-focused)
- `config/orchestrator_config.accuracy.yaml` (accuracy-focused)
- `config/agents.yaml` (agent registry)
- `config/agent_training.yaml` (training hyperparameters)

## Decision semantics
There is a single decision field:
- `decision`: `accept|reject|defer`

Metrics use:
- Classification: `compromise_prob >= 0.5`
- Utility: realized cost from `decision`

## Notes on A2A SDK
`orchestrator/a2a.py` uses the official A2A Python SDK (`a2aproject/a2a-python`) for agent-card resolution when available, with HTTP JSON fallback for the project’s `POST /a2a/infer` contract.
