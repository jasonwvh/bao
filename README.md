# BAO (Bayesian Agent Orchestrator)

Containerized multi-agent orchestrator with a deterministic Bayesian cascade controller, VOI-based query gating, and a shared SQLite state backend. Agents run as independent HTTP services; the orchestrator connects to them via a registry in `config/agents.yaml`.

**Key traits**
- Control plane vs data plane split
- Registry-driven agent discovery (YAML)
- A2A HTTP+JSON contract for inference/health/capabilities
- Shared state backend (SQLite) keyed by `agent_id`
- Posterior-first belief updates with optional strict-likelihood mode
- Expected-cost action selection (`accept` / `reject` / `defer`)
- Approximate VOI query gating from configuration

**Architecture diagram**

```mermaid
flowchart LR
  subgraph ControlPlane["Control Plane"]
    REG["Agent Registry (YAML)"]
    ORCH["Orchestrator\nDeterministic Cascade\nBelief + VOI + Decision"]
    POL["Config-Driven Policy\nCosts + Query"]
  end

  subgraph DataPlane["Data Plane"]
    A2A["A2A HTTP Client"]
    STATE["Shared State Backend\nSQLite"]
  end

  subgraph Agents["Agent Services (Containers)"]
    IF["ocsvm\nPort 8081"]
    AE["lstm_autoencoder\nPort 8082"]
    LLM["wgan_gp\nPort 8084"]
  end

  REG --> ORCH
  POL --> ORCH
  ORCH --> A2A
  A2A --> IF
  A2A --> AE
  A2A --> LLM
  ORCH <--> STATE
```

## Quickstart

### 1) Train models (optional - models can be pre-built)
```bash
make train
```

### 2) Build and start agents
```bash
make build
make up
```

### 3) Run orchestrator replay
Requires a labeled dataset with a `label` column.

```bash
python3 main.py \
  --dataset /path/to/replay.csv \
  --config config/orchestrator_config.yaml \
  --max-flows 1000
```

Useful replay options:
```bash
python3 main.py \
  --dataset data/UNSW_NB15_testing-set.csv \
  --config config/orchestrator_config.yaml \
  --prediction-source decision \
  --reset-state
```

### 4) Check agent health
```bash
make health
```

### 5) Run benchmarks
All benchmark scripts use A2A black-box calls through `config/agents.yaml`.

```bash
# First-agent baseline (LSTM)
python3 agents/lstm_autoencoder/benchmark.py \
  --dataset data/UNSW_NB15_testing-set.csv \
  --prediction-source decision \
  --output-dir artifacts/replay

# BAO replay
python3 main.py \
  --dataset data/UNSW_NB15_testing-set.csv \
  --config config/orchestrator_config.yaml \
  --max-agents 1 \
  --agent-sequence lstm_autoencoder,ocsvm \
  --prediction-source decision \
  --output-dir artifacts/replay
```

### 6) Stop agents
```bash
make down
```

## Agents

| Agent | Port | Model | Description |
|-------|------|-------|-------------|
| `ocsvm` | 8081 | sklearn One-Class SVM | Lightweight one-class anomaly detector |
| `lstm_autoencoder` | 8082 | PyTorch Hybrid LSTM Autoencoder | Temporal reconstruction anomaly detector |
| `wgan_gp` | 8084 | PyTorch WGAN-GP | Generative anomaly detector |

## A2A HTTP Contract

**Inference** `POST /a2a/infer`

```json
{
  "request_id": "uuid",
  "flow_id": "string",
  "timestamp": 0.0,
  "flow_features": {},
  "context": {
    "belief": {"p_mal": 0.5, "uncertainty": 0.69},
    "requested_capabilities": []
  }
}
```

**Response**

```json
{
  "agent_id": "ocsvm",
  "proba": [0.7, 0.3],
  "prediction": {"label": "benign", "probability": 0.3},
  "uncertainty": {"epistemic": 0.1, "aleatoric": 0.2, "total_entropy": 0.3},
  "cost": 1.0,
  "latency_ms": 12.0,
  "metadata": {}
}
```

**Health** `GET /a2a/health` -> `{"status":"ok","agent_id":"...","version":"..."}`

**Capabilities** `GET /a2a/capabilities` -> `{"agent_id":"...","capabilities":[...],"cost":...}`

## Configuration

- `config/agents.yaml`: registry for containerized agents
- `config/orchestrator_config.yaml`: source of truth for update mode, fusion, decision costs, query policy, VOI, and benchmark behavior

## Project layout

```
.
├── main.py                          # Entry point for replay
├── Makefile                         # Build/run commands
├── docker-compose.yml               # Agent containers
├── config/
│   ├── agents.yaml                  # Agent registry
│   └── orchestrator_config.yaml     # Orchestrator config
├── orchestrator/
│   ├── integrated_system.py         # Orchestrator runtime
│   ├── control/                     # Policy, registry, scheduler
│   └── data_plane/                  # A2A client, state backend
└── agents/
    ├── ocsvm/service.py             # One-Class SVM agent
    ├── lstm_autoencoder/service.py  # Hybrid LSTM autoencoder agent
    └── wgan_gp/service.py           # WGAN-GP agent
```

## Notes
- All inference is via A2A HTTP - agents are black boxes
- One-agent parity is guaranteed in `posterior_first` mode when `query.max_agents=1`
- Benchmark runs can reset state and emit reproducibility manifests
