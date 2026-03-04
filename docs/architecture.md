# BAO Architecture (Lean)

## Runtime shape
The runtime is intentionally small and built around seven modules:
- `orchestrator/runtime.py`: orchestration loop
- `orchestrator/belief.py`: Bayesian posterior + reliability tracking
- `orchestrator/decisioning.py`: expected-cost decision and VOI math
- `orchestrator/a2a.py`: A2A transport adapter (official SDK card resolution + HTTP JSON infer)
- `orchestrator/state.py`: sqlite state backend
- `orchestrator/data.py`: dataset loading + preprocessing adapter
- `orchestrator/config.py`: config parsing

## Decision semantics
Single decision field only:
- `decision`: `accept|reject|defer`

No `action_decision` exists.

## Query flow
Default sequence is ordered by config (typically `ocsvm -> lstm_autoencoder -> wgan_gp`).

Per flow:
1. Query first agent.
2. Update posterior from agent probability.
3. Compute combined uncertainty.
4. If uncertainty low, stop.
5. If uncertainty high and next agent exists, apply VOI gate against next-agent cost.
6. Query next agent only if VOI gain passes threshold.
7. Finalize decision via expected action cost.
8. If all agents exhausted and uncertainty remains high near 0.5, defer.

## Benchmark interface
Single entrypoint:
- `benchmark.py --mode bao|agent|all`

Each run writes exactly:
- `benchmark.json`
- `replay_results.json`
- `run_manifest.json`
- `state.sqlite`

Run directories are never reused. If `--run-id` collides, the benchmark runner appends a suffix.
