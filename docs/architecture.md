# Bayesian Agent Orchestrator (BAO) — System Architecture v5

## Runtime truth (current implementation)

- **Default engine:** deterministic orchestrator loop (`orchestration.engine: deterministic`)
- **Optional engine:** LangGraph parity runtime (`orchestration.engine: langgraph`)
- **Cheapest-first policy:** first queried agent is selected dynamically from healthy enabled agents when `orchestration.first_agent_strategy: dynamic_cheapest`
- **Decision semantics:** `decision` is the classification output (`accept`/`reject` from posterior threshold), while `action_decision` is the expected-cost action used for utility accounting
- **Routing objective:** expected gain with utilization-band adjustments (`query.utilization_targets`) + warmup gating (`query.utilization_warmup_flows`) + seeded exploration/top-up controls (`query.exploration_*`)
- **Adaptive gate:** optional entropy gate in adaptive routing (`query.apply_uncertainty_gate_in_adaptive`) with ordered escalation (`query.escalation_ordered`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAFFIC INGESTION LAYER                         │
│                                                                     │
│        Packet Capture → Feature Extraction → Streaming Buffer       │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SHARED CONTEXT LAYER                           │
│                                                                     │
│   Shared Resources: belief states, observation models, threat intel │
│   Shared Services:  belief update, VOI computation, agent routing   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│           BAYESIAN ORCHESTRATION LAYER  [LangGraph Graph]           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │             BELIEF STATE MANAGER  [LangGraph Node]           │   │
│  │                                                              │   │
│  │  • Maintains posterior belief over latent threat state       │   │
│  │  • Tracks epistemic uncertainty                              │   │
│  │  • Tracks per-detector reliability                           │   │
│  │  • Detects belief drift over time                            │   │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │  LangGraph State passed forward   │
│                                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  VOI ROUTER  [LangGraph Node]                │   │
│  │                              [Conditional Edge Logic]        │   │
│  │                                                              │   │
│  │  • Computes expected loss under current belief               │   │
│  │  • Estimates VOI for each available detector                 │   │
│  │  • Routes to next detector only if VOI > invocation cost     │   │
│  │  • Filters by detector availability and reliability          │   │
│  │                                                              │   │
│  │  Conditional edges:                                          │   │
│  │    VOI positive  → invoke next detector                      │   │
│  │    VOI negative  → proceed to Decision Node                  │   │
│  └──────────┬────────────────────────────────────┬─────────────┘   │
│             │                                    │                  │
│             ▼                                    ▼                  │
│  ┌──────────────────────────┐     ┌─────────────────────────────┐  │
│  │  DETECTOR POOL           │     │  AGENT COMMUNICATION BUS    │  │
│  │  [LangGraph Tool Nodes]  │     │  [A2A Protocol]             │  │
│  │                          │     │                             │  │
│  │  Agent A                 │     │  • Capability advertisement │  │
│  │  Network traffic         │     │  • Evidence sharing         │  │
│  │  Lightweight classifier  │◄───►│  • Consensus protocol       │  │
│  │                          │     │  • Uncertainty negotiation  │  │
│  │  Agent B                 │     │  • Drift alerts             │  │
│  │  Network traffic         │     │                             │  │
│  │  Deep / uncertain        │     │  Each agent exposes an A2A  │  │
│  │                          │     │  endpoint. The bus mediates │  │
│  │  Agent C                 │     │  inter-agent communication  │  │
│  │  Temporal / spatial      │     │  independently of the main  │  │
│  │  network context         │     │  orchestration graph.       │  │
│  │                          │     └─────────────────────────────┘  │
│  │  Agent D                 │                                       │
│  │  System log reasoning    │                                       │
│  │  [LangChain pipeline]    │                                       │
│  └──────────┬───────────────┘                                       │
│             │  Calibrated likelihoods returned to graph state       │
│             ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        OBSERVATION MODEL CALIBRATOR  [LangGraph Node]        │   │
│  │                                                              │   │
│  │  • Calibrates raw detector outputs into likelihoods          │   │
│  │  • Tracks and updates per-detector reliability               │   │
│  │  • Detects and responds to model/data drift                  │   │
│  │  • Maintains experience buffer for continual recalibration   │   │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DECISION NODE  [LangGraph Node]                 │   │
│  │                             [Conditional Edge Logic]         │   │
│  │                                                              │   │
│  │  Three-way classifier:                                       │   │
│  │    Accept  — belief below low threshold, low uncertainty     │   │
│  │    Reject  — belief above high threshold, low uncertainty    │   │
│  │    Defer   — high uncertainty OR belief in ambiguous zone    │   │
│  │                                                              │   │
│  │  Adaptive thresholds:                                        │   │
│  │    • Cost-sensitive  (C_FN >> C_FP → lower alert threshold)  │   │
│  │    • Workload-aware  (high queue → raise deferral threshold) │   │
│  │    • Cost parameters maintained as updatable distributions   │   │
│  │                                                              │   │
│  │  Conditional edges:                                          │   │
│  │    Accept / Reject → Actions & Observability layer           │   │
│  │    Defer           → HITL Deferral Handler                   │   │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼  (Defer path only)               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           HITL DEFERRAL HANDLER  [LangGraph Node]            │   │
│  │                                [Human-in-the-Loop interrupt] │   │
│  │                                                              │   │
│  │  • Priority-ordered analyst queue                            │   │
│  │  • Packages belief state, evidence trail, and reasoning      │   │
│  │  • Graph execution pauses; resumes on analyst response       │   │
│  │  • Forwards analyst response to Feedback Integrator          │   │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                   │
│                                 ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            FEEDBACK INTEGRATOR  [LangGraph Node]             │   │
│  │                                                              │   │
│  │  Receives analyst response and routes three signal types:    │   │
│  │                                                              │   │
│  │  Label signal    → Observation Model Calibrator              │   │
│  │                     (updates per-detector reliability)       │   │
│  │                                                              │   │
│  │  Override signal → Decision Node cost distributions          │   │
│  │                     (recalibrates C_FN / C_FP priors)        │   │
│  │                                                              │   │
│  │  Queue signal    → Decision Node deferral threshold          │   │
│  │                     (workload-aware η adjustment)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ACTIONS & OBSERVABILITY                         │
│                                                                     │
│   Actions:   Accept (pass flow) │ Reject (block/alert) │ Defer      │
│                                                                     │
│   Metrics:                                                          │
│     Detection      — precision, recall, F1, AUC, calibration error │
│     Efficiency     — cost per flow, latency, detector utilization   │
│     Uncertainty    — expected calibration error, Brier score        │
│     Collaboration  — deferral rate, analyst accuracy, queue depth   │
│     Drift          — belief divergence, recalibration frequency     │
│     A2A            — consensus rate, evidence sharing volume        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Framework Roles

**LangGraph** is an optional runtime for orchestration parity and inspection. The deterministic runtime is production default for performance. Both engines execute the same policy math and share config/state/A2A contracts; parity and performance guardrails determine whether LangGraph can be promoted.

**A2A Protocol** governs lateral communication between detector agents, independently of the main orchestration graph. Each agent advertises its capabilities (input modality, cost, uncertainty type) and exposes an endpoint through which it can share intermediate evidence, flag anomalies, or participate in consensus on ambiguous flows. The communication bus mediates this without routing through the orchestrator, keeping inter-agent coordination decoupled from the main belief update cycle.

**LangChain** is scoped to Agent D's internal reasoning pipeline. Because Agent D reasons over unstructured system log data, it requires a structured prompting and retrieval pipeline that the other agents (which run fixed model inference) do not. LangChain manages that chain internally; from the orchestrator's perspective, Agent D remains a black box that returns a calibrated likelihood like any other agent.

---

## Key Architectural Properties

**State continuity** — The shared state backend maintains belief and reliability continuity independent of engine choice. Every belief update, routing decision, and calibration change is persisted with reproducible replay traces.

**Decoupled coordination** — A2A communication between agents does not pass through the orchestrator. This means agents can share evidence asynchronously without blocking the main graph execution or coupling agent internals to the orchestration logic.

**Auditable human integration** — Deferral and action decisions are explicit (`decision` vs `action_decision`) and logged per flow, so predictive quality and utility costs are auditable independently.

**Feedback as graph re-entry** — Analyst signals processed by the Feedback Integrator update the shared graph state (reliability estimates, cost distributions, deferral thresholds), meaning subsequent flows through the graph automatically reflect accumulated human feedback without requiring explicit retraining cycles.
