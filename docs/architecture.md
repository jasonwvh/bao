# Bayesian Agent Orchestrator (BAO) — System Architecture v2

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

**LangGraph** governs the entire orchestration layer as a stateful directed graph. Each major component is a node holding a slice of the shared graph state (current belief, evidence collected, cost parameters, uncertainty estimates). Routing between components is implemented as conditional edges — for example, the VOI Router branches to a detector or directly to the Decision Node, and the Decision Node branches to Actions or HITL depending on the outcome. The graph execution naturally pauses at the HITL node and resumes when analyst input arrives, which is a native LangGraph capability. This makes the full decision trace — every node visited, every state transition — inspectable and auditable.

**A2A Protocol** governs lateral communication between detector agents, independently of the main orchestration graph. Each agent advertises its capabilities (input modality, cost, uncertainty type) and exposes an endpoint through which it can share intermediate evidence, flag anomalies, or participate in consensus on ambiguous flows. The communication bus mediates this without routing through the orchestrator, keeping inter-agent coordination decoupled from the main belief update cycle.

**LangChain** is scoped to Agent D's internal reasoning pipeline. Because Agent D reasons over unstructured system log data, it requires a structured prompting and retrieval pipeline that the other agents (which run fixed model inference) do not. LangChain manages that chain internally; from the orchestrator's perspective, Agent D remains a black box that returns a calibrated likelihood like any other agent.

---

## Key Architectural Properties

**State continuity** — LangGraph maintains a single shared state object across all nodes. Every belief update, VOI computation, and calibration change is recorded in this state, giving full decision provenance without additional logging infrastructure.

**Decoupled coordination** — A2A communication between agents does not pass through the orchestrator. This means agents can share evidence asynchronously without blocking the main graph execution or coupling agent internals to the orchestration logic.

**Auditable human integration** — The HITL node is a formal interrupt in the LangGraph execution flow, not a side channel. The graph cannot proceed past deferral without an analyst response, making human oversight architecturally enforced rather than optional.

**Feedback as graph re-entry** — Analyst signals processed by the Feedback Integrator update the shared graph state (reliability estimates, cost distributions, deferral thresholds), meaning subsequent flows through the graph automatically reflect accumulated human feedback without requiring explicit retraining cycles.