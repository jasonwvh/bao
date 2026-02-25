---
marp: true
theme: default
paginate: true
---

# Introduction

- Modern NIDS rely on single models or static ensembles  
- Struggle with zero-day attacks, concept drift, and class imbalance  
- High false-positive rates → alert fatigue in SOCs  
- Multiple heterogeneous detectors exist but lack rational coordination  
- Proposal: **Bayesian Agent Orchestrator (BAO)**  
  - Maintains belief over threat state  
  - Uses uncertainty + Value of Information (VoI)  
  - Coordinates ML agents and human analysts  

---

# Problem Statements

- **P1: Orchestration Problem**  
  - Existing NIDS deploy multiple models but without a unified decision-making layer
  - No central belief state over network threat status

- **P2: Value-of-Information Problem**  
  - No cost-aware mechanism to decide when to invoke expensive agents  
  - No sequential belief update for evidence gathering  

- **P3: Human Collaboration Problem**  
  - Alert fatigue and automation bias reduce trust  
  - No formal rule for when to defer to analysts  

---

# Research Questions

- **RQ1:**  
  - How can Bayesian inference be used to orchestrate multiple NIDS agents under uncertainty?  

- **RQ2:**  
  - How can Value of Information guide sequential activation of cheap vs expensive detectors?  

- **RQ3:**  
  - How can uncertainty thresholds determine optimal human-in-the-loop deferral?  

---

# Research Objectives

- **O1: Develop Bayesian Control Layer**  
  - Maintain posterior belief over latent threat state  
  - Treat agents as noisy observation sources  

- **O2: Implement VoI-Based Decision Policy**  
  - Compare expected utility of acting vs gathering more evidence  
  - Minimise total operational cost  

- **O3: Integrate Human Collaboration Mechanism**  
  - Route high-uncertainty cases to analysts  
  - Reduce false positives and alert fatigue  

---

# Literature Review

- **Agent Orchestration**  
  - Multi-agent RL and optimisation improve accuracy (>98–99%)  
  - Lack probabilistic coordination and cost-awareness  

- **Bayesian & Value of Information**  
  - Uncertainty-aware models estimate uncertainty and defer low-confidence cases
  - VoI models quantify the benefit of investigating additional alerts before acting

---

# Literature Review

- **Human–AI Collaboration**  
  - Entropy-based deferral improves rare attack detection  
  - Automation bias and trust issues persist  

- **Gap Identified**  
  - No unified framework combining:  
    - Multi-agent orchestration  
    - Bayesian belief updating  
    - VoI-based routing  
    - Human-in-the-loop decision control  