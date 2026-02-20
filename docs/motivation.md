# Bayesian Agent Orchestrator for Network Intrusion Detection Systems (NIDS)

## Abstract

Modern network intrusion detection relies on multiple heterogeneous detectors (machine-learning classifiers, rule-based engines, anomaly detectors) to flag malicious traffic. Rather than naively aggregating their outputs, this proposal envisions a Bayesian Agent Orchestrator (BAO) that fuses detector alerts probabilistically and makes cost-sensitive decisions under uncertainty. The orchestrator maintains a posterior belief over the hidden threat state (malicious or benign) and selects actions (raise an alert, declare benign, defer to an analyst, or invoke another detector) to minimize expected cost. It uses a value-of-information criterion to decide whether invoking an additional detector is worth its cost. We will show that this sequential decision framework achieves lower total expected cost and better analyst workload efficiency than fixed voting or cascade-based ensembles.

## Research Background

Network intrusion detection systems (IDS) commonly combine multiple sensors or models to improve detection. Ensemble methods—training many classifiers and combining their outputs—are known to boost accuracy over any single detector[[1]](https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/what_is_the_latest_type_of_hybrid_metaheuristic_might_work_well_for_intrusion_detection/attachment/59d6523d79197b80779aa869/AS%3A511259284918272%401498905120773/download/A+survey+of+intrusion+detection+systems+based+on+ensemble+and+hybrid+classifiers+Aburomman2016.pdf#:~:text=Training%20many%20classifiers%20at%20the,ensemble%2C%20also%20known%20as%20a). In practice these detectors include fast signature or heuristic filters as well as slower anomaly and deep-learning models. However, because benign traffic vastly outnumbers attacks, modern IDS suffer from very high false-positive rates[[2]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=has%20widely%20known%20high%20false,effect%20us%02ing%20an%20idea%20borrowed). Administrators can become desensitized by this “cry wolf” effect[[2]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=has%20widely%20known%20high%20false,effect%20us%02ing%20an%20idea%20borrowed), undermining security. Furthermore, most ML/DL detectors produce poorly calibrated confidence scores[[3]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=time%20IDS%20applications,and%20hamper%20effective%20decision%20making), so fixed-threshold rules do not reliably reflect true uncertainty. As a result, static fusion rules or voting ensembles often either trigger too many false alarms or miss rare threats.

Traditional IDS design also neglects cost-awareness. Classic work on cost-sensitive IDS models this explicitly[[4]](https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf#:~:text=detection%20models,that%20can%20produce%20detection%20models): detectors should be optimized not merely for accuracy but to minimize the overall expected loss (damage from intrusions, wasted effort on false alarms, and analyst time). Existing multi-detector systems lack this principled approach: they simply maximize hit rates or use hand-tuned thresholds without formally trading off false negatives versus false positives. For example, fixed cascades will always run the same sequence of checks regardless of how close a case is to the decision boundary.

Recent research has begun to address uncertainty and human involvement. For example, uncertainty-aware IDS architectures route high-entropy (ambiguous) alerts to analysts for review[[5]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=,Attack%E2%80%93XSS%20%2B25%20pp%2C%20and%20DoS). Active perception frameworks for cyber defense suggest controlling sensors to optimize information gain[[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows). In that spirit, one can imagine treating cheap, noisy detectors as “pre-screeners” and invoking expensive, accurate tools only when needed[[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows). However, there is no unified system that (1) treats all detector outputs as noisy evidence, (2) updates a belief state via Bayes’ rule, and (3) chooses among actions (alert, stop, defer, or acquire more evidence) based on expected cost. This gap motivates our Bayesian orchestrator approach.

## Research Problems

1. Uncertainty modeling in IDS ensembles: Current IDS pipelines lack a formal belief state. They do not properly calibrate or combine detector scores as likelihoods, so they ignore the actual probability of threat implied by all evidence[[3]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=time%20IDS%20applications,and%20hamper%20effective%20decision%20making). This prevents the system from reasoning about confidence or when additional data might change the decision.
    
2. Cost-sensitive decision-making: Existing ensemble and cascade methods maximize detection rates, but overlook asymmetric costs[[4]](https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf#:~:text=detection%20models,that%20can%20produce%20detection%20models). In security, the loss from a missed attack can far exceed the cost of a false alarm. With fixed thresholds, IDS cannot minimize total expected cost across false positives, false negatives, analyst reviews, and detector usage.
    
3. Inefficient evidence gathering and deferral: Without a VOI-based policy, IDS typically run all configured detectors or follow a rigid sequence, wasting computation. There is no principled stopping rule (“is another test worth it?”) in most systems[[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows)[[7]](https://arxiv.org/html/2601.01522v1#:~:text=decision%20theory%20provides%20the%20answer,that%20standard%20LLM%20architectures%20lack). Similarly, decisions to hand off to a human analyst are ad hoc, leading either to alert overload or delayed threat responses. In short, modern IDS lack a mechanism to dynamically allocate sensing and human review where it most reduces risk.
    

## Research Questions

1. Cost Reduction: Can a Bayesian orchestrator reduce the total expected cost of intrusion detection (balancing missed attack costs, false alarm costs, and analyst time) compared to fixed ensemble and threshold-based approaches?
    
2. Efficiency and VOI: Does using value-of-information–driven detector invocation yield lower computation and fewer unnecessary alerts than static methods? In other words, can adaptive evidence gathering reduce overhead?
    
3. Analyst Integration under Imbalance: Can uncertainty-based deferral to human analysts improve overall detection efficiency and robustness, especially under realistic class imbalance and asymmetric cost regimes?
    

## Research Objectives

1. Bayesian fusion model: Develop a probabilistic framework for combining heterogeneous detector outputs. Calibrate each detector’s output $o_i$ into likelihood functions $p(o_i \mid Z)$ for the binary threat state $Z\in{0,1}$ (benign, malicious). Under a conditional-independence approximation, derive the posterior probability $p(Z=1 \mid o_{1:k})$ via Bayes’ rule.
    
2. Cost-sensitive decision policy: Formulate an action-selection rule that minimizes expected cost given the current belief. Encode cost parameters (miss cost $C_{FN}$, false alarm cost $C_{FP}$, analyst review cost $C_H$, detector cost $C_{D_i}$). Implement approximate value-of-information computations: for each unused detector $d_j$, estimate the expected reduction in risk from observing its output versus its invocation cost. Trigger an additional detector only when its VOI is positive[[7]](https://arxiv.org/html/2601.01522v1#:~:text=decision%20theory%20provides%20the%20answer,that%20standard%20LLM%20architectures%20lack).
    
3. Prototype implementation and evaluation: Build the BAO on top of heterogeneous IDS components, using benchmark network datasets (e.g. CIC-IDS2017[[8]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=increases%20the%20number%20of%20false,accuracy%20when%20trained%20and%20tested), UNSW-NB15[[8]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=increases%20the%20number%20of%20false,accuracy%20when%20trained%20and%20tested)). Calibrate detector scores, fit the likelihood models, and simulate the sequential decision loop. Compare BAO against baselines (majority vote, average-probability ensemble, fixed cascade, and single best detector) using metrics such as total expected cost, false-positive/false-negative rates, computational usage, and analyst deferral rate.
    

## Scope

- We assume detector outputs can be calibrated into likelihoods (e.g. via Platt scaling). The work focuses on orchestration; perfect calibration is assumed for modeling, although real outputs may be miscalibrated[[3]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=time%20IDS%20applications,and%20hamper%20effective%20decision%20making).
    
- We restrict to binary threat detection (malicious vs. benign) for tractability. Extensions to multi-class attack types are beyond this master’s scope.
    
- Costs ($C_{FN}, C_{FP}, C_H, C_{D_i}$) will be specified via simulation or sensitivity analysis, not derived from real SOC data. This abstracts realistic security priorities (e.g. false negatives ≫ false positives).
    
- The orchestrator treats detectors as conditionally independent evidence sources. In practice detectors may correlate; this simplification may affect real-world performance but is adopted here for feasibility.
    
- Adversarial adaptation (attackers learning about the orchestrator) is not modeled. This study emphasizes feasibility and empirical gains under fixed assumptions.
    

## Contributions

- System-level Bayesian orchestration: We design and implement a novel controller that sits above multiple IDS detectors, maintaining a Bayesian belief over the threat and making unified decisions. This differs from prior work that focuses on individual classifiers or fixed ensembles.
    
- VOI-driven detector selection: We develop a practical approximation of value-of-information for intrusion detection, enabling dynamic invocation of expensive detectors only when they are expected to reduce risk enough. This sequential sensing strategy is inspired by active perception[[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows) but tailored to off-the-shelf IDS models.
    
- Cost-sensitive deferral mechanism: We incorporate deferral to human analysts into the Bayesian decision loop. The orchestrator explicitly “chooses” to escalate ambiguous cases based on posterior uncertainty and cost parameters, aiming to reduce analyst workload and detection delay.
    
- Empirical validation: Through experiments on modern IDS datasets, we will demonstrate that BAO achieves lower total expected cost and more efficient resource usage than conventional ensembles. This will show the value of principled Bayesian decision-making (posterior updating, VOI, cost-sensitive actions) at the system level, rather than ad-hoc fusion of alerts.
    

Sources: Established IDS datasets and literature inform this proposal (e.g. ensemble methods[[1]](https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/what_is_the_latest_type_of_hybrid_metaheuristic_might_work_well_for_intrusion_detection/attachment/59d6523d79197b80779aa869/AS%3A511259284918272%401498905120773/download/A+survey+of+intrusion+detection+systems+based+on+ensemble+and+hybrid+classifiers+Aburomman2016.pdf#:~:text=Training%20many%20classifiers%20at%20the,ensemble%2C%20also%20known%20as%20a), cost-sensitive IDS design[[4]](https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf#:~:text=detection%20models,that%20can%20produce%20detection%20models), active perception ideas[[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows), and uncertainty-aware detection[[3]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=time%20IDS%20applications,and%20hamper%20effective%20decision%20making)). The BAO approach integrates these concepts into a cohesive thesis.

---

[[1]](https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/what_is_the_latest_type_of_hybrid_metaheuristic_might_work_well_for_intrusion_detection/attachment/59d6523d79197b80779aa869/AS%3A511259284918272%401498905120773/download/A+survey+of+intrusion+detection+systems+based+on+ensemble+and+hybrid+classifiers+Aburomman2016.pdf#:~:text=Training%20many%20classifiers%20at%20the,ensemble%2C%20also%20known%20as%20a) A survey of intrusion detection systems based on ensemble and hybrid classifiers

[https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/what_is_the_latest_type_of_hybrid_metaheuristic_might_work_well_for_intrusion_detection/attachment/59d6523d79197b80779aa869/AS%3A511259284918272%401498905120773/download/A+survey+of+intrusion+detection+systems+based+on+ensemble+and+hybrid+classifiers+Aburomman2016.pdf](https://www.researchgate.net/profile/Mohamed_Mourad_Lafifi/post/what_is_the_latest_type_of_hybrid_metaheuristic_might_work_well_for_intrusion_detection/attachment/59d6523d79197b80779aa869/AS%3A511259284918272%401498905120773/download/A+survey+of+intrusion+detection+systems+based+on+ensemble+and+hybrid+classifiers+Aburomman2016.pdf)

[[2]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=has%20widely%20known%20high%20false,effect%20us%02ing%20an%20idea%20borrowed) [[6]](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf#:~:text=in%20neuroscience,This%20allows) Active Perception for Cyber Intrusion Detection and Defense

[https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf](https://cdn.aaai.org/ocs/ws/ws0206/12621-57420-1-PB.pdf)

[[3]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=time%20IDS%20applications,and%20hamper%20effective%20decision%20making) [[5]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=,Attack%E2%80%93XSS%20%2B25%20pp%2C%20and%20DoS) [[8]](https://www.mdpi.com/2313-576X/11/4/120#:~:text=increases%20the%20number%20of%20false,accuracy%20when%20trained%20and%20tested) Uncertainty-Aware Adaptive Intrusion Detection Using Hybrid CNN-LSTM with cWGAN-GP Augmentation and Human-in-the-Loop Feedback

[https://www.mdpi.com/2313-576X/11/4/120](https://www.mdpi.com/2313-576X/11/4/120)

[[4]](https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf#:~:text=detection%20models,that%20can%20produce%20detection%20models) acm_ccs.dvi

[https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf](https://ids.cs.columbia.edu/sites/default/files/wenke-acmccs2k-cost.pdf)

[[7]](https://arxiv.org/html/2601.01522v1#:~:text=decision%20theory%20provides%20the%20answer,that%20standard%20LLM%20architectures%20lack) Bayesian Orchestration of Multi-LLM Agents for Cost-Aware Sequential Decision-Making

[https://arxiv.org/html/2601.01522v1](https://arxiv.org/html/2601.01522v1)