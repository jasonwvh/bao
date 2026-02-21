from __future__ import annotations

from typing import Any, Dict, List


def weighted_consensus(agent_outputs: List[Dict[str, Any]], reliability_lookup: Dict[str, float]) -> Dict[str, Any]:
    if not agent_outputs:
        return {"probability": 0.5, "agreement": 1.0, "participants": []}

    weights = []
    probs = []
    participants = []
    for out in agent_outputs:
        aid = str(out.get("agent_id", "unknown"))
        p = float(out.get("proba", [0.5, 0.5])[1])
        unc = float(out.get("uncertainty", {}).get("epistemic", 0.5))
        rel = float(reliability_lookup.get(aid, 0.5))

        w = rel / (1.0 + unc)
        weights.append(w)
        probs.append(p)
        participants.append(aid)

    total_w = sum(weights) or 1.0
    prob = sum(w * p for w, p in zip(weights, probs)) / total_w
    agreement = 1.0 - min(1.0, (max(probs) - min(probs)) if len(probs) > 1 else 0.0)
    return {"probability": prob, "agreement": agreement, "participants": participants}
