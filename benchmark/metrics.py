from __future__ import annotations

from typing import Dict, List, Optional


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    probabilities: Optional[List[float]] = None,
    costs: Optional[List[float]] = None,
    approach: str = "unknown",
) -> Dict[str, float | int | str]:
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    auc = 0.5
    if probabilities is not None and len(probabilities) == len(labels):
        try:
            auc = _compute_auc(labels, probabilities)
        except Exception:
            auc = 0.5

    total_cost = sum(costs) if costs else 0.0
    avg_cost = total_cost / max(len(costs), 1) if costs else 0.0

    return {
        "approach": approach,
        "flows_processed": total,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "auc": round(auc, 6),
        "total_cost": round(total_cost, 4),
        "avg_cost_per_flow": round(avg_cost, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _compute_auc(labels: List[int], probabilities: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probabilities))
    except ImportError:
        pass
    except Exception:
        pass

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = list(zip(probabilities, labels))
    pairs.sort(key=lambda x: x[0], reverse=True)

    concordant = 0
    discordant = 0
    ties = 0

    for i, (p_i, l_i) in enumerate(pairs):
        for j, (p_j, l_j) in enumerate(pairs[i + 1 :], start=i + 1):
            if l_i == l_j:
                continue
            if p_i > p_j:
                concordant += 1
            elif p_i < p_j:
                discordant += 1
            else:
                ties += 1

    total_pairs = concordant + discordant + ties
    if total_pairs == 0:
        return 0.5

    return (concordant + 0.5 * ties) / total_pairs
