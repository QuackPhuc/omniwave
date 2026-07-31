"""Quality metrics: Top-k accuracy and expected calibration error."""

from __future__ import annotations

import torch
from torch import Tensor


def topk_accuracy(
    logits: Tensor,
    targets: Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute Top-k accuracy for each k in *topk*.

    Returns dict mapping ``"top1"``, ``"top5"``, etc. to accuracy in [0, 1].
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if logits.size(1) < 1:
        raise ValueError("logits must contain at least one class")

    maxk = min(max(topk), logits.size(1))
    batch_size = targets.size(0)
    if batch_size == 0:
        return {f"top{k}": 0.0 for k in topk}

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))

    result: dict[str, float] = {}
    for k in topk:
        effective_k = min(k, logits.size(1))
        correct_k = correct[:, :effective_k].reshape(-1).float().sum().item()
        result[f"top{k}"] = correct_k / batch_size
    return result


def expected_calibration_error(
    logits: Tensor,
    targets: Tensor,
    n_bins: int = 15,
) -> float:
    """Compute the expected calibration error (ECE).

    Uses uniform confidence bins.  Returns ECE in [0, 1].
    """
    probs = torch.softmax(logits.float(), dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = 0.0
    n_total = targets.size(0)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        n_in_bin = in_bin.sum().item()
        if n_in_bin == 0:
            continue
        avg_confidence = confidences[in_bin].mean().item()
        avg_accuracy = accuracies[in_bin].mean().item()
        ece += (n_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece
