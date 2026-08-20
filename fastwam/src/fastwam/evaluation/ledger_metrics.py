from __future__ import annotations

import torch


def _as_float_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach()


def failure_localization_accuracy(debt, failed_claim, mask=None) -> float:
    """Top-1 accuracy of locating the causal claim responsible for failure."""

    debt = _as_float_tensor(debt)
    failed_claim = torch.as_tensor(failed_claim, dtype=torch.long)
    if debt.ndim != 2 or failed_claim.shape != debt.shape[:1]:
        raise ValueError("Expected debt [N,C] and failed_claim [N].")
    valid = torch.ones_like(failed_claim, dtype=torch.bool)
    if mask is not None:
        valid &= torch.as_tensor(mask, dtype=torch.bool)
    if not bool(valid.any()):
        return 0.0
    return float((debt.argmax(dim=-1)[valid] == failed_claim[valid]).float().mean())


def rollback_accuracy(rollback_logits, rollback_target, mask=None) -> float:
    """Accuracy of the predicted local rollback checkpoint."""

    logits = _as_float_tensor(rollback_logits)
    target = torch.as_tensor(rollback_target, dtype=torch.long)
    if logits.ndim != 2 or target.shape != logits.shape[:1]:
        raise ValueError("Expected rollback_logits [N,R] and rollback_target [N].")
    valid = torch.ones_like(target, dtype=torch.bool)
    if mask is not None:
        valid &= torch.as_tensor(mask, dtype=torch.bool)
    if not bool(valid.any()):
        return 0.0
    return float((logits.argmax(dim=-1)[valid] == target[valid]).float().mean())


def debt_calibration_error(predicted_debt, observed_failure, bins: int = 10) -> float:
    """Expected calibration error between causal debt and observed failures."""

    debt = _as_float_tensor(predicted_debt).flatten().clamp(0.0, 1.0)
    failure = _as_float_tensor(observed_failure).flatten().clamp(0.0, 1.0)
    if debt.shape != failure.shape:
        raise ValueError("Predicted debt and observed failure shapes must match.")
    if bins <= 0:
        raise ValueError("`bins` must be positive.")
    if debt.numel() == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, bins + 1)
    error = torch.zeros(())
    for index in range(bins):
        if index == bins - 1:
            selected = (debt >= edges[index]) & (debt <= edges[index + 1])
        else:
            selected = (debt >= edges[index]) & (debt < edges[index + 1])
        if bool(selected.any()):
            weight = selected.float().mean()
            error = error + weight * (debt[selected].mean() - failure[selected].mean()).abs()
    return float(error)


def repair_efficiency(debt_before, debt_after, action_cost) -> float:
    """Mean causal-debt reduction per unit repair cost."""

    before = _as_float_tensor(debt_before)
    after = _as_float_tensor(debt_after)
    cost = _as_float_tensor(action_cost)
    if before.shape != after.shape or before.shape != cost.shape:
        raise ValueError("Debt-before, debt-after, and action-cost shapes must match.")
    if before.numel() == 0:
        return 0.0
    return float(((before - after) / cost.clamp(min=1e-6)).mean())


def world_prediction_mae(predicted_next_debt, observed_next_debt, mask=None) -> float:
    predicted = _as_float_tensor(predicted_next_debt)
    observed = _as_float_tensor(observed_next_debt)
    if predicted.shape != observed.shape:
        raise ValueError("Predicted and observed next-debt shapes must match.")
    error = (predicted - observed).abs()
    if mask is not None:
        selected = torch.as_tensor(mask, dtype=torch.bool)
        if selected.shape != error.shape:
            raise ValueError("World-prediction mask shape must match debt.")
        error = error[selected]
    return float(error.mean()) if error.numel() else 0.0


def dependency_accuracy(predicted_dependency, observed_dependency, threshold: float = 0.5) -> float:
    predicted = _as_float_tensor(predicted_dependency)
    observed = _as_float_tensor(observed_dependency)
    if predicted.shape != observed.shape:
        raise ValueError("Predicted and observed dependency shapes must match.")
    if predicted.numel() == 0:
        return 0.0
    return float(((predicted >= threshold) == (observed >= threshold)).float().mean())


def object_identity_consistency(slot_assignment) -> float:
    assignment = _as_float_tensor(slot_assignment)
    if assignment.ndim != 3 or assignment.shape[-1] != assignment.shape[-2]:
        raise ValueError("Slot assignment must be [N,S,S].")
    if assignment.numel() == 0:
        return 0.0
    # A persistent slot should have one decisive predecessor even when identities
    # legitimately permute, hence row maximum rather than diagonal mass.
    return float(assignment.max(dim=-1).values.mean())


def recovery_success_rate(repair_triggered, recovered) -> float:
    triggered = torch.as_tensor(repair_triggered, dtype=torch.bool)
    recovered = torch.as_tensor(recovered, dtype=torch.bool)
    if triggered.shape != recovered.shape:
        raise ValueError("Repair and recovered arrays must match.")
    if not bool(triggered.any()):
        return 0.0
    return float(recovered[triggered].float().mean())


def unnecessary_repair_rate(repair_triggered, failure_without_repair) -> float:
    triggered = torch.as_tensor(repair_triggered, dtype=torch.bool)
    needed = torch.as_tensor(failure_without_repair, dtype=torch.bool)
    if triggered.shape != needed.shape:
        raise ValueError("Repair and counterfactual-need arrays must match.")
    if not bool(triggered.any()):
        return 0.0
    return float((~needed[triggered]).float().mean())


def mean_recovery_steps(repair_triggered, recovery_steps) -> float:
    triggered = torch.as_tensor(repair_triggered, dtype=torch.bool)
    steps = _as_float_tensor(recovery_steps)
    if triggered.shape != steps.shape:
        raise ValueError("Repair and recovery-step arrays must match.")
    valid = triggered & torch.isfinite(steps)
    return float(steps[valid].mean()) if bool(valid.any()) else 0.0
