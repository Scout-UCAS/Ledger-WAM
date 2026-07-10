"""Evaluation metrics for Ledger-WAM.

The functions in this module are deliberately stateless and depend only on
PyTorch.  Inputs may be tensors or values accepted by :func:`torch.as_tensor`.
Every public metric function returns a JSON-serializable dictionary containing
plain Python ``float`` and ``int`` values.

Masks use ``True`` for a valid observation.  Values outside a mask are ignored
before range and finiteness checks, which makes it safe to keep sentinel or NaN
values in padded locations.  Metrics with no valid observations return zero
values and a count of zero.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch


Tensor = torch.Tensor
TensorLike = object
MetricValue = Union[int, float]
MetricDict = Dict[str, MetricValue]


def _as_tensor(
    value: TensorLike,
    name: str,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Convert an input without changing an existing tensor's device."""

    try:
        if isinstance(value, Tensor):
            tensor = value.detach()
            if device is not None and tensor.device != device:
                tensor = tensor.to(device=device)
            return tensor
        return torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("%s must be tensor-like" % name) from exc


def _broadcast_mask(mask: Optional[TensorLike], reference: Tensor, name: str) -> Tensor:
    """Return a boolean mask broadcast to ``reference.shape``.

    In addition to regular broadcasting, a prefix mask such as ``[B]`` is
    accepted for a reference shaped ``[B, N]`` and expanded over trailing
    dimensions.  Prefix masks are convenient for episode-level filtering.
    """

    if mask is None:
        return torch.ones(reference.shape, dtype=torch.bool, device=reference.device)

    value = _as_tensor(mask, name, reference.device).to(dtype=torch.bool)
    if tuple(value.shape) == tuple(reference.shape):
        return value

    try:
        return torch.broadcast_to(value, reference.shape)
    except RuntimeError:
        pass

    if value.ndim <= reference.ndim and tuple(value.shape) == tuple(
        reference.shape[: value.ndim]
    ):
        expanded_shape = tuple(value.shape) + (1,) * (reference.ndim - value.ndim)
        return value.reshape(expanded_shape).expand(reference.shape)

    raise ValueError(
        "%s shape %s is not compatible with data shape %s"
        % (name, tuple(value.shape), tuple(reference.shape))
    )


def _mask_for_shape(
    mask: TensorLike,
    shape: Tuple[int, ...],
    device: torch.device,
    name: str,
) -> Optional[Tensor]:
    """Try to broadcast ``mask`` to ``shape`` and return ``None`` on failure."""

    reference = torch.empty(shape, dtype=torch.bool, device=device)
    try:
        return _broadcast_mask(mask, reference, name)
    except ValueError:
        return None


def _require_same_shape(first: Tensor, second: Tensor, names: str) -> None:
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(
            "%s must have the same shape, got %s and %s"
            % (names, tuple(first.shape), tuple(second.shape))
        )


def _require_finite(values: Tensor, name: str) -> None:
    if values.numel() and not bool(torch.isfinite(values).all()):
        raise ValueError("valid %s values must be finite" % name)


def _require_binary(values: Tensor, name: str) -> None:
    _require_finite(values, name)
    if values.numel() and not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("valid %s values must be binary (0 or 1)" % name)


def _require_probabilities(values: Tensor, name: str) -> None:
    _require_finite(values, name)
    if values.numel() and not bool(((values >= 0) & (values <= 1)).all()):
        raise ValueError("valid %s values must be in [0, 1]" % name)


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("%s must be in [0, 1]" % name)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _selected_pair(
    predictions: TensorLike,
    targets: TensorLike,
    mask: Optional[TensorLike],
    prediction_name: str,
    target_name: str,
) -> Tuple[Tensor, Tensor]:
    prediction_tensor = _as_tensor(predictions, prediction_name)
    target_tensor = _as_tensor(targets, target_name, prediction_tensor.device)
    _require_same_shape(
        prediction_tensor,
        target_tensor,
        "%s and %s" % (prediction_name, target_name),
    )
    valid = _broadcast_mask(mask, target_tensor, "mask")
    return prediction_tensor[valid], target_tensor[valid]


def claim_root_cause_metrics(
    predictions: TensorLike,
    targets: TensorLike,
    mask: Optional[TensorLike] = None,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> MetricDict:
    """Compute micro precision, recall, and F1 for root-cause claims.

    ``targets`` contains a binary label per candidate causal claim.  Predictions
    are probabilities unless ``from_logits=True``.  All dimensions are flattened
    after masking, so the result is a micro average across episodes and slots.
    A zero denominator maps to zero rather than NaN.
    """

    _require_unit_interval(float(threshold), "threshold")
    predicted, target = _selected_pair(
        predictions,
        targets,
        mask,
        "claim predictions",
        "claim targets",
    )
    target = target.to(dtype=torch.float64)
    _require_binary(target, "claim targets")

    predicted = predicted.to(dtype=torch.float64)
    _require_finite(predicted, "claim predictions")
    if from_logits:
        predicted = torch.sigmoid(predicted)
    else:
        _require_probabilities(predicted, "claim predictions")

    predicted_positive = predicted >= float(threshold)
    actual_positive = target == 1
    true_positive_count = int((predicted_positive & actual_positive).sum().item())
    predicted_positive_count = int(predicted_positive.sum().item())
    actual_positive_count = int(actual_positive.sum().item())
    count = int(target.numel())

    precision = _safe_ratio(true_positive_count, predicted_positive_count)
    recall = _safe_ratio(true_positive_count, actual_positive_count)
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "claim_root_cause_precision": precision,
        "claim_root_cause_recall": recall,
        "claim_root_cause_f1": f1,
        "claim_root_cause_true_positives": true_positive_count,
        "claim_root_cause_predicted_positives": predicted_positive_count,
        "claim_root_cause_actual_positives": actual_positive_count,
        "claim_root_cause_count": count,
    }


def _normalize_ks(k: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    values: Sequence[int]
    if isinstance(k, bool):
        raise ValueError("k must contain positive integers")
    if isinstance(k, int):
        values = (k,)
    else:
        if isinstance(k, (str, bytes)):
            raise ValueError("k must contain positive integers")
        values = tuple(k)
    if not values:
        raise ValueError("k must not be empty")

    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("k must contain positive integers")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def top_k_accuracy(
    scores: TensorLike,
    targets: TensorLike,
    k: Union[int, Sequence[int]] = (1, 3, 5),
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Compute root-cause localization accuracy at one or more values of ``k``.

    The last score dimension contains candidate claims/classes.  ``targets`` may
    be integer class indices with shape ``scores.shape[:-1]`` or binary multi-hot
    labels with the same shape as ``scores``.  A sample is correct when any valid
    root cause occurs in its top-k predictions.

    A sample-shaped mask excludes complete samples.  For multi-hot targets, a
    score-shaped mask instead excludes padded candidate claims before ranking.
    The latter form is also accepted for index targets and causes samples whose
    target candidate is masked out to be ignored.
    """

    ks = _normalize_ks(k)
    score_tensor = _as_tensor(scores, "top-k scores")
    target_tensor = _as_tensor(targets, "top-k targets", score_tensor.device)

    result: MetricDict = {"top_%d_accuracy" % value: 0.0 for value in ks}
    result["top_k_count"] = 0

    # An unshaped empty list is treated as an empty data set for convenience.
    if score_tensor.numel() == 0 and score_tensor.ndim < 1:
        return result
    if score_tensor.ndim < 1:
        raise ValueError("top-k scores must have at least one dimension")

    sample_shape = tuple(score_tensor.shape[:-1])
    class_count = int(score_tensor.shape[-1])
    index_targets = tuple(target_tensor.shape) == sample_shape
    if tuple(target_tensor.shape) == sample_shape + (1,):
        target_tensor = target_tensor.squeeze(-1)
        index_targets = True
    multi_hot_targets = tuple(target_tensor.shape) == tuple(score_tensor.shape)
    if not index_targets and not multi_hot_targets:
        raise ValueError(
            "top-k targets must be class indices shaped %s or multi-hot labels "
            "shaped %s" % (sample_shape, tuple(score_tensor.shape))
        )

    sample_reference = torch.empty(
        sample_shape, dtype=torch.bool, device=score_tensor.device
    )
    sample_mask = torch.ones(sample_shape, dtype=torch.bool, device=score_tensor.device)
    candidate_mask = torch.ones_like(score_tensor, dtype=torch.bool)
    if mask is not None:
        # Exact candidate-shaped masks take precedence over prefix expansion.
        raw_mask = _as_tensor(mask, "mask", score_tensor.device)
        if tuple(raw_mask.shape) == tuple(score_tensor.shape):
            candidate_mask = raw_mask.to(dtype=torch.bool)
        else:
            maybe_sample_mask = _mask_for_shape(
                mask, sample_shape, score_tensor.device, "mask"
            )
            if maybe_sample_mask is not None:
                sample_mask = maybe_sample_mask
            else:
                candidate_mask = _broadcast_mask(mask, score_tensor, "mask")

    if class_count == 0 or sample_reference.numel() == 0:
        return result

    active_candidates = candidate_mask & sample_mask.unsqueeze(-1)
    _require_finite(score_tensor[active_candidates], "top-k scores")
    valid_samples = sample_mask & candidate_mask.any(dim=-1)

    if index_targets:
        index_values = target_tensor.to(dtype=torch.float64)
        selected_indices = index_values[valid_samples]
        _require_finite(selected_indices, "top-k targets")
        if selected_indices.numel() and not bool(
            (selected_indices == selected_indices.round()).all()
        ):
            raise ValueError("valid top-k class targets must be integers")
        if selected_indices.numel() and not bool(
            ((selected_indices >= 0) & (selected_indices < class_count)).all()
        ):
            raise ValueError("valid top-k class targets are out of range")
        target_indices = index_values.to(dtype=torch.long)

        # A candidate mask can mark a target as padded/missing.  Such a sample
        # has no evaluable root-cause label and is left out of the denominator.
        safe_indices = target_indices.clamp(0, class_count - 1)
        target_is_valid = candidate_mask.gather(-1, safe_indices.unsqueeze(-1)).squeeze(
            -1
        )
        valid_samples = valid_samples & target_is_valid
        positive_targets = None
    else:
        target_values = target_tensor.to(dtype=torch.float64)
        _require_binary(target_values[active_candidates], "top-k targets")
        positive_targets = (target_values == 1) & candidate_mask
        target_indices = None

    count = int(valid_samples.sum().item())
    result["top_k_count"] = count
    if count == 0:
        return result

    ranked_scores = score_tensor.to(dtype=torch.float64).masked_fill(
        ~candidate_mask, -torch.inf
    )
    largest_k = min(max(ks), class_count)
    top_indices = ranked_scores.topk(largest_k, dim=-1).indices

    for requested_k in ks:
        effective_k = min(requested_k, class_count)
        selected = top_indices[..., :effective_k]
        if target_indices is not None:
            hits = (selected == target_indices.unsqueeze(-1)).any(dim=-1)
        else:
            assert positive_targets is not None
            hits = positive_targets.gather(-1, selected).any(dim=-1)
        accuracy = float(hits[valid_samples].to(dtype=torch.float64).mean().item())
        result["top_%d_accuracy" % requested_k] = accuracy
    return result


def topk_accuracy(
    scores: TensorLike,
    targets: TensorLike,
    k: Union[int, Sequence[int]] = (1, 3, 5),
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Alias for :func:`top_k_accuracy`."""

    return top_k_accuracy(scores, targets, k=k, mask=mask)


def debt_calibration_metrics(
    predictions: TensorLike,
    targets: TensorLike,
    mask: Optional[TensorLike] = None,
    num_bins: int = 10,
    from_logits: bool = False,
) -> MetricDict:
    """Compute Brier score and expected calibration error for causal debt.

    Debt is interpreted as the predicted probability that a claim leads to a
    failure.  ECE bins this probability directly and compares mean debt with the
    empirical failure frequency in each non-empty equal-width bin.
    """

    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    predicted, target = _selected_pair(
        predictions,
        targets,
        mask,
        "debt predictions",
        "debt targets",
    )
    predicted = predicted.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)
    _require_finite(predicted, "debt predictions")
    _require_binary(target, "debt targets")
    if from_logits:
        predicted = torch.sigmoid(predicted)
    else:
        _require_probabilities(predicted, "debt predictions")

    count = int(target.numel())
    if count == 0:
        return {"debt_brier": 0.0, "debt_ece": 0.0, "debt_count": 0}

    brier = float(((predicted - target) ** 2).mean().item())
    ece = 0.0
    for bin_index in range(num_bins):
        lower = float(bin_index) / float(num_bins)
        upper = float(bin_index + 1) / float(num_bins)
        if bin_index == num_bins - 1:
            in_bin = (predicted >= lower) & (predicted <= upper)
        else:
            in_bin = (predicted >= lower) & (predicted < upper)
        bin_count = int(in_bin.sum().item())
        if bin_count == 0:
            continue
        mean_prediction = float(predicted[in_bin].mean().item())
        empirical_frequency = float(target[in_bin].mean().item())
        ece += (float(bin_count) / float(count)) * abs(
            mean_prediction - empirical_frequency
        )

    return {"debt_brier": brier, "debt_ece": ece, "debt_count": count}


def rollback_metrics(
    predictions: TensorLike,
    targets: TensorLike,
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Compute exact rollback-stage accuracy and mean absolute stage distance.

    Predictions may be stage indices/positions with the same shape as targets,
    or class scores with one additional trailing stage dimension.
    """

    predicted = _as_tensor(predictions, "rollback predictions")
    target = _as_tensor(targets, "rollback targets", predicted.device)
    if predicted.ndim == target.ndim + 1 and tuple(predicted.shape[:-1]) == tuple(
        target.shape
    ):
        if predicted.shape[-1] <= 0:
            if target.numel() == 0:
                return {
                    "rollback_accuracy": 0.0,
                    "rollback_mean_distance": 0.0,
                    "rollback_count": 0,
                }
            raise ValueError("rollback class predictions need at least one stage")
        valid = _broadcast_mask(mask, target, "mask")
        score_valid = valid.unsqueeze(-1).expand_as(predicted)
        _require_finite(predicted[score_valid], "rollback predictions")
        predicted = predicted.argmax(dim=-1)
    else:
        _require_same_shape(
            predicted,
            target,
            "rollback predictions and rollback targets",
        )
        valid = _broadcast_mask(mask, target, "mask")

    predicted = predicted[valid].to(dtype=torch.float64)
    target = target[valid].to(dtype=torch.float64)
    _require_finite(predicted, "rollback predictions")
    _require_finite(target, "rollback targets")

    count = int(target.numel())
    if count == 0:
        return {
            "rollback_accuracy": 0.0,
            "rollback_mean_distance": 0.0,
            "rollback_count": 0,
        }
    return {
        "rollback_accuracy": float((predicted == target).double().mean().item()),
        "rollback_mean_distance": float((predicted - target).abs().mean().item()),
        "rollback_count": count,
    }


def repair_metrics(
    successes: TensorLike,
    action_counts: TensorLike,
    debt_before: TensorLike,
    debt_after: TensorLike,
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Compute repair success rate, action efficiency, and mean debt drop.

    Each valid element represents one attempted repair.  Debt drop is signed
    (``debt_before - debt_after``), so repairs that increase debt reduce the
    aggregate rather than being silently clipped.
    """

    success_tensor = _as_tensor(successes, "repair successes")
    action_tensor = _as_tensor(
        action_counts, "repair action counts", success_tensor.device
    )
    before_tensor = _as_tensor(debt_before, "debt before repair", success_tensor.device)
    after_tensor = _as_tensor(debt_after, "debt after repair", success_tensor.device)
    for tensor, names in (
        (action_tensor, "repair successes and action counts"),
        (before_tensor, "repair successes and debt before repair"),
        (after_tensor, "repair successes and debt after repair"),
    ):
        _require_same_shape(success_tensor, tensor, names)

    valid = _broadcast_mask(mask, success_tensor, "mask")
    success_tensor = success_tensor[valid].to(dtype=torch.float64)
    action_tensor = action_tensor[valid].to(dtype=torch.float64)
    before_tensor = before_tensor[valid].to(dtype=torch.float64)
    after_tensor = after_tensor[valid].to(dtype=torch.float64)
    _require_binary(success_tensor, "repair successes")
    _require_finite(action_tensor, "repair action counts")
    _require_probabilities(before_tensor, "debt before repair")
    _require_probabilities(after_tensor, "debt after repair")
    if action_tensor.numel() and not bool((action_tensor >= 0).all()):
        raise ValueError("valid repair action counts must be non-negative")

    count = int(success_tensor.numel())
    if count == 0:
        return {
            "repair_success_rate": 0.0,
            "repair_mean_actions": 0.0,
            "repair_mean_debt_drop": 0.0,
            "repair_count": 0,
        }
    return {
        "repair_success_rate": float(success_tensor.mean().item()),
        "repair_mean_actions": float(action_tensor.mean().item()),
        "repair_mean_debt_drop": float((before_tensor - after_tensor).mean().item()),
        "repair_count": count,
    }


def local_rollback_metrics(
    is_local: TensorLike,
    is_rollback: Optional[TensorLike] = None,
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Compute the fraction of rollback events handled by local rollback.

    If ``is_rollback`` is omitted, every valid input element is assumed to
    describe a rollback event.  Otherwise the denominator includes only entries
    where ``is_rollback`` is true.  A local flag outside that set is ignored.
    """

    local_tensor = _as_tensor(is_local, "local rollback flags")
    valid = _broadcast_mask(mask, local_tensor, "mask")
    local_values = local_tensor[valid].to(dtype=torch.float64)
    _require_binary(local_values, "local rollback flags")

    if is_rollback is None:
        rollback_values = torch.ones_like(local_values, dtype=torch.bool)
    else:
        rollback_tensor = _as_tensor(is_rollback, "rollback flags", local_tensor.device)
        _require_same_shape(
            local_tensor, rollback_tensor, "local rollback and rollback flags"
        )
        rollback_values_raw = rollback_tensor[valid].to(dtype=torch.float64)
        _require_binary(rollback_values_raw, "rollback flags")
        rollback_values = rollback_values_raw.to(dtype=torch.bool)

    local_flags = local_values.to(dtype=torch.bool) & rollback_values
    local_count = int(local_flags.sum().item())
    total_count = int(rollback_values.sum().item())
    return {
        "local_rollback_ratio": _safe_ratio(local_count, total_count),
        "local_rollback_count": local_count,
        "total_rollback_count": total_count,
    }


def local_rollback_ratio(
    is_local: TensorLike,
    is_rollback: Optional[TensorLike] = None,
    mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Alias for :func:`local_rollback_metrics`."""

    return local_rollback_metrics(is_local, is_rollback=is_rollback, mask=mask)


def _require_pair(
    first: Optional[TensorLike],
    second: Optional[TensorLike],
    group_name: str,
) -> bool:
    if first is None and second is None:
        return False
    if first is None or second is None:
        raise ValueError("%s metrics require both predictions and targets" % group_name)
    return True


def compute_ledger_metrics(
    *,
    claim_predictions: Optional[TensorLike] = None,
    claim_targets: Optional[TensorLike] = None,
    claim_mask: Optional[TensorLike] = None,
    claim_threshold: float = 0.5,
    claim_from_logits: bool = False,
    topk_scores: Optional[TensorLike] = None,
    topk_targets: Optional[TensorLike] = None,
    topk_mask: Optional[TensorLike] = None,
    topk_values: Union[int, Sequence[int]] = (1, 3, 5),
    debt_predictions: Optional[TensorLike] = None,
    debt_targets: Optional[TensorLike] = None,
    debt_mask: Optional[TensorLike] = None,
    debt_num_bins: int = 10,
    debt_from_logits: bool = False,
    rollback_predictions: Optional[TensorLike] = None,
    rollback_targets: Optional[TensorLike] = None,
    rollback_mask: Optional[TensorLike] = None,
    repair_successes: Optional[TensorLike] = None,
    repair_action_counts: Optional[TensorLike] = None,
    repair_debt_before: Optional[TensorLike] = None,
    repair_debt_after: Optional[TensorLike] = None,
    repair_mask: Optional[TensorLike] = None,
    local_rollback_flags: Optional[TensorLike] = None,
    rollback_event_flags: Optional[TensorLike] = None,
    rollback_event_mask: Optional[TensorLike] = None,
) -> MetricDict:
    """Compute any available subset of Ledger-WAM evaluation metrics.

    Metric groups are optional.  A completely missing group is skipped, while a
    partially supplied group raises ``ValueError`` to avoid silently reporting a
    misleading zero.  Passing empty tensors is different from omitting a group:
    the corresponding zero-valued metrics and count are included in the result.
    """

    metrics: MetricDict = {}
    if _require_pair(claim_predictions, claim_targets, "claim root-cause"):
        metrics.update(
            claim_root_cause_metrics(
                claim_predictions,
                claim_targets,
                mask=claim_mask,
                threshold=claim_threshold,
                from_logits=claim_from_logits,
            )
        )
    if _require_pair(topk_scores, topk_targets, "top-k"):
        metrics.update(
            top_k_accuracy(
                topk_scores,
                topk_targets,
                k=topk_values,
                mask=topk_mask,
            )
        )
    if _require_pair(debt_predictions, debt_targets, "debt calibration"):
        metrics.update(
            debt_calibration_metrics(
                debt_predictions,
                debt_targets,
                mask=debt_mask,
                num_bins=debt_num_bins,
                from_logits=debt_from_logits,
            )
        )
    if _require_pair(rollback_predictions, rollback_targets, "rollback"):
        metrics.update(
            rollback_metrics(
                rollback_predictions,
                rollback_targets,
                mask=rollback_mask,
            )
        )

    repair_values = (
        repair_successes,
        repair_action_counts,
        repair_debt_before,
        repair_debt_after,
    )
    present_repair_values = sum(value is not None for value in repair_values)
    if present_repair_values:
        if present_repair_values != len(repair_values):
            raise ValueError(
                "repair metrics require successes, action counts, debt before, "
                "and debt after"
            )
        metrics.update(
            repair_metrics(
                repair_successes,
                repair_action_counts,
                repair_debt_before,
                repair_debt_after,
                mask=repair_mask,
            )
        )

    if local_rollback_flags is not None:
        metrics.update(
            local_rollback_metrics(
                local_rollback_flags,
                is_rollback=rollback_event_flags,
                mask=rollback_event_mask,
            )
        )
    elif rollback_event_flags is not None:
        raise ValueError(
            "local rollback metrics require local_rollback_flags when "
            "rollback_event_flags are supplied"
        )
    return metrics


__all__ = [
    "claim_root_cause_metrics",
    "compute_ledger_metrics",
    "debt_calibration_metrics",
    "local_rollback_metrics",
    "local_rollback_ratio",
    "repair_metrics",
    "rollback_metrics",
    "top_k_accuracy",
    "topk_accuracy",
]
