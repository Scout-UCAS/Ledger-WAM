from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ledger import CausalLedgerOutput


@dataclass
class CandidateWorldPrediction:
    """Predicted post-repair observation and causal ledger for every candidate."""

    candidate_actions: torch.Tensor
    next_hidden: torch.Tensor
    predicted_observation: torch.Tensor
    confidence_logits: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    dependency: torch.Tensor
    repair_cost: torch.Tensor
    observability: torch.Tensor
    importance: torch.Tensor
    debt: torch.Tensor
    global_risk: torch.Tensor
    predicted_failure_risk: torch.Tensor
    expected_debt_reduction: torch.Tensor

    def summary(self, batch_index: int = 0) -> dict[str, Any]:
        return {
            "candidate_global_risk": self.global_risk[batch_index]
            .detach()
            .float()
            .cpu()
            .tolist(),
            "candidate_failure_risk": self.predicted_failure_risk[batch_index]
            .detach()
            .float()
            .cpu()
            .tolist(),
            "expected_debt_reduction": self.expected_debt_reduction[batch_index]
            .detach()
            .float()
            .cpu()
            .tolist(),
        }


class CausalWorldPredictor(nn.Module):
    """Action-conditioned one-step world model used to evaluate repair candidates.

    Unlike a scalar repair-value head, this module explicitly predicts the next
    object/claim latent state, an observation embedding, and every debt component for
    every candidate action sequence.  It is intentionally lightweight compared with
    video diffusion so closed-loop planning remains fast.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_claims: int,
        num_object_slots: int,
        action_encoder_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.num_claims = int(num_claims)
        self.num_object_slots = int(num_object_slots)

        self.action_input = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.action_encoder = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=int(action_encoder_layers),
            batch_first=True,
        )
        self.object_pool = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.transition = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)
        self.observation_decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.component_head = nn.Linear(self.hidden_dim, 6)
        self.failure_head = nn.Linear(self.hidden_dim * 2, 1)
        self.debt_weight_raw = nn.Parameter(torch.zeros(5))
        self.debt_bias = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self,
        ledger: CausalLedgerOutput,
        candidate_actions: torch.Tensor,
    ) -> CandidateWorldPrediction:
        if candidate_actions.ndim != 4:
            raise ValueError("`candidate_actions` must have shape [B,R,T,A].")
        batch_size, num_candidates, horizon, action_dim = candidate_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"Expected candidate action dim {self.action_dim}, got {action_dim}."
            )
        if ledger.hidden.shape[0] != batch_size:
            raise ValueError("Ledger and candidate-action batch sizes must match.")

        flat_actions = candidate_actions.reshape(
            batch_size * num_candidates, horizon, action_dim
        )
        _, action_state = self.action_encoder(self.action_input(flat_actions))
        action_context = action_state[-1].view(batch_size, num_candidates, self.hidden_dim)
        object_context = self.object_pool(
            (ledger.object_presence.unsqueeze(-1) * ledger.object_slots).sum(dim=1)
            / ledger.object_presence.sum(dim=1, keepdim=True).clamp(min=1e-6)
        )
        object_context = object_context[:, None].expand(-1, num_candidates, -1)

        previous = ledger.hidden[:, None].expand(-1, num_candidates, -1, -1)
        transition_input = torch.cat(
            (
                action_context[:, :, None].expand(-1, -1, self.num_claims, -1),
                object_context[:, :, None].expand(-1, -1, self.num_claims, -1),
            ),
            dim=-1,
        )
        next_hidden = self.transition(
            transition_input.reshape(-1, self.hidden_dim * 2),
            previous.reshape(-1, self.hidden_dim),
        ).view(batch_size, num_candidates, self.num_claims, self.hidden_dim)

        components = self.component_head(next_hidden)
        confidence_logits = components[..., 0]
        confidence = torch.sigmoid(confidence_logits)
        uncertainty = torch.sigmoid(components[..., 1])
        dependency = torch.sigmoid(components[..., 2])
        repair_cost = torch.sigmoid(components[..., 3])
        observability = torch.sigmoid(components[..., 4])
        importance = torch.sigmoid(components[..., 5])
        debt_features = torch.stack(
            (1.0 - confidence, uncertainty, dependency, repair_cost, 1.0 - observability),
            dim=-1,
        )
        weights = F.softplus(self.debt_weight_raw).to(next_hidden)
        debt = torch.sigmoid((debt_features * weights).sum(dim=-1) + self.debt_bias)
        global_risk = (debt * importance).sum(dim=-1) / importance.sum(dim=-1).clamp(
            min=1e-6
        )

        weighted_hidden = (importance.unsqueeze(-1) * next_hidden).sum(dim=2)
        weighted_hidden = weighted_hidden / importance.sum(dim=2, keepdim=True).clamp(
            min=1e-6
        )
        predicted_observation = self.observation_decoder(
            torch.cat((weighted_hidden, object_context), dim=-1)
        )
        predicted_failure_risk = torch.sigmoid(
            self.failure_head(torch.cat((weighted_hidden, action_context), dim=-1)).squeeze(-1)
        )
        expected_debt_reduction = ledger.global_risk[:, None] - global_risk
        return CandidateWorldPrediction(
            candidate_actions=candidate_actions,
            next_hidden=next_hidden,
            predicted_observation=predicted_observation,
            confidence_logits=confidence_logits,
            confidence=confidence,
            uncertainty=uncertainty,
            dependency=dependency,
            repair_cost=repair_cost,
            observability=observability,
            importance=importance,
            debt=debt,
            global_risk=global_risk,
            predicted_failure_risk=predicted_failure_risk,
            expected_debt_reduction=expected_debt_reduction,
        )
