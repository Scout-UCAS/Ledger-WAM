from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectSlotOutput:
    slots: torch.Tensor
    presence: torch.Tensor
    assignment: torch.Tensor


class DynamicObjectSlotEncoder(nn.Module):
    """Extracts object-centric slots and keeps their identity across replans.

    Slot identity is made recurrent by matching newly inferred slots to the previous
    memory with a differentiable cosine-similarity assignment.  The module does not
    assume a fixed semantic class list: absent slots are explicitly represented by
    the predicted presence probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_slots: int = 12,
        iterations: int = 3,
        slot_mlp_dim: Optional[int] = None,
        identity_temperature: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.iterations = int(iterations)
        self.identity_temperature = float(identity_temperature)
        self.eps = float(eps)
        mlp_dim = int(slot_mlp_dim or hidden_dim * 2)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.slot_identity = nn.Parameter(
            torch.randn(1, self.num_slots, self.hidden_dim) / self.hidden_dim**0.5
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.slot_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, self.hidden_dim),
        )
        self.presence_head = nn.Linear(self.hidden_dim, 1)

    def _initial_slots(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        # Deterministic identity anchors are preferable at inference.  Stochasticity
        # comes from the visual/action diffusion backbone rather than slot indexing.
        return (
            self.slot_mu.to(reference)
            + self.slot_identity.to(reference)
        ).expand(batch_size, -1, -1)

    def _align_to_previous(
        self,
        slots: torch.Tensor,
        previous_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = F.normalize(slots.float(), dim=-1)
        previous = F.normalize(previous_slots.float(), dim=-1)
        similarity = torch.einsum("bih,bjh->bij", previous, current)
        assignment = torch.softmax(
            similarity / max(self.identity_temperature, self.eps), dim=-1
        ).to(slots.dtype)
        aligned = torch.einsum("bij,bjh->bih", assignment, slots)
        return aligned, assignment

    def forward(
        self,
        visual_tokens: torch.Tensor,
        previous_slots: Optional[torch.Tensor] = None,
    ) -> ObjectSlotOutput:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                "`visual_tokens` must have shape [B,S,input_dim], got "
                f"{tuple(visual_tokens.shape)}."
            )
        inputs = self.input_projection(self.input_norm(visual_tokens))
        keys = self.key(inputs)
        values = self.value(inputs)
        slots = self._initial_slots(inputs.shape[0], inputs)
        scale = self.hidden_dim**-0.5

        for _ in range(self.iterations):
            slots_prev = slots
            logits = torch.einsum("bnh,bsh->bns", self.query(self.slot_norm(slots)), keys)
            attention = torch.softmax(logits * scale, dim=1) + self.eps
            attention = attention / attention.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bns,bsh->bnh", attention, values)
            slots = self.gru(
                updates.reshape(-1, self.hidden_dim),
                slots_prev.reshape(-1, self.hidden_dim),
            ).view_as(slots_prev)
            slots = slots + self.mlp(slots)

        if previous_slots is None:
            assignment = torch.eye(
                self.num_slots, device=slots.device, dtype=slots.dtype
            ).unsqueeze(0).expand(slots.shape[0], -1, -1)
        else:
            if tuple(previous_slots.shape) != tuple(slots.shape):
                raise ValueError(
                    "Previous object-slot shape must match current slots: "
                    f"{tuple(previous_slots.shape)} vs {tuple(slots.shape)}."
                )
            slots, assignment = self._align_to_previous(slots, previous_slots.to(slots))

        presence = torch.sigmoid(self.presence_head(slots).squeeze(-1))
        return ObjectSlotOutput(slots=slots, presence=presence, assignment=assignment)
