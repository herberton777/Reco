"""Exact factorized fourth-order (ANOVA) interactions for HOFM4 + JointBCD.

Adds genuine 4-way interactions among distinct fields:
  [user ID, item ID, 15 temporal preference/BCD features].
The original HOFM3 scores (and their existing BCD penalties) are preserved.
Requires PyTorch; intentionally isolated from the MovieLens Small baseline.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def fourth_anova(fields: torch.Tensor) -> torch.Tensor:
    """Compute sum_{i<j<k<l} <v_i*v_j*v_k*v_l, 1> in O(B * F * D).

    fields: [batch, num_fields, latent_rank]. This is exactly the degree-4
    elementary symmetric polynomial per latent dimension, not a cubic power
    or approximation through pairwise products.
    """
    if fields.ndim != 3 or fields.shape[1] < 4:
        raise ValueError("Expected [batch, at least four fields, latent rank]")
    b, _, dim = fields.shape
    e1 = fields.new_zeros((b, dim))
    e2 = fields.new_zeros((b, dim))
    e3 = fields.new_zeros((b, dim))
    e4 = fields.new_zeros((b, dim))
    for j in range(fields.shape[1]):
        v = fields[:, j, :]
        e4 = e4 + e3 * v
        e3 = e3 + e2 * v
        e2 = e2 + e1 * v
        e1 = e1 + v
    return e4.sum(dim=-1)


class FourthOrderJointBCD(nn.Module):
    """Trainable degree-4 FM branch for user/item and temporal BCD fields."""

    def __init__(self, n_users: int, n_items: int, bcd_dim: int = 15,
                 rank: int = 24, strength: float = 1.0,
                 input_dropout: float = 0.20, init_std: float = 0.075):
        super().__init__()
        if min(n_users, n_items, bcd_dim, rank) < 1 or bcd_dim + 2 < 4:
            raise ValueError("Invalid number of users, items, BCD fields or rank")
        if not (0 <= input_dropout < 1):
            raise ValueError("input_dropout must be in [0,1)")
        if not (0 < strength <= 100) or not (0 < init_std <= 1):
            raise ValueError("Invalid fourth-order strength or initialization")
        self.user_field = nn.Embedding(n_users, rank)
        self.movie_field = nn.Embedding(n_items, rank)
        self.bcd_field = nn.Parameter(torch.empty(bcd_dim, rank))
        self.strength = float(strength)
        self.input_dropout = float(input_dropout)
        nn.init.normal_(self.user_field.weight, std=init_std)
        nn.init.normal_(self.movie_field.weight, std=init_std)
        nn.init.normal_(self.bcd_field, std=init_std)

    def forward(self, users, items, bcd):
        if bcd.ndim != 2 or bcd.shape[1] != self.bcd_field.shape[0]:
            raise ValueError("Incompatible JointBCD shape for degree-4 field")
        if bcd.shape[0] != users.shape[0] or items.shape != users.shape:
            raise ValueError("User/item/BCD batch dimensions differ")
        if not torch.isfinite(bcd).all():
            raise FloatingPointError("Nonfinite BCD input")
        # Input dropout applies to the fourth-order branch only; the original
        # launcher retains its own independent BCD feature dropout.
        bcd = F.dropout(bcd, p=self.input_dropout, training=self.training)
        # tanh prevents extreme temporal features from exploding quartic terms.
        bcd = bcd.tanh()
        fields = torch.cat([
            self.user_field(users.long()).unsqueeze(1),
            self.movie_field(items.long()).unsqueeze(1),
            bcd.unsqueeze(-1) * self.bcd_field.unsqueeze(0),
        ], dim=1)
        return self.strength * fourth_anova(fields)

    def sampled_l2(self, positive, negative):
        """Batchwise factor L2; avoid scanning huge embedding tables per step."""
        return (
            self.user_field(positive["user"].long()).square().mean()
            + self.movie_field(positive["item"].long()).square().mean()
            + self.movie_field(negative["item"].long()).square().mean()
            + self.bcd_field.square().mean()
        )
