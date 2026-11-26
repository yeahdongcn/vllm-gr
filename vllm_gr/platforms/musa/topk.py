# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager full-vocabulary candidate selection for the MUSA provider."""

import torch


def full_vocabulary_topk_out(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_counts: torch.Tensor,
) -> None:
    """Write ranked candidates, breaking ties by the lower token ID."""
    scale = torch.where(temperatures == 0, 1.0, temperatures)
    scaled = logits.float() / scale.reshape(-1, 1)
    scaled = scaled.masked_fill(~torch.isfinite(scaled), -torch.inf)
    indices = torch.argsort(scaled, dim=-1, descending=True, stable=True)
    indices = indices[:, : token_ids.shape[1]]
    selected = scaled.gather(1, indices)
    valid = torch.isfinite(selected)
    normalized = selected - torch.logsumexp(scaled, dim=1, keepdim=True)
    token_ids.copy_(indices.masked_fill(~valid, -1))
    logprobs.copy_(normalized.masked_fill(~valid, -torch.inf))
    valid_mask.copy_(valid)
    valid_counts.copy_(valid.sum(dim=1, dtype=torch.int32))
