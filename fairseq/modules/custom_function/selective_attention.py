import torch
import torch.nn.functional as F
from typing import Optional, Tuple

import logging
import os
import sys

from torch import nn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("multihead_attention")


class SingleHeadAttention(nn.Module):
    def __init__(self, dropout: float = 0.0, k_select: int = 16):
        """
        Initialize SingleHeadAttention with selective attention.

        Args:
            dropout (float): Dropout probability for attention probabilities.
            k_select (int): Number of top-k positions to select for attention.
        """
        super().__init__()
        self.dropout = dropout
        self.k_select = k_select  # Number of positions to select (top-k)

    def forward(
            self,
            q: torch.Tensor,  # [B_eff, Tq, D_head] where B_eff is B*H
            k: torch.Tensor,  # [B_eff, Tk, D_head]
            v: torch.Tensor,  # [B_eff, Tk, D_head]
            attn_mask: Optional[torch.Tensor] = None,         # [Tq, Tk] or [B_eff, Tq, Tk]
            key_padding_mask: Optional[torch.Tensor] = None,  # [B_eff, Tk]
            need_weights: bool = True,
            before_softmax: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with selective (top-k) attention.

        Args:
            q: Query tensor, shape [B_eff, Tq, D_head].
            k: Key tensor, shape [B_eff, Tk, D_head].
            v: Value tensor, shape [B_eff, Tk, D_head].
            attn_mask: Attention mask, shape [Tq, Tk] or [B_eff, Tq, Tk].
            key_padding_mask: Padding mask, shape [B_eff, Tk].
            need_weights: Whether to return attention probabilities.
            before_softmax: Whether to return raw scores and original v.

        Returns:
            Tuple of (output, attn_probs):
                - output: Attention output, shape [B_eff, Tq, D_head].
                - attn_probs: Attention probabilities, shape [B_eff, Tq, Tk] or None.
        """
        B_eff, Tq, D_head = q.shape
        Tk = k.size(1)

        if Tk == 0:
            logger.warning("Empty key sequence (Tk=0), returning zero output")
            output = torch.zeros(B_eff, Tq, D_head, device=q.device)
            return output, None if need_weights else None

        # Compute attention scores: [B_eff, Tq, Tk]
        attn_scores = torch.bmm(q, k.transpose(1, 2))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [Tq, Tk]
                attn_scores += attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:  # [B_eff, Tq, Tk]
                attn_scores += attn_mask
            else:
                raise ValueError(f"attn_mask has an unexpected dimension: {attn_mask.dim()}")

        # Apply key padding mask
        if key_padding_mask is not None:  # [B_eff, Tk]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf')
            )

        # Determine number of positions to select
        k_effective = min(self.k_select, Tk)
        k_effective = max(1, k_effective)

        # Selective attention: select top-k positions
        try:
            topk_scores, topk_indices = torch.topk(
                attn_scores, k=k_effective, dim=-1, largest=True, sorted=True
            )  # [B_eff, Tq, k_effective]
        except RuntimeError as e:
            logger.error(f"topk failed: Tk={Tk}, k_effective={k_effective}, attn_scores shape={attn_scores.shape}")
            raise e

        # Create a mask for top-k positions: [B_eff, Tq, Tk]
        topk_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_indices, True)

        # Apply top-k mask
        attn_scores = attn_scores.masked_fill(~topk_mask, float('-inf'))

        if before_softmax:
            return attn_scores, v

        # Compute softmax over selected positions
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Gather selected values: [B_eff, Tq, k_effective, D_head]
        selected_v = v.unsqueeze(1).expand(-1, Tq, -1, -1)  # [B_eff, Tq, Tk, D_head]
        selected_v = torch.gather(
            selected_v,
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, D_head)
        )

        # Gather the probabilities corresponding to the top-k indices
        # attn_probs has shape [B_eff, Tq, Tk]
        # topk_indices has shape [B_eff, Tq, k_effective]
        attn_probs_selected = torch.gather(attn_probs, dim=2, index=topk_indices) # Shape: [B_eff, Tq, k_effective]
        
        # Multiply selected probabilities with selected values and sum
        # selected_v has shape [B_eff, Tq, k_effective, D_head]
        # attn_probs_selected.unsqueeze(-1) has shape [B_eff, Tq, k_effective, 1]
        output = (attn_probs_selected.unsqueeze(-1) * selected_v).sum(dim=2) # Shape: [B_eff, Tq, D_head]
        
        logger.info(
            f"output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, "
            f"mean={output.mean().item():.6f}"
        )
        
        return output, attn_probs if need_weights else None


class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, dropout=0.0, k_select: int = 16): # Pass necessary params like num_heads, head_dim if needed
        super().__init__()
        self.dropout = dropout # Or pass directly to SingleHeadAttention
        # If num_heads and head_dim are fixed for this custom MHA, define them here
        # self.num_heads = num_heads
        # self.head_dim = head_dim
        self.single_head_attention = SingleHeadAttention(dropout=dropout, k_select=k_select)

    def forward(
            self,
            query: torch.Tensor,      # [B*H, Tq, D_head]
            key: torch.Tensor,        # [B*H, Tk, D_head]
            value: torch.Tensor,      # [B*H, Tk, D_head]
            key_padding_mask: Optional[torch.Tensor] = None,  # [B, Tk] (original batch_size)
            attn_mask: Optional[torch.Tensor] = None,         # [Tq, Tk] or [B*H, Tq, Tk]
            need_weights: bool = True,
            before_softmax: bool = False,
            # Add bsz and num_heads if they are not fixed attributes of this class
            # Or derive them if possible and key_padding_mask is always present.
            # For this example, let's assume B*H is the first dim of query.
            # And that key_padding_mask, if provided, needs expansion.
            bsz: Optional[int] = None, # Actual batch size
            num_heads: Optional[int] = None # Number of heads
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        BxH, Tq, D_head = query.size()
        Tk = key.size(1)

        expanded_kpm = None
        if key_padding_mask is not None:
            if bsz is None or num_heads is None:
                raise ValueError("bsz and num_heads must be provided to MultiHeadAttentionCustom if key_padding_mask is used.")
            if bsz * num_heads != BxH:
                raise ValueError(f"bsz ({bsz}) * num_heads ({num_heads}) != query.size(0) ({BxH})")
            # key_padding_mask is [B, Tk] -> expand to [B*H, Tk]
            # bsz here is kv_bsz, num_heads is self.num_heads from the parent MHA
            # target_b_eff_k = key.size(0) # This is kv_bsz * num_heads
            expanded_kpm = key_padding_mask.unsqueeze(1).expand(bsz, num_heads, Tk).reshape(bsz * num_heads, Tk)
            # Ensure this expanded_kpm.size(0) == key.size(0)
            if expanded_kpm.size(0) != key.size(0):
                raise ValueError(
                    f"Expanded key_padding_mask batch {expanded_kpm.size(0)} "
                    f"does not match key batch {key.size(0)}"
                )

        # attn_mask is assumed to be broadcastable or already [B*H, Tq, Tk]
        # Pass the already reshaped Q, K, V (where batch dim is B*H)
        return self.single_head_attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=expanded_kpm, # Pass the expanded mask
            need_weights=need_weights,
            before_softmax=before_softmax
        )

