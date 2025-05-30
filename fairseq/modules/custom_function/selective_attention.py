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


class SingleHeadAttentionSelective(nn.Module):
    def __init__(self, dropout: float = 0.0, k_select: Optional[int] = None,
                 adaptive_k: bool = True, min_k_ratio: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.k_select = k_select  # Fixed number of positions to select
        self.adaptive_k = adaptive_k  # Whether to adapt k based on sequence length
        self.min_k_ratio = min_k_ratio  # Minimum ratio of positions to keep

    def _determine_k_effective(self, seq_len: int) -> int:
        """Determine effective k based on sequence length and settings"""
        if self.k_select is None:
            # No selection - use full attention
            return seq_len

        if self.adaptive_k:
            # Adaptive: use a percentage of sequence length, but respect min/max bounds
            adaptive_k = max(
                int(seq_len * self.min_k_ratio),  # At least min_k_ratio of sequence
                min(self.k_select, seq_len)       # But not more than k_select or seq_len
            )
            return adaptive_k
        else:
            # Fixed k_select
            return min(self.k_select, seq_len)

    def forward(
            self,
            q: torch.Tensor,  # [B_eff, Tq, D_head]
            k: torch.Tensor,  # [B_eff, Tk, D_head]
            v: torch.Tensor,  # [B_eff, Tk, D_head]
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            before_softmax: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B_eff, Tq, D_head = q.shape
        Tk = k.size(1)

        if Tk == 0:
            logger.warning("Empty key sequence (Tk=0), returning zero output")
            output = torch.zeros(B_eff, Tq, D_head, device=q.device)
            return output, None

        # Compute attention scores: [B_eff, Tq, Tk]
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        # Apply attention mask BEFORE selection
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [Tq, Tk]
                attn_scores += attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:  # [B_eff, Tq, Tk]
                attn_scores += attn_mask
            else:
                raise ValueError(f"attn_mask has an unexpected dimension: {attn_mask.dim()}")

        # Apply key padding mask BEFORE selection
        if key_padding_mask is not None:  # [B_eff, Tk]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf')
            )

        # Determine effective k
        k_effective = self._determine_k_effective(Tk)
        logger.info(
            f"k_effective: {k_effective}"
        )

        # If k_effective equals Tk, use standard attention (no selection)
        if k_effective >= Tk:
            if before_softmax:
                return attn_scores, v

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            output = torch.bmm(attn_probs, v)

            logger.info(
                f"output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, "
                f"mean={output.mean().item():.6f}"
            )
            
            return output, attn_probs if need_weights else None

        # SELECTIVE ATTENTION: Select top-k positions
        try:
            # Get top-k attention scores and their indices
            topk_scores, topk_indices = torch.topk(
                attn_scores, k=k_effective, dim=-1, largest=True, sorted=True
            )  # Both: [B_eff, Tq, k_effective]
        except RuntimeError as e:
            logger.error(f"topk failed: Tk={Tk}, k_effective={k_effective}, attn_scores shape={attn_scores.shape}")
            # Fallback to full attention
            if before_softmax:
                return attn_scores, v
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            output = torch.bmm(attn_probs, v)

            logger.info(
                f"output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, "
                f"mean={output.mean().item():.6f}"
            )
            
            return output, attn_probs if need_weights else None

        if before_softmax:
            # For before_softmax, we still need to return full-size scores
            # Create masked version with only top-k positions
            masked_scores = torch.full_like(attn_scores, float('-inf'))
            masked_scores.scatter_(-1, topk_indices, topk_scores)

            logger.info(
                f"masked_scores stats: min={masked_scores.min().item():.6f}, max={masked_scores.max().item():.6f}, "
                f"mean={masked_scores.mean().item():.6f}"
            )
            
            return masked_scores, v

        # Compute softmax over ONLY the selected positions
        # This is more efficient than masking the full matrix
        topk_probs = F.softmax(topk_scores, dim=-1)  # [B_eff, Tq, k_effective]
        topk_probs = F.dropout(topk_probs, p=self.dropout, training=self.training)

        # Gather selected values efficiently
        # Method : Using gather (memory efficient for small k_effective)
        if k_effective <= Tk // 4:  # Use gather for small selections
            # Expand indices to match value dimensions
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D_head)  # [B_eff, Tq, k_effective, D_head]

            # Expand values to allow gathering
            expanded_v = v.unsqueeze(1).expand(-1, Tq, -1, -1)  # [B_eff, Tq, Tk, D_head]

            # Gather selected values
            selected_v = torch.gather(expanded_v, dim=2, index=expanded_indices)  # [B_eff, Tq, k_effective, D_head]

            # Compute weighted sum
            output = (topk_probs.unsqueeze(-1) * selected_v).sum(dim=2)  # [B_eff, Tq, D_head]

        else:  # Use masking for larger selections (more memory but potentially faster)
            # Create sparse attention matrix
            sparse_attn_probs = torch.zeros_like(attn_scores)  # [B_eff, Tq, Tk]
            sparse_attn_probs.scatter_(-1, topk_indices, topk_probs)

            # Standard matrix multiplication
            output = torch.bmm(sparse_attn_probs, v)  # [B_eff, Tq, D_head]

        logger.info(
            f"output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, "
            f"mean={output.mean().item():.6f}"
        )
        # Prepare return weights
        if need_weights:
            # Return sparse attention weights in original shape
            full_attn_probs = torch.zeros_like(attn_scores)
            full_attn_probs.scatter_(-1, topk_indices, topk_probs)
            return output, full_attn_probs
        else:
            return output, None


class MultiHeadAttentionSelective(nn.Module):
    def __init__(self, dropout=0.0, k_select: Optional[int] = None,
                 adaptive_k: bool = True, min_k_ratio: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.single_head_attention = SingleHeadAttentionSelective(
            dropout=dropout,
            k_select=k_select,
            adaptive_k=adaptive_k,
            min_k_ratio=min_k_ratio
        )

    def forward(
            self,
            query: torch.Tensor,      # [B*H, Tq, D_head]
            key: torch.Tensor,        # [B*H, Tk, D_head]
            value: torch.Tensor,      # [B*H, Tk, D_head]
            key_padding_mask: Optional[torch.Tensor] = None,  # [B, Tk]
            attn_mask: Optional[torch.Tensor] = None,         # [Tq, Tk] or [B*H, Tq, Tk]
            need_weights: bool = True,
            before_softmax: bool = False,
            bsz: Optional[int] = None, # Actual batch size
            num_heads: Optional[int] = None # Number of heads
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        BxH, Tq, D_head = query.size()
        Tk = key.size(1)

        expanded_kpm = None
        if key_padding_mask is not None:
            if bsz is None or num_heads is None:
                raise ValueError("bsz and num_heads must be provided if key_padding_mask is used.")
            if bsz * num_heads != BxH:
                raise ValueError(f"bsz ({bsz}) * num_heads ({num_heads}) != query.size(0) ({BxH})")

            expanded_kpm = key_padding_mask.unsqueeze(1).expand(bsz, num_heads, Tk).reshape(bsz * num_heads, Tk)

        return self.single_head_attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=expanded_kpm,
            need_weights=need_weights,
            before_softmax=before_softmax
        )


# Example usage configurations:
def create_attention_configs():
    """Different configurations for selective attention"""

    # 1. Conservative selective attention (good starting point)
    conservative_config = MultiHeadAttentionSelective(
        dropout=0.1,
        k_select=64,        # Select top 64 positions
        adaptive_k=True,    # Adapt based on sequence length
        min_k_ratio=0.25    # Always keep at least 25% of positions
    )

    # 2. Aggressive selective attention (for very long sequences)
    aggressive_config = MultiHeadAttentionSelective(
        dropout=0.1,
        k_select=32,        # Select top 32 positions
        adaptive_k=True,    # Adapt based on sequence length  
        min_k_ratio=0.1     # Keep at least 10% of positions
    )

    # 3. Full attention (no selection)
    full_attention_config = MultiHeadAttentionSelective(
        dropout=0.1,
        k_select=None,      # No selection - full attention
        adaptive_k=False,
        min_k_ratio=1.0
    )

    # 4. Fixed selective attention (always select exactly k positions)
    fixed_config = MultiHeadAttentionSelective(
        dropout=0.1,
        k_select=48,        # Always select exactly 48 positions
        adaptive_k=False,   # No adaptation
        min_k_ratio=0.0
    )

    return {
        'conservative': conservative_config,
        'aggressive': aggressive_config,
        'full': full_attention_config,
        'fixed': fixed_config
    }