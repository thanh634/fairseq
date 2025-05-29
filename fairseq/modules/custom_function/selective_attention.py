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
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

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
            output = torch.zeros(B_eff, Tq, D_head, device=q.device)
            return output, None

        # Standard attention computation
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [B_eff, Tq, Tk]

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_scores += attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_scores += attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float('-inf')
            )

        if before_softmax:
            return attn_scores, v

        # Standard softmax and attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Simple matrix multiplication - like Fairseq
        output = torch.bmm(attn_probs, v)  # [B_eff, Tq, D_head]
        
        logger.info(
            f"output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, "
            f"mean={output.mean().item():.6f}"
        )
        return output, attn_probs if need_weights else None


class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.single_head_attention = SingleHeadAttention(dropout=dropout)

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
