import torch
import torch.nn.functional as F
from torch import Tensor, nn

import logging
import os
import sys
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("multihead_attention")

class SelectiveAttention(nn.Module):
    def __init__(self, head_dim, dropout_prob=0.0, selective=False):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.selective = selective

    def forward(self, q, k, v, mask=None):
        bsz, seq_len, _ = q.size()
        scale = self.head_dim ** 0.5

        scores = torch.bmm(q, k.transpose(1, 2)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        if self.selective:
            # --- Selective Masking ---
            s = F.relu(scores.clone())
            s[..., 0] = 0

            if seq_len == k.size(1):
                eye = torch.eye(seq_len, device=s.device, dtype=s.dtype).unsqueeze(0)
                s = s * (1 - eye)

            s = torch.roll(s, 1, -2)
            s[..., 0, :] = 0

            s = torch.cumsum(s, dim=-1)
            scores = scores - s

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Logging stats
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            logger.warning("⚠️ NaN or Inf detected in attention weights")

        logger.info(f"[Selective={self.selective}] attn_weights stats: "
                    f"min={attn_weights.min().item():.6f}, max={attn_weights.max().item():.6f}, "
                    f"mean={attn_weights.mean().item():.6f}")

        output = torch.bmm(attn_weights, v)
        return output, attn_weights
