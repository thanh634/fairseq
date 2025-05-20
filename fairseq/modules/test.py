import torch

from fairseq.modules import MultiheadAttention

embed_dim = 64
kv_embed_dim = 1208
num_heads = 16
src_len = 20
tgt_len = 30
bsz = 10

model = MultiheadAttention(embed_dim, num_heads, kdim=kv_embed_dim, vdim=kv_embed_dim,
                           bias=True, add_bias_kv=False, add_zero_attn=True)

query = torch.rand((src_len, bsz, embed_dim))
key = torch.rand((src_len, bsz, kv_embed_dim))
value = torch.rand((src_len, bsz, kv_embed_dim))

attn_mask = torch.randint(0, 2, (src_len, src_len)).float()
attn_mask.masked_fill_(attn_mask == 0, float('-inf'))
attn_mask.masked_fill_(attn_mask > 0, float('0.0'))

seq_mask = torch.randint(0, 2, (1, src_len))
key_padding_mask = seq_mask
for i in range(bsz-1):
    key_padding_mask = torch.cat([key_padding_mask, seq_mask], axis=0)
key_padding_mask = key_padding_mask == 1

# Apply torch.nn version
model.skip_embed_dim_check = True
torch_output, torch_weight = model(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

# Apply fairseq version
model.skip_embed_dim_check = False
fairseq_output, fairseq_weight = model(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

print("torch and fairseq generate same results: outputs are same ? ",
      torch.allclose(torch_output, fairseq_output, atol=5e-6, rtol=1e-6),
      ", weights are same ? ",
      torch.allclose(torch_weight, fairseq_weight, atol=5e-6, rtol=1e-6)
      )