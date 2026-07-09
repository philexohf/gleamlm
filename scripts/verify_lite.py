"""verify Lite model"""

import torch

from gleamlm.models.model import GleamLMModel

m = GleamLMModel(
    vocab_size=12002,
    d_model=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=6,
    d_ff=2048,
    dropout=0.0,
    max_seq_len=2048,
    use_flash_attn=True,
)
total = sum(p.numel() for p in m.parameters())
embed = sum(p.numel() for n, p in m.named_parameters() if "token_embed" in n)
ffn_params = sum(
    p.numel() for n, p in m.named_parameters() if "W_gate" in n or "W_up" in n or "W_down" in n
)

print(f"Total:    {total:>12,}  ({total / 1e6:.1f}M)")
print(f"Embed:    {embed:>12,}  ({100 * embed / total:.0f}%)")
print(f"FFN:      {ffn_params:>12,}  ({100 * ffn_params / total:.0f}%)")

x = torch.randint(0, 12000, (2, 64))
logits, _ = m(x)
loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 12002), x[:, 1:].reshape(-1))
loss.backward()
print(f"Forward/Backward OK (loss={loss.item():.2f})")
