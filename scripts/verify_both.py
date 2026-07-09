import torch

from gleamlm.models.model import GleamLMModel

print("=== 40M (flash_attn=False, default) ===")
m40 = GleamLMModel(
    vocab_size=12002,
    d_model=512,
    num_layers=12,
    num_heads=8,
    num_kv_heads=4,
    d_ff=1365,
    dropout=0.1,
    max_seq_len=1024,
)
print(f"Params: {sum(p.numel() for p in m40.parameters()):,}")
x = torch.randint(0, 12000, (2, 64))
logits, _ = m40(x)
loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 12002), x[:, 1:].reshape(-1))
loss.backward()
print(f"Forward/Backward OK (loss={loss.item():.2f})")

print("\n=== 87M (flash_attn=True) ===")
m87 = GleamLMModel(
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
print(f"Params: {sum(p.numel() for p in m87.parameters()):,}")
x = torch.randint(0, 12000, (2, 64))
logits, _ = m87(x)
loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 12002), x[:, 1:].reshape(-1))
loss.backward()
print(f"Forward/Backward OK (loss={loss.item():.2f})")

print("\nBoth configs work, no interference.")
