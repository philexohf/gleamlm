import torch
import torch.nn as nn


class LayerLimitedModel(nn.Module):
    """Wrapper that limits the number of active layers"""

    def __init__(self, original_model, num_active_layers):
        super().__init__()
        self.original = original_model
        self.num_active_layers = num_active_layers
        self.d_model = original_model.d_model

    def forward(self, input_ids, past_kv_list=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x = self.original.token_embed(input_ids)
        x = self.original.embed_dropout(x)

        offset = (
            past_kv_list[0][0].size(2)
            if past_kv_list is not None and past_kv_list and past_kv_list[0][0] is not None
            else 0
        )

        total_len = offset + seq_len
        cos = self.original.rope_cos[offset:total_len].unsqueeze(1).unsqueeze(0).to(device)
        sin = self.original.rope_sin[offset:total_len].unsqueeze(1).unsqueeze(0).to(device)

        attn_mask = self.original._create_causal_mask(seq_len, device, offset=offset)

        new_kv_list = []
        for i, layer in enumerate(self.original.layers):
            if i >= self.num_active_layers:
                # Bypass: just pass through with identity (dummy KV on correct device)
                current_kv = (torch.zeros(1, device=device), torch.zeros(1, device=device))
                new_kv_list.append(current_kv)
                continue
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, current_kv = layer(x, cos, sin, attn_mask, past_kv)
            new_kv_list.append(current_kv)

        x = self.original.final_norm(x)
        logits = self.original.lm_head(x)
        return logits, new_kv_list

    def get_num_params(self):
        return self.original.get_num_params()
