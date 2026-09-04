"""GleamLMForCausalLM — HF PreTrainedModel 包装，对接 transformers / TRL / lm-eval。"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.generation import GenerationMixin
except ImportError:
    from transformers.generation_utils import GenerationMixin

from gleamlm.models.attention_variants import AliBiGQA, NoPEGQA, SlidingWindowGQA
from gleamlm.models.model import GQA, MLP, MoE, GleamLMModel

from .hf_config import GleamLMConfig


def _legacy_cache_from_hf(cache):
    """DynamicCache → legacy (k, v) tuple list。"""
    if cache is None:
        return None
    if isinstance(cache, DynamicCache):
        if not cache.layers or any(layer.keys is None for layer in cache.layers):
            return None
        return [(layer.keys, layer.values) for layer in cache.layers]
    return cache


def _hf_cache_from_legacy(cache, use_dynamic: bool):
    """Legacy tuple list → DynamicCache。"""
    if cache is None or not use_dynamic:
        return cache
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(cache):
        new_cache.update(k, v, layer_idx=layer_idx)
    return new_cache


ATTN_REGISTRY: dict[str, type] = {
    "gqa": GQA,
    "nope": NoPEGQA,
    "alibi": AliBiGQA,
    "sliding": SlidingWindowGQA,
}
FFN_REGISTRY: dict[str, type] = {"mlp": MLP, "moe": MoE}


def load_from_checkpoint(model, checkpoint: dict, strict: bool = False):
    """训练 checkpoint → HF wrapper 权重加载。

    处理 DDP 的 module. 前缀 + core 侧的 model. 前缀规整。
    """
    sd = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
    if isinstance(sd, dict):
        if any(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        sd = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in sd.items()}
    return model.load_state_dict(sd, strict=strict)


class GleamLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GleamLMConfig
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"rope_cos", r"rope_sin"]
    _keys_to_ignore_on_load_unexpected = [r"rope_cos", r"rope_sin"]
    _tied_weights_keys = {"model.lm_head.weight": "model.token_embed.weight"}

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # RoPE buffers are non-persistent and lost during meta-device materialization
        model.model._recompute_rope_cache()
        return model

    def __init__(self, config: GleamLMConfig):
        super().__init__(config)
        self.generation_config = GenerationConfig(
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        attn_variant = ATTN_REGISTRY.get(config.attn_type, GQA)
        ffn_variant = FFN_REGISTRY.get(config.ffn_type, MLP)

        # hf_config 把 layer_configs 里的变体 class 序列化为类名，这里映射回来
        layer_configs = config.layer_configs
        if layer_configs:
            attn_by_name = {c.__name__: c for c in ATTN_REGISTRY.values()}
            ffn_by_name = {c.__name__: c for c in FFN_REGISTRY.values()}
            layer_configs = [
                {
                    **cfg,
                    **(
                        {"attn_variant": attn_by_name.get(cfg["attn_variant"], GQA)}
                        if isinstance(cfg.get("attn_variant"), str) else {}
                    ),
                    **(
                        {"ffn_variant": ffn_by_name.get(cfg["ffn_variant"], MLP)}
                        if isinstance(cfg.get("ffn_variant"), str) else {}
                    ),
                }
                for cfg in layer_configs
            ]

        self.model = GleamLMModel(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            d_ff=config.intermediate_size,
            max_seq_len=config.max_position_embeddings,
            dropout=0.0,
            pad_token_id=config.pad_token_id,
            tie_weights=config.tie_word_embeddings,
            use_flash_attn=config.use_flash_attn,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            attn_variant=attn_variant,
            ffn_variant=ffn_variant,
            num_experts=config.num_experts,
            top_k=config.top_k,
            rope_scale=config.rope_scale,
            rope_factor=config.rope_factor,
            rope_theta=config.rope_theta,
            layer_configs=layer_configs,
        )
        self.post_init()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | DynamicCache | None = None,
        labels: torch.Tensor | None = None, output_hidden_states: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_dynamic_cache = isinstance(past_key_values, DynamicCache)
        past_kv_list = _legacy_cache_from_hf(past_key_values)
        logits, new_kv, aux_loss, hidden = self.model(
            input_ids, past_kv_list=past_kv_list,
            attention_mask=attention_mask,
            use_cache=kwargs.get("use_cache", True),
            output_hidden_states=output_hidden_states,
        )
        new_kv = _hf_cache_from_legacy(new_kv, use_dynamic_cache)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # HF Trainer 用 -100 标记忽略位置，core 训练用 pad_token_id；
            # 统一映射到 pad_token_id 后按 ignore_index 一次忽略
            if torch.any(shift_labels == -100):
                shift_labels = shift_labels.masked_fill(
                    shift_labels == -100, self.config.pad_token_id
                )
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            if aux_loss is not None and aux_loss > 0:
                loss = loss + 0.01 * aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_kv,
            hidden_states=hidden if output_hidden_states else None,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.token_embed

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.token_embed = value

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.lm_head if hasattr(self.model, "lm_head") else None

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = new_embeddings

    def _set_gradient_checkpointing(self, enable: bool, gradient_checkpointing_func=None) -> None:
        self.model.use_gradient_checkpointing = enable

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs
    ):
        has_past = False
        if isinstance(past_key_values, DynamicCache):
            has_past = bool(past_key_values.layers and any(
                layer.keys is not None for layer in past_key_values.layers
            ))
        elif past_key_values is not None:
            has_past = len(past_key_values) > 0

        if has_past:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            reordered.append((k.index_select(0, beam_idx), v.index_select(0, beam_idx)))
        return reordered


# 注册 AutoModel/AutoConfig，使 from_pretrained / TRL ref_model / lm-eval 可用
AutoConfig.register(GleamLMConfig.model_type, GleamLMConfig)
AutoModelForCausalLM.register(GleamLMConfig, GleamLMForCausalLM)
