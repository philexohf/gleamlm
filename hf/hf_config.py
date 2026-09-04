"""HF PretrainedConfig wrapper — 把 GleamLM 核心配置映射为 HF 标准键名。"""

from transformers import PretrainedConfig


class GleamLMConfig(PretrainedConfig):
    model_type = "gleam_lm"

    def __init__(
        self,
        vocab_size: int = 12002,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        intermediate_size: int = 1365,
        max_position_embeddings: int = 1024,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        use_flash_attn: bool = False,
        use_gradient_checkpointing: bool = False,
        attn_type: str = "gqa",
        ffn_type: str = "mlp",
        num_experts: int = 8,
        top_k: int = 2,
        rope_scale: float = 1.0,
        rope_factor: float = 8.0,
        rope_theta: float = 10000.0,
        layer_configs: list[dict] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_flash_attn = use_flash_attn
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.num_experts = num_experts
        # 回读时 super().__init__ 会把保存的 _top_k 设进来，必须优先用，
        # 否则非默认 top_k 被构造参数默认值 (2) 静默覆盖
        self._top_k = kwargs.get("_top_k", top_k)
        self.rope_scale = rope_scale
        self.rope_factor = rope_factor
        self.rope_theta = rope_theta
        self.architectures = ["GleamLMForCausalLM"]
        # layer_configs 里的变体可能是 class 对象，JSON 无法序列化；
        # 统一转成类名（GleamLMForCausalLM 侧再映射回 class）
        if layer_configs:
            layer_configs = [
                {k: (v.__name__ if isinstance(v, type) else v) for k, v in cfg.items()}
                for cfg in layer_configs
            ]
        self.layer_configs = layer_configs

    @property
    def auto_map(self) -> dict:
        # TRL create_model_from_path 会读 config.auto_map（`cls.__name__ in auto_map`）；
        # AutoModel 注册后无需远程代码映射，返回空 dict 满足 `in` 检查，
        # 且不写入 config.json（null 会让 AutoConfig.from_pretrained 崩溃）
        return {}

    # 用 _top_k 避免 GenerationConfig 把 MoE top_k 误当生成参数
    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = value

    def _get_generation_parameters(self) -> dict[str, any]:
        # transformers <4.31 没有该方法，按需跳过（此时 GenerationConfig
        # 本来就会过滤生成参数）
        if not hasattr(super(), "_get_generation_parameters"):
            return {}
        params = super()._get_generation_parameters()
        params.pop("top_k", None)
        return params

    # core 风格别名（兼容旧代码用 GleamLMConfig.xxx 代替 core 键名）
    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def d_ff(self) -> int:
        return self.intermediate_size

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings


def gleamlm_config_from_core(cfg: dict) -> GleamLMConfig:
    """core 风格 dict → HF GleamLMConfig（显式键名翻译）。"""
    return GleamLMConfig(
        vocab_size=cfg.get("vocab_size", 12002),
        hidden_size=cfg.get("d_model", 512),
        num_hidden_layers=cfg.get("num_layers", 12),
        num_attention_heads=cfg.get("num_heads", 8),
        num_key_value_heads=cfg.get("num_kv_heads", 4),
        intermediate_size=cfg.get("d_ff", 1365),
        max_position_embeddings=cfg.get("max_seq_len", 1024),
        pad_token_id=cfg.get("pad_token_id", 0),
        tie_word_embeddings=cfg.get("tie_weights", True),
        use_flash_attn=cfg.get("use_flash_attn", False),
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", False),
        attn_type=cfg.get("attn_type", "gqa"),
        ffn_type=cfg.get("ffn_type", "mlp"),
        num_experts=cfg.get("num_experts", 8),
        top_k=cfg.get("top_k", 2),
        rope_scale=cfg.get("rope_scale", 1.0),
        rope_factor=cfg.get("rope_factor", 8.0),
        rope_theta=cfg.get("rope_theta", 10000.0),
        layer_configs=cfg.get("layer_configs"),
    )
