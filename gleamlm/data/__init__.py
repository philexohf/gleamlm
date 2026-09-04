"""GleamLM data module — 预处理引擎 + 管线编排 + 打包 + 数据集加载。"""

from gleamlm.data.dataset import (
    estimate_tokens_per_row,
    lm_collate,
    tokenize_and_group,
)
from gleamlm.data.pipeline import run_pipeline
from gleamlm.data.dpo_data import DPODataset, dpad_collate
from gleamlm.data.preprocess import (
    MinHash,
    MinHashIndex,
    SimHashIndex,
    clean_file,
    clean_text,
    compute_stats,
    dedup_file,
    filter_qa,
    hamming_distance,
    minhash_dedup_file,
    normalize,
    parse_qa,
    score_quality_file,
    score_text,
    simhash,
    stream_split,
)
from gleamlm.data.rl_data import RLHFDataset, tokenize_prompts
from gleamlm.data.sft_data import SFTDataset, SYSTEM_PROMPTS

__all__ = [
    # dataset
    "tokenize_and_group",
    "estimate_tokens_per_row",
    "lm_collate",
    # pipeline
    "run_pipeline",
    # preprocess
    "normalize",
    "clean_text",
    "clean_file",
    "simhash",
    "hamming_distance",
    "SimHashIndex",
    "MinHash",
    "MinHashIndex",
    "dedup_file",
    "minhash_dedup_file",
    "parse_qa",
    "filter_qa",
    "score_text",
    "score_quality_file",
    "stream_split",
    "compute_stats",
    # post-train datasets
    "SFTDataset",
    "SYSTEM_PROMPTS",
    "DPODataset",
    "dpad_collate",
    "RLHFDataset",
    "tokenize_prompts",
]
