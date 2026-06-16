from .xfind_model import XfindModel
from .xfind_config import XfindConfig, get_args


def load_model_for_inference(model_path, device='cuda', checkpoint=None):
    """从 checkpoint 加载模型用于推理/评估
    三级回退：checkpoint['args'] → checkpoint['config'] → 默认值
    """
    import torch

    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location=device)

    # 从 checkpoint 获取配置
    if 'args' in checkpoint:
        args = checkpoint['args']
        config = {
            'vocab_size': getattr(args, 'vocab_size', 32000),
            'd_model': getattr(args, 'd_model', 512),
            'num_layers': getattr(args, 'num_layers', 8),
            'num_heads': getattr(args, 'num_heads', 8),
            'num_kv_heads': getattr(args, 'num_kv_heads', 4),
            'd_ff': getattr(args, 'd_ff', 1365),
            'dropout': 0.0,
            'max_seq_len': getattr(args, 'max_seq_len', 1024),
            'pad_token_id': 0,
            'tie_weights': False,
        }
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'vocab_size': 32000, 'd_model': 512, 'num_layers': 8,
            'num_heads': 8, 'num_kv_heads': 4, 'd_ff': 1365,
            'dropout': 0.0, 'max_seq_len': 1024, 'pad_token_id': 0,
            'tie_weights': False,
        }

    # 推理固定配置
    config['dropout'] = 0.0
    config['tie_weights'] = False

    # 构建模型
    model = XfindModel(**config).to(device)

    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # FP16 量化模型自动转换
    if checkpoint.get('dtype') == 'float16':
        model = model.half()

    return model, config
