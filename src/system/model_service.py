"""
模型加载与初始化

统一模型 NeuralAffectiveModel：LSTM 或 Transformer 编码器，按 --backend 选择。
"""
import os

from src.system.config_types import AppConfig
from src.neural_model import NeuralAffectiveModel, _TORCH_AVAILABLE


def _is_transformer_checkpoint(path: str) -> bool:
    torch_path = path.replace(".npz", "_transformer.pt") if path.endswith(".npz") else path + "_transformer.pt"
    return os.path.exists(torch_path)


def load_or_create_model(config: AppConfig) -> NeuralAffectiveModel:
    """
    加载或创建统一模型。
    存在 transformer 格式则加载 Transformer；否则加载 LSTM。
    backend=torch 创建 Transformer+大上下文；backend=numpy 创建 LSTM。
    """
    if os.path.exists(config.model_path):
        model = NeuralAffectiveModel.load(config.model_path)
        if config.model_name != "模型" and model.model_name != config.model_name:
            model.model_name = config.model_name
            print(f"[load] 模型名称已更新为「{config.model_name}」")
        if config.creator_name != "创造者" and model.creator_name != config.creator_name:
            model.creator_name = config.creator_name
            print(f"[load] 创造者名称已更新为「{config.creator_name}」")
        enc = "Transformer" if model._use_transformer else "LSTM"
        ctx = f" (上下文 {model.context_max_len})" if model._use_transformer else ""
        print(f"[load] 已加载 {enc} 模型「{model.model_name}」: {config.model_path}{ctx}")
        return model

    for legacy in ("checkpoints/data.json", "checkpoints/neural_affective_model.json"):
        if config.model_path == "checkpoints/model.npz" and os.path.exists(legacy):
            model = NeuralAffectiveModel.load(legacy)
            print(f"[migrate] 从 {legacy} 加载，后续保存到 {config.model_path}")
            return model

    use_torch = config.backend == "torch" and _TORCH_AVAILABLE
    if config.backend == "torch" and not _TORCH_AVAILABLE:
        print("[warn] PyTorch 未安装，回退到 LSTM 后端")
    if _is_transformer_checkpoint(config.model_path) and not use_torch:
        raise RuntimeError("存在 Transformer  checkpoint，请安装 PyTorch 或删除后重建")

    model = NeuralAffectiveModel(
        hidden_dim=config.hidden_dim,
        lr=config.lr,
        seed=config.seed,
        sensory_dim=config.sensory_dim,
        use_transformer=use_torch,
        context_max_len=config.context_max_len,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        model_name=config.model_name,
        creator_name=config.creator_name,
    )
    steps = model.seed_identity_logic(passes_per_step=8)
    enc = "Transformer" if model._use_transformer else "LSTM"
    ctx = f"，上下文 {config.context_max_len}" if model._use_transformer else ""
    print(f"[init] 新 {enc} 模型「{model.model_name}」已创建{ctx}，内置身份逻辑链 {steps} 步。")
    return model
