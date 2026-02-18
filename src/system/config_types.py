"""
配置类型定义

所有命令行参数对应的数据结构，供 config、runtime、各模块使用。
"""
from dataclasses import dataclass


@dataclass
class AppConfig:
    """应用配置，由 parse_args 填充。"""

    # --- 模型与训练 ---
    model_path: str = "checkpoints/data.npz"
    backend: str = "numpy"  # numpy | torch
    hidden_dim: int = 128
    context_max_len: int = 8192  # 大上下文窗口（仅 torch 后端生效）
    transformer_layers: int = 6
    transformer_heads: int = 8
    lr: float = 0.04
    seed: int = 42
    learning_passes: int = 10
    replay_steps: int = 320

    # --- 生成控制 ---
    max_len: int = 0
    temperature: float = 0.9
    thought_trials: int = 7

    # --- 感官与展示 ---
    sensory_dim: int = 8
    sensor_file: str = ""
    show_thought: bool = False

    # --- 模式 ---
    mode: str = "interactive"

    # --- teacher ---
    teacher_episodes: int = 500
    teacher_log_every: int = 50
    teacher_learning_passes: int = 8
    teacher_replay_steps: int = 160
    teacher_user_facts: str = ""
    teacher_user_facts_file: str = ""
    teacher_chat_enabled: bool = True
    teacher_chat_max_turns: int = 2
    teacher_ai_api_key: str = ""
    teacher_ai_model: str = "gemini-2.0-flash"
    teacher_ai_batches: int = 20
    teacher_reinforce_rounds: int = 3
    teacher_pass_threshold: float = 0.75
    teacher_loop: bool = False

    # --- correct ---
    correct_purge_memory: bool = False
