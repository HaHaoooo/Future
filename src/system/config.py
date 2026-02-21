"""
命令行参数解析

将 argparse 结果转换为 AppConfig，供 runtime 与各模块使用。
"""
import argparse
import os

from src.system.config_types import AppConfig


def parse_args() -> AppConfig:
    """解析命令行，返回 AppConfig。"""
    parser = argparse.ArgumentParser(description="Future interactive learning system")

    # --- 通用 ---
    parser.add_argument("--model", type=str, default="checkpoints/model.npz")
    parser.add_argument("--name", type=str, default="模型",
                        help="模型名称（如：小来、Future），问「你是谁」时回答此名称")
    parser.add_argument("--creator", type=str, default="创造者",
                        help="创造者名称，问「我是谁」/「谁创造了你」时回答此名称")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"],
                        help="numpy=LSTM, torch=Transformer+大上下文")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐层维度")
    parser.add_argument("--context-max-len", type=int, default=8192, help="上下文窗口（仅 torch）")
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--thought-trials", type=int, default=7)
    parser.add_argument("--learning-passes", type=int, default=10)
    parser.add_argument("--replay-steps", type=int, default=320)
    parser.add_argument("--sensory-dim", type=int, default=8)
    parser.add_argument("--sensor-file", type=str, default="")
    parser.add_argument("--show-thought", action="store_true")

    # --- 模式 ---
    parser.add_argument(
        "--mode", type=str, default="interactive",
        choices=["interactive", "teacher", "correct"]
    )

    # --- teacher ---
    parser.add_argument("--teacher-episodes", type=int, default=500)
    parser.add_argument("--teacher-log-every", type=int, default=50)
    parser.add_argument("--teacher-learning-passes", type=int, default=8)
    parser.add_argument("--teacher-replay-steps", type=int, default=160)
    parser.add_argument("--teacher-user-facts", type=str, default="")
    parser.add_argument("--teacher-user-facts-file", type=str, default="")
    parser.add_argument("--no-teacher-chat", action="store_true")
    parser.add_argument("--teacher-chat-max-turns", type=int, default=2)
    parser.add_argument("--teacher-ai-api-key", type=str, default="")
    parser.add_argument("--teacher-ai-model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--teacher-ai-batches", type=int, default=20)
    parser.add_argument("--teacher-reinforce-rounds", type=int, default=3)
    parser.add_argument("--teacher-pass-threshold", type=float, default=0.75)
    parser.add_argument("--teacher-loop", action="store_true",
                        help="循环训练模式：训练完成后自动重新开始，Ctrl+C 停止")

    # --- correct ---
    parser.add_argument("--correct-purge-memory", action="store_true")

    args = parser.parse_args()
    return AppConfig(
        model_path=args.model,
        model_name=args.name,
        creator_name=args.creator,
        backend=args.backend,
        hidden_dim=args.hidden_dim,
        context_max_len=args.context_max_len,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        lr=args.lr,
        seed=args.seed,
        max_len=args.max_len,
        temperature=args.temperature,
        thought_trials=args.thought_trials,
        learning_passes=args.learning_passes,
        replay_steps=args.replay_steps,
        sensory_dim=args.sensory_dim,
        sensor_file=args.sensor_file,
        show_thought=args.show_thought,
        mode=args.mode,
        teacher_episodes=args.teacher_episodes,
        teacher_log_every=args.teacher_log_every,
        teacher_learning_passes=args.teacher_learning_passes,
        teacher_replay_steps=args.teacher_replay_steps,
        teacher_user_facts=args.teacher_user_facts,
        teacher_user_facts_file=args.teacher_user_facts_file,
        teacher_chat_enabled=not args.no_teacher_chat,
        teacher_chat_max_turns=args.teacher_chat_max_turns,
        teacher_ai_api_key=args.teacher_ai_api_key or os.environ.get("GEMINI_API_KEY", ""),
        teacher_ai_model=args.teacher_ai_model,
        teacher_ai_batches=args.teacher_ai_batches,
        teacher_reinforce_rounds=args.teacher_reinforce_rounds,
        teacher_pass_threshold=args.teacher_pass_threshold,
        teacher_loop=args.teacher_loop,
        correct_purge_memory=args.correct_purge_memory,
    )
