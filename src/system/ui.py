"""
界面输出

Banner、模型输出、思考报告的格式化打印。
"""
from typing import Dict, Sequence


def print_banner() -> None:
    """打印启动 Banner。"""
    print("=== Future 对话系统 ===")
    print("纯对话模式：你输入 -> 模型思考与输出。")
    print("已启用：神经网络 + 感官推断 + 情感系统（全模式开启，贴近人类复杂智慧）。")
    print("输入 quit / exit / q 退出。")
    print()


def print_model_output(
    tokens: Sequence[str],
    thought_report: Dict[str, object],
    show_thought: bool,
    sensory_text: str,
) -> None:
    """打印感官、情感、模型输出及可选的思考报告。"""
    print(f"[感官] {sensory_text}")
    if "emotion_state" in thought_report:
        es = thought_report["emotion_state"]
        if isinstance(es, (list, tuple)) and len(es) >= 3:
            print(f"[情感] [{es[0]:.2f}, {es[1]:.2f}, {es[2]:.2f}]")
    print(f"模型> {' '.join(tokens)}")
    if show_thought:
        print(
            "[思考报告] "
            f"trials={thought_report['thought_trials']}, "
            f"best_score={float(thought_report['best_score']):.4f}, "
            f"best_conf={float(thought_report['best_confidence']):.4f}, "
            f"候选长度={thought_report['candidate_lengths']}, "
            f"目标长度={thought_report.get('target_length', '-')}, "
            f"实际长度={thought_report.get('actual_length', '-')}"
        )
