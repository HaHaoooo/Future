"""
交互会话编排

纯对话模式：循环 输入 → 推断感官 → deliberate_generate → 打印输出 → 保存模型。
"""
from src.system.sensory_io import SensoryContext, collect_sensory_payload, sensory_payload_text
from src.system.config_types import AppConfig
from src.system.ui import print_model_output
from src.neural_model import NeuralAffectiveModel


def run_interactive_session(model: NeuralAffectiveModel, config: AppConfig) -> None:
    """
    纯对话模式：你与模型的对话。
    智慧体：工作记忆、目标持久化、自我评估、不确定性表达。
    """
    from src.agent import get_continuation_intent
    sensory_context = SensoryContext()

    while True:
        prompt = input("你> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break

        continuation_intent = get_continuation_intent(
            prompt, sensory_context.last_intent, sensory_context.last_prompt,
        )
        sensory_payload = collect_sensory_payload(prompt=prompt, context=sensory_context, config=config)
        trace, thought_report = model.deliberate_generate(
            prompt=prompt,
            sensory_payload=sensory_payload,
            max_len=config.max_len,
            temperature=config.temperature,
            thought_trials=config.thought_trials,
            conversation_history=sensory_context.conversation_history,
            continuation_intent=continuation_intent,
        )
        sensory_context.update_from_thought_report(thought_report)
        intent = thought_report.get("inferred_intent")
        sensory_context.set_last_intent(intent, prompt)

        if not trace.tokens:
            print("模型> （空输出）")
            print("  [提示] 若频繁空输出，可运行 python3 main.py --mode teacher 进行基础对话训练")
            print()
            continue

        model_reply = "".join(trace.tokens).strip()
        sensory_context.push_turn(prompt, model_reply)
        print_model_output(
            trace.tokens,
            thought_report,
            config.show_thought,
            sensory_text=sensory_payload_text(sensory_payload),
        )
        model.save(config.model_path)
