"""
系统启动流程

解析配置 → 加载/创建模型 → 按 mode 分发到 teacher / correct / 交互对话。
"""
from src.system.config import parse_args
from src.system.model_service import load_or_create_model
from src.system.session import run_interactive_session
from src.system.ui import print_banner


def run_app() -> None:
    """主入口：解析配置、加载模型、分发到对应模式。"""
    config = parse_args()
    model = load_or_create_model(config)

    if config.mode == "teacher":
        from teacher import run_teacher_session
        run_teacher_session(model, config)
        return

    if config.mode == "correct":
        from corrector import run_correct_session
        run_correct_session(model, config)
        return

    print_banner()
    run_interactive_session(model, config)
