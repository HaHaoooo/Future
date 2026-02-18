"""
Future: 神经网络交互学习系统

入口模块。解析配置、加载模型，根据 --mode 分发到交互对话 / teacher / correct。
"""
from src.system.runtime import run_app


if __name__ == "__main__":
    run_app()
