"""
公共工具模块

消除跨文件重复定义，不导入模型模块以避免循环依赖。
"""
from typing import Sequence


def clip(x: float, low: float = -1.0, high: float = 1.0) -> float:
    """将 x 限制在 [low, high] 区间内。"""
    return max(low, min(high, x))


def flatten_tokens(tokens: Sequence[str]) -> str:
    """将 token 序列拼接为字符串，空序列返回空串。"""
    return "".join(tokens).strip() if tokens else ""
