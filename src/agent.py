"""
智慧体属性

工作记忆、目标持久化、自我评估、不确定性表达。
与 neural_model 配合，不切除情感与人格。
"""
from typing import List, Optional, Sequence, Tuple


# =============================================================================
# 常量
# =============================================================================

# 延续/追问类提示：沿用上一轮意图
_CONTINUATION_CUES = frozenset({
    "再说一遍", "再说一次", "详细点", "详细说说", "继续", "还有呢", "然后呢",
    "然后", "还有", "多说点", "展开说说", "重复",
})


# =============================================================================
# 工作记忆与目标持久化
# =============================================================================


def is_continuation_prompt(prompt: str) -> bool:
    """是否为延续上一轮的追问（需沿用 last_intent）。"""
    p = prompt.strip().lower()
    if len(p) <= 4 and p in _CONTINUATION_CUES:
        return True
    return p in _CONTINUATION_CUES


def build_context_with_history(
    history: Sequence[Tuple[str, str]],
    current_prompt: str,
    tokenize_fn,
    max_history_turns: int = 2,
    max_tokens_per_turn: int = 25,
) -> str:
    """
    工作记忆：将对话历史与当前输入拼接为完整上下文。
    格式：「上轮：{u}→{m}」+ 当前，限制长度以控制计算量。
    """
    if not history:
        return current_prompt.strip()
    recent = list(history[-max_history_turns:])
    parts: List[str] = []
    for u, m in recent:
        ut = tokenize_fn(u)
        mt = tokenize_fn(m)
        u_str = "".join(ut[:max_tokens_per_turn])
        m_str = "".join(mt[:max_tokens_per_turn])
        parts.append(f"{u_str}→{m_str}")
    prefix = " ".join(parts) + " "
    return (prefix + current_prompt.strip()).strip()


def should_express_uncertainty(mean_confidence: float, threshold: float = 0.32) -> bool:
    """元认知：低置信时表达不确定性（前置「嗯，」）。"""
    return mean_confidence < threshold and mean_confidence > 0.0


def get_continuation_intent(
    prompt: str,
    last_intent: Optional[str],
    last_prompt: Optional[str],
) -> Optional[str]:
    """目标持久化：延续追问时沿用上一轮意图。"""
    if not last_intent:
        return None
    if not is_continuation_prompt(prompt):
        return None
    return last_intent
