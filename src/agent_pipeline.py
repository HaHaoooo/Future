"""
智慧体推理流水线

显式逻辑化流程：感知 → 理解 → 情感 → 回忆 → 生成。
每个阶段产出明确，便于追踪与调试。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 流水线阶段常量（便于日志与追踪）
STAGE_WORKING_MEMORY = "working_memory"
STAGE_UNDERSTANDING = "understanding"
STAGE_EMOTION = "emotion"
STAGE_RECALL = "recall"
STAGE_PERSPECTIVE = "perspective"
STAGE_LOGIC = "logic"


# =============================================================================
# 推理上下文
# =============================================================================


@dataclass
class AgentReasoningContext:
    """
    智慧体推理上下文：流水线各阶段的显式产出。
    生成前一次性准备，生成循环中按步使用。
    """
    full_prompt: str = ""
    prompt_tokens: List[str] = field(default_factory=list)
    prompt_ids: List[int] = field(default_factory=list)
    intent: Optional[str] = None
    sensory_vec: List[float] = field(default_factory=list)
    emotion_vec: List[float] = field(default_factory=list)
    bias_ids: List[int] = field(default_factory=list)
    retrieval_boosts: Dict[int, float] = field(default_factory=dict)
    min_len: int = 1
    target_len: int = 8
    hard_max_len: Optional[int] = None

    def stages_executed(self) -> List[str]:
        """返回已执行阶段名（便于调试）。"""
        out = [STAGE_WORKING_MEMORY]
        if self.intent is not None or self.prompt_tokens:
            out.append(STAGE_UNDERSTANDING)
        if self.emotion_vec:
            out.append(STAGE_EMOTION)
        if self.retrieval_boosts or self.bias_ids:
            out.append(STAGE_RECALL)
        return out


# =============================================================================
# 准备推理上下文
# =============================================================================


def prepare_reasoning_context(
    model: Any,
    prompt: str,
    sensory_payload: Optional[Dict[str, float]] = None,
    conversation_history: Optional[Sequence[Tuple[str, str]]] = None,
    continuation_intent: Optional[str] = None,
    max_len: int = 0,
) -> AgentReasoningContext:
    """
    按逻辑顺序准备智慧体推理上下文：
    1. 工作记忆：拼接历史与当前输入
    2. 理解：推断意图（或沿用延续追问的上一轮意图）
    3. 情感：感官更新 + 输入语义→情感影响
    4. 回忆：意图偏置 token + 检索增强
    5. 长度预算：自适应 min/target/max
    """
    sensory_payload = sensory_payload or {}
    from src.agent import build_context_with_history
    from src.sensory import encode_sensory_vector

    full_prompt = build_context_with_history(
        conversation_history or [], prompt, model.tokenize,
        max_history_turns=2, max_tokens_per_turn=20,
    )
    prompt_tokens = model.tokenize(full_prompt)
    for t in prompt_tokens:
        model.ensure_token(t)
    prompt_ids = [model.token_to_id[t] for t in prompt_tokens]

    intent = continuation_intent if continuation_intent else model._infer_intent(prompt_tokens)

    model.emotion_core.update_from_sensory(sensory_payload)
    val_imp, ar_imp = model._infer_emotional_impact_from_input(prompt_tokens, intent)
    model.emotion_core.update_from_input_meaning(val_imp, ar_imp)
    sensory_vec = encode_sensory_vector(sensory_payload, model.sensory_dim)
    emotion_vec = list(model._effective_emotion())

    bias_ids = model._get_intent_bias_tokens(intent) if intent else []
    retrieval_boosts = model._retrieve_from_memory(prompt_ids, bias_ids, top_k=5)

    hard_max_len: Optional[int]
    if max_len is None or max_len <= 0:
        hard_max_len = None
    else:
        hard_max_len = max(2, max_len)
    min_len, target_len = model._adaptive_length_budget(
        prompt_tokens=prompt_tokens,
        sensory_payload=sensory_payload,
        hard_max_len=hard_max_len,
    )

    return AgentReasoningContext(
        full_prompt=full_prompt,
        prompt_tokens=list(prompt_tokens),
        prompt_ids=prompt_ids,
        intent=intent,
        sensory_vec=sensory_vec,
        emotion_vec=emotion_vec,
        bias_ids=bias_ids,
        retrieval_boosts=retrieval_boosts,
        min_len=min_len,
        target_len=target_len,
        hard_max_len=hard_max_len,
    )
