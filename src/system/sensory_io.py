"""
会话期感官推断

从文本与学习状态自动估计感官 payload，可叠加外部文件。
供 session、teacher、corrector 使用。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.sensory import load_sensor_file, merge_sensory_payloads, payload_to_pairs
from src.system.config_types import AppConfig
from src.utils import clip as _clip


# =============================================================================
# 感官上下文
# =============================================================================


@dataclass
class SensoryContext:
    """会话期感官推断的上下文：对话历史、最近损失、置信度、意图等。"""
    turn_index: int = 0
    recent_avg_loss: float = 1.0
    recent_correction_ratio: float = 0.0
    recent_confidence: float = 0.5
    conversation_history: List[Tuple[str, str]] = field(default_factory=list)
    last_intent: Optional[str] = None
    last_prompt: Optional[str] = None

    def push_turn(self, user_msg: str, model_msg: str) -> None:
        """工作记忆：追加一轮对话，保留最近 4 轮。"""
        self.conversation_history.append((user_msg, model_msg))
        if len(self.conversation_history) > 4:
            self.conversation_history = self.conversation_history[-4:]

    def set_last_intent(self, intent: Optional[str], prompt: Optional[str]) -> None:
        """目标持久化：记录上一轮意图与问句。"""
        self.last_intent = intent
        self.last_prompt = prompt

    def update_from_train_report(self, report: Dict[str, float]) -> None:
        """根据训练报告更新 recent_avg_loss、recent_correction_ratio。"""
        self.turn_index += 1
        avg_loss = float(report.get("avg_loss", self.recent_avg_loss))
        supervised = max(1.0, float(report.get("supervised_updates", 0.0)))
        contrastive = max(0.0, float(report.get("contrastive_updates", 0.0)))
        correction_ratio = min(1.0, contrastive / supervised)
        self.recent_avg_loss = 0.75 * self.recent_avg_loss + 0.25 * avg_loss
        self.recent_correction_ratio = 0.7 * self.recent_correction_ratio + 0.3 * correction_ratio

    def update_from_thought_report(self, thought_report: Dict[str, object]) -> None:
        """根据思考报告更新 recent_confidence。"""
        conf = float(thought_report.get("best_confidence", self.recent_confidence))
        self.recent_confidence = 0.7 * self.recent_confidence + 0.3 * conf


# =============================================================================
# 感官推断与收集
# =============================================================================


def infer_sensory_payload(
    prompt: str,
    context: SensoryContext,
    file_payload: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    从文本特征 + 上轮学习状态推断感官。
    输出 noise/stress/confidence/energy/valence/arousal。
    """
    import re
    file_payload = file_payload or {}
    text = prompt.strip()
    _tok_pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+|[^\w\s]")
    tokens = _tok_pattern.findall(text) or text.split() or [text]
    n_tokens = max(1, len(tokens))
    n_chars = max(1, len(text))
    unique_ratio = len(set(t.lower() for t in tokens)) / n_tokens
    punct_count = sum(1 for ch in text if ch in "!?.,;:()[]{}")
    punct_ratio = punct_count / n_chars
    uppercase_ratio = sum(1 for ch in text if ch.isupper()) / n_chars
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / n_chars
    question_ratio = (text.count("?") + text.count("？")) / max(1, n_chars)

    complexity = _clip((n_tokens - 5) / 15.0, 0.0, 1.0)
    volatility = _clip(2.8 * punct_ratio + 2.2 * uppercase_ratio + 1.2 * digit_ratio, 0.0, 1.0)
    curiosity = _clip(6.0 * question_ratio, 0.0, 1.0)
    lexical_stability = _clip(unique_ratio, 0.0, 1.0)

    stress = _clip(
        0.35 * volatility
        + 0.30 * context.recent_correction_ratio
        + 0.20 * _clip((context.recent_avg_loss - 1.2) / 2.5, 0.0, 1.0)
        + 0.15 * complexity,
        -1.0, 1.0,
    )
    noise = _clip(0.55 * volatility + 0.25 * (1.0 - lexical_stability) + 0.20 * complexity, -1.0, 1.0)
    confidence = _clip(
        0.55 * context.recent_confidence + 0.25 * lexical_stability + 0.20 * (1.0 - context.recent_correction_ratio),
        0.0, 1.0,
    )
    energy = _clip(0.45 * complexity + 0.30 * curiosity + 0.25 * (1.0 - noise), -1.0, 1.0)

    exclam = text.count("!") + text.count("！")
    lt = text.lower()
    positive_cues = sum(1 for w in ("好", "爱", "谢", "开心", "快乐", "喜欢", "棒", "yes", "great", "love") if w in text or w in lt)
    negative_cues = sum(1 for w in ("错", "讨厌", "烦", "no", "bad", "sad", "恨") if w in text or w in lt)
    valence = _clip(
        0.15 * (positive_cues - negative_cues) / max(1, n_tokens) * 10
        - 0.1 * context.recent_correction_ratio
        + 0.08 * (context.recent_confidence - 0.5),
        -1.0, 1.0,
    )
    arousal = _clip(0.3 * exclam / max(1, n_chars) * 20 + 0.2 * curiosity + 0.15 * volatility, -1.0, 1.0)

    inferred = {
        "noise": noise, "stress": stress, "confidence": confidence,
        "energy": energy, "valence": valence, "arousal": arousal,
    }
    return merge_sensory_payloads(inferred, file_payload)


def collect_sensory_payload(
    prompt: str,
    context: SensoryContext,
    config: AppConfig,
    sensory_override: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """合并文件 payload、推断 payload、override。"""
    file_payload = load_sensor_file(config.sensor_file)
    base = infer_sensory_payload(prompt=prompt, context=context, file_payload=file_payload)
    if sensory_override:
        allowed = ("stress", "arousal", "noise", "energy", "valence", "confidence", "force_calm")
        base = {**base, **{k: v for k, v in sensory_override.items() if k in base or k in allowed}}
    return base


def sensory_payload_text(payload: Dict[str, float]) -> str:
    """将 payload 转为可读字符串。"""
    pairs = list(payload_to_pairs(payload))
    return "(无)" if not pairs else ", ".join(pairs)
