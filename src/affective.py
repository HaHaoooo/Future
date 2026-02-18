"""
内在状态（情感）模块

三维模糊向量，无语义预设。状态由感官、推理、学习等多源信号驱动，
具体含义由模型通过训练自主学习运用。
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from src.utils import clip as _clip


# =============================================================================
# 数据类型
# =============================================================================


@dataclass
class EmotionState:
    """三维模糊内在状态，不做语义标注，由网络自主解读。"""
    dim0: float = 0.0
    dim1: float = 0.0
    dim2: float = 0.0


# =============================================================================
# 情感核心
# =============================================================================


class FuzzyEmotionCore:
    """
    内在状态动力学：state = decay * state + alpha * influence。
    不预设「置信度→愉悦」等词汇映射，多源信号混合后微弱推动，
    含义由模型在学习中自行形成。
    """

    def __init__(self, state: Optional[EmotionState] = None) -> None:
        self.state = state or EmotionState()

    def as_vector(self) -> Tuple[float, float, float]:
        """返回当前三维状态向量。"""
        return (self.state.dim0, self.state.dim1, self.state.dim2)

    def integrate(
        self,
        sensory: Optional[Dict[str, float]] = None,
        inference_signal: Optional[Tuple[float, float]] = None,
        learning_signal: Optional[Sequence[float]] = None,
    ) -> None:
        """
        统一内在动力：多源信号混合后更新 state。
        - sensory: noise, stress, confidence, energy 等
        - inference_signal: (mean_confidence, uncertainty)
        - learning_signal: [correctness_ratio, correction_ratio, loss_improvement, ...]
        force_calm 时 decay 加快、alpha 增大，快速拉回平静。
        """
        s0, s1, s2 = self.state.dim0, self.state.dim1, self.state.dim2
        force_calm = float(sensory.get("force_calm", 0.0)) if sensory else 0.0
        decay = 0.75 if force_calm > 0.5 else 0.94
        alpha = 0.12 if force_calm > 0.5 else 0.09
        inf0, inf1, inf2 = 0.0, 0.0, 0.0

        if sensory:
            n = _clip(float(sensory.get("noise", 0.0)), -1.0, 1.0)
            st = _clip(float(sensory.get("stress", 0.0)), -1.0, 1.0)
            c = _clip(float(sensory.get("confidence", 0.5)), 0.0, 1.0)
            e = _clip(float(sensory.get("energy", 0.0)), -1.0, 1.0)
            inf0 += 0.35 * (c - 0.5) - 0.22 * st
            inf1 += 0.28 * (n + e)
            inf2 += 0.22 * (c - n)

        if inference_signal:
            conf, unc = inference_signal[0], inference_signal[1]
            inf0 += 0.28 * (conf - 0.5)
            inf1 += 0.2 * unc
            inf2 += 0.22 * (conf - unc)

        if learning_signal and len(learning_signal) >= 3:
            acc = _clip(learning_signal[0], 0.0, 1.0)
            corr = _clip(learning_signal[1], 0.0, 1.0)
            improve = _clip(learning_signal[2], -1.0, 1.0)
            replay = _clip(learning_signal[3], 0.0, 1.0) if len(learning_signal) > 3 else 0.0
            contrastive = _clip(learning_signal[4], 0.0, 1.0) if len(learning_signal) > 4 else 0.0
            q = 0.4 * acc + 0.3 * (1 - corr) + 0.3 * improve
            pressure = 0.4 * corr + 0.3 * contrastive + 0.2 * replay
            inf0 += 0.32 * (q - 0.5)
            inf1 += 0.22 * pressure
            inf2 += 0.2 * (q - 0.5)

        self.state.dim0 = _clip(s0 * decay + alpha * inf0, -1.0, 1.0)
        self.state.dim1 = _clip(s1 * decay + alpha * inf1, -1.0, 1.0)
        self.state.dim2 = _clip(s2 * decay + alpha * inf2, -1.0, 1.0)

    def update_from_sensory(self, sensory_payload: Dict[str, float]) -> None:
        """仅由感官更新。"""
        self.integrate(sensory=sensory_payload)

    def update_from_input_meaning(
        self,
        valence_influence: float,
        arousal_influence: float,
        alpha: float = 0.15,
    ) -> None:
        """
        根据输入语义与逻辑链判断，推动情感变化。
        与 integrate 的 sensory 中 valence/arousal 叠加，形成「理解→情感」链。
        """
        s0, s1, s2 = self.state.dim0, self.state.dim1, self.state.dim2
        decay = 0.96
        v = _clip(valence_influence, -1.0, 1.0)
        a = _clip(arousal_influence, -1.0, 1.0)
        inf0 = 0.4 * v
        inf1 = 0.35 * a
        inf2 = 0.25 * v
        self.state.dim0 = _clip(s0 * decay + alpha * inf0, -1.0, 1.0)
        self.state.dim1 = _clip(s1 * decay + alpha * inf1, -1.0, 1.0)
        self.state.dim2 = _clip(s2 * decay + alpha * inf2, -1.0, 1.0)

    def update_from_inference(self, mean_confidence: float, uncertainty: float) -> None:
        """由推理置信度与不确定性更新。"""
        self.integrate(inference_signal=(mean_confidence, uncertainty))

    def update_from_learning(
        self,
        correctness_ratio: float,
        correction_ratio: float,
        loss_improvement: float,
        replay_intensity: float = 0.0,
        contrastive_intensity: float = 0.0,
    ) -> None:
        """由学习反馈更新。"""
        self.integrate(learning_signal=(
            correctness_ratio, correction_ratio, loss_improvement,
            replay_intensity, contrastive_intensity,
        ))
