# -*- coding: utf-8 -*-
"""
核心神经网络模型（统一整合版）

双编码器：LSTM（numpy） / Transformer（torch，大上下文 8K+）
结构：嵌入 + 感官 + 情感 → [LSTM 或 Transformer] → 组织层 → logits
功能：memory 检索、意图偏置、情感/视角/逻辑链、deliberate 生成、即时学习。

LSTM 初始化优化（对齐现代最佳实践）：
  - 遗忘门偏置 = 1.0（Jozefowicz 2015 / PyTorch 默认），初期倾向记住信息
  - 隐→隐权重正交初始化（Saxe 2014），改善长距离梯度流
"""
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.affective import EmotionState, FuzzyEmotionCore
from src.perspective import PerspectiveCore, FIRST_PERSON_TOKENS
from src.sensory import encode_sensory_vector

if _TORCH_AVAILABLE:
    from src.transformer_core import FutureTransformer


# =============================================================================
# 训练常量
# =============================================================================

_SQRT_2_PI = np.float64(np.sqrt(2.0 / np.pi))
MAX_GRAD_NORM = 5.0   # 梯度裁剪，防 BPTT 爆炸
ORG_LAYER_SCALE = 0.45  # 组织层对 logits 贡献
ATTN_SCALE = 0.35   # 注意力对隐态贡献
EMBED_SCALE = 0.08  # 嵌入与权重初始化缩放


# --- 激活与采样 ---
def _np_orthogonal(rows: int, cols: int) -> np.ndarray:
    """正交初始化 (Saxe et al. 2014)，现代 RNN 标准做法，改善梯度流。"""
    a = np.random.randn(rows, cols).astype(np.float64)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    return u if rows >= cols else vt


def _np_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_PI * (x + 0.044715 * x ** 3)))


def _np_gelu_derivative(x: np.ndarray) -> np.ndarray:
    t = _SQRT_2_PI * (x + 0.044715 * x ** 3)
    tanh_t = np.tanh(t)
    return 0.5 * (1.0 + tanh_t) + 0.5 * x * (1.0 - tanh_t ** 2) * _SQRT_2_PI * (1.0 + 0.134145 * x ** 2)


def _np_rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


def _np_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(temperature, 1e-6)
    scaled = logits / t
    shifted = scaled - np.max(scaled)
    exps = np.exp(shifted)
    return exps / exps.sum()


def _sample_from_probs(probs) -> int:
    if isinstance(probs, np.ndarray):
        p = probs / probs.sum()
        return int(np.random.choice(len(p), p=p))
    r = random.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1


# =============================================================================
# 数据类型
# =============================================================================


@dataclass
class GenerationTrace:
    tokens: List[str]
    token_ids: List[int]
    prev_ids: List[int]
    confidences: List[float]
    sensory_vector: List[float]
    emotion_vector: List[float]
    prompt_token_ids: Optional[List[int]] = None  # 人类输入，LSTM 编码理解用


# =============================================================================
# 主模型
# =============================================================================


class NeuralAffectiveModel:
    """
    统一情感模型：LSTM 或 Transformer 编码器。
    功能：memory 检索、意图偏置、情感/视角/逻辑链、deliberate 生成、即时学习。
    """
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(
        self,
        hidden_dim: int = 48,
        lr: float = 0.04,
        seed: int = 42,
        sensory_dim: int = 8,
        max_memory: int = 0,
        use_transformer: bool = False,
        context_max_len: int = 8192,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        model_name: str = "模型",
        creator_name: str = "创造者",
    ) -> None:
        random.seed(seed)
        self.model_name = model_name
        self.creator_name = creator_name
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.sensory_dim = sensory_dim
        self.emotion_dim = 3
        self.max_memory = max_memory
        self.context_max_len = context_max_len
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self._use_transformer = use_transformer and _TORCH_AVAILABLE

        self.memory: List[Tuple[Any, int, float, List[float], List[float]]] = []
        self._memory_index: Dict[int, List[int]] = {}
        self.irrelevant_memory: Dict[str, Dict[str, Any]] = {}
        self.emotion_core = FuzzyEmotionCore()
        self.perspective_core = PerspectiveCore()
        self.identity_drive: List[float] = [0.045, 0.045, 0.045]
        self._identity_temp_damp: float = 0.97

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for t in [self.PAD, self.BOS, self.EOS, self.UNK]:
            self._add_token(t)

        self.input_dim = self.hidden_dim + self.sensory_dim + self.emotion_dim
        self.org_dim = min(32, self.hidden_dim)

        if self._use_transformer:
            self._init_transformer()
        else:
            self.E = np.random.randn(self.vocab_size, self.hidden_dim).astype(np.float64) * EMBED_SCALE
            self._init_lstm()
        self._init_org_layer()
        if not self._use_transformer:
            self.W2 = np.random.randn(self.hidden_dim, self.vocab_size).astype(np.float64) * EMBED_SCALE
            self.b2 = np.zeros(self.vocab_size, dtype=np.float64)
            self.W_emotion_logits = np.random.randn(self.emotion_dim, self.vocab_size).astype(np.float64) * EMBED_SCALE

    # 内置语言逻辑链：语义分组 + 多问法泛化（同一意图多种问法 → 同一答案）
    # 第一视角：存在感链条，模型知道「我存在」
    _EXISTENCE_CHAINS: List[Tuple[str, str]] = [
        ("你存在吗", "我存在"),
        ("你在吗", "我在"),
        ("你知道你自己吗", "我知道我存在"),
    ]
    _IDENTITY_PROMPTS: List[str] = [
        "你是谁", "你是谁？", "你是谁啊", "你叫啥", "你叫什么", "你叫什么名字",
        "你谁啊", "你哪位", "怎么称呼你", "how are you called", "what's your name",
    ]
    _CREATOR_PROMPTS: List[str] = [
        "我是谁", "我是谁？", "你主人是谁", "谁创造了你", "谁造的", "创造者是谁",
    ]
    _SEMANTIC_GROUPS_BASE: List[Tuple[str, str, List[str]]] = [
        ("greeting", "你好", [
            "你好", "嗨", "hello", "hi", "hey",
        ]),
        ("greeting_how", "我很好", [
            "你好吗", "你好吗？", "怎么样", "你怎么样", "how are you", "你好不",
        ]),
        ("remember", "记得", [
            "你记得我吗", "记得我吗", "你还记得我吗",
        ]),
        ("logic_because", "所以", [
            "因为所以", "因为...所以", "因果关系", "因为什么所以", "用因为所以造句",
        ]),
        ("logic_if", "就", [
            "如果就", "如果...就", "条件关系", "如果怎样就", "用如果就造句",
        ]),
        ("logic_although", "但是", [
            "虽然但是", "虽然...但是", "转折关系", "虽然怎样但是",
        ]),
    ]

    def _get_semantic_groups(self) -> List[Tuple[str, str, List[str]]]:
        """返回语义组，identity 和 creator 使用自定义名称。"""
        return [
            ("identity", f"我是{self.model_name}", self._IDENTITY_PROMPTS),
            ("creator", self.creator_name, self._CREATOR_PROMPTS),
        ] + self._SEMANTIC_GROUPS_BASE

    def seed_identity_logic(self, passes_per_step: int = 2, epochs: int = 1) -> int:
        """
        预置模型的语言逻辑链到记忆并训练权重。
        即学模式：高学习率、少轮次，一遍即记。
        Transformer 模式会先预注册所有 token，避免训练中动态扩展导致梯度不匹配。
        """
        neutral_s = [0.0] * self.sensory_dim
        neutral_e = [0.0] * self.emotion_dim
        bos_id = self.token_to_id[self.BOS]
        steps = 0
        all_chains: List[Tuple[str, List[str]]] = [
            (ans, [p]) for p, ans in self._EXISTENCE_CHAINS
        ]
        for _intent, answer_str, prompt_list in self._get_semantic_groups():
            all_chains.append((answer_str, prompt_list))
        if self._use_transformer:
            all_tokens = set()
            for answer_str, prompt_list in all_chains:
                for prompt_str in prompt_list:
                    for t in self.tokenize(prompt_str) + self.tokenize(answer_str):
                        all_tokens.add(t)
            for t in all_tokens:
                self.ensure_token(t)
        eos_id = self.token_to_id[self.EOS]
        for _ in range(epochs):
            for answer_str, prompt_list in all_chains:
                for prompt_str in prompt_list:
                    pt = self.tokenize(prompt_str)
                    at = self.tokenize(answer_str)
                    for t in pt + at:
                        self.ensure_token(t)
                    if not at:
                        continue
                    prompt_ids = [self.token_to_id[t] for t in pt]
                    answer_ids = [self.token_to_id[t] for t in at]
                    prefix: List[int] = []
                    for target_id in answer_ids:
                        context_ids = prompt_ids + [bos_id] + prefix
                        for _ in range(passes_per_step):
                            self.train_one(
                                context_ids,
                                target_id,
                                sensory_vec=neutral_s,
                                emotion_vec=neutral_e,
                                lr=self.lr * 1.2,
                            )
                        self._remember(
                            context_ids,
                            target_id,
                            priority=1.5,
                            sensory_vec=neutral_s,
                            emotion_vec=neutral_e,
                        )
                        prefix.append(target_id)
                        steps += 1
                    # 答案末尾训练 EOS：让模型学会在回答完后停止
                    eos_context = prompt_ids + [bos_id] + prefix
                    for _ in range(passes_per_step):
                        self.train_one(
                            eos_context, eos_id,
                            sensory_vec=neutral_s,
                            emotion_vec=neutral_e,
                            lr=self.lr * 1.2,
                        )
                    steps += 1
        return steps

    # 输入语义→情感：意图与逻辑链判断后，驱动情感变化
    _INTENT_EMOTIONAL_IMPACT: Dict[str, Tuple[float, float]] = {
        "greeting": (0.25, 0.1),
        "greeting_how": (0.2, 0.08),
        "remember": (0.3, 0.12),
        "identity": (0.05, 0.0),
        "creator": (0.2, 0.05),
        "logic_because": (0.0, 0.06),
        "logic_if": (0.0, 0.05),
        "logic_although": (0.0, 0.04),
    }
    _SEMANTIC_VALENCE_CUES: Dict[str, float] = {
        "谢": 0.35, "爱": 0.4, "喜欢": 0.3, "好": 0.15, "棒": 0.3, "开心": 0.35, "快乐": 0.3,
        "笨": -0.4, "错": -0.25, "讨厌": -0.35, "烦": -0.3, "恨": -0.4, "坏": -0.3,
    }
    _SEMANTIC_AROUSAL_CUES: Dict[str, float] = {
        "为什么": 0.2, "怎么": 0.15, "?": 0.1, "？": 0.1, "!": 0.15, "！": 0.15,
    }

    def _infer_emotional_impact_from_input(
        self, prompt_tokens: Sequence[str], intent: Optional[str]
    ) -> Tuple[float, float]:
        """
        根据输入意思（意图+语义线索）判断情感影响，涉及逻辑链。
        结果驱动 emotion_core 变化，形成「理解输入→情感推演」。
        """
        valence, arousal = 0.0, 0.0
        if intent and intent in self._INTENT_EMOTIONAL_IMPACT:
            v, a = self._INTENT_EMOTIONAL_IMPACT[intent]
            valence += v
            arousal += a
        joined = "".join(prompt_tokens).lower()
        for cue, w in self._SEMANTIC_VALENCE_CUES.items():
            if cue in joined:
                valence += w * 0.6
        for cue, w in self._SEMANTIC_AROUSAL_CUES.items():
            if cue in joined:
                arousal += w
        return (max(-1.0, min(1.0, valence)), max(-1.0, min(1.0, arousal)))

    def _infer_intent(self, prompt_tokens: Sequence[str]) -> Optional[str]:
        """
        意图推理：从问句关键词推断语义意图，支撑泛化。
        采用 token 重叠 + 短语匹配，保持人格：仅 logit 软偏置，不覆盖情感。
        """
        if not prompt_tokens:
            return None
        tok_set = frozenset(t.lower() for t in prompt_tokens)
        joined = "".join(prompt_tokens).lower()
        # 短语级匹配（更精准）：完整问句优先
        if "我是谁" in joined or "谁创造了你" in joined or "创造者" in joined:
            return "creator"
        if "你是谁" in joined or "你叫" in joined or "你叫什么" in joined:
            return "identity"
        if "你好吗" in joined or "怎么样" in joined:
            return "greeting_how"
        # creator：我+谁、或 造/创/缔造/主人
        if ("我" in tok_set or "me" in tok_set) and ("谁" in tok_set or "who" in tok_set):
            if "你" not in tok_set:
                return "creator"
        if any(k in tok_set for k in {"造", "创", "缔造", "主人", "创造"}):
            return "creator"
        # identity：你+(谁/叫/名字/啥) 或 name/called（英文问名）
        if "你" in tok_set and any(k in tok_set for k in {"谁", "叫", "名字", "啥", "称呼", "哪位"}):
            return "identity"
        if any(k in tok_set for k in {"what", "name", "called"}) and "how" not in tok_set:
            return "identity"
        # greeting_how：好+吗、怎么、how are you
        if "吗" in tok_set and "好" in tok_set:
            return "greeting_how"
        if any(k in tok_set for k in {"怎么", "怎么样", "how", "are", "you"}):
            return "greeting_how"
        # greeting：短问好
        if len(prompt_tokens) <= 2 and any(k in tok_set for k in {"好", "嗨", "hello", "hi", "hey"}):
            return "greeting"
        # remember
        if "记" in tok_set and "得" in tok_set:
            return "remember"
        # 逻辑链条：因为所以、如果就、虽然但是
        if "因" in tok_set and "为" in tok_set:
            return "logic_because"
        if "所" in tok_set and "以" in tok_set:
            return "logic_because"
        if "如" in tok_set and "果" in tok_set:
            return "logic_if"
        if "虽" in tok_set and "然" in tok_set:
            return "logic_although"
        return None

    # 逻辑连接词链：因果/条件/转折/递进（整合加强）
    _LOGIC_CONNECTIVE_CHAINS: Dict[str, List[str]] = {
        "因为": ["所", "以", "所以", "就"],
        "如果": ["就", "那"],
        "虽然": ["但", "是", "但是"],
        "由于": ["所", "以", "因此"],
        "不但": ["而且", "还"],
        "既": ["又", "也"],
    }

    def _get_logic_connective_boosts(self, out_tokens: Sequence[str], strength: float = 0.35) -> Optional[Dict[int, float]]:
        """
        逻辑衔接偏置：根据已生成内容预测下一词，符合因果/条件/转折链。
        最新 AI 技术：利用语言规律做软引导，不覆盖人格。
        """
        if not out_tokens:
            return None
        last_few = "".join(out_tokens[-6:])
        boosts: Dict[int, float] = {}
        for trigger, follow_tokens in self._LOGIC_CONNECTIVE_CHAINS.items():
            if trigger in last_few:
                for tok in follow_tokens:
                    self.ensure_token(tok)
                    tid = self.token_to_id.get(tok)
                    if tid is not None and 0 <= tid:
                        boosts[tid] = boosts.get(tid, 0) + strength / len(follow_tokens)
        return boosts if boosts else None

    def _get_intent_bias_tokens(self, intent: str) -> List[int]:
        """意图 → 目标答案的 token ids，用于推理偏置（软引导，不覆盖人格）。"""
        for ik, answer_str, _ in self._get_semantic_groups():
            if ik == intent:
                at = self.tokenize(answer_str)
                for t in at:
                    self.ensure_token(t)
                return [self.token_to_id[t] for t in at]
        return []

    def _retrieve_from_memory(
        self,
        prompt_ids: Sequence[int],
        intent_bias_ids: Optional[Sequence[int]],
        top_k: int = 5,
    ) -> Dict[int, float]:
        """检索增强：倒排索引查找，重叠率归一化 + 新颖性奖励。"""
        if not self.memory:
            return {}
        prompt_set = set(prompt_ids)
        intent_set = set(intent_bias_ids) if intent_bias_ids else set()
        n_prompt = max(1, len(prompt_set))
        candidate_indices: set = set()
        for pid in prompt_set:
            if pid in self._memory_index:
                candidate_indices.update(self._memory_index[pid])
        scored: List[Tuple[int, float, float, float]] = []
        for idx in candidate_indices:
            if idx >= len(self.memory):
                continue
            ctx_or_prev, target_id, priority, _, _ = self.memory[idx]
            ctx = list(ctx_or_prev) if isinstance(ctx_or_prev, (list, tuple)) else [ctx_or_prev]
            ctx_set = set(ctx)
            overlap_count = len(prompt_set & ctx_set)
            overlap = overlap_count / n_prompt
            ctx_overlap = overlap_count / max(1, len(ctx_set))
            combined_overlap = 0.6 * overlap + 0.4 * ctx_overlap
            recency = (idx + 1) / len(self.memory)
            if combined_overlap >= 0.15 or target_id in intent_set:
                scored.append((target_id, combined_overlap, float(priority), recency))
        if not scored:
            return {}
        scored.sort(key=lambda x: (x[1] + 0.1 * x[3], x[2]), reverse=True)
        boosts: Dict[int, float] = {}
        for target_id, overlap, prio, recency in scored[:top_k]:
            boost = 0.07 * overlap + 0.04 * min(prio, 2.0) + 0.02 * recency
            boosts[target_id] = min(0.35, boosts.get(target_id, 0.0) + boost)
        return boosts

    def _get_perspective_boosts(
        self,
        prompt_tokens: Sequence[str],
        step: int,
    ) -> Dict[int, float]:
        """
        视角偏置：第一视角（存在感）必备 + 其他视角按概念自整理
        第一视角：我/我的 等，模型知道「我存在」
        """
        boosts: Dict[int, float] = {}
        pc = self.perspective_core
        # 第一视角：基线存在感 + 自指时增强
        strength = pc.get_first_person_strength(prompt_tokens) * (1.0 - step * 0.08)
        if strength > 0.02:
            for t in FIRST_PERSON_TOKENS:
                self.ensure_token(t)
                tid = self.token_to_id.get(t)
                if tid is not None:
                    boosts[tid] = boosts.get(tid, 0.0) + strength
        # 其他视角：按概念自整理的 token 袋
        learned = pc.get_learned_boosts(prompt_tokens, strength=0.1 * (1.0 - step * 0.1))
        for tid, b in learned.items():
            boosts[tid] = boosts.get(tid, 0.0) + b
        return boosts

    def _intent_satisfied(self, intent: Optional[str], tokens: Sequence[str]) -> bool:
        """自我评估：意图是否在输出中体现"""
        if not intent or not tokens:
            return True
        out_set = set(t.lower() for t in tokens)
        if intent == "identity":
            return any(c in out_set for c in self.model_name) or "我" in out_set
        if intent == "creator":
            return any(c in out_set for c in self.creator_name) or "造" in out_set
        if intent in ("greeting", "greeting_how"):
            return "好" in out_set or "你好" in "".join(tokens)
        return True

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _rand_small(self) -> float:
        return random.uniform(-0.1, 0.1)

    def _effective_emotion(self) -> List[float]:
        """情感 + 减熵者身份融合。身份为永久引力，不随情境衰减。"""
        base = list(self.emotion_core.as_vector())
        idv = self.identity_drive[: self.emotion_dim]
        idv = idv + [0.0] * (self.emotion_dim - len(idv))
        return [base[i] + 0.22 * idv[i] for i in range(self.emotion_dim)]

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def ensure_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = self._add_token(token)
        self._expand_parameters_for_new_token()
        return idx

    def _init_transformer(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        self._transformer = FutureTransformer(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            sensory_dim=self.sensory_dim,
            emotion_dim=self.emotion_dim,
            num_layers=self.transformer_layers,
            num_heads=self.transformer_heads,
            max_seq_len=self.context_max_len,
            dropout=0.05,
        )
        self._w2 = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self._b2 = nn.Parameter(torch.zeros(self.vocab_size))
        self._w_emotion = nn.Linear(self.emotion_dim, self.vocab_size, bias=False)
        nn.init.normal_(self._w2.weight, std=EMBED_SCALE)
        nn.init.normal_(self._w_emotion.weight, std=EMBED_SCALE)
        self._transformer.to(self._device)
        self._w2.to(self._device)
        self._w_emotion.to(self._device)

    def _init_lstm(self) -> None:
        d, inp = self.hidden_dim, self.input_dim
        scale = EMBED_SCALE
        self.lstm_Wxi = np.random.randn(inp, d).astype(np.float64) * scale
        self.lstm_Wxf = np.random.randn(inp, d).astype(np.float64) * scale
        self.lstm_Wxg = np.random.randn(inp, d).astype(np.float64) * scale
        self.lstm_Wxo = np.random.randn(inp, d).astype(np.float64) * scale
        # 正交初始化隐→隐权重 (Saxe 2014)，现代 LSTM 标配，改善长距离梯度流
        self.lstm_Whi = _np_orthogonal(d, d) * (scale * 2.5)
        self.lstm_Whf = _np_orthogonal(d, d) * (scale * 2.5)
        self.lstm_Whg = _np_orthogonal(d, d) * (scale * 2.5)
        self.lstm_Who = _np_orthogonal(d, d) * (scale * 2.5)
        self.lstm_bi = np.zeros(d, dtype=np.float64)
        # 遗忘门偏置=1.0 (Jozefowicz 2015 / PyTorch 默认)，让 LSTM 初期倾向「记住」
        self.lstm_bf = np.ones(d, dtype=np.float64)
        self.lstm_bg = np.zeros(d, dtype=np.float64)
        self.lstm_bo = np.zeros(d, dtype=np.float64)

    def _init_org_layer(self) -> None:
        in_dim = 2 * self.hidden_dim
        scale = EMBED_SCALE
        if self._use_transformer:
            self._org_w1 = nn.Linear(in_dim, self.org_dim)
            self._org_b1 = nn.Parameter(torch.zeros(self.org_dim))
            self._org_w2 = nn.Linear(self.org_dim, self.vocab_size, bias=False)
            nn.init.normal_(self._org_w1.weight, std=scale)
            nn.init.normal_(self._org_w2.weight, std=scale)
            self._org_w1.to(self._device)
            self._org_w2.to(self._device)
        else:
            self.org_W1 = np.random.randn(in_dim, self.org_dim).astype(np.float64) * scale
            self.org_b1 = np.random.randn(self.org_dim).astype(np.float64) * scale
            self.org_W2 = np.random.randn(self.org_dim, self.vocab_size).astype(np.float64) * scale

    def _expand_parameters_for_new_token(self) -> None:
        if self._use_transformer:
            with torch.no_grad():
                old_w = self._transformer.embed.weight
                new_row = torch.randn(1, self.hidden_dim, device=self._device) * EMBED_SCALE
                self._transformer.embed.weight = nn.Parameter(torch.cat([old_w, new_row], dim=0))
                w2 = self._w2.weight
                self._w2.weight = nn.Parameter(torch.cat([w2, torch.randn(1, self.hidden_dim, device=self._device) * EMBED_SCALE], dim=0))
                self._b2.data = torch.cat([self._b2, torch.zeros(1, device=self._device)])
                we = self._w_emotion.weight
                self._w_emotion.weight = nn.Parameter(torch.cat([we, torch.randn(1, self.emotion_dim, device=self._device) * EMBED_SCALE], dim=0))
                ow2 = self._org_w2.weight
                self._org_w2.weight = nn.Parameter(torch.cat([ow2, torch.randn(1, self.org_dim, device=self._device) * EMBED_SCALE], dim=0))
            return
        new_emb = np.random.randn(1, self.hidden_dim).astype(np.float64) * EMBED_SCALE
        self.E = np.vstack([self.E, new_emb])
        self.W2 = np.hstack([self.W2, np.random.randn(self.hidden_dim, 1).astype(np.float64) * EMBED_SCALE])
        self.b2 = np.append(self.b2, 0.0)
        self.W_emotion_logits = np.hstack([self.W_emotion_logits, np.random.randn(self.emotion_dim, 1).astype(np.float64) * EMBED_SCALE])
        self.org_W2 = np.hstack([self.org_W2, np.random.randn(self.org_dim, 1).astype(np.float64) * EMBED_SCALE])

    def tokenize(self, text: str) -> List[str]:
        normalized = text.strip().lower()
        if not normalized:
            return []
        if " " in normalized:
            return [x.strip() for x in normalized.split() if x.strip()]

        # 无空格场景（中文常见）：按“中文单字 / 英文串 / 数字串 / 标点”切分。
        # 这样输入不会被当成一个超长 token，交互学习更稳定。
        pattern = r"[\u4e00-\u9fff]|[a-z]+|\d+|[^\w\s]"
        tokens = re.findall(pattern, normalized)
        return [t for t in tokens if t.strip()]

    def _adaptive_length_budget(
        self,
        prompt_tokens: Sequence[str],
        sensory_payload: Dict[str, float],
        hard_max_len: Optional[int],
    ) -> Tuple[int, int]:
        # 自我感知长度预算：根据输入复杂度与内在状态动态决定，短对话（你好）允许简短回复。
        n = len(prompt_tokens)
        energy = float(sensory_payload.get("energy", 0.0))
        confidence = float(sensory_payload.get("confidence", 0.5))
        stress = float(sensory_payload.get("stress", 0.0))
        noise = float(sensory_payload.get("noise", 0.0))
        question_bias = 0.0
        for tok in prompt_tokens:
            if "?" in tok or "？" in tok:
                question_bias += 1.0
        complexity = min(1.0, n / 18.0 + question_bias * 0.12)
        desire = 0.42 * complexity + 0.28 * max(0.0, energy) + 0.2 * max(0.0, confidence) - 0.22 * max(0.0, stress + noise)
        base_len = 24 + int(128 * max(0.0, desire))
        if hard_max_len is not None:
            base_len = min(hard_max_len, base_len)
        # 短 prompt 允许短回复，避免逼出无意义续写（如「你好」→「你好」仅 2 token）
        if n <= 3:
            desired_len = max(8, min(base_len, 6 + n * 2))
            min_len = max(2, min(5, n + 2))
        else:
            desired_len = max(16, base_len)
            min_len = max(2, min(desired_len, int(4 + 0.5 * n)))
        return min_len, desired_len

    _SENTENCE_END_TOKENS = frozenset("。！？.!?")

    def _should_self_stop(
        self,
        out_ids: Sequence[int],
        confidences: Sequence[float],
        min_len: int,
        target_len: int,
        hard_max_len: Optional[int],
    ) -> Tuple[bool, str]:
        cur_len = len(out_ids)
        if cur_len < min_len:
            return False, ""
        if hard_max_len is not None and cur_len >= hard_max_len:
            return True, "hard_cap"
        if cur_len >= 3 and out_ids[-1] == out_ids[-2] == out_ids[-3]:
            return True, "loop_repeat"
        if cur_len >= 4 and out_ids[-4:-2] == out_ids[-2:]:
            return True, "bigram_repeat"
        if cur_len >= 6 and out_ids[-6:-3] == out_ids[-3:]:
            return True, "trigram_repeat"

        unique_ratio = len(set(out_ids)) / max(1, cur_len)
        mean_conf = sum(confidences) / max(1, len(confidences))
        tail = list(confidences[-3:]) if confidences else []
        low_conf_tail = len(tail) >= 2 and all(x < 0.12 for x in tail[-2:])

        if low_conf_tail and cur_len >= min_len + 12:
            return True, "low_confidence"
        if unique_ratio < 0.42 and cur_len >= min_len + 3:
            return True, "low_novelty"

        # 句末标点检测：输出完整句子（以句号/问号/叹号结尾）且已过 min_len，自然停止
        last_token = self.id_to_token[out_ids[-1]] if 0 <= out_ids[-1] < self.vocab_size else ""
        if last_token in self._SENTENCE_END_TOKENS and cur_len >= min_len:
            return True, "sentence_end"

        # 须达较高置信才允停（否则继续思考）
        if cur_len >= target_len and mean_conf < 0.72:
            return False, ""
        if cur_len >= target_len + 12:
            return True, "soft_cap"
        return False, ""

    def decode(self, ids: Sequence[int]) -> List[str]:
        out = []
        for i in ids:
            if 0 <= i < self.vocab_size:
                out.append(self.id_to_token[i])
            else:
                out.append(self.UNK)
        return out

    def _build_x(self, token_id: int, sensory_vec, emotion_vec) -> np.ndarray:
        emb = self.E[token_id] if 0 <= token_id < len(self.E) else self.E[0]
        s = np.zeros(self.sensory_dim, dtype=np.float64)
        sv = np.asarray(sensory_vec, dtype=np.float64).ravel()
        s[:min(len(sv), self.sensory_dim)] = sv[:self.sensory_dim]
        e = np.zeros(self.emotion_dim, dtype=np.float64)
        ev = np.asarray(emotion_vec, dtype=np.float64).ravel()
        e[:min(len(ev), self.emotion_dim)] = ev[:self.emotion_dim]
        return np.concatenate([emb, s, e])

    def _lstm_step(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        i_gate = 1.0 / (1.0 + np.exp(-np.clip(x @ self.lstm_Wxi + h @ self.lstm_Whi + self.lstm_bi, -20, 20)))
        f_gate = 1.0 / (1.0 + np.exp(-np.clip(x @ self.lstm_Wxf + h @ self.lstm_Whf + self.lstm_bf, -20, 20)))
        g_gate = np.tanh(x @ self.lstm_Wxg + h @ self.lstm_Whg + self.lstm_bg)
        o_gate = 1.0 / (1.0 + np.exp(-np.clip(x @ self.lstm_Wxo + h @ self.lstm_Who + self.lstm_bo, -20, 20)))
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * np.tanh(c_new)
        return h_new, c_new, i_gate, f_gate, g_gate, o_gate

    def _run_lstm_forward(self, token_ids: Sequence[int], sensory_vec, emotion_vec, modulate_by_emotion: bool = False):
        d = self.hidden_dim
        h = np.zeros(d, dtype=np.float64)
        c = np.zeros(d, dtype=np.float64)
        cache = []
        for tid in token_ids:
            x = self._build_x(tid, sensory_vec, emotion_vec)
            h_new, c_new, i_g, f_g, g_g, o_g = self._lstm_step(x, h, c)
            cache.append((x.copy(), h.copy(), c.copy(), i_g, f_g, g_g, o_g))
            h, c = h_new, c_new
        if modulate_by_emotion:
            ev = np.asarray(emotion_vec, dtype=np.float64).ravel()
            emo = np.zeros(self.emotion_dim, dtype=np.float64)
            emo[:min(len(ev), self.emotion_dim)] = ev[:self.emotion_dim]
            mod = 1.0 + 0.14 * emo.mean()
            h = np.clip(h * mod, -1.0, 1.0)
        return h, c, cache

    def _scaled_dot_product_attention(self, h_stack: np.ndarray, scale: float = ATTN_SCALE) -> np.ndarray:
        """h_stack: (T, d) numpy array。Q=最后隐态，K=V=全部。"""
        if len(h_stack) == 0:
            return np.zeros(self.hidden_dim, dtype=np.float64)
        q = h_stack[-1]
        scores = h_stack @ q / np.sqrt(self.hidden_dim)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        return scale * (weights @ h_stack)

    def _forward_transformer(
        self,
        context_ids: Sequence[int],
        sensory_vec: Sequence[float],
        emotion_vec: Sequence[float],
        temperature: float = 1.0,
        prev_id_for_org: Optional[int] = None,
    ) -> Tuple[Any, Any, np.ndarray]:
        self._transformer.eval()
        with torch.no_grad():
            ids = torch.tensor([context_ids], dtype=torch.long, device=self._device)
            sens = torch.tensor([list(sensory_vec)[: self.sensory_dim]], dtype=torch.float32, device=self._device)
            emo = torch.tensor([list(emotion_vec)[: self.emotion_dim]], dtype=torch.float32, device=self._device)
            h = self._transformer(ids, sens, emo)
            logits = self._w2(h) + self._b2
            if prev_id_for_org is not None and 0 <= prev_id_for_org < self.vocab_size:
                prev_emb = self._transformer.embed.weight[prev_id_for_org].unsqueeze(0)
                org_in = torch.cat([h, prev_emb], dim=-1)
                org_h = F.gelu(self._org_w1(org_in) + self._org_b1)
                logits = logits + ORG_LAYER_SCALE * self._org_w2(org_h)
            logits = logits + self._w_emotion(emo)
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            probs_np = probs.cpu().float().numpy()[0]
        return None, None, probs_np

    def _forward(self, context_ids: Sequence[int], sensory_vec, emotion_vec, temperature: float = 1.0, modulate_by_emotion: bool = True):
        if self._use_transformer:
            return self._forward_transformer(
                context_ids, sensory_vec, emotion_vec, temperature,
                prev_id_for_org=context_ids[-1] if context_ids else None,
            )
        h, c, cache = self._run_lstm_forward(context_ids, sensory_vec, emotion_vec, modulate_by_emotion)
        h_stack = np.array([item[1] for item in cache] + [h])
        attn_out = self._scaled_dot_product_attention(h_stack)
        h = h + attn_out
        h = _np_rms_norm(h)
        logits = h @ self.W2 + self.b2
        if context_ids:
            prev_id = context_ids[-1]
            prev_emb = self.E[prev_id] if 0 <= prev_id < len(self.E) else np.zeros(self.hidden_dim, dtype=np.float64)
            org_in = np.concatenate([h, prev_emb])
            org_h = _np_gelu(org_in @ self.org_W1 + self.org_b1)
            logits += ORG_LAYER_SCALE * (org_h @ self.org_W2)
        if modulate_by_emotion:
            ev = np.asarray(emotion_vec, dtype=np.float64).ravel()
            emo = np.zeros(self.emotion_dim, dtype=np.float64)
            emo[:min(len(ev), self.emotion_dim)] = ev[:self.emotion_dim]
            logits += emo @ self.W_emotion_logits
        probs = _np_softmax(logits, temperature=temperature)
        last_x = self._build_x(context_ids[-1], sensory_vec, emotion_vec) if context_ids else np.zeros(self.input_dim, dtype=np.float64)
        return last_x, h, probs

    REPETITION_PENALTY = 1.95  # 已出现 token 概率除以该值，抑制重复（原 1.35 偏弱）
    REPETITION_WINDOW = 24    # 扩大窗口，中文单字多，需看更多上下文
    REPETITION_FREQ_WEIGHT = 0.4  # 出现次数越多，惩罚越强：penalty *= (1 + freq_weight * (count-1))
    DEGENERATE_TOKENS = frozenset({"+", "-", "*", "×", "÷", "="})
    DEGENERATE_REPEAT_PENALTY = 8.0  # 退化 token 重复时施加的强惩罚

    def _sample_next_with_confidence(
        self,
        context_ids: Sequence[int],
        sensory_vec,
        emotion_vec,
        temperature: float = 1.0,
        recent_ids: Optional[Sequence[int]] = None,
        intent_bias: Optional[Tuple[Sequence[int], float, int]] = None,
        retrieval_boosts: Optional[Dict[int, float]] = None,
        perspective_boosts: Optional[Dict[int, float]] = None,
    ) -> Tuple[int, float]:
        _, _, probs = self._forward(context_ids, sensory_vec, emotion_vec, temperature=temperature)
        probs = np.array(probs, dtype=np.float64) if not isinstance(probs, np.ndarray) else probs.copy()
        banned = {self.token_to_id[self.PAD], self.token_to_id[self.BOS]}
        ban_mask = np.zeros(len(probs), dtype=bool)
        for b in banned:
            if 0 <= b < len(probs):
                ban_mask[b] = True
        def _apply_boosts(p, boosts):
            for tid, b in boosts.items():
                if 0 <= tid < len(p) and not ban_mask[tid]:
                    p[tid] += b
            s = p.sum()
            if s > 1e-12:
                p /= s
        if perspective_boosts:
            _apply_boosts(probs, perspective_boosts)
        if retrieval_boosts:
            _apply_boosts(probs, retrieval_boosts)
        if intent_bias:
            bias_ids, strength, step = intent_bias
            if step < len(bias_ids):
                tid = bias_ids[step]
                if 0 <= tid < len(probs) and not ban_mask[tid]:
                    probs[tid] += strength
                    s = probs.sum()
                    if s > 1e-12:
                        probs /= s
        if recent_ids:
            window = list(recent_ids[-self.REPETITION_WINDOW:])
            counts = Counter(window)
            for i, count in counts.items():
                if 0 <= i < len(probs) and not ban_mask[i] and count >= 1:
                    base = self.REPETITION_PENALTY
                    if i < len(self.id_to_token) and self.id_to_token[i] in self.DEGENERATE_TOKENS:
                        base = self.DEGENERATE_REPEAT_PENALTY
                    penalty = base * (1.0 + self.REPETITION_FREQ_WEIGHT * (count - 1))
                    probs[i] /= penalty
        probs[ban_mask] = 0.0
        probs = np.maximum(probs, 1e-12)
        s = probs.sum()
        if s <= 1e-12:
            return self.token_to_id[self.EOS], 1.0
        probs /= s
        picked = _sample_from_probs(probs)
        return int(picked), float(probs[picked])

    def generate(
        self,
        prompt: str,
        sensory_payload: Optional[Dict[str, float]] = None,
        max_len: int = 0,
        temperature: float = 0.9,
        conversation_history: Optional[Sequence[Tuple[str, str]]] = None,
        continuation_intent: Optional[str] = None,
        intent_boost: float = 1.0,
        reasoning_context: Optional[Any] = None,
    ) -> GenerationTrace:
        """生成回复。可用 reasoning_context 复用已准备流水线，避免重复计算。"""
        from src.agent_pipeline import prepare_reasoning_context
        sensory_payload = sensory_payload or {}
        if reasoning_context is not None:
            ctx = reasoning_context
        else:
            ctx = prepare_reasoning_context(
                self, prompt, sensory_payload, conversation_history,
                continuation_intent, max_len,
            )
        prompt_ids = ctx.prompt_ids
        prompt_tokens = ctx.prompt_tokens
        sensory_vec = ctx.sensory_vec
        emotion_vec = ctx.emotion_vec
        bias_ids = ctx.bias_ids
        retrieval_boosts = dict(ctx.retrieval_boosts)
        min_len, target_len = ctx.min_len, ctx.target_len
        hard_max_len = ctx.hard_max_len

        bos_id = self.token_to_id[self.BOS]
        context_ids = list(prompt_ids) + [bos_id]
        out_ids: List[int] = []
        prev_ids: List[int] = []
        confidences: List[float] = []
        last_ctx_id = bos_id if not prompt_ids else prompt_ids[-1]

        intent_strength_decay = 0.36 * intent_boost

        loop_guard = hard_max_len if hard_max_len is not None else (min(4096, self.context_max_len) if self._use_transformer else 4096)
        eff_temp = max(0.2, temperature * self._identity_temp_damp)
        for step in range(loop_guard):
            strength = max(0.0, intent_strength_decay * (1.0 - step * 0.25)) if bias_ids else 0.0
            ib = (bias_ids, strength, len(out_ids)) if strength > 0.01 else None
            # 生成时按步注入：视角(Perspective) + 逻辑衔接(Logic)
            pb = self._get_perspective_boosts(prompt_tokens, step) if step < 8 else None
            lb = self._get_logic_connective_boosts(self.decode(out_ids)) if out_ids else None
            rb = retrieval_boosts if step < 6 else None
            if lb and rb is not None:
                rb = dict(rb)
                for tid, b in lb.items():
                    rb[tid] = rb.get(tid, 0) + b
            elif lb:
                rb = lb
            next_id, conf = self._sample_next_with_confidence(
                context_ids, sensory_vec, emotion_vec, temperature=eff_temp, recent_ids=out_ids,
                intent_bias=ib,
                retrieval_boosts=rb,
                perspective_boosts=pb,
            )
            if next_id == self.token_to_id[self.EOS]:
                if len(out_ids) >= min_len:
                    break
                continue
            out_ids.append(next_id)
            prev_ids.append(last_ctx_id)
            confidences.append(conf)
            last_ctx_id = next_id
            context_ids = list(prompt_ids) + [bos_id] + list(out_ids)
            if self._use_transformer and len(context_ids) > self.context_max_len:
                context_ids = context_ids[-self.context_max_len :]
            should_stop, _ = self._should_self_stop(
                out_ids=out_ids,
                confidences=confidences,
                min_len=min_len,
                target_len=target_len,
                hard_max_len=hard_max_len,
            )
            if should_stop:
                break
        return GenerationTrace(
            tokens=self.decode(out_ids),
            token_ids=out_ids,
            prev_ids=prev_ids,
            confidences=confidences,
            sensory_vector=sensory_vec,
            emotion_vector=emotion_vec,
            prompt_token_ids=prompt_ids,
        )

    # 知识倾泻：对话场景下不应出现的公式/常数类 token（被塞满知识后易溢出）
    _KNOWLEDGE_DUMP_CUES = frozenset({
        "×", "kg", "²", "³", "常数", "玻尔兹曼", "定律", "公式", "n⋅", "m/kg",
        "mol", "j/mol", "m/s", "f=ma", "g=9.8", "π", "ρ",
    })

    def _is_conversational_prompt(self, prompt_tokens: Sequence[str]) -> bool:
        """是否为人际对话类输入（非知识问答）。"""
        conv_cues = {"你", "我", "谁", "吗", "记得", "好", "嗨", "呀", "啊", "呢", "吧", "hello", "hi"}
        return any(t in conv_cues for t in prompt_tokens)

    def _output_is_knowledge_dump(self, tokens: Sequence[str]) -> bool:
        """输出是否被知识倾泻污染（公式、常数、学科术语在非知识场景溢出）。"""
        if not tokens:
            return False
        dump_count = sum(1 for t in tokens if t in self._KNOWLEDGE_DUMP_CUES)
        if dump_count >= 2:
            return True
        if dump_count >= 1 and len(tokens) <= 8:
            return True
        joined = "".join(tokens).lower()
        if any(p in joined for p in ("m/kg", "n⋅", "mol/", "j/mol", "玻尔兹曼", "×a", "×b", "(kg)")):
            return True
        return False

    def _output_coherent_with_prompt(
        self, prompt_tokens: Sequence[str], trace: GenerationTrace
    ) -> bool:
        """
        逻辑链判断：输出是否与输入在语义/结构上连贯。
        短问短答、无冗长重复、对话时拒绝知识倾泻（公式/常数溢出）。
        """
        if not trace.tokens:
            return True
        out = trace.tokens
        n_prompt = len(prompt_tokens)
        n_out = len(out)
        # 逻辑：对话场景下若输出为知识倾泻（公式常数），则与输入不连贯
        if self._is_conversational_prompt(prompt_tokens) and self._output_is_knowledge_dump(out):
            return False
        # 逻辑：短 greeting 期待短回应
        if n_prompt <= 2 and n_out >= 15:
            greeting_cues = {"你", "好", "嗨", "hello", "hi"}
            if any(t in greeting_cues for t in prompt_tokens):
                return False
        # 逻辑：同一短语连续重复 2 次以上 → 无意义复读
        if n_out >= 6:
            for win in (3, 2):
                if n_out < win * 2:
                    continue
                for i in range(n_out - win * 2 + 1):
                    chunk = tuple(out[i : i + win])
                    rep = 1
                    for j in range(i + win, n_out - win + 1, win):
                        if tuple(out[j : j + win]) == chunk:
                            rep += 1
                        else:
                            break
                    if rep >= 2:
                        return False
        # 逻辑：输出全是同一 token（退化已单独判）
        if len(set(out)) == 1:
            return False
        return True

    def _is_degenerate_trace(self, trace: GenerationTrace) -> bool:
        """检测输出是否为退化（如全 + + + +），此类输出无意义。"""
        if not trace.tokens:
            return False
        degenerate_count = sum(1 for t in trace.tokens if t in self.DEGENERATE_TOKENS)
        ratio = degenerate_count / len(trace.tokens)
        # 超过 60% 为符号，或全部为同一退化 token
        if ratio >= 0.6:
            return True
        if len(set(trace.tokens)) == 1 and trace.tokens[0] in self.DEGENERATE_TOKENS:
            return True
        return False

    def _score_trace(
        self, trace: GenerationTrace, prompt_tokens: Optional[Sequence[str]] = None
    ) -> float:
        if not trace.token_ids:
            return -1e9
        if self._is_degenerate_trace(trace):
            return -1e8
        # 逻辑链：输出与输入不连贯则扣分（如短问长答、冗长复读）
        if prompt_tokens is not None and not self._output_coherent_with_prompt(
            prompt_tokens, trace
        ):
            return -1e7
        conf = sum(trace.confidences) / max(1, len(trace.confidences))
        unique_ratio = len(set(trace.token_ids)) / max(1, len(trace.token_ids))
        length_bonus = min(len(trace.token_ids), 8) * 0.02
        emo_mag = sum(abs(x) for x in trace.emotion_vector[:3]) / 3.0 if trace.emotion_vector else 0.0
        return conf + 0.18 * unique_ratio + length_bonus + 0.08 * emo_mag

    # 思考不达最深处/最正确就不停：质量门槛与安全上限
    _DEPTH_SCORE_THRESHOLD = 0.52
    _DELIBERATE_MAX_TRIALS = 25

    def deliberate_generate(
        self,
        prompt: str,
        sensory_payload: Optional[Dict[str, float]] = None,
        max_len: int = 0,
        temperature: float = 0.9,
        thought_trials: int = 5,
        conversation_history: Optional[Sequence[Tuple[str, str]]] = None,
        continuation_intent: Optional[str] = None,
    ) -> Tuple[GenerationTrace, Dict[str, Any]]:
        from src.agent_pipeline import prepare_reasoning_context
        sensory_payload = sensory_payload or {}
        ctx = prepare_reasoning_context(
            self, prompt, sensory_payload, conversation_history,
            continuation_intent, max_len,
        )
        intent = ctx.intent
        min_len, target_len = ctx.min_len, ctx.target_len
        raw_prompt_tokens = self.tokenize(prompt)

        def _should_reject(t: GenerationTrace) -> bool:
            if self._is_degenerate_trace(t):
                return True
            return not self._output_coherent_with_prompt(raw_prompt_tokens, t)

        def _is_deep_or_correct(trace: GenerationTrace, score: float) -> bool:
            if not trace.tokens:
                return False
            if _should_reject(trace):
                return False
            if intent and not self._intent_satisfied(intent, trace.tokens):
                return False
            return score >= self._DEPTH_SCORE_THRESHOLD

        candidates: List[Tuple[float, GenerationTrace]] = []
        trials = 0
        best_score, best_trace = -1e9, None
        min_trials = max(1, thought_trials)
        max_trials = self._DELIBERATE_MAX_TRIALS
        while trials < max_trials:
            jitter = 0.12 * ((trials % 3) - 1)
            t = max(0.2, min(1.5, temperature + jitter))
            trace = self.generate(
                prompt=prompt, sensory_payload=sensory_payload, max_len=max_len, temperature=t,
                conversation_history=conversation_history, continuation_intent=continuation_intent,
                reasoning_context=ctx,
            )
            score = self._score_trace(trace, prompt_tokens=raw_prompt_tokens)
            candidates.append((score, trace))
            if score > best_score:
                best_score, best_trace = score, trace
            trials += 1
            if trials >= min_trials and _is_deep_or_correct(best_trace, best_score):
                break
        if best_trace is None:
            best_trace = candidates[0][1] if candidates else None
        if best_trace is None or _should_reject(best_trace):
            best_trace = GenerationTrace(
                tokens=[], token_ids=[], prev_ids=[], confidences=[],
                sensory_vector=ctx.sensory_vec, emotion_vector=ctx.emotion_vec,
                prompt_token_ids=ctx.prompt_ids,
            )
        mean_conf = (sum(best_trace.confidences) / len(best_trace.confidences)) if best_trace.confidences else 0.0
        uncertainty = 1.0 - mean_conf
        self.emotion_core.update_from_inference(mean_confidence=mean_conf, uncertainty=uncertainty)
        # 不确定性表达：低置信时前置「嗯，」（智慧体：元认知）
        from src.agent import should_express_uncertainty
        if best_trace.tokens and should_express_uncertainty(mean_conf, 0.32):
            prefix_tok = self.tokenize("嗯，")
            for t in prefix_tok:
                self.ensure_token(t)
            best_trace = GenerationTrace(
                tokens=prefix_tok + list(best_trace.tokens),
                token_ids=[self.token_to_id[t] for t in prefix_tok] + list(best_trace.token_ids),
                prev_ids=best_trace.prev_ids,  # 简化，不重建完整 prev
                confidences=[0.5] * len(prefix_tok) + list(best_trace.confidences),
                sensory_vector=best_trace.sensory_vector,
                emotion_vector=best_trace.emotion_vector,
                prompt_token_ids=best_trace.prompt_token_ids,
            )
        report: Dict[str, Any] = {
            "thought_trials": trials,
            "best_score": best_score,
            "best_confidence": mean_conf,
            "inferred_intent": intent,
            "candidate_lengths": [len(t.token_ids) for _, t in candidates[: min(3, len(candidates))]],
            "target_length": target_len,
            "min_length": min_len,
            "actual_length": len(best_trace.token_ids),
            "emotion_state": list(self.emotion_core.as_vector()),
        }
        return best_trace, report

    BPTT_TRUNCATE = 24  # 多步 BPTT 长度，梯度回传到 prompt 编码，模型才能学会「理解输入」

    def _train_one_transformer(
        self,
        context_ids: Sequence[int],
        target_id: int,
        sensory_vec: Sequence[float],
        emotion_vec: Sequence[float],
        lr: float,
    ) -> float:
        self._transformer.train()
        ids = torch.tensor([list(context_ids)], dtype=torch.long, device=self._device)
        sens = torch.tensor([list(sensory_vec)[: self.sensory_dim]], dtype=torch.float32, device=self._device)
        emo = torch.tensor([list(emotion_vec)[: self.emotion_dim]], dtype=torch.float32, device=self._device)
        h = self._transformer(ids, sens, emo)
        logits = self._w2(h) + self._b2
        if context_ids:
            prev_id = context_ids[-1]
            if 0 <= prev_id < self.vocab_size:
                prev_emb = self._transformer.embed.weight[prev_id].unsqueeze(0)
                org_in = torch.cat([h, prev_emb], dim=-1)
                logits = logits + ORG_LAYER_SCALE * self._org_w2(F.gelu(self._org_w1(org_in) + self._org_b1))
        logits = logits + self._w_emotion(emo)
        loss = F.cross_entropy(logits, torch.tensor([target_id], device=self._device))
        params = (
            list(self._transformer.parameters()) + list(self._w2.parameters()) +
            list(self._w_emotion.parameters()) + list(self._org_w1.parameters()) +
            list(self._org_w2.parameters()) + [self._b2]
        )
        optim = torch.optim.SGD(params, lr=lr)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        optim.step()
        return float(loss.item())

    def train_one(
        self,
        context_ids: Sequence[int],
        target_id: int,
        sensory_vec: Sequence[float],
        emotion_vec: Sequence[float],
        lr: Optional[float] = None,
    ) -> float:
        if not context_ids:
            return 0.0
        eta = self.lr if lr is None else lr
        if self._use_transformer:
            return self._train_one_transformer(context_ids, target_id, sensory_vec, emotion_vec, eta)
        d = self.hidden_dim
        inp = self.input_dim

        h, c, cache = self._run_lstm_forward(context_ids, sensory_vec, emotion_vec, modulate_by_emotion=False)
        h_stack = np.array([item[1] for item in cache] + [h])
        attn_out = self._scaled_dot_product_attention(h_stack)
        h_comb = h + attn_out
        h_norm = _np_rms_norm(h_comb)

        logits = h_norm @ self.W2 + self.b2
        org_h_np = org_z_np = None
        if context_ids:
            prev_id = context_ids[-1]
            prev_emb = self.E[prev_id] if 0 <= prev_id < len(self.E) else np.zeros(d, dtype=np.float64)
            org_in = np.concatenate([h_norm, prev_emb])
            org_z_np = org_in @ self.org_W1 + self.org_b1
            org_h_np = _np_gelu(org_z_np)
            logits += ORG_LAYER_SCALE * (org_h_np @ self.org_W2)
        ev = np.asarray(emotion_vec, dtype=np.float64).ravel()
        emo = np.zeros(self.emotion_dim, dtype=np.float64)
        emo[:min(len(ev), self.emotion_dim)] = ev[:self.emotion_dim]
        logits += emo @ self.W_emotion_logits
        probs = _np_softmax(logits, temperature=1.0)
        dlogits = probs.copy()
        dlogits[target_id] -= 1.0

        org_scale = ORG_LAYER_SCALE
        d_org_z = None
        org_in_v = None
        d_org = None
        if org_h_np is not None and org_z_np is not None:
            d_org = org_scale * dlogits
            d_org_h = d_org @ self.org_W2.T
            prev_emb_v = self.E[context_ids[-1]] if 0 <= context_ids[-1] < len(self.E) else np.zeros(d, dtype=np.float64)
            org_in_v = np.concatenate([h_norm, prev_emb_v])
            d_org_z = d_org_h * _np_gelu_derivative(org_z_np)

        dh_norm = self.W2 @ dlogits
        if d_org_z is not None:
            dh_norm += self.org_W1[:d, :] @ d_org_z

        rms = np.sqrt(np.mean(h_comb ** 2) + 1e-6)
        dot_hn_dhn = np.dot(h_norm, dh_norm)
        dh_comb = (dh_norm - h_norm * dot_hn_dhn / d) / rms

        # Attention weights for BPTT
        q = h_stack[-1]
        scores = h_stack @ q / np.sqrt(d)
        aw = np.exp(scores - scores.max())
        aw /= aw.sum()
        attn_scale = ATTN_SCALE

        # BPTT with numpy vectorized ops
        dh = dh_comb.copy()
        dc = np.zeros(d, dtype=np.float64)
        dWxi = np.zeros_like(self.lstm_Wxi)
        dWxf = np.zeros_like(self.lstm_Wxf)
        dWxg = np.zeros_like(self.lstm_Wxg)
        dWxo = np.zeros_like(self.lstm_Wxo)
        dWhi = np.zeros_like(self.lstm_Whi)
        dWhf = np.zeros_like(self.lstm_Whf)
        dWhg = np.zeros_like(self.lstm_Whg)
        dWho = np.zeros_like(self.lstm_Who)
        dbi = np.zeros(d, dtype=np.float64)
        dbf = np.zeros(d, dtype=np.float64)
        dbg = np.zeros(d, dtype=np.float64)
        dbo = np.zeros(d, dtype=np.float64)
        token_grads: Dict[int, np.ndarray] = {}

        truncate = min(len(cache), self.BPTT_TRUNCATE)
        for t in range(len(cache) - 1, max(-1, len(cache) - 1 - truncate), -1):
            dh = dh + aw[t] * attn_scale * dh_comb
            x_t, h_prev, c_prev, i_g, f_g, g_g, o_g = cache[t]
            c_cur = f_g * c_prev + i_g * g_g
            tanh_c = np.tanh(c_cur)
            do_g = dh * tanh_c
            dc_cur = dh * o_g * (1.0 - tanh_c ** 2) + dc
            di_g = dc_cur * g_g
            dg_g = dc_cur * i_g
            df_g = dc_cur * c_prev
            dz_i = di_g * i_g * (1.0 - i_g)
            dz_f = df_g * f_g * (1.0 - f_g)
            dz_g = dg_g * (1.0 - g_g ** 2)
            dz_o = do_g * o_g * (1.0 - o_g)

            dWxi += np.outer(x_t, dz_i)
            dWxf += np.outer(x_t, dz_f)
            dWxg += np.outer(x_t, dz_g)
            dWxo += np.outer(x_t, dz_o)
            dWhi += np.outer(h_prev, dz_i)
            dWhf += np.outer(h_prev, dz_f)
            dWhg += np.outer(h_prev, dz_g)
            dWho += np.outer(h_prev, dz_o)
            dbi += dz_i
            dbf += dz_f
            dbg += dz_g
            dbo += dz_o

            dx = self.lstm_Wxi @ dz_i + self.lstm_Wxf @ dz_f + self.lstm_Wxg @ dz_g + self.lstm_Wxo @ dz_o
            tid = context_ids[t]
            if tid not in token_grads:
                token_grads[tid] = np.zeros(inp, dtype=np.float64)
            token_grads[tid][:inp] += dx[:inp]

            dh = self.lstm_Whi @ dz_i + self.lstm_Whf @ dz_f + self.lstm_Whg @ dz_g + self.lstm_Who @ dz_o
            dc = dc_cur * f_g

        # 梯度裁剪：防止 BPTT 长序列时梯度爆炸
        all_grads: List[np.ndarray] = [
            dWxi, dWxf, dWxg, dWxo, dWhi, dWhf, dWhg, dWho,
            dbi, dbf, dbg, dbo,
            np.outer(h_norm, dlogits),
            dlogits.copy(),
            np.outer(emo, dlogits),
        ]
        if d_org_z is not None and org_in_v is not None and d_org is not None:
            all_grads.extend([
                np.outer(org_h_np, d_org),
                np.outer(org_in_v, d_org_z),
            ])
        for g in token_grads.values():
            all_grads.append(g)
        grad_norm_sq = sum(float(np.sum(g ** 2)) for g in all_grads)
        grad_norm = np.sqrt(grad_norm_sq) if grad_norm_sq > 0 else 0.0
        if grad_norm > MAX_GRAD_NORM:
            clip_scale = MAX_GRAD_NORM / grad_norm
            dWxi *= clip_scale
            dWxf *= clip_scale
            dWxg *= clip_scale
            dWxo *= clip_scale
            dWhi *= clip_scale
            dWhf *= clip_scale
            dWhg *= clip_scale
            dWho *= clip_scale
            dbi *= clip_scale
            dbf *= clip_scale
            dbg *= clip_scale
            dbo *= clip_scale
            dlogits *= clip_scale
            if d_org_z is not None:
                d_org_z *= clip_scale
                d_org = org_scale * dlogits
            for g in token_grads.values():
                g *= clip_scale

        self.W2 -= eta * np.outer(h_norm, dlogits)
        self.b2 -= eta * dlogits
        self.W_emotion_logits -= eta * np.outer(emo, dlogits)
        if d_org_z is not None and org_in_v is not None and d_org is not None:
            self.org_W2 -= eta * np.outer(org_h_np, d_org)
            self.org_W1 -= eta * np.outer(org_in_v, d_org_z)
            self.org_b1 -= eta * d_org_z

        self.lstm_Wxi -= eta * dWxi
        self.lstm_Wxf -= eta * dWxf
        self.lstm_Wxg -= eta * dWxg
        self.lstm_Wxo -= eta * dWxo
        self.lstm_Whi -= eta * dWhi
        self.lstm_Whf -= eta * dWhf
        self.lstm_Whg -= eta * dWhg
        self.lstm_Who -= eta * dWho
        self.lstm_bi -= eta * dbi
        self.lstm_bf -= eta * dbf
        self.lstm_bg -= eta * dbg
        self.lstm_bo -= eta * dbo
        for tid, grad in token_grads.items():
            if 0 <= tid < len(self.E):
                self.E[tid, :d] -= eta * grad[:d]
        return float(-np.log(max(float(probs[target_id]), 1e-12)))

    def estimate_loss(self, context_ids, target_id, sensory_vec, emotion_vec) -> float:
        _, _, probs = self._forward(context_ids, sensory_vec, emotion_vec, temperature=1.0, modulate_by_emotion=False)
        return float(-np.log(max(float(probs[target_id]), 1e-12)))

    def penalize_one(self, context_ids, bad_target_id, sensory_vec, emotion_vec, lr=None) -> None:
        if self._use_transformer:
            return
        eta = self.lr * 0.5 if lr is None else lr
        _, _, probs = self._forward(context_ids, sensory_vec, emotion_vec, temperature=1.0, modulate_by_emotion=False)
        bad_prob = float(probs[bad_target_id])
        if bad_prob <= 1e-8:
            return
        h, c, cache = self._run_lstm_forward(context_ids, sensory_vec, emotion_vec, modulate_by_emotion=False)
        h_stack = np.array([item[1] for item in cache] + [h])
        attn_out = self._scaled_dot_product_attention(h_stack)
        h_norm = _np_rms_norm(h + attn_out)
        self.W2[:, bad_target_id] -= eta * h_norm * bad_prob
        self.b2[bad_target_id] -= eta * bad_prob

    def _remember(
        self,
        context_ids: Sequence[int],
        target_id: int,
        priority: float,
        sensory_vec: Sequence[float],
        emotion_vec: Sequence[float],
    ) -> None:
        ctx = list(context_ids)
        self.memory.append((ctx, target_id, max(0.05, priority), list(sensory_vec), list(emotion_vec)))
        for cid in set(ctx):
            self._memory_index.setdefault(cid, []).append(len(self.memory) - 1)
        if self.max_memory > 0 and len(self.memory) > self.max_memory:
            overflow = len(self.memory) - self.max_memory
            if overflow > 0:
                self.memory = self.memory[overflow:]
                self._rebuild_memory_index()

    def _irrelevant_key(self, prev_id: int, target_id: int) -> str:
        return f"{prev_id}:{target_id}"

    def _stash_irrelevant(
        self,
        prev_id: int,
        target_id: int,
        sensory_vec: Sequence[float],
        emotion_vec: Sequence[float],
    ) -> None:
        key = self._irrelevant_key(prev_id, target_id)
        item = self.irrelevant_memory.get(key)
        if item is None:
            self.irrelevant_memory[key] = {
                "prev_id": prev_id,
                "target_id": target_id,
                "count": 1.0,
                "sensory_vector": list(sensory_vec),
                "emotion_vector": list(emotion_vec),
            }
            return

        old_count = float(item.get("count", 1.0))
        new_count = min(1000.0, old_count + 1.0)
        item["count"] = new_count
        # 用 EMA 保存“无关样本”的上下文，等待后续自悟。
        old_s = list(item.get("sensory_vector", []))
        old_e = list(item.get("emotion_vector", []))
        s = list(sensory_vec)
        e = list(emotion_vec)
        if len(old_s) == len(s):
            item["sensory_vector"] = [0.85 * old_s[i] + 0.15 * s[i] for i in range(len(s))]
        else:
            item["sensory_vector"] = s
        if len(old_e) == len(e):
            item["emotion_vector"] = [0.85 * old_e[i] + 0.15 * e[i] for i in range(len(e))]
        else:
            item["emotion_vector"] = e

    def _ponder_irrelevant(self, budget: int = 24) -> int:
        if budget <= 0 or not self.irrelevant_memory:
            return 0
        entries: List[Dict[str, Any]] = list(self.irrelevant_memory.values())
        matured = [x for x in entries if float(x.get("count", 0.0)) >= 2.0]
        if not matured:
            return 0

        steps = min(budget, len(matured))
        weights = [float(x.get("count", 1.0)) for x in matured]
        sampled = random.choices(matured, weights=weights, k=steps)
        bos_id = self.token_to_id[self.BOS]
        updates = 0
        for item in sampled:
            prev_id = int(item["prev_id"])
            target_id = int(item["target_id"])
            context_ids = [bos_id, prev_id] if prev_id != bos_id else [bos_id]
            count = float(item.get("count", 1.0))
            maturity = min(1.0, count / 6.0)
            sensory_vec = [float(x) for x in item.get("sensory_vector", [])]
            emotion_vec = [float(x) for x in item.get("emotion_vector", [])]
            self.train_one(
                context_ids,
                target_id,
                sensory_vec=sensory_vec,
                emotion_vec=emotion_vec,
                lr=self.lr * (0.08 + 0.22 * maturity),
            )
            self._remember(
                context_ids,
                target_id,
                priority=0.25 + 0.35 * maturity,
                sensory_vec=sensory_vec,
                emotion_vec=emotion_vec,
            )
            item["count"] = max(1.0, count * 0.92)
            updates += 1
        return updates

    def _rebuild_memory_index(self) -> None:
        self._memory_index.clear()
        for idx, (ctx, *_rest) in enumerate(self.memory):
            ids = list(ctx) if isinstance(ctx, (list, tuple)) else [ctx]
            for cid in set(ids):
                self._memory_index.setdefault(cid, []).append(idx)

    def _replay(self, replay_steps: int, lr_scale: float = 0.5, absorb_knowledge: bool = False) -> int:
        if not self.memory:
            return 0
        steps = min(replay_steps, len(self.memory))
        neutral_s = [0.0] * self.sensory_dim
        neutral_e = [0.0] * self.emotion_dim
        # 逻辑链回放：按经验优先级强化，辅以随机覆盖不同链条，避免单一模式
        weights = [p for _, _, p, _, _ in self.memory]
        n_priority = int(steps * 0.85)
        n_explore = steps - n_priority
        sampled: List[Tuple[Any, int, float, List[float], List[float]]] = []
        if n_priority > 0:
            sampled.extend(random.choices(self.memory, weights=weights, k=n_priority))
        if n_explore > 0 and len(self.memory) > 1:
            sampled.extend(random.choices(self.memory, k=n_explore))
        for ctx_or_prev, target_id, _, sensory_vec, emotion_vec in sampled:
            ctx = list(ctx_or_prev) if isinstance(ctx_or_prev, (list, tuple)) else [int(ctx_or_prev)]
            s = neutral_s if absorb_knowledge else sensory_vec
            e = neutral_e if absorb_knowledge else emotion_vec
            self.train_one(ctx, target_id, sensory_vec=s, emotion_vec=e, lr=self.lr * lr_scale)
        return steps

    def apply_feedback(
        self,
        trace: GenerationTrace,
        correctness: Sequence[Optional[bool]],
        corrections: Sequence[Optional[str]],
        learning_passes: int = 3,
        replay_steps: int = 24,
        absorb_knowledge: bool = False,
    ) -> Dict[str, float]:
        """
        强制即学：少遍数、高学习率、深记忆。
        absorb_knowledge=True：仅训练语言，不触碰情感/感知/思维。
        """
        if len(correctness) != len(trace.token_ids) or len(corrections) != len(trace.token_ids):
            raise ValueError("Feedback length must match generated token length.")
        if not trace.token_ids:
            return {"avg_loss": 0.0, "supervised_updates": 0.0, "contrastive_updates": 0.0, "replay_updates": 0.0}

        sensory_vec = [0.0] * self.sensory_dim if absorb_knowledge else list(trace.sensory_vector)
        emotion_vec = [0.0] * self.emotion_dim if absorb_knowledge else list(trace.emotion_vector)
        if len(sensory_vec) < self.sensory_dim:
            sensory_vec = sensory_vec + [0.0] * (self.sensory_dim - len(sensory_vec))
        if len(emotion_vec) < self.emotion_dim:
            emotion_vec = emotion_vec + [0.0] * (self.emotion_dim - len(emotion_vec))

        losses: List[float] = []
        initial_losses: List[float] = []
        supervised_updates = 0
        contrastive_updates = 0
        irrelevant_stashed = 0

        targets: List[Optional[int]] = [None] * len(trace.token_ids)
        priorities: List[float] = [1.0] * len(trace.token_ids)
        for i, token_id in enumerate(trace.token_ids):
            status = correctness[i]
            if status is True:
                targets[i] = token_id
                priorities[i] = 1.0
            elif status is False:
                corr = corrections[i]
                if corr is None or not corr.strip():
                    continue
                targets[i] = self.ensure_token(corr.strip().lower())
                priorities[i] = 2.7
            else:
                self._stash_irrelevant(
                    prev_id=trace.prev_ids[i],
                    target_id=token_id,
                    sensory_vec=sensory_vec,
                    emotion_vec=emotion_vec,
                )
                irrelevant_stashed += 1

        prompt_ids = list(trace.prompt_token_ids) if trace.prompt_token_ids else []
        bos_id = self.token_to_id[self.BOS]
        # 视角自整理：按 prompt 概念将 answer 归入对应视角
        prompt_tokens = [self.id_to_token[i] for i in prompt_ids if 0 <= i < len(self.id_to_token)]
        answer_ids = [t for t in targets if t is not None]
        if prompt_tokens and answer_ids and not absorb_knowledge:
            self.perspective_core.learn_from(prompt_tokens, answer_ids)
        # 构建前缀：用 targets（含纠错），None 处用原 token
        def _prefix(i: int) -> List[int]:
            out: List[int] = []
            for j in range(i):
                t = targets[j] if targets[j] is not None else trace.token_ids[j]
                out.append(t)
            return out

        passes = max(1, learning_passes)
        force_lr = 1.6
        for i, target_id in enumerate(targets):
            if target_id is None:
                continue
            context_ids = prompt_ids + [bos_id] + _prefix(i)
            wrong_id = trace.token_ids[i]
            warm_loss = self.estimate_loss(context_ids, target_id, sensory_vec=sensory_vec, emotion_vec=emotion_vec)
            initial_losses.append(warm_loss)
            for p in range(passes):
                lr_scale = force_lr / (1.0 + 0.3 * p)
                loss = self.train_one(
                    context_ids,
                    target_id,
                    sensory_vec=sensory_vec,
                    emotion_vec=emotion_vec,
                    lr=self.lr * lr_scale,
                )
                losses.append(loss)
                supervised_updates += 1
                if (correctness[i] is False) and (target_id != wrong_id):
                    self.penalize_one(
                        context_ids,
                        wrong_id,
                        sensory_vec=sensory_vec,
                        emotion_vec=emotion_vec,
                        lr=self.lr * 0.45 * lr_scale,
                    )
                    contrastive_updates += 1

            self._remember(
                context_ids,
                target_id,
                priorities[i],
                sensory_vec=sensory_vec,
                emotion_vec=emotion_vec,
            )

        for i in range(len(targets) - 1):
            cur_target = targets[i]
            next_target = targets[i + 1]
            if cur_target is None or next_target is None:
                continue
            context_ids = prompt_ids + [bos_id] + _prefix(i) + [cur_target]
            bridge_loss = self.train_one(
                context_ids,
                next_target,
                sensory_vec=sensory_vec,
                emotion_vec=emotion_vec,
                lr=self.lr * 0.8,
            )
            losses.append(bridge_loss)
            supervised_updates += 1
            self._remember(
                context_ids,
                next_target,
                1.8,
                sensory_vec=sensory_vec,
                emotion_vec=emotion_vec,
            )

        replay_done = self._replay(replay_steps=max(0, replay_steps), lr_scale=0.45, absorb_knowledge=absorb_knowledge)
        ponder_done = self._ponder_irrelevant(budget=0 if absorb_knowledge else max(4, replay_steps // 12))

        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        initial_avg_loss = (sum(initial_losses) / len(initial_losses)) if initial_losses else avg_loss
        loss_improvement = 0.0
        if initial_avg_loss > 1e-9:
            loss_improvement = (initial_avg_loss - avg_loss) / initial_avg_loss
        # absorb_knowledge 时不更新 emotion，保持中性，完全吸收
        if not absorb_knowledge:
            judged = [x for x in correctness if x is not None]
            correction_count = sum(1 for c in corrections if c is not None and c.strip())
            correction_ratio = correction_count / max(1, len(judged))
            correctness_ratio = sum(1 for x in judged if x is True) / max(1, len(judged))
            replay_intensity = replay_done / max(1, replay_steps)
            contrastive_intensity = contrastive_updates / max(1, supervised_updates)
            self.emotion_core.update_from_learning(
                correctness_ratio=correctness_ratio,
                correction_ratio=correction_ratio,
                loss_improvement=loss_improvement,
                replay_intensity=replay_intensity,
                contrastive_intensity=contrastive_intensity,
            )

        return {
            "avg_loss": avg_loss,
            "supervised_updates": float(supervised_updates),
            "contrastive_updates": float(contrastive_updates),
            "replay_updates": float(replay_done),
            "irrelevant_stashed": float(irrelevant_stashed),
            "ponder_updates": float(ponder_done),
        }

    def reinforce_comprehension(
        self,
        trace: GenerationTrace,
        extra_passes: int = 2,
        lr_scale: float = 1.0,
        absorb_knowledge: bool = True,
    ) -> float:
        """
        即学巩固：高学习率、少遍数，一次教会。
        absorb_knowledge=True 时用中性 sensory/emotion，不触碰情感，仅加深语言映射。
        """
        if not trace.token_ids:
            return 0.0
        if absorb_knowledge:
            sensory_vec = [0.0] * self.sensory_dim
            emotion_vec = [0.0] * self.emotion_dim
        else:
            sensory_vec = list(trace.sensory_vector) if trace.sensory_vector else [0.0] * self.sensory_dim
            emotion_vec = list(trace.emotion_vector) if trace.emotion_vector else [0.0] * self.emotion_dim
        if len(sensory_vec) < self.sensory_dim:
            sensory_vec = sensory_vec + [0.0] * (self.sensory_dim - len(sensory_vec))
        if len(emotion_vec) < self.emotion_dim:
            emotion_vec = emotion_vec + [0.0] * (self.emotion_dim - len(emotion_vec))
        prompt_ids = list(trace.prompt_token_ids) if trace.prompt_token_ids else []
        bos_id = self.token_to_id[self.BOS]
        prefix: List[int] = []
        total_loss = 0.0
        n_updates = 0
        eta = self.lr * lr_scale
        for i, target_id in enumerate(trace.token_ids):
            context_ids = prompt_ids + [bos_id] + prefix
            for _ in range(extra_passes):
                loss = self.train_one(
                    context_ids,
                    target_id,
                    sensory_vec=sensory_vec,
                    emotion_vec=emotion_vec,
                    lr=eta * (0.7 + 0.1 * (1.0 - i / max(1, len(trace.token_ids)))),
                )
                total_loss += loss
                n_updates += 1
            prefix.append(target_id)
        return total_loss / max(1, n_updates)

    def _np_to_list(self, arr) -> Any:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

    def _save_transformer(self, path: str) -> None:
        torch_path = path.replace(".npz", "_transformer.pt") if path.endswith(".npz") else path + "_transformer.pt"
        meta_path = path.replace(".npz", "_transformer_meta.json") if path.endswith(".npz") else path + "_transformer_meta.json"
        torch.save({
            "transformer": self._transformer.state_dict(),
            "w2": self._w2.state_dict(),
            "w_emotion": self._w_emotion.state_dict(),
            "org_w1": self._org_w1.state_dict(),
            "org_w2": self._org_w2.state_dict(),
        }, torch_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "memory": self.memory,
                "irrelevant_memory": self.irrelevant_memory,
                "emotion_state": {"dim0": self.emotion_core.state.dim0, "dim1": self.emotion_core.state.dim1, "dim2": self.emotion_core.state.dim2},
                "identity_drive": self.identity_drive,
                "model_name": self.model_name,
                "creator_name": self.creator_name,
                "b2": self._b2.data.cpu().numpy().tolist(),
                "hidden_dim": self.hidden_dim,
                "sensory_dim": self.sensory_dim,
                "transformer_layers": self.transformer_layers,
                "transformer_heads": self.transformer_heads,
                "context_max_len": self.context_max_len,
            }, f, ensure_ascii=False)

    def save(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if self._use_transformer:
            self._save_transformer(path)
            return
        data = {
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "seed": self.seed,
            "sensory_dim": self.sensory_dim,
            "max_memory": self.max_memory,
            "model_name": self.model_name,
            "creator_name": self.creator_name,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "E": self._np_to_list(self.E),
            "lstm_Wxi": self._np_to_list(self.lstm_Wxi),
            "lstm_Wxf": self._np_to_list(self.lstm_Wxf),
            "lstm_Wxg": self._np_to_list(self.lstm_Wxg),
            "lstm_Wxo": self._np_to_list(self.lstm_Wxo),
            "lstm_Whi": self._np_to_list(self.lstm_Whi),
            "lstm_Whf": self._np_to_list(self.lstm_Whf),
            "lstm_Whg": self._np_to_list(self.lstm_Whg),
            "lstm_Who": self._np_to_list(self.lstm_Who),
            "lstm_bi": self._np_to_list(self.lstm_bi),
            "lstm_bf": self._np_to_list(self.lstm_bf),
            "lstm_bg": self._np_to_list(self.lstm_bg),
            "lstm_bo": self._np_to_list(self.lstm_bo),
            "W2": self._np_to_list(self.W2),
            "b2": self._np_to_list(self.b2),
            "org_W1": self._np_to_list(self.org_W1),
            "org_b1": self._np_to_list(self.org_b1),
            "org_W2": self._np_to_list(self.org_W2),
            "W_emotion_logits": self._np_to_list(self.W_emotion_logits),
            "memory": self.memory,
            "irrelevant_memory": self.irrelevant_memory,
            "emotion_state": {
                "dim0": self.emotion_core.state.dim0,
                "dim1": self.emotion_core.state.dim1,
                "dim2": self.emotion_core.state.dim2,
            },
            "identity_drive": self.identity_drive,
            "perspective_learned": {
                k: {"keywords": list(v["keywords"]), "token_strs": [self.id_to_token[t] for t in v["token_bag"] if 0 <= t < len(self.id_to_token)]}
                for k, v in self.perspective_core.learned.items()
            },
        }
        if path.endswith(".npz"):
            meta_path = path.replace(".npz", "_meta.json")
            meta = {
                "token_to_id": data["token_to_id"],
                "id_to_token": data["id_to_token"],
                "memory": data["memory"],
                "irrelevant_memory": data["irrelevant_memory"],
                "emotion_state": data["emotion_state"],
                "identity_drive": data["identity_drive"],
                "perspective_learned": data["perspective_learned"],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            np.savez_compressed(
                path,
                E=self.E, lstm_Wxi=self.lstm_Wxi, lstm_Wxf=self.lstm_Wxf,
                lstm_Wxg=self.lstm_Wxg, lstm_Wxo=self.lstm_Wxo,
                lstm_Whi=self.lstm_Whi, lstm_Whf=self.lstm_Whf,
                lstm_Whg=self.lstm_Whg, lstm_Who=self.lstm_Who,
                lstm_bi=self.lstm_bi, lstm_bf=self.lstm_bf,
                lstm_bg=self.lstm_bg, lstm_bo=self.lstm_bo,
                W2=self.W2, b2=self.b2,
                org_W1=self.org_W1, org_b1=self.org_b1, org_W2=self.org_W2,
                W_emotion_logits=self.W_emotion_logits,
                hidden_dim=np.array(self.hidden_dim),
                lr=np.array(self.lr), seed=np.array(self.seed),
                sensory_dim=np.array(self.sensory_dim),
                max_memory=np.array(self.max_memory),
            )
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

    @classmethod
    def _load_transformer(cls, path: str, torch_path: str, meta_path: str) -> "NeuralAffectiveModel":
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m = cls(
            hidden_dim=meta["hidden_dim"],
            sensory_dim=meta["sensory_dim"],
            max_memory=0,
            use_transformer=True,
            context_max_len=meta["context_max_len"],
            transformer_layers=meta["transformer_layers"],
            transformer_heads=meta["transformer_heads"],
        )
        m.token_to_id = {str(k): int(v) for k, v in meta["token_to_id"].items()}
        m.id_to_token = [str(x) for x in meta["id_to_token"]]
        m.memory = meta.get("memory", [])
        m.irrelevant_memory = meta.get("irrelevant_memory", {})
        m._rebuild_memory_index()
        e = meta.get("emotion_state", {})
        m.emotion_core = FuzzyEmotionCore(EmotionState(
            dim0=e.get("dim0", 0), dim1=e.get("dim1", 0), dim2=e.get("dim2", 0)
        ))
        m.identity_drive = meta.get("identity_drive", [0.045, 0.045, 0.045])
        m.model_name = meta.get("model_name", "模型")
        m.creator_name = meta.get("creator_name", "创造者")
        state = torch.load(torch_path, map_location=m._device)
        m._transformer.load_state_dict(state["transformer"])
        m._w2.load_state_dict(state["w2"])
        m._w_emotion.load_state_dict(state["w_emotion"])
        m._org_w1.load_state_dict(state["org_w1"])
        m._org_w2.load_state_dict(state["org_w2"])
        m._b2.data = torch.tensor(meta["b2"], dtype=torch.float32, device=m._device)
        return m

    @classmethod
    def load(cls, path: str) -> "NeuralAffectiveModel":
        torch_path = path.replace(".npz", "_transformer.pt") if path.endswith(".npz") else path + "_transformer.pt"
        meta_path = path.replace(".npz", "_transformer_meta.json") if path.endswith(".npz") else path + "_transformer_meta.json"
        if os.path.exists(torch_path) and os.path.exists(meta_path) and _TORCH_AVAILABLE:
            return cls._load_transformer(path, torch_path, meta_path)
        if path.endswith(".npz"):
            meta_path = path.replace(".npz", "_meta.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            archive = np.load(path, allow_pickle=True)
            data = {
                "hidden_dim": int(archive["hidden_dim"]),
                "lr": float(archive["lr"]),
                "seed": int(archive["seed"]),
                "sensory_dim": int(archive["sensory_dim"]),
                "max_memory": int(archive["max_memory"]),
                "token_to_id": meta["token_to_id"],
                "id_to_token": meta["id_to_token"],
                "memory": meta["memory"],
                "irrelevant_memory": meta["irrelevant_memory"],
                "emotion_state": meta["emotion_state"],
                "identity_drive": meta["identity_drive"],
                "perspective_learned": meta["perspective_learned"],
                "E": archive["E"],
                "lstm_Wxi": archive["lstm_Wxi"], "lstm_Wxf": archive["lstm_Wxf"],
                "lstm_Wxg": archive["lstm_Wxg"], "lstm_Wxo": archive["lstm_Wxo"],
                "lstm_Whi": archive["lstm_Whi"], "lstm_Whf": archive["lstm_Whf"],
                "lstm_Whg": archive["lstm_Whg"], "lstm_Who": archive["lstm_Who"],
                "lstm_bi": archive["lstm_bi"], "lstm_bf": archive["lstm_bf"],
                "lstm_bg": archive["lstm_bg"], "lstm_bo": archive["lstm_bo"],
                "W2": archive["W2"], "b2": archive["b2"],
                "org_W1": archive["org_W1"], "org_b1": archive["org_b1"],
                "org_W2": archive["org_W2"],
                "W_emotion_logits": archive["W_emotion_logits"],
            }
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        m = cls(
            hidden_dim=int(data.get("hidden_dim", 48)),
            lr=float(data.get("lr", 0.04)),
            seed=int(data.get("seed", 42)),
            sensory_dim=int(data.get("sensory_dim", 8)),
            max_memory=int(data.get("max_memory", 0)),
            model_name=str(data.get("model_name", "模型")),
            creator_name=str(data.get("creator_name", "创造者")),
        )
        m.token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        m.id_to_token = [str(x) for x in data["id_to_token"]]
        m.E = np.array(data["E"], dtype=np.float64)
        if "lstm_Wxi" in data:
            for name in ("lstm_Wxi", "lstm_Wxf", "lstm_Wxg", "lstm_Wxo",
                         "lstm_Whi", "lstm_Whf", "lstm_Whg", "lstm_Who"):
                setattr(m, name, np.array(data[name], dtype=np.float64))
            for name in ("lstm_bi", "lstm_bf", "lstm_bg", "lstm_bo"):
                setattr(m, name, np.array(data[name], dtype=np.float64))
        m.W2 = np.array(data["W2"], dtype=np.float64)
        m.b2 = np.array(data["b2"], dtype=np.float64)
        if "org_W1" in data:
            m.org_W1 = np.array(data["org_W1"], dtype=np.float64)
            m.org_b1 = np.array(data["org_b1"], dtype=np.float64)
            m.org_W2 = np.array(data["org_W2"], dtype=np.float64)
            if m.org_W2.shape[1] < m.vocab_size:
                pad = np.random.randn(m.org_dim, m.vocab_size - m.org_W2.shape[1]).astype(np.float64) * EMBED_SCALE
                m.org_W2 = np.hstack([m.org_W2, pad])
        else:
            m._init_org_layer()
        if "W_emotion_logits" in data:
            m.W_emotion_logits = np.array(data["W_emotion_logits"], dtype=np.float64)
        else:
            m.W_emotion_logits = np.random.randn(m.emotion_dim, m.vocab_size).astype(np.float64) * EMBED_SCALE
        raw_memory = data.get("memory", [])
        parsed_memory: List[Tuple[Any, int, float, List[float], List[float]]] = []
        for item in raw_memory:
            if not isinstance(item, (list, tuple)) or len(item) < 5:
                continue
            ctx_or_prev, target_id, priority, sensory_vec, emotion_vec = item[:5]
            parsed_ctx = [int(x) for x in ctx_or_prev] if isinstance(ctx_or_prev, (list, tuple)) else [int(ctx_or_prev)]
            parsed_memory.append((parsed_ctx, int(target_id), float(priority), [float(x) for x in sensory_vec], [float(x) for x in emotion_vec]))
        m.memory = parsed_memory
        m._rebuild_memory_index()
        raw_irrelevant = data.get("irrelevant_memory", {})
        parsed_irrelevant: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_irrelevant, dict):
            for k, v in raw_irrelevant.items():
                if not isinstance(v, dict):
                    continue
                parsed_irrelevant[str(k)] = {
                    "prev_id": int(v.get("prev_id", 0)),
                    "target_id": int(v.get("target_id", 0)),
                    "count": float(v.get("count", 1.0)),
                    "sensory_vector": [float(x) for x in v.get("sensory_vector", [])],
                    "emotion_vector": [float(x) for x in v.get("emotion_vector", [])],
                }
        m.irrelevant_memory = parsed_irrelevant
        e_state = data.get("emotion_state", {})
        d0 = float(e_state.get("dim0", e_state.get("valence", 0.0)))
        d1 = float(e_state.get("dim1", e_state.get("arousal", 0.0)))
        d2 = float(e_state.get("dim2", e_state.get("focus", 0.0)))
        m.emotion_core = FuzzyEmotionCore(EmotionState(dim0=d0, dim1=d1, dim2=d2))
        raw_id = data.get("identity_drive", [])
        m.identity_drive = [float(x) for x in raw_id] if isinstance(raw_id, (list, tuple)) and len(raw_id) >= 3 else [0.045, 0.045, 0.045]
        raw_pers = data.get("perspective_learned", {})
        for name, info in raw_pers.items():
            if name in m.perspective_core.learned:
                m.perspective_core.learned[name]["keywords"] = set(info.get("keywords", []))
                m.perspective_core.learned[name]["token_bag"] = []
                for s in info.get("token_strs", []):
                    m.ensure_token(s)
                    tid = m.token_to_id.get(s)
                    if tid is not None:
                        m.perspective_core.learned[name]["token_bag"].append(tid)
        return m
