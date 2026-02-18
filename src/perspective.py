"""
视角模块

第一视角（存在感）为必备；其他视角（物理、时间、数学、情感、逻辑）由模型按概念自整理。
双字关键词，防止日常对话误触发。
"""
from typing import Dict, List, Optional, Sequence, Set


# =============================================================================
# 第一视角
# =============================================================================

# 第一视角：模型知道「我存在」
FIRST_PERSON_KEYWORDS = frozenset({"谁", "你", "叫", "名字", "啥", "我", "什么", "称呼", "哪位"})
FIRST_PERSON_TOKENS = ("我", "我的", "小", "来", "是", "在", "存")

# 其他视角种子：模型可从训练中扩展
PERSPECTIVE_SEEDS: Dict[str, Set[str]] = {
    "物理": {"物理", "力学", "质量", "速度", "牛顿", "定律", "重力", "惯性", "加速"},
    "时间": {"时间", "过去", "现在", "未来", "之前", "之后", "同时"},
    "数学": {"数学", "计算", "加减", "乘除", "方程", "公式"},
    "情感": {"感觉", "喜欢", "难过", "开心", "爱", "讨厌", "害怕"},
    "逻辑": {"因为", "所以", "如果", "虽然", "但是", "因果", "条件", "转折", "逻辑"},
}


# =============================================================================
# 视角核心
# =============================================================================


class PerspectiveCore:
    """
    视角核心：第一视角恒在，其他视角按概念自整理。
    训练时 learn_from 将 answer token 归入触发的概念；生成时 get_learned_boosts 做软提升。
    """

    def __init__(self) -> None:
        self.first_person_baseline: float = 0.25
        self.first_person_self_reflective: float = 0.55
        self.learned: Dict[str, Dict] = {}
        for name, kws in PERSPECTIVE_SEEDS.items():
            self.learned[name] = {"keywords": set(kws), "token_bag": []}

    def is_self_referential(self, prompt_tokens: Sequence[str]) -> bool:
        """是否在问「我」（身份、创造者、你好吗 等）。"""
        s = set(t.lower() for t in prompt_tokens)
        if "你" in s and any(k in s for k in {"谁", "叫", "好", "吗", "记", "得"}):
            return True
        if "我" in s and "谁" in s:
            return True
        return False

    def get_first_person_strength(self, prompt_tokens: Sequence[str]) -> float:
        """第一视角强度：自指时强，否则为基线。"""
        if self.is_self_referential(prompt_tokens):
            return self.first_person_self_reflective
        return self.first_person_baseline

    def detect_concepts(self, prompt_tokens: Sequence[str]) -> List[str]:
        """从 prompt 检测触发的概念视角。"""
        s = set(t.lower() for t in prompt_tokens)
        out: List[str] = []
        for name, data in self.learned.items():
            if s & data["keywords"]:
                out.append(name)
        return out

    def learn_from(
        self,
        prompt_tokens: Sequence[str],
        answer_token_ids: Sequence[int],
        token_id_limit: int = 30,
    ) -> None:
        """训练时：按 prompt 概念，将 answer token 归入对应视角。"""
        concepts = self.detect_concepts(prompt_tokens)
        for c in concepts:
            bag = self.learned[c]["token_bag"]
            for tid in answer_token_ids[:token_id_limit]:
                if tid not in bag:
                    bag.append(tid)
                if len(bag) > 80:
                    bag[:] = bag[-80:]

    def get_learned_boosts(
        self,
        prompt_tokens: Sequence[str],
        strength: float = 0.12,
    ) -> Dict[int, float]:
        """从概念视角获取 token 软提升。"""
        concepts = self.detect_concepts(prompt_tokens)
        boosts: Dict[int, float] = {}
        for c in concepts:
            bag = self.learned[c].get("token_bag", [])
            for tid in bag:
                boosts[tid] = boosts.get(tid, 0.0) + strength / max(1, len(concepts))
        return boosts
