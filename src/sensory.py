"""
感官模块

外部感官数据解析与向量化。支持内联格式、JSON 文件、合并多源。
"""
import json
import os
from typing import Dict, Iterable, List

from src.utils import clip as _clip


# =============================================================================
# 常量
# =============================================================================

# 语义键到维度的映射，减少 hash 碰撞
_SENSORY_KEY_MAP: Dict[str, int] = {
    "stress": 0, "noise": 1, "confidence": 2, "energy": 3,
    "valence": 4, "arousal": 5, "force_calm": 6, "camera": 7,
}


# =============================================================================
# 工具
# =============================================================================


def _stable_hash(token: str) -> int:
    """FNV-1a 风格哈希，用于未映射键的索引。"""
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def parse_inline_sensory(raw: str) -> Dict[str, float]:
    """
    解析内联格式："noise=0.2 stress=0.5 camera=dark"
    非数值会 hash 映射到 [-1, 1]。
    """
    out: Dict[str, float] = {}
    if not raw.strip():
        return out
    parts = [p.strip() for p in raw.split() if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        k = key.strip().lower()
        v = value.strip().lower()
        try:
            out[k] = _clip(float(v), -1.0, 1.0)
        except ValueError:
            hashed = (_stable_hash(v) % 2000) / 1000.0 - 1.0
            out[k] = _clip(hashed, -1.0, 1.0)
    return out


def load_sensor_file(path: str) -> Dict[str, float]:
    """从 JSON 文件加载感官数据。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        key = str(k).strip().lower()
        try:
            out[key] = _clip(float(v), -1.0, 1.0)
        except (TypeError, ValueError):
            hashed = (_stable_hash(str(v)) % 2000) / 1000.0 - 1.0
            out[key] = _clip(hashed, -1.0, 1.0)
    return out


def merge_sensory_payloads(*payloads: Dict[str, float]) -> Dict[str, float]:
    """合并多源感官，后者覆盖前者。"""
    merged: Dict[str, float] = {}
    for payload in payloads:
        for k, v in payload.items():
            merged[k] = _clip(float(v), -1.0, 1.0)
    return merged


def encode_sensory_vector(payload: Dict[str, float], sensory_dim: int) -> List[float]:
    """
    将感官 payload 编码为定长向量。
    优先使用语义键映射，其余键用 hash 填充空位。
    """
    vec = [0.0] * max(1, sensory_dim)
    if not payload:
        return vec
    for key, value in payload.items():
        k = key.strip().lower()
        idx = _SENSORY_KEY_MAP.get(k)
        if idx is not None and idx < len(vec):
            vec[idx] = _clip(float(value), -1.0, 1.0)
        else:
            idx = _stable_hash(k) % len(vec)
            vec[idx] = _clip(vec[idx] + float(value), -1.0, 1.0)
    for i in range(len(vec)):
        vec[i] = _clip(vec[i], -1.0, 1.0)
    return vec


def payload_to_pairs(payload: Dict[str, float]) -> Iterable[str]:
    """将 payload 转为 "key=value" 字符串序列，用于展示。"""
    for k, v in sorted(payload.items()):
        yield f"{k}={v:.2f}"
