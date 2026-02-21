"""
反馈解析

交互对话中逐 token 反馈的输入解析与收集。
支持 1/0/2（对/错/无关）及多种同义词。
"""
from typing import List, Optional, Sequence, Tuple


def parse_feedback(raw: str, n: int) -> List[Optional[bool]]:
    parts = [x.strip().lower() for x in raw.split() if x.strip()]
    if len(parts) != n:
        raise ValueError(f"反馈数量需要是 {n} 个。")

    yes = {"1", "y", "yes", "+", "t", "true", "对", "正确"}
    no = {"0", "n", "no", "-", "f", "false", "错", "错误"}
    neutral = {"2", "u", "unk", "?", "x", "skip", "na", "n/a", "无关", "中立", "不确定"}
    out: List[Optional[bool]] = []
    for part in parts:
        if part in yes:
            out.append(True)
        elif part in no:
            out.append(False)
        elif part in neutral:
            out.append(None)
        else:
            raise ValueError(f"无法识别反馈标记: {part}")
    return out


def collect_feedback(tokens: Sequence[str]) -> Tuple[List[Optional[bool]], List[Optional[str]]]:
    while True:
        fb_raw = input("逐词反馈（1=对, 0=错, 2=无关）：").strip()
        try:
            correctness = parse_feedback(fb_raw, len(tokens))
            break
        except ValueError as exc:
            print(f"[输入有误] {exc}")

    corrections: List[Optional[str]] = [None] * len(tokens)
    for idx, status in enumerate(correctness):
        if status is not False:
            continue
        corr = input(f"第 {idx} 个词 '{tokens[idx]}' 的正确词（可回车跳过）：").strip()
        corrections[idx] = corr if corr else None
    return correctness, corrections
