"""
Trace 构建工具

从 prompt + 期望答案 构建 GenerationTrace，供 teacher / corrector 调用 apply_feedback。
"""
from typing import List

from src.neural_model import GenerationTrace, NeuralAffectiveModel


def build_trace_from_answer(
    model: NeuralAffectiveModel,
    prompt: str,
    desired_answer: str,
    sensory_vector: List[float],
    emotion_vector: List[float],
) -> GenerationTrace:
    """
    从 prompt 与期望答案构建 trace，用于强化学习。
    末尾附加 EOS token，让 apply_feedback 同时学会在答案结束时停止。
    prev_ids 为前一 token 的 id 链，用于 BPTT。
    """
    pt = model.tokenize(prompt)
    dt = model.tokenize(desired_answer)
    for t in pt + dt:
        model.ensure_token(t)
    if not dt:
        dt = [model.UNK]
        model.ensure_token(model.UNK)
    # 附加 EOS：让模型学会在答案结束后停止
    dt_with_eos = dt + [model.EOS]
    aid = [model.token_to_id[t] for t in dt_with_eos]
    prompt_ids = [model.token_to_id[t] for t in pt]
    bos_id = model.token_to_id[model.BOS]
    pid = bos_id if not pt else model.token_to_id[pt[-1]]
    prev_ids: List[int] = []
    for i in aid:
        prev_ids.append(pid)
        pid = i
    return GenerationTrace(
        tokens=dt_with_eos,
        token_ids=aid,
        prev_ids=prev_ids,
        confidences=[1.0] * len(aid),
        sensory_vector=list(sensory_vector),
        emotion_vector=list(emotion_vector),
        prompt_token_ids=prompt_ids,
    )
