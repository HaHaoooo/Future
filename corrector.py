"""
我来审查：人为主导的纠错模式

流程：你提问 → 模型回答 → 你输入期望答案 → 编辑距离对齐 → 逐 token 对错 → 模型学习。

【吐知识】若模型曾被 teacher 灌输大量知识，memory 塞满旧链，replay 会强化这些链。
correct 模式已关闭 replay；若仍无效，可加 --correct-purge-memory 清空 memory。
"""
from typing import List, Optional, Tuple

from src.neural_model import GenerationTrace, NeuralAffectiveModel
from src.system.sensory_io import SensoryContext, collect_sensory_payload
from src.system.config_types import AppConfig
from src.utils import flatten_tokens as _flatten
from src.trace_builder import build_trace_from_answer as _build_correction_trace


# =============================================================================
# 编辑距离对齐
# =============================================================================


def _align_tokens_for_correctness(
    model_tokens: List[str],
    desired_tokens: List[str],
) -> Tuple[List[Optional[bool]], List[Optional[str]]]:
    """
    Levenshtein DP：模型输出 token 与期望 token 最优对齐。
    支持插入、删除、替换。返回 (correctness, corrections)，长度 == len(model_tokens)。
    """
    M, D = model_tokens, desired_tokens
    m, n = len(M), len(D)
    if m == 0:
        return [], []

    INS, DEL, SUB, MATCH = 0, 1, 2, 3
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        ops[i][0] = DEL
    for j in range(1, n + 1):
        dp[0][j] = j
        ops[0][j] = INS
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if M[i - 1] == D[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = MATCH
            else:
                candidates = [
                    (dp[i - 1][j] + 1, DEL),
                    (dp[i][j - 1] + 1, INS),
                    (dp[i - 1][j - 1] + 1, SUB),
                ]
                dp[i][j], ops[i][j] = min(candidates, key=lambda x: x[0])

    # 回溯构造对齐
    aligned: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] in (MATCH, SUB):
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ops[i][j] == DEL):
            aligned.append((i - 1, None))
            i -= 1
        else:
            j -= 1
    aligned.reverse()

    correctness: List[Optional[bool]] = []
    corrections: List[Optional[str]] = []
    for mi, di in aligned:
        if mi is None:
            continue
        if di is not None and M[mi] == D[di]:
            correctness.append(True)
            corrections.append(None)
        elif di is not None:
            correctness.append(False)
            corrections.append(D[di])
        else:
            correctness.append(False)
            corrections.append(None)
    return correctness, corrections


def _print_token_correctness(
    tokens: List[str],
    correctness: List[Optional[bool]],
    corrections: List[Optional[str]],
) -> None:
    """打印每个 token 的 ✓/✗ 及纠错建议。"""
    parts: List[str] = []
    for tok, ok, corr in zip(tokens, correctness, corrections):
        mark = "✓" if ok is True else ("✗" if ok is False else "?")
        parts.append(f"「{tok}」{mark}→「{corr}」" if corr else f"「{tok}」{mark}")
    print("  " + " ".join(parts))
    n_ok = sum(1 for c in correctness if c is True)
    n_bad = sum(1 for c in correctness if c is False)
    n_irr = sum(1 for c in correctness if c is None)
    total = len(tokens)
    if total > 0:
        acc = n_ok / total
        print(f"  对={n_ok} 错={n_bad} 无关={n_irr} 正确率={acc:.0%}")


# =============================================================================
# 入口
# =============================================================================


def run_correct_session(model: NeuralAffectiveModel, config: AppConfig) -> None:
    """
    人为主导纠错交互：
    1. 你提问，模型回答
    2. 你输入期望答案
    3. 编辑距离对齐 → 展示逐 token 对错
    4. apply_feedback 纠错 + reinforce_comprehension 巩固 prompt→答案
    """
    print("=" * 50)
    print("  Future 我来审查 —— 人为主导，模型理解感悟")
    print("=" * 50)
    print("  用法：输入问题 → 模型回答 → 你输入期望答案 → 模型学习")
    print("  命令：quit 退出  skip 跳过本轮（不纠错）")
    print("=" * 50)

    if getattr(config, "correct_purge_memory", False):
        model.memory.clear()
        model.irrelevant_memory.clear()
        print("[审查] 已清空 memory，避免旧知识反强化。")
        model.save(config.model_path)

    ctx = SensoryContext()
    from src.agent import get_continuation_intent

    while True:
        try:
            prompt = input("\n你（问题）> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        continuation_intent = get_continuation_intent(prompt, ctx.last_intent, ctx.last_prompt)
        payload = collect_sensory_payload(prompt, ctx, config)
        trace, rep = model.deliberate_generate(
            prompt, payload, config.max_len, config.temperature, config.thought_trials,
            conversation_history=ctx.conversation_history,
            continuation_intent=continuation_intent,
        )
        ctx.update_from_thought_report(rep)
        ctx.set_last_intent(rep.get("inferred_intent"), prompt)
        ctx.push_turn(prompt, _flatten(trace.tokens))
        pred = _flatten(trace.tokens)
        print(f"\n模型回答> {pred}")
        if not trace.tokens:
            print("  （空输出，可输入期望答案让模型学习）")
            pred = ""

        try:
            desired = input("你（期望答案，回车跳过）> ").strip()
        except EOFError:
            break
        if not desired:
            print("[审查] 已跳过，无纠错。")
            continue
        if desired.lower() in {"quit", "exit", "q"}:
            break
        if desired.lower() in {"skip", "s"}:
            print("[审查] 已跳过。")
            continue

        desired_tokens = model.tokenize(desired)
        correctness, corrections = _align_tokens_for_correctness(trace.tokens, desired_tokens)
        n = len(trace.tokens)
        correctness = (correctness + [False] * n)[:n]
        corrections = (corrections + [None] * n)[:n]

        print("\n[审查] 逐词对错：")
        _print_token_correctness(trace.tokens, correctness, corrections)

        passes = max(3, config.teacher_learning_passes)
        train_rep = model.apply_feedback(
            trace, correctness, corrections,
            learning_passes=passes,
            replay_steps=0,
            absorb_knowledge=True,
        )
        model.save(config.model_path)

        correct_trace = _build_correction_trace(
            model, prompt, desired,
            [0.0] * model.sensory_dim, [0.0] * model.emotion_dim,
        )
        model.reinforce_comprehension(correct_trace, extra_passes=2, lr_scale=1.0, absorb_knowledge=True)
        model.apply_feedback(
            correct_trace,
            [True] * len(correct_trace.token_ids),
            [None] * len(correct_trace.token_ids),
            learning_passes=2,
            replay_steps=0,
            absorb_knowledge=True,
        )
        model.save(config.model_path)
        ctx.update_from_train_report(train_rep)

        avg_loss = train_rep.get("avg_loss", 0.0)
        print(f"[审查] 已学习，loss={avg_loss:.4f}，模型已根据你的答案理解感悟。")
