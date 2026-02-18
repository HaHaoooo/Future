#!/bin/bash
# run_train：统一训练入口
# 用法：./run_train.sh teacher   # 完形填空预训练（默认）
#       ./run_train.sh loop      # 循环训练（不断重复，Ctrl+C 停止）
#       ./run_train.sh correct   # 人为主导纠错

MODEL_PATH="checkpoints/xiaolai.npz"
MODE="${1:-teacher}"

case "$MODE" in
  teacher)
    python3 main.py --mode teacher \
      --model "$MODEL_PATH" \
      --teacher-learning-passes 3 \
      --teacher-replay-steps 24
    ;;
  loop)
    python3 main.py --mode teacher \
      --model "$MODEL_PATH" \
      --teacher-learning-passes 3 \
      --teacher-replay-steps 24 \
      --teacher-loop
    ;;
  correct)
    python3 main.py --mode correct \
      --model "$MODEL_PATH" \
      --teacher-learning-passes 3 \
      --temperature 0.85 \
      --thought-trials 5
    ;;
  *)
    echo "用法: $0 {teacher|correct|loop}"
    echo "  teacher  完形填空预训练（默认）"
    echo "  loop     循环训练（Ctrl+C 停止）"
    echo "  correct  人为主导纠错"
    exit 1
    ;;
esac
