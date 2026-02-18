# Future: 神经网络交互学习系统

核心闭环：**你输入 → 模型输出 → 你判对错 → 模型立刻学习**。

---

## 一、模型架构

### 1.1 整体结构（统一整合）

```
输入 token 序列
    ↓
嵌入层 E [vocab × hidden_dim] + 感官向量 [sensory_dim] + 情感向量 [emotion_dim]
    ↓
[LSTM 或 Transformer]（numpy=LSTM / torch=6 层 Transformer+RoPE+8K 上下文）
    ↓
Scaled Dot-Product Attention（Q=最后隐态，K=V=历史隐态堆叠）
    ↓
RMSNorm + 组织层（org_layer，soft 意图→token 偏置）
    ↓
输出 logits [vocab] → softmax(temperature) → 采样下一 token
```

- **词表**：动态扩展，`<PAD> <BOS> <EOS> <UNK>` + 训练中出现的 token
- **Token 化**：中文字符级 + 英文按空格/标点
- **训练**：BPTT 逐词反传，梯度裁剪（max_norm=5.0）

### 1.2 模块组成


| 模块                    | 说明                                                   |
| --------------------- | ---------------------------------------------------- |
| **LSTM**              | 单层，输入 = 嵌入 + 感官 + 情感拼接                               |
| **Attention**         | scaled dot-product，最后隐态 attend 历史隐态                  |
| **组织层**               | 意图→目标 token 的软偏置，不覆盖情感与身份                            |
| **FuzzyEmotionCore**  | 三维内在状态，decay + 多源影响（感官/推理/学习）                        |
| **PerspectiveCore**   | 视角偏置（第一人称存在感、双字关键词）                                  |
| **Memory**            | (context, target, priority, sensory, emotion)，倒排索引检索 |
| **Irrelevant memory** | 「无关」反馈存入，供模型后续自悟                                     |


### 1.3 智慧体推理流水线

每次生成前按顺序准备：

1. **工作记忆**：拼接对话历史 + 当前输入
2. **理解**：意图推断（关键词→语义分组）或沿用延续追问意图
3. **情感**：感官更新 + 输入语义→情感影响（无双重叠加）
4. **回忆**：意图偏置 token + 倒排索引检索增强
5. **生成**：按步注入视角偏置 + 逻辑衔接偏置

---

## 二、使用方式

### 2.1 启动模式


| 模式              | 说明                     | 入口                       |
| --------------- | ---------------------- | ------------------------ |
| **interactive** | 纯对话，你输入→模型答→可反馈对错      | `python3 main.py`        |
| **teacher**     | 完形填空预训练，奠基+四域+造句       | `./run_train.sh teacher` |
| **correct**     | 人为主导纠错，逐 token 对错+期望答案 | `./run_train.sh correct` |


### 2.2 统一模型（LSTM / Transformer 整合）

**NeuralAffectiveModel** 单一类，按 `--backend` 选择编码器：

| 后端    | 说明                   | 启动方式              |
| ----- | -------------------- | ----------------- |
| numpy | LSTM，纯 NumPy           | `--backend numpy` |
| torch | Transformer，8K+ 上下文    | `--backend torch` |

```bash
# Transformer（高性能 + 强逻辑）
python3 main.py --backend torch --context-max-len 8192

# LSTM（轻量）
python3 main.py --backend numpy
```

### 2.3 交互式对话（默认）

```bash
python3 main.py --model checkpoints/data.npz
```

- 首次运行创建新模型，后续自动加载
- 输入 `quit` / `exit` / `q` 结束
- 反馈格式：`1 2 0` 表示 正确/无关/错误；对错误词可输入修正词

### 2.4 训练（run_train）

```bash
./run_train.sh teacher   # 完形填空预训练（默认）
./run_train.sh correct   # 人为主导纠错
```

- **teacher**：语料完形填空 → 身份种子 → 33 课自动验证（奠基/四域/造句）
- **correct**：输入问题 → 模型答 → 你给期望答案 → 编辑距离对齐 → 纠错学习

**correct 若出现「吐知识」**：模型曾被灌输大量知识导致 memory 塞满旧链条。correct 模式已关闭 replay；若仍无效，可加 `--correct-purge-memory` 清空 memory（会丢失历史经验）。

### 2.5 从头开始

```bash
rm -rf checkpoints/data.npz
./run_train.sh teacher
./run_train.sh correct   # 可选：教身份与纠错
```

---

## 三、训练模块结构

### 3.1 teacher.py（完形填空预训练）

```
_run_cloze()            # context→下一词，逐词 train_one
seed_xiaolai_logic()    # 身份/问候/因果逻辑链植入
_run_auto_validation()  # 每课 提问→答→判断→纠错→强化，未达标不停止
```

**课程**：奠基 9 课 + 四域 4 课 + 造句 20 课。语料 = `_PRETRAIN` + 课程 explanation / question-answer。

### 3.2 corrector.py（人为主导纠错）

```
_align_tokens_for_correctness()  # 编辑距离对齐（Levenshtein DP）
run_correct_session()            # 交互：输入问题→模型答→你给期望→对齐→纠错
```

**编辑距离**：支持插入/删除/替换，按模型 token 序列对齐到期望 token 序列。

### 3.3 trace_builder.py

`build_trace_from_answer()`：从 question + reference_answer 构建 GenerationTrace，供 teacher/corrector 调用 `apply_feedback`。

---

## 四、可用配置

### 4.1 通用


| 参数                     | 类型    | 默认                     | 说明                |
| ---------------------- | ----- | ---------------------- | ----------------- |
| `--model`              | str   | `checkpoints/data.npz` | 模型路径              |
| `--backend`            | str   | numpy                  | `numpy` / `torch` |
| `--context-max-len`    | int   | 8192                   | 上下文窗口（仅 torch）    |
| `--transformer-layers` | int   | 6                      | Transformer 层数    |
| `--transformer-heads`  | int   | 8                      | 注意力头数             |
| `--hidden-dim`         | int   | 128                    | 隐层维度              |
| `--lr`                 | float | 0.04                   | 学习率               |
| `--seed`               | int   | 42                     | 随机种子              |
| `--max-len`            | int   | 0                      | 输出硬上限，0=不设限       |
| `--temperature`        | float | 0.9                    | 采样温度              |
| `--thought-trials`     | int   | 7                      | 内部候选思考次数          |
| `--learning-passes`    | int   | 10                     | 每条反馈放大学习次数        |
| `--replay-steps`       | int   | 320                    | 每轮反馈后经验回放步数       |
| `--sensory-dim`        | int   | 8                      | 感官向量维度            |
| `--sensor-file`        | str   | ""                     | 外部感官 JSON 路径      |
| `--show-thought`       | flag  | False                  | 显示思考评分报告          |


### 4.2 模式


| 参数       | 类型  | 默认          | 说明                                    |
| -------- | --- | ----------- | ------------------------------------- |
| `--mode` | str | interactive | `interactive` / `teacher` / `correct` |


### 4.3 teacher


| 参数                          | 类型   | 默认    | 说明        |
| --------------------------- | ---- | ----- | --------- |
| `--teacher-episodes`        | int  | 500   | 训练轮数      |
| `--teacher-log-every`       | int  | 50    | 每 N 课打印进度 |
| `--teacher-learning-passes` | int  | 8     | 每课学习次数    |
| `--teacher-replay-steps`    | int  | 160   | 每课回放步数    |
| `--teacher-user-facts`      | str  | ""    | 用         |
| `--teacher-user-facts-file` | str  | ""    | 事实文件路径    |
| `--no-teacher-chat`         | flag | False | 关闭课间对话    |
| `--teacher-chat-max-turns`  | int  | 2     | 课间对话最大轮次  |


### 4.4 correct


| 参数                       | 类型   | 默认    | 说明           |
| ------------------------ | ---- | ----- | ------------ |
| `--correct-purge-memory` | flag | False | 启动时清空 memory |


### 4.5 示例

```bash
# 交互式，强学习
python3 main.py --thought-trials 9 --learning-passes 12 --replay-steps 480 --show-thought

# teacher 自定义
python3 main.py --mode teacher --teacher-learning-passes 12 --teacher-replay-steps 200

# correct 清空 memory（慎用）
python3 main.py --mode correct --correct-purge-memory
```

---

## 五、依赖

```bash
pip install -r requirements.txt  # numpy
```

