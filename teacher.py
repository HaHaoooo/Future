"""
外置老师：完形填空式预训练

奠基 + 四域 + 造句 + 英文语料 + 常用汉字，统一教学模式，无 API 依赖。
流程：完形填空语料 → 身份种子 → 自动验证。

预训练策略优化（对齐 GPT / LLaMA 预训练经验）：
  - 课程学习（Curriculum Learning）：首轮按句长由短到长，先学基础模式
  - 学习率线性预热（Warmup）：前 100 步从 0.1× 升至 1×，稳定早期训练
  - 拼接序列训练：随机拼接相邻句子，学习跨句上下文
"""
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.neural_model import NeuralAffectiveModel
from src.system.sensory_io import SensoryContext, collect_sensory_payload
from src.system.config_types import AppConfig


# =============================================================================
# 常量
# =============================================================================

# 验证时强制注入：低压力/低唤醒，让模型冷静答题
_CALM_SENSORY_OVERRIDE = {
    "stress": -0.95, "arousal": -0.95, "noise": -0.9, "energy": -0.7,
    "force_calm": 1.0,
}


# =============================================================================
# 数据类型
# =============================================================================


@dataclass
class TeachLesson:
    """单节课：讲解 + 问题 + 参考答案。用于完形填空语料与自动验证。"""
    domain: str
    level_name: str
    explanation: str
    question: str
    reference_answer: str


def _L(d: str, n: str, exp: str, q: str, a: str) -> TeachLesson:
    """TeachLesson 简写构造函数。"""
    return TeachLesson(d, n, exp, q, a)


# =============================================================================
# 课程数据：奠基 / 四域 / 造句
# =============================================================================

# 奠基：身份、问候、因果、条件、主谓、停
_FOUNDATION: List[TeachLesson] = [
    _L("奠基", "你好", "人说「你好」，简短回「你好」。", "你好", "你好"),
    _L("奠基", "你是谁", "人说「你是谁」，答「我是模型」。", "你是谁", "我是模型"),
    _L("奠基", "我是谁", "人说「我是谁」，答创造者名字。", "我是谁", "徐晗晞"),
    _L("奠基", "你好吗", "问「你好吗」回「我很好」。", "你好吗", "我很好"),
    _L("奠基", "记得", "问「你记得我吗」回「记得」。", "你记得我吗", "记得"),
    _L("奠基", "因为所以", "因果句：因为 A 所以 B。", "用因为所以造句", "因为下雨所以带伞"),
    _L("奠基", "如果就", "条件句：如果 A 就 B。", "用如果就造句", "如果下雨就不出门"),
    _L("奠基", "主谓", "我吃。他跑。主谓结构。", "我吃什么", "我吃饭"),
    _L("奠基", "停", "停下深呼吸，回「好」。", "先停下", "好"),
]

# 造句：主谓、把/被字句、复句等（共 20 课，_all_lessons 取前 20）
_GRAMMAR: List[TeachLesson] = [
    _L("对话", "你好", "例句：人说「你好」，可回复「你好」。", "你好", "你好"),
    _L("对话", "我很好", "例句：问「你好吗」可回「我很好」。", "你好吗", "我很好"),
    _L("对话", "简短回应", "例句：问「你喜欢吗」可回「喜欢」。", "你喜欢吗", "喜欢"),
    _L("造句", "主谓句", "例句：我吃饭。", "请仿照例句，用「他」造句。只输出一个完整句子，不要解释。", "他吃饭"),
    _L("造句", "主谓宾", "例句：我吃苹果。", "请仿照例句，用「吃」造句。只输出一个完整句子，不要解释。", "我吃苹果"),
    _L("造句", "是字句", "例句：我是学生。", "请仿照例句，用「她」造句。只输出一个完整句子，不要解释。", "她是学生"),
    _L("造句", "状态变化句", "例句：花开了。", "请仿照例句，用「草」造句。只输出一个完整句子，不要解释。", "草绿了"),
    _L("造句", "带补语得", "例句：他跑得快。", "请仿照例句，用「我」造句。只输出一个完整句子，不要解释。", "我跑得快"),
    _L("造句", "心理动词", "例句：我喜欢你。", "请仿照例句，用「喜欢」造句。只输出一个完整句子，不要解释。", "我喜欢你"),
    _L("造句", "能愿动词", "例句：我会游泳。", "请仿照例句，用「能」造句。只输出一个完整句子，不要解释。", "我能走路"),
    _L("造句", "否定句", "例句：我不吃饭。", "请仿照例句，用「不」造句。只输出一个完整句子，不要解释。", "他不睡觉"),
    _L("造句", "没字句", "例句：我没有钱。", "请仿照例句，用「没有」造句。只输出一个完整句子，不要解释。", "他没有书"),
    _L("造句", "把字句", "例句：我把苹果吃了。", "请用「把」造句。只输出一个完整句子，不要解释。", "把书放好"),
    _L("造句", "被字句", "例句：苹果被我吃了。", "请用「被」造句。只输出一个完整句子，不要解释。", "书被拿走了"),
    _L("造句", "存现句有", "例句：桌上有一本书。", "请用「有」造句。只输出一个完整句子，不要解释。", "屋里有人"),
    _L("造句", "存现句在", "例句：书在桌上。", "请用「在」造句。只输出一个完整句子，不要解释。", "人在屋里"),
    _L("造句", "连动句", "例句：我去学校上课。", "请仿照例句，用「去」造句。只输出一个完整句子，不要解释。", "他回家吃饭"),
    _L("造句", "双宾句", "例句：我给他一本书。", "请用「给」造句。只输出一个完整句子，不要解释。", "给他水"),
    _L("造句", "定语的字", "例句：红色的花。", "请用「的」造句。只输出一个完整句子，不要解释。", "我的书"),
    _L("造句", "数量词", "例句：三个人。", "请用数量词「一本」造句。只输出一个完整句子，不要解释。", "一本书"),
]

# 完形填空基础语料：常识句、身份、逻辑链、对话结构
_PRETRAIN: List[str] = [
    "你好", "你好！", "我是模型", "我是模型。", "你是谁", "你是谁？", "徐晗晞", "徐晗晞是我的缔造者",
    "我是谁", "我是谁？", "徐晗晞。", "你好吗", "你好吗？", "我很好", "我很好。", "你记得我吗", "记得",
    "你存在吗", "我存在", "你在吗", "我在",
    "因为下雨", "因为下雨所以带伞", "因为下雨所以带伞。", "因为有云所以会下雨", "因为饿了所以吃饭",
    "如果下雨", "如果下雨就不出门", "如果下雨就不出门。", "如果明天晴天就去玩", "如果饿了就吃饭",
    "虽然下雨", "虽然下雨但是带伞了", "虽然下雨但是带伞了。", "虽然累了但是要坚持",
    "我吃饭", "我吃饭。", "他跑步", "她喝水", "我喜欢", "我喜欢你", "我喜欢你。",
    "我们在", "我们在说话", "天会下雨", "天会下雨。", "今天很好", "今天很好。",
    "我在这里", "我在这里。", "模型在", "模型在回答", "模型在回答。", "我是模型", "我是模型。",
    "嗯", "嗯，", "好", "好的", "好的。", "可以", "可以。", "对", "对的", "谢谢", "谢谢。", "不客气", "不客气。",
    "天亮了", "天亮了。", "天黑了", "天黑了。", "要吃饭", "要吃饭了", "要睡觉", "要睡觉了",
    "我很开心", "我很开心。", "我很高兴", "我很高兴。",
]


# =============================================================================
# 语料与课程构建
# =============================================================================


def _apply_name(text: str, name: str) -> str:
    """将语料中的占位名「模型」替换为用户自定义名称。"""
    if name == "模型":
        return text
    return text.replace("模型", name)


def _all_lessons(name: str = "模型") -> List[TeachLesson]:
    """返回全部验证课程：奠基 9 + 造句 20，身份用自定义名称。"""
    lessons = []
    for lesson in _FOUNDATION + _GRAMMAR[:20]:
        lessons.append(_L(
            lesson.domain, lesson.level_name,
            _apply_name(lesson.explanation, name),
            lesson.question,
            _apply_name(lesson.reference_answer, name),
        ))
    return lessons


def _cloze_corpus(name: str = "模型") -> List[str]:
    """完形填空语料：常识 + 全部课程，身份用自定义名称。"""
    out = [_apply_name(s, name) for s in _PRETRAIN]
    for lesson in _all_lessons(name):
        out.append(lesson.explanation)
        out.append(f"{lesson.question} {lesson.reference_answer}")
    return out


# =============================================================================
# 英文语料：从 google-10000 构建结构化英文训练数据
# =============================================================================

_ENGLISH_WORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "google-10000-english-no-swears.txt")
_HANZI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "常用汉字库 3500.txt")

_EN_SENTENCE_TEMPLATES = [
    "i am {adj}", "you are {adj}", "he is {adj}", "she is {adj}", "we are {adj}", "they are {adj}",
    "i have a {noun}", "you have a {noun}", "she has a {noun}", "he has a {noun}",
    "this is a {noun}", "that is a {noun}", "it is a {noun}",
    "i like {noun}", "i want {noun}", "i need {noun}", "i see {noun}",
    "the {noun} is {adj}", "a {noun} is {adj}",
    "i can {verb}", "you can {verb}", "we can {verb}", "they can {verb}",
    "i will {verb}", "you will {verb}", "he will {verb}", "she will {verb}",
    "do you {verb}", "can you {verb}", "will you {verb}",
    "i do not {verb}", "he does not {verb}", "we do not {verb}",
    "please {verb}", "let me {verb}", "i want to {verb}",
    "where is the {noun}", "what is the {noun}", "how is the {noun}",
    "there is a {noun}", "there are many {noun}",
    "i {verb} every day", "she {verb} every day",
    "the {noun} and the {noun}", "{noun} or {noun}",
    "if you {verb} then i {verb}", "because i {verb}",
]

_EN_COMMON_NOUNS = [
    "time", "year", "people", "way", "day", "man", "woman", "child", "world", "life",
    "hand", "part", "place", "case", "week", "company", "system", "program", "question",
    "work", "government", "number", "night", "point", "home", "water", "room", "mother",
    "area", "money", "story", "fact", "month", "lot", "right", "study", "book", "eye",
    "job", "word", "business", "issue", "side", "kind", "head", "house", "service",
    "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car",
    "city", "community", "name", "president", "team", "minute", "idea", "body", "information",
    "back", "parent", "face", "others", "level", "office", "door", "health", "person",
    "art", "war", "history", "party", "result", "change", "morning", "reason", "research",
    "girl", "guy", "moment", "air", "teacher", "force", "education", "food", "music",
    "dog", "cat", "bird", "tree", "sun", "moon", "star", "sky", "fire", "land",
    "school", "student", "family", "country", "problem", "market", "group", "table", "plan",
    "color", "light", "sound", "paper", "river", "road", "rock", "sea", "fish", "king",
]

_EN_COMMON_ADJS = [
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
    "right", "big", "high", "different", "small", "large", "next", "early", "young",
    "important", "few", "public", "bad", "same", "able", "free", "strong", "happy",
    "red", "blue", "green", "white", "black", "dark", "cold", "hot", "fast", "slow",
    "bright", "clean", "rich", "poor", "safe", "deep", "wide", "open", "close", "full",
]

_EN_COMMON_VERBS = [
    "go", "see", "know", "come", "think", "look", "want", "give", "use", "find",
    "tell", "ask", "work", "call", "try", "need", "feel", "become", "leave", "put",
    "mean", "keep", "let", "begin", "show", "hear", "play", "run", "move", "live",
    "believe", "happen", "write", "sit", "stand", "lose", "pay", "meet", "include",
    "learn", "change", "lead", "understand", "watch", "follow", "stop", "speak",
    "read", "spend", "grow", "open", "walk", "win", "teach", "offer", "remember",
    "love", "help", "start", "eat", "drink", "sleep", "sing", "fly", "swim", "jump",
]

_EN_ZH_PAIRS = [
    ("hello", "你好"), ("thank you", "谢谢"), ("goodbye", "再见"), ("yes", "是"), ("no", "不"),
    ("please", "请"), ("sorry", "对不起"), ("water", "水"), ("food", "食物"), ("book", "书"),
    ("sun", "太阳"), ("moon", "月亮"), ("star", "星星"), ("sky", "天空"), ("tree", "树"),
    ("dog", "狗"), ("cat", "猫"), ("bird", "鸟"), ("fish", "鱼"), ("fire", "火"),
    ("man", "男人"), ("woman", "女人"), ("child", "孩子"), ("friend", "朋友"), ("teacher", "老师"),
    ("school", "学校"), ("home", "家"), ("city", "城市"), ("country", "国家"), ("world", "世界"),
    ("time", "时间"), ("day", "天"), ("night", "夜"), ("year", "年"), ("morning", "早上"),
    ("love", "爱"), ("happy", "快乐"), ("good", "好"), ("big", "大"), ("small", "小"),
    ("go", "去"), ("come", "来"), ("eat", "吃"), ("drink", "喝"), ("sleep", "睡"),
    ("see", "看"), ("hear", "听"), ("read", "读"), ("write", "写"), ("think", "想"),
    ("i", "我"), ("you", "你"), ("he", "他"), ("she", "她"), ("we", "我们"),
    ("what", "什么"), ("where", "哪里"), ("when", "什么时候"), ("why", "为什么"), ("how", "怎么"),
    ("one", "一"), ("two", "二"), ("three", "三"), ("four", "四"), ("five", "五"),
    ("red", "红"), ("blue", "蓝"), ("green", "绿"), ("white", "白"), ("black", "黑"),
    ("music", "音乐"), ("art", "艺术"), ("science", "科学"), ("history", "历史"), ("math", "数学"),
    ("computer", "电脑"), ("internet", "互联网"), ("phone", "电话"), ("car", "车"), ("house", "房子"),
]

_EN_STRUCTURE_SENTENCES = [
    "i am here", "you are there", "we are together",
    "i have a book", "she has a cat", "they have a dog",
    "the sun is bright", "the moon is beautiful", "the sky is blue",
    "i like music", "he likes food", "she likes books",
    "i want to learn", "i want to help", "i want to go",
    "i can read", "i can write", "i can think",
    "this is good", "that is bad", "it is ok",
    "where are you", "who are you", "what is this",
    "i do not know", "i do not understand", "i am not sure",
    "thank you very much", "you are welcome", "nice to meet you",
    "good morning", "good night", "good luck",
    "how are you", "i am fine", "i am good",
    "please help me", "let me think", "wait a moment",
    "i love you", "i miss you", "i believe you",
    "the water is cold", "the fire is hot", "the wind is strong",
    "he runs fast", "she reads well", "they work hard",
    "if you go then i go", "because i love you", "although it is hard",
    "i eat food every day", "she drinks water every day",
    "the cat and the dog", "the sun and the moon",
    "one two three four five", "red blue green white black",
]


def _load_english_words() -> List[str]:
    """从 google-10000-english-no-swears.txt 加载全部英文单词。"""
    if not os.path.exists(_ENGLISH_WORD_FILE):
        print(f"[warn] 英文词表未找到: {_ENGLISH_WORD_FILE}")
        return []
    with open(_ENGLISH_WORD_FILE, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words


def _build_english_corpus(words: List[str]) -> List[str]:
    """从英文词表构建结构化英文语料。"""
    corpus: List[str] = []

    for en, zh in _EN_ZH_PAIRS:
        corpus.append(f"{en} means {zh}")
        corpus.append(f"{zh} is {en}")

    corpus.extend(_EN_STRUCTURE_SENTENCES)

    rng = random.Random(42)
    nouns = _EN_COMMON_NOUNS
    adjs = _EN_COMMON_ADJS
    verbs = _EN_COMMON_VERBS
    for tmpl in _EN_SENTENCE_TEMPLATES:
        for _ in range(3):
            s = tmpl
            while "{noun}" in s:
                s = s.replace("{noun}", rng.choice(nouns), 1)
            while "{adj}" in s:
                s = s.replace("{adj}", rng.choice(adjs), 1)
            while "{verb}" in s:
                s = s.replace("{verb}", rng.choice(verbs), 1)
            corpus.append(s)

    chunk_size = 5
    for i in range(0, min(len(words), 2000), chunk_size):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= 2:
            corpus.append(" ".join(chunk))

    return corpus


def _register_english_vocab(model: NeuralAffectiveModel, words: List[str]) -> int:
    """将全部英文单词注册为模型词汇。"""
    count = 0
    for w in words:
        if w and w not in model.token_to_id:
            model.ensure_token(w)
            count += 1
    return count


# =============================================================================
# 常用汉字库：加载 3500 常用字，注册词汇 + 构建字序语料
# =============================================================================


def _load_hanzi_chars() -> List[str]:
    """从常用汉字库 3500.txt 加载全部汉字，返回单字列表。"""
    if not os.path.exists(_HANZI_FILE):
        print(f"[warn] 常用汉字库未找到: {_HANZI_FILE}")
        return []
    with open(_HANZI_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    return chars


def _register_hanzi_vocab(model: NeuralAffectiveModel, chars: List[str]) -> int:
    """将常用汉字注册为模型词汇，返回新增数量。"""
    count = 0
    for c in chars:
        if c not in model.token_to_id:
            model.ensure_token(c)
            count += 1
    return count


def _build_hanzi_corpus(chars: List[str]) -> List[str]:
    """从 3500 常用字构建完形填空语料：字组序列 + 高频双字词。"""
    corpus: List[str] = []
    rng = random.Random(42)

    shuffled = list(chars)
    rng.shuffle(shuffled)
    chunk_size = 6
    for i in range(0, len(shuffled), chunk_size):
        chunk = shuffled[i:i + chunk_size]
        if len(chunk) >= 2:
            corpus.append("".join(chunk))

    common_words = [
        "我们", "他们", "她们", "你们", "大家", "自己", "什么", "怎么", "这里", "那里",
        "因为", "所以", "如果", "可以", "已经", "正在", "非常", "特别", "一起", "应该",
        "时候", "地方", "东西", "朋友", "学生", "老师", "工作", "学习", "生活", "问题",
        "知道", "觉得", "希望", "喜欢", "开始", "发现", "认为", "需要", "感觉", "明白",
        "今天", "明天", "昨天", "现在", "以前", "以后", "上面", "下面", "里面", "外面",
        "高兴", "快乐", "难过", "开心", "努力", "认真", "仔细", "简单", "复杂", "重要",
        "太阳", "月亮", "星星", "天空", "大地", "河流", "山水", "花草", "春天", "秋天",
        "父亲", "母亲", "孩子", "家庭", "社会", "国家", "世界", "历史", "文化", "科学",
        "吃饭", "喝水", "走路", "说话", "读书", "写字", "唱歌", "画画", "跑步", "游泳",
        "电话", "电脑", "手机", "学校", "医院", "公园", "商店", "图书", "报纸", "杂志",
    ]
    for w in common_words:
        if all(c in chars for c in w):
            corpus.append(w)

    templates = [
        "我是{}", "他是{}", "她是{}", "我有{}", "他有{}", "我在{}", "我要{}",
        "很{}", "不{}", "最{}", "都{}", "也{}", "会{}", "能{}", "想{}",
    ]
    fill_words = [w for w in common_words if len(w) == 2]
    for tmpl in templates:
        for _ in range(3):
            corpus.append(tmpl.format(rng.choice(fill_words)))

    return corpus


# =============================================================================
# 判定：客观题 / 造句题
# =============================================================================


def _keywords(text: str) -> List[str]:
    """提取文本中的有意义词（2+ 中文字、3+ 英文、数字），过滤虚词。"""
    w = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}|\d+", text.lower())
    return [x for x in w if x not in {"一个", "这个", "那个", "以及", "因此", "可以", "我们", "他们"}]


def _extract_required_chars(question: str) -> List[str]:
    """从问句中提取「」内的必含字，造句判定用。"""
    matches = re.findall(r"「([^」]+)」", question)
    chars, skip = [], {"…", ".", " ", "\t"}
    for m in matches:
        for c in m:
            if c not in skip and c not in chars:
                chars.append(c)
    return chars


def _extract_required_phrases(question: str) -> List[str]:
    """从问句中提取「」内的必含短语（如 然后、因为、所以）。"""
    matches = re.findall(r"「([^」]+)」", question)
    phrases = []
    for m in matches:
        for frag in re.findall(r"[\u4e00-\u9fff]{2,}", m):
            if frag not in phrases:
                phrases.append(frag)
    return phrases


def _judge_exact(predicted: str, reference: str) -> bool:
    """精确匹配：去除标点空白后完全一致。"""
    clean = lambda s: re.sub(r"[。！？，、\s.!?,\u3000]+", "", s.strip())
    return clean(predicted) == clean(reference)


def _has_garbage(text: str) -> bool:
    """检测是否含有明显垃圾内容：中英混杂碎片、过多无关重复。"""
    # 英文碎片穿插在中文回答中（如 "badge动好", "lcdgbp"）
    en_frags = re.findall(r"[a-zA-Z]{3,}", text)
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if cn_chars and len(en_frags) >= 2:
        return True
    # 同一个字/词连续重复 3 次以上
    if re.search(r"(.{2,})\1{2,}", text):
        return True
    return False


def _judge_objective(predicted: str, reference: str) -> bool:
    """客观题：精确匹配 或 关键词匹配，同时惩罚过长/垃圾回答。"""
    if _judge_exact(predicted, reference):
        return True
    # 回答长度不应超过参考答案的 3 倍
    if len(predicted) > max(20, len(reference) * 3):
        return False
    if _has_garbage(predicted):
        return False
    pk, rk = set(_keywords(predicted)), set(_keywords(reference))
    return bool(rk and len(pk & rk) / max(1, len(rk)) >= 0.6)


def _judge_sentence_grammar(predicted: str, question: str, reference: str) -> bool:
    """造句题严格判定：精确匹配优先，否则必须同时满足格式+必含字+关键词。"""
    pred = predicted.strip()
    if _judge_exact(pred, reference):
        return True
    if len(pred) < 2 or len(pred) > 40:
        return False
    if _has_garbage(pred):
        return False
    if re.search(r"\d|[a-zA-Z]{2,}", pred):
        return False
    if pred.count("。") + pred.count("！") + pred.count("？") > 1:
        return False
    for prefix in ("好的", "根据", "答案是", "例句", "仿照", "请看", "如下"):
        if pred.startswith(prefix):
            return False
    phrases, chars = _extract_required_phrases(question), _extract_required_chars(question)
    if phrases and not all(p in pred for p in phrases):
        return False
    if chars and not all(c in pred for c in chars):
        return False
    pk, rk = set(_keywords(pred)), set(_keywords(reference))
    if rk and len(pk & rk) / max(1, len(rk)) < 0.5:
        return False
    return True


# =============================================================================
# 完形填空预训练
# =============================================================================


_CLOZE_WARMUP_STEPS = 100  # 前 100 步线性预热，避免初期大步更新破坏嵌入


def _run_cloze(
    model: NeuralAffectiveModel,
    corpus: List[str],
    epochs: int = 2,
    passes_per_pair: int = 1,
    lr_scale: float = 1.2,
) -> int:
    """
    完形填空即学（优化版）：
    ① 课程学习 —— 首轮按句长由短到长，让模型先掌握基础模式再学复杂结构；
    ② 学习率预热 —— 前 100 步线性从 0.1× 升至 1×，稳定早期训练；
    ③ 拼接序列 —— 随机拼接相邻句子，学习跨句上下文（模拟 GPT 预训练长文本）；
    ④ 句末 EOS 训练不变。
    """
    bos_id = model.token_to_id[model.BOS]
    eos_id = model.token_to_id[model.EOS]
    neutral_s = [0.0] * model.sensory_dim
    neutral_e = [0.0] * model.emotion_dim
    total = 0
    corpus = list(corpus)

    # ---------- 拼接序列语料（随机取相邻 2-3 句拼成长序列）----------
    rng = random.Random(42)
    concat_corpus: List[str] = []
    indices = list(range(len(corpus)))
    rng.shuffle(indices)
    i = 0
    while i < len(indices) - 1:
        span = rng.randint(2, min(3, len(indices) - i))
        combined = " ".join(corpus[indices[i + j]] for j in range(span))
        concat_corpus.append(combined)
        i += span

    for ep in range(epochs):
        if ep == 0:
            # 课程学习：首轮按句长排序（短→长），让模型先学简单模式
            work = sorted(corpus, key=len)
        else:
            random.shuffle(corpus)
            work = corpus

        # 第二轮起追加拼接序列，强化跨句理解
        if ep >= 1:
            random.shuffle(concat_corpus)
            work = work + concat_corpus

        for s in work:
            tokens = model.tokenize(s)
            for t in tokens:
                model.ensure_token(t)
            if len(tokens) < 2:
                continue
            ids = [model.token_to_id[t] for t in tokens]
            for i in range(len(ids) - 1):
                # 学习率预热：前 _CLOZE_WARMUP_STEPS 步线性升温
                if total < _CLOZE_WARMUP_STEPS:
                    warmup_mult = 0.1 + 0.9 * (total / _CLOZE_WARMUP_STEPS)
                else:
                    warmup_mult = 1.0
                effective_lr = model.lr * lr_scale * warmup_mult

                for _ in range(passes_per_pair):
                    model.train_one(
                        [bos_id] + ids[: i + 1], ids[i + 1],
                        sensory_vec=neutral_s, emotion_vec=neutral_e,
                        lr=effective_lr,
                    )
                    total += 1
            for _ in range(passes_per_pair):
                model.train_one(
                    [bos_id] + ids, eos_id,
                    sensory_vec=neutral_s, emotion_vec=neutral_e,
                    lr=model.lr * lr_scale,
                )
                total += 1
        print(f"[预训练] epoch {ep + 1}/{epochs} 完成，累计 {total} 步")
    return total


# =============================================================================
# 强化验证：纠错后重复验证直至通过
# =============================================================================


def _validate_and_reinforce(
    model: NeuralAffectiveModel,
    question: str,
    reference: str,
    judge_fn,
    train_s: List[float], train_e: List[float], ctx: SensoryContext,
    config: AppConfig, passes: int, replay: int, sensory_override: Optional[Dict[str, float]] = None,
) -> bool:
    from src.trace_builder import build_trace_from_answer as _build_trace
    MAX_ROUNDS = 5
    for r in range(MAX_ROUNDS):
        if r > 0:
            tt = _build_trace(model, question, reference, train_s, train_e)
            model.apply_feedback(tt, [True] * len(tt.token_ids), [None] * len(tt.token_ids),
                passes, 0, absorb_knowledge=True)
            model.save(config.model_path)
            print(f"[强化] 第 {r} 轮追加训练…")
        payload = collect_sensory_payload(question, ctx, config, sensory_override=sensory_override)
        trace, _ = model.deliberate_generate(question, payload, config.max_len, 0.7, 1)
        pred = "".join(trace.tokens)
        if judge_fn(pred):
            return True
    print(f"[强化] 达到上限 {MAX_ROUNDS} 轮，直接灌入答案。")
    tt = _build_trace(model, question, reference, train_s, train_e)
    model.apply_feedback(tt, [True] * len(tt.token_ids), [None] * len(tt.token_ids),
        passes, 0, absorb_knowledge=True)
    return False


def _run_auto_validation(
    model: NeuralAffectiveModel, config: AppConfig, lessons: List[TeachLesson]
) -> None:
    from src.utils import flatten_tokens as _flatten
    from src.trace_builder import build_trace_from_answer as _build_trace
    ctx = SensoryContext()
    passes, replay = config.teacher_learning_passes, config.teacher_replay_steps
    for step, lesson in enumerate(lessons, 1):
        print(f"\n[自动验证] 第 {step}/{len(lessons)} 课 【{lesson.domain}·{lesson.level_name}】 {lesson.question} → {lesson.reference_answer}")
        q, is_sentence = lesson.question, lesson.domain == "造句"
        judge = (lambda p: _judge_sentence_grammar(p, lesson.question, lesson.reference_answer)) if is_sentence else (lambda p, _ref=lesson.reference_answer: _judge_objective(p, _ref))
        MAX_ATTEMPTS = 3
        neutral_s = [0.0] * model.sensory_dim
        neutral_e = [0.0] * model.emotion_dim

        tt = _build_trace(model, q, lesson.reference_answer, neutral_s, neutral_e)
        model.apply_feedback(tt, [True] * len(tt.token_ids), [None] * len(tt.token_ids), passes, 0, absorb_knowledge=True)

        passed = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            payload = collect_sensory_payload(q, ctx, config, sensory_override=_CALM_SENSORY_OVERRIDE)
            trace, _ = model.deliberate_generate(q, payload, config.max_len, 0.7, 1)
            pred = _flatten(trace.tokens)
            if judge(pred):
                print(f"  ✓ 第 {attempt} 次通过: {pred[:60]}{'…' if len(pred) > 60 else ''}")
                passed = True
                break
            tt = _build_trace(model, q, lesson.reference_answer, neutral_s, neutral_e)
            model.apply_feedback(tt, [True] * len(tt.token_ids), [None] * len(tt.token_ids), passes, 0, absorb_knowledge=True)
            print(f"  第 {attempt} 次未通过，追加学习…")

        if not passed:
            print(f"  ✗ {MAX_ATTEMPTS} 次未通过，已强制灌入答案。")
        model.save(config.model_path)


# =============================================================================
# 入口
# =============================================================================


def _run_one_cycle(
    model: NeuralAffectiveModel,
    config: AppConfig,
    corpus: List[str],
    hanzi_corpus: List[str],
    en_corpus: List[str],
    lessons: List[TeachLesson],
    cycle: int,
) -> None:
    """执行一轮训练周期。cycle=0 为首次完整训练，cycle>=2 跳过完形填空只做强化验证。"""
    tag = f"[cycle {cycle}]" if cycle > 0 else "[teacher]"
    full_train = cycle <= 1

    if full_train:
        print(f"{tag} 第一步：中文完形填空…")
        _run_cloze(model, corpus, epochs=2, passes_per_pair=1)

        if hanzi_corpus:
            print(f"{tag} 第二步：常用汉字语料学习…")
            _run_cloze(model, hanzi_corpus, epochs=1, passes_per_pair=1, lr_scale=1.0)

        if en_corpus:
            print(f"{tag} 第三步：英文语料学习…")
            _run_cloze(model, en_corpus, epochs=1, passes_per_pair=1, lr_scale=1.0)
    else:
        print(f"{tag} 跳过完形填空（已学过），直接进入强化阶段")

    print(f"{tag} 身份种子…")
    model.seed_identity_logic(passes_per_step=2, epochs=1)
    print(f"{tag} 自动化验证…")
    _run_auto_validation(model, config, lessons)
    model.save(config.model_path)
    print(f"\n{tag} 完成。模型已保存: {config.model_path}")


def run_teacher_session(model: NeuralAffectiveModel, config: AppConfig) -> None:
    """完形填空预训练主流程。支持 --teacher-loop 循环训练，Ctrl+C 安全退出。"""
    name = model.model_name
    print("=" * 50)
    print(f"  Future 老师 —— 完形填空预训练（名称：{name}）")
    if config.teacher_loop:
        print("  (循环模式：Ctrl+C 安全退出)")
    print("=" * 50)
    corpus = _cloze_corpus(name)
    lessons = _all_lessons(name)

    hanzi_chars = _load_hanzi_chars()
    hanzi_corpus: List[str] = []
    if hanzi_chars:
        n_hanzi = _register_hanzi_vocab(model, hanzi_chars)
        hanzi_corpus = _build_hanzi_corpus(hanzi_chars)
        print(f"[teacher] 常用汉字注册 {n_hanzi} 个（共 {len(hanzi_chars)} 字），汉字语料 {len(hanzi_corpus)} 句")

    en_words = _load_english_words()
    en_corpus: List[str] = []
    if en_words:
        n_reg = _register_english_vocab(model, en_words)
        en_corpus = _build_english_corpus(en_words)
        print(f"[teacher] 英文词汇注册 {n_reg} 个，英文语料 {len(en_corpus)} 句")

    total_corpus = corpus + hanzi_corpus + en_corpus
    print(f"[teacher] 总语料 {len(total_corpus)} 句（中文 {len(corpus)} + 汉字 {len(hanzi_corpus)} + 英文 {len(en_corpus)}），验证 {len(lessons)} 课")

    if not config.teacher_loop:
        _run_one_cycle(model, config, corpus, hanzi_corpus, en_corpus, lessons, 0)
        return

    cycle = 1
    try:
        while True:
            print(f"\n{'=' * 50}")
            print(f"  第 {cycle} 轮循环训练开始")
            print(f"{'=' * 50}")
            _run_one_cycle(model, config, corpus, hanzi_corpus, en_corpus, lessons, cycle)
            cycle += 1
    except KeyboardInterrupt:
        model.save(config.model_path)
        print(f"\n[teacher] 循环训练已停止（共完成 {cycle - 1} 轮）。模型已保存: {config.model_path}")
