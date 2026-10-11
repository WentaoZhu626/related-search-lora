"""Chat template 与 prompt：训练和推理都从这里取，保证两边完全一致。

修改 system prompt 只需要改下面的 ``SYSTEM_PROMPT``（并顺手改 ``PROMPT_VERSION``）。
训练结束后，当时使用的 prompt 会原样保存到 ``final_adapter/system_prompt.txt``，
推理/评估默认读取这份文件，因此即使之后又改了这里，旧 adapter 也不会被错误的 prompt 使用。

相比旧版（"你是一个搜索推荐系统……"）的改动：
  * 措辞对齐清洗数据时使用的“一跳关系”标准，不再提 Pinterest；
  * 明确数量是 1～7 个、宁缺毋滥（清洗后的数据里相关词数量 0～7 不等，不再固定 5 个）；
  * 明确用简体中文、可独立搜索、不要同义改写/过宽大类词。
"""

from __future__ import annotations

PROMPT_VERSION = "rs-prompt-v3-2026-09-21"

SYSTEM_PROMPT = (
    "你是 Nalook 的相关搜索推荐模块。输入一个搜索词，输出用户接下来最可能继续搜索、"
    "且与它有明确直接关系的相关搜索词，例如别名、上下位类别、具体类型、同类对象、"
    "成员或作品角色、品牌产品关联、风格材质变体、工具材料、用途场景、常见搭配、"
    "替代对象、相邻步骤、问题与解决方案。"
    "不要输出字面相似但意图不同的词、过宽的大类词，或对搜索词的同义改写。"
    "每个词简洁自然、可独立搜索，使用简体中文（公认的品牌、人名、型号可保留原文）。"
    "输出 1 到 7 个词，按相关程度从高到低排列，宁缺毋滥。"
    "只输出相关搜索词，用“ | ”分隔，不要解释。"
)

MAX_TERMS = 7


def build_messages(query: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query.strip()},
    ]


def prompt_text(tokenizer, query: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """模型开始回答之前的完整文本（含 assistant 起始标记）。

    ``enable_thinking=False`` 对 Qwen3-4B-Instruct-2507 的模板没有影响（会被忽略），
    但如果以后换成带思考模式的 Qwen3-8B/14B，可以直接沿用。
    """
    return tokenizer.apply_chat_template(
        build_messages(query, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(
    tokenizer,
    query: str,
    answer: str | None,
    max_length: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict | None:
    """把一条样本编码成训练输入。

    * 正样本：input = prompt + 答案 + <|im_end|>，labels 只在“答案 + <|im_end|>”上有效；
    * 负样本（answer 为 None）：input = prompt，labels 全部为 -100（只训练判断头）；
    * gate_position = prompt 最后一个 token（assistant 起始标记之后），因果注意力保证看不到答案；
    * 超长样本返回 None（由调用方统计并丢弃），不做截断——截断会让答案缺少结束符。
    """
    prompt_ids = tokenizer(
        prompt_text(tokenizer, query, system_prompt), add_special_tokens=False
    )["input_ids"]
    if answer:
        answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    else:
        answer_ids = []
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "gate_position": len(prompt_ids) - 1,
    }


def check_template_consistency(tokenizer, query: str = "测试", answer: str = "甲 | 乙", system_prompt: str = SYSTEM_PROMPT) -> dict:
    """自检：手工拼接的“prompt + 答案 + eos”是否与官方模板渲染的完整对话一致。

    不一致不影响训练/推理的自洽性（两边都用我们自己拼接的格式），
    但可以提前发现模板变化，例如模板在 assistant 段里插入了额外标记。
    """
    canonical = tokenizer.apply_chat_template(
        build_messages(query, system_prompt) + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    manual = prompt_text(tokenizer, query, system_prompt) + answer + tokenizer.eos_token
    return {
        "consistent": canonical.startswith(manual),
        "canonical_tail": canonical[len(manual) - 30 :] if len(canonical) >= len(manual) else canonical[-30:],
        "manual_tail": manual[-30:],
    }
