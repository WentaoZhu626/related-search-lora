"""手动体验：输入 query，看判断头概率和相关搜索词。

    python try_gate.py --query "汉服" --query "咖啡杯"
    python try_gate.py                      # 交互模式，空行退出
    python try_gate.py --always-generate    # 即使门控判为“不展示”也照样生成，方便看模型在拒答样本上的输出
"""

from __future__ import annotations

import argparse
import json

from rs_infer import generate_terms, gate_probabilities, load_model

DEFAULT_ROOT = "/root/autodl-tmp/related_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=f"{DEFAULT_ROOT}/outputs/ood_v3_qwen3_4b_gate_lora/final_adapter")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--query", action="append", default=[], help="可重复；不给则进入交互模式")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--always-generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--merge", action="store_true", help="合并 LoRA 后推理（略快，占用同样显存）")
    return parser.parse_args()


def run(lm, queries: list[str], args) -> list[dict]:
    threshold = args.threshold if args.threshold is not None else lm.threshold
    probabilities = gate_probabilities(lm, queries)
    todo = [i for i, p in enumerate(probabilities) if args.always_generate or p >= threshold]
    raw, terms = generate_terms(
        lm,
        [queries[i] for i in todo],
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    generated = {index: (raw[k], terms[k]) for k, index in enumerate(todo)}
    results = []
    for i, query in enumerate(queries):
        shown = probabilities[i] >= threshold
        raw_text, parsed = generated.get(i, ("", []))
        results.append(
            {
                "query": query,
                "gate_probability": round(probabilities[i], 4),
                "threshold": round(threshold, 4),
                "show": shown,
                "related_searches": parsed if shown else [],
                "generated_anyway": parsed if (not shown and parsed) else None,
                "raw_output": raw_text,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    lm = load_model(args.adapter, args.base_model, merge=args.merge)
    if args.query:
        print(json.dumps(run(lm, args.query, args), ensure_ascii=False, indent=2))
        return
    print("输入 query，空行退出。")
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            break
        if not query:
            break
        print(json.dumps(run(lm, [query], args)[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
