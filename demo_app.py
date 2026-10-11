"""网页体验 Demo：给 mentor 一个可以直接点开、自己输入 query 体验的页面。

用法（在 AutoDL 上）：
    pip install gradio
    python demo_app.py --adapter /root/autodl-tmp/related_research/outputs/full_qwen3_4b_gate_lora/final_adapter

启动后会打印两个链接：
  * Local URL   —— 只有这台机器自己能打开，用不上
  * Public URL  —— gradio 官方中转出来的公网链接（形如 https://xxxx.gradio.live），
                    72 小时内有效。把这个链接发给 mentor，他在自己电脑/手机浏览器打开就能用，
                    不需要装任何东西、不需要 GPU、不需要连 AutoDL。

注意：
  * 这台 AutoDL 实例必须保持"运行中"，mentor 才能访问；用完记得手动关机（不然一直计费）。
  * 这个链接是公网可访问的（网址随机、不好猜，但没有身份验证），演示完想让它失效，
    直接在终端 Ctrl+C 停掉这个进程即可。
  * 如果想加一层密码保护，加 --username 和 --password（mentor 打开链接时会先弹出登录框）。
  * 建议用 tmux/screen 跑这个脚本，防止 SSH 断开后进程被杀掉：
        tmux new -s demo
        python demo_app.py --username mentor --password <随便设一个>
        # Ctrl+B 再按 D 退出 tmux 但保持后台运行；下次用 tmux attach -t demo 回来看
"""

from __future__ import annotations

import argparse

import gradio as gr

from rs_infer import gate_probabilities, generate_terms, load_model

DEFAULT_ROOT = "/root/autodl-tmp/related_research"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=f"{DEFAULT_ROOT}/outputs/full_qwen3_4b_gate_lora/final_adapter",
        help="要展示的 adapter 目录（默认指向全量重训后的模型）",
    )
    parser.add_argument("--base-model", default=None, help="缺省则读 adapter_config.json 里记录的路径")
    parser.add_argument("--merge", action="store_true", help="合并 LoRA 后推理（略快，显存占用相同）")
    parser.add_argument("--share", dest="share", action="store_true", default=True)
    parser.add_argument("--no-share", dest="share", action="store_false", help="只在本机内网使用，不生成公网链接")
    parser.add_argument("--username", default=None, help="设置后 mentor 打开链接需要先登录")
    parser.add_argument("--password", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("加载模型中……")
    lm = load_model(args.adapter, args.base_model, merge=args.merge)
    print(f"加载完成，门控阈值 = {lm.threshold:.4f}")

    def respond(query: str, always_generate: bool):
        query = (query or "").strip()
        if not query:
            return "请输入一个搜索词", "", ""
        probability = gate_probabilities(lm, [query])[0]
        shown = probability >= lm.threshold
        words: list[str] = []
        if shown or always_generate:
            _, terms = generate_terms(lm, [query])
            words = terms[0]
        status = "✅ 展示相关搜索" if shown else "\U0001F6AB 不展示（模型判断为不相关 / 信息不足）"
        terms_text = "、".join(words) if words else "（无）"
        detail = f"门控概率：{probability:.4f}（当前阈值 {lm.threshold:.4f}）"
        return status, terms_text, detail

    with gr.Blocks(title="相关搜索 Demo") as demo:
        gr.Markdown(
            "## 相关搜索推荐 Demo\n"
            "输入一个搜索词，看模型判断要不要展示相关搜索，以及会给出哪些词。\n"
            "（Qwen3-4B-Instruct-2507 + LoRA + 判断头，已在语义簇留出的 near-OOD 划分上做过评估）"
        )
        with gr.Row():
            query_box = gr.Textbox(label="搜索词", placeholder="例如：汉服、咖啡杯、露营装备……", scale=3)
            always = gr.Checkbox(label="不相关也生成看看", value=False, scale=1)
        submit = gr.Button("试试看", variant="primary")
        status_box = gr.Textbox(label="判断结果", interactive=False)
        terms_box = gr.Textbox(label="相关搜索词", interactive=False)
        detail_box = gr.Textbox(label="详情", interactive=False)

        submit.click(respond, inputs=[query_box, always], outputs=[status_box, terms_box, detail_box])
        query_box.submit(respond, inputs=[query_box, always], outputs=[status_box, terms_box, detail_box])

    auth = (args.username, args.password) if args.username and args.password else None
    demo.launch(share=args.share, server_port=args.port, auth=auth)


if __name__ == "__main__":
    main()
