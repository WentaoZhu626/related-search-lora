# 相关搜索 v3：Qwen3-4B-Instruct + LoRA + 判断头，带 OOD 划分

这是一套**独立于旧 `model_training/` 的新代码**（所有模块以 `rs_` 开头，互不冲突），旧代码没有被修改。
主体保持不变：Qwen3-4B-Instruct-2507 + LoRA + gate head，联合损失 `L = L_gen + L_gate`。

## 和旧版相比改了什么

| 项目 | 旧版 | 新版 |
|---|---|---|
| 数据划分 | 随机划分 | `prepare_ood_split.py`：按语义簇整体留出 → `test_ood`；缓冲带去掉最相似的训练候选；剩余随机分出 `validation` / `test_id` |
| 评估 | 单个测试集 | 同一个模型评估 `test_id` 与 `test_ood`，输出对比表和 bootstrap 95% 置信区间 |
| system prompt | “你是一个搜索推荐系统…” | `rs_prompt.py` 里的新 prompt（一跳关系、1～7 个词、宁缺毋滥、简体中文），并随 adapter 保存为 `system_prompt.txt`，推理自动读取那一份 |
| 翻译步骤 | 有 | 没有（数据已清洗，直接读 `cleaned_dataset.csv`） |
| 生成损失 | 全部位置算 logits | 只在答案位置算 logits，省显存 |
| 评估显存 | Trainer 会累积整批 logits | `prediction_loss_only=True` + 精简输出 |
| 超长样本 | 截断 | 丢弃并计数（截断会丢掉结束符） |
| checkpoint | 只有 LoRA | 同时保存 `gate_head.pt`，支持 `--resume-from-checkpoint` |
| 解码 | repetition_penalty 1.1 | 默认 1.0（会惩罚合理的重复字，如“红色”“红裙”），可用参数改回 |
| 指标 | 精确匹配 | 另增：强制生成、端到端、embedding 软匹配、输出健康度、词表覆盖率 |

## 文件

```
prepare_ood_split.py   数据划分（语义簇留出 + 缓冲带 + 分层随机）
train_gate_lora.py     训练（LoRA + 判断头）
evaluate_gate.py       同一模型评估 test_id / test_ood，写 summary.md
try_gate.py            手动输入 query 体验
make_full_train_dir.py OOD 实验结束后，生成“全量训练”的数据目录
run_all.sh             一键脚本（split / train / eval / all）
rs_prompt.py           system prompt 与 chat template 拼接（训练和推理共用）
rs_infer.py rs_model.py rs_metrics.py rs_embed.py rs_quality.py rs_common.py   公共模块
tests/                 CPU 冒烟测试（玩具模型，不需要 GPU 和网络）
```

## 在 AutoDL 上的流程

### 0. 上传与环境

把整个文件夹传到 `/root/autodl-tmp/related_research/code/related_search_ood_v3/`，把
`cleaned_dataset.csv` 传到 `/root/autodl-tmp/related_research/data/`。

```bash
cd /root/autodl-tmp/related_research/code/related_search_ood_v3
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

下载两个模型（都是 safetensors，torch 2.5.1 可以直接加载；不要用只有 `.bin` 的模型，例如 BGE-M3 会被 transformers 拒绝）。
AutoDL 上任选一种：

```bash
# 方式 A：ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir /root/autodl-tmp/related_research/models/Qwen3-4B-Instruct-2507
modelscope download --model Qwen/Qwen3-Embedding-0.6B   --local_dir /root/autodl-tmp/related_research/models/Qwen3-Embedding-0.6B

# 方式 B：HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/autodl-tmp/related_research/models/Qwen3-4B-Instruct-2507
huggingface-cli download Qwen/Qwen3-Embedding-0.6B   --local-dir /root/autodl-tmp/related_research/models/Qwen3-Embedding-0.6B
```

### 1. 先跑 CPU 冒烟测试（可选，1 分钟）

```bash
python tests/smoke_test.py
```

### 2. 数据划分

先只看聚类结果，不写数据集：

```bash
python prepare_ood_split.py \
  --input /root/autodl-tmp/related_research/data/cleaned_dataset.csv \
  --output-dir /root/autodl-tmp/related_research/data/ood_split_v3 \
  --embedding-model /root/autodl-tmp/related_research/models/Qwen3-Embedding-0.6B \
  --report-only
```

打开 `ood_split_v3/cluster_report.csv`，每个簇有规模、正样本率、离群程度、代表 query、是否被自动选为 OOD。
自动选择的簇合理就直接去掉 `--report-only` 再跑一次（向量有缓存，很快）；不满意就手工指定：

```bash
python prepare_ood_split.py ... --ood-cluster-ids 3,17,25,31
```

跑完看 `split_report.json`：`test_ood` 的 `nearest_train_similarity` 应明显低于 `test_id`，
`gold_term_seen_in_train_rate` 也应更低。这是“OOD 集确实更远”的证据；如果两者差不多，说明 OOD 划分没起作用（簇选得太近），换簇或调大 `--buffer-fraction`。

### 3. 训练

```bash
python train_gate_lora.py \
  --model /root/autodl-tmp/related_research/models/Qwen3-4B-Instruct-2507 \
  --data-dir /root/autodl-tmp/related_research/data/ood_split_v3 \
  --output-dir /root/autodl-tmp/related_research/outputs/ood_v3_qwen3_4b_gate_lora
```

* 先加 `--smoke` 跑 5 步确认环境没问题（产物写到 `*_smoke` 目录，不会污染正式目录）。
* 4090 24GB：默认 batch 8 × 累积 4（有效 batch 32）+ 梯度检查点，应当放得下。
  显存有富余可加 `--no-gradient-checkpointing` 提速 20～30%；如果 OOM，改 `--batch-size 4 --gradient-accumulation 8`。
* 训练结束会在 `validation`（同分布）上选门控阈值并写入 `final_adapter/gate_config.json`。
  测试集不参与任何选择。
* 粗估 1～1.5 小时（3 epoch，约 1.6 万条，序列约 200～250 token），以实际日志为准。
* 日志里的 `lm_loss` / `gate_loss` 是分开记录的，方便判断是生成还是门控在变差。

### 4. 评估

```bash
python evaluate_gate.py \
  --adapter /root/autodl-tmp/related_research/outputs/ood_v3_qwen3_4b_gate_lora/final_adapter \
  --base-model /root/autodl-tmp/related_research/models/Qwen3-4B-Instruct-2507 \
  --data-dir /root/autodl-tmp/related_research/data/ood_split_v3 \
  --soft-embedding-model /root/autodl-tmp/related_research/models/Qwen3-Embedding-0.6B
```

输出在 `outputs/.../eval/`：`summary.md`（ID vs OOD 对比）、`*_metrics.json`、`*_predictions.jsonl`（逐条预测，可以直接翻看 OOD 上的失败样本）。

怎么读：

* 门控看 AUROC 和 balanced accuracy；阈值是在 ID 验证集上定的，OOD 上如果 recall/specificity 明显失衡，说明概率分布发生了偏移。
* “强制生成”是在教师标注应展示的样本上强制生成，衡量生成能力本身，不受门控误判影响。
* 精确匹配 F1 天然很低（约 75% 的标准答案词从没在训练答案里出现过），**主要看软匹配 F1**，同时对比“模型输出词在训练集中出现过的比例”——OOD 上这个比例若明显更高，说明模型在回退到记忆过的词。
* 差距要结合 bootstrap 置信区间看，区间明显重叠就不能说 OOD 更差。
* **结论范围**：只代表“同一 Pinterest 风格分布内、训练时没见过的语义主题”；没有真实 Nalook 线上 query 和教师标注，做不了远 OOD。

### 5. 体验

```bash
python try_gate.py --adapter .../final_adapter --query "汉服" --query "咖啡杯"
python try_gate.py --adapter .../final_adapter            # 交互模式
python try_gate.py --adapter .../final_adapter --query "..." --always-generate   # 门控拒绝时也看看会生成什么
```

### 6. 上线模型：用全部数据重训

OOD 实验只是为了得到结论。最终部署的模型用全部数据、同样的超参数再训一次：

```bash
python make_full_train_dir.py --split-dir .../data/ood_split_v3 --output-dir .../data/full_split_v3
python train_gate_lora.py --data-dir .../data/full_split_v3 --output-dir .../outputs/full_qwen3_4b_gate_lora
```

`validation` 依然留出，只用来选阈值和观察曲线。

### 7. 让 mentor 自己体验（网页 Demo）

比起一份报告，让 mentor 自己动手试几个词更有说服力。`demo_app.py` 用 Gradio 包了一个网页：

```bash
pip install gradio
python demo_app.py --adapter /root/autodl-tmp/related_research/outputs/full_qwen3_4b_gate_lora/final_adapter
```

终端会打印一个 `https://xxxx.gradio.live` 的公网链接（72 小时内有效），直接发给 mentor，
他在自己电脑或手机浏览器打开就能用——不用装任何东西、不用连 AutoDL、不用懂代码。

几点要注意：

* 这台 AutoDL 实例要保持"运行中"mentor 才能访问，用完记得关机，不然一直计费。
* 链接是公网可访问的（网址随机不好猜，但没有身份验证）；想加密码就用
  `--username xxx --password xxx`，mentor 打开时会先弹登录框。
* 建议用 `tmux` 跑，防止 SSH 断开后进程被杀：`tmux new -s demo`，起完 `Ctrl+B` `D` 退出但保持后台运行。
* 想直接演示不生成公网链接，用 `--no-share`（只有这台机器自己能打开）。

如果 mentor 更想要"跑一下命令行"的原始体验，也可以直接把 `try_gate.py` 的用法（见第 5 步）
发给他，前提是他能连上这台 AutoDL 实例（比如通过实例分享/SSH）——网页 Demo 通常更省事。

## 改 prompt

只需要改 `rs_prompt.py` 里的 `SYSTEM_PROMPT`（并改 `PROMPT_VERSION`），重新训练即可。
每个 adapter 会把训练时用的 prompt 存成 `final_adapter/system_prompt.txt`，评估和体验都读这份，
所以之后改了代码里的 prompt 也不会让旧 adapter 用错 prompt。

## 已知限制

* 训练脚本在本地只用玩具模型做过 CPU 冒烟测试（损失、门控、保存、加载、推理一致性都通过），该说明写于初版代码阶段；后续真实 GPU 训练日志与实验总结已保存在本仓库 `results/` 和 `docs/`。新环境首次运行仍建议先 `--smoke`。
* `--embedding-model tfidf` 只是调试兜底，语义质量远不如真 embedding，不要用来做正式划分。
* 环境要求 `transformers>=4.56,<5`；更旧的版本可能没有 `dtype=` 参数，代码里已做兼容，但没有逐版本验证。
