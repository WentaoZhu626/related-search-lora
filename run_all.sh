#!/usr/bin/env bash
# 在 AutoDL 上一键跑完：划分 -> 训练 -> 评估。每一步都可以单独重跑。
#
#   bash run_all.sh split     # 只做划分（先 --report-only 看簇，见 README）
#   bash run_all.sh train
#   bash run_all.sh eval
#   bash run_all.sh all
#
# 路径都放在 /root/autodl-tmp（数据盘），需要改的话改下面几个变量。
set -euo pipefail
cd "$(dirname "$0")"

ROOT=${ROOT:-/root/autodl-tmp/related_research}
CSV=${CSV:-$ROOT/data/cleaned_dataset.csv}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/data/ood_split_v3}
BASE_MODEL=${BASE_MODEL:-$ROOT/models/Qwen3-4B-Instruct-2507}
EMBED_MODEL=${EMBED_MODEL:-$ROOT/models/Qwen3-Embedding-0.6B}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/ood_v3_qwen3_4b_gate_lora}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

step=${1:-all}

if [[ $step == split || $step == all ]]; then
  python prepare_ood_split.py --input "$CSV" --output-dir "$SPLIT_DIR" \
    --embedding-model "$EMBED_MODEL" ${SPLIT_EXTRA_ARGS:-}
fi

if [[ $step == train || $step == all ]]; then
  python train_gate_lora.py --model "$BASE_MODEL" --data-dir "$SPLIT_DIR" --output-dir "$OUT_DIR" ${TRAIN_EXTRA_ARGS:-}
fi

if [[ $step == eval || $step == all ]]; then
  python evaluate_gate.py --adapter "$OUT_DIR/final_adapter" --base-model "$BASE_MODEL" --data-dir "$SPLIT_DIR" \
    --soft-embedding-model "$EMBED_MODEL" ${EVAL_EXTRA_ARGS:-}
fi
