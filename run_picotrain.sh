#!/bin/bash
# Shell utility to automate dependencies, preprocessing, and training in Picotron.

set -e

if [ -z "$1" ]; then
    echo "Usage: ./run_picotrain.sh <path_to_config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

echo "[Picotron Boot] Verifying python installation and registering console scripts..."
pip install -q -e .

echo ""
echo "[Picotron Preprocess] Running dataset tokenization pipeline..."
picotron-preprocess "$CONFIG_PATH"

echo ""
echo "[Picotron Train] Launching model training..."
# Parse dp_size from the YAML config
DP_SIZE=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_PATH')); print(cfg.get('parallel', {}).get('dp_size', 1))")

if [ "$DP_SIZE" -eq 1 ]; then
    echo "[Picotron Train] Starting single-GPU/CPU training run..."
    python3 train.py "$CONFIG_PATH"
else
    echo "[Picotron Train] Starting multi-GPU distributed DDP training (GPUs: $DP_SIZE)..."
    torchrun --nproc_per_node="$DP_SIZE" train.py "$CONFIG_PATH"
fi
