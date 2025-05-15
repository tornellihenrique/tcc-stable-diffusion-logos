#!/usr/bin/env bash

# ---------- Configurações (edite conforme necessário) ----------
SERVER="127.0.0.1:8188"
PROMPT_FILE="text2img.json"
TEXTS_FILE="./prompts/test1-wordmark.txt"
OUTPUT_DIR="./output"
WIDTH=1024
HEIGHT=1024
CFG=4,5,6
STEPS=15,20,25
SEED=100

# Converge        (dpmpp_2m karras, lms karras, euler normal)
# Don't Converge  (dpmpp_2s_ancestral normal, dpmpp_2s_ancestral karras)
# Kinda Converge  (dpmpp_2m_sde karras)
SAMPLER="dpmpp_2m_sde"
SCHEDULER="karras"

BATCH_SIZE=4
BASE_STEPS=1.0
LORA_DIR="/home/henrique/comfy/ComfyUI/models/loras"

# Defina aqui suas LoRAs: categoria ou categoria:strength ou categoria:strength:epoch
LORAS=(
  "wordmark:1.0"
  "vintage:1.0"
)

PREFIX="test1_wordmark_minimalistic2"

# ---------- Monta argumentos de LoRA ----------
LORA_ARGS=()
for L in "${LORAS[@]}"; do
  LORA_ARGS+=("--lora" "$L")
done

# ---------- Comando para executar ----------
python execute-flow.py \
  --server      "$SERVER" \
  --prompt-file "$PROMPT_FILE" \
  --texts-file  "$TEXTS_FILE" \
  --output-dir  "$OUTPUT_DIR" \
  --width       "$WIDTH" \
  --height      "$HEIGHT" \
  --cfg         "$CFG" \
  --steps       "$STEPS" \
  --seed        "$SEED" \
  --sampler     "$SAMPLER" \
  --scheduler   "$SCHEDULER" \
  --batch-size  "$BATCH_SIZE" \
  --base-steps  "$BASE_STEPS" \
  --lora-dir    "$LORA_DIR" \
  --prefix      "$PREFIX" \
  "${LORA_ARGS[@]}"
