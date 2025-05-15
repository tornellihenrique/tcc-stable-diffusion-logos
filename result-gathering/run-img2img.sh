#!/usr/bin/env bash

# ---------- Configurações (edite conforme necessário) ----------
SERVER="127.0.0.1:8188"
PROMPT_FILE="img2img.json"
TEXTS_FILE="./prompts/ufu1.txt"
OUTPUT_DIR="./output"
INPUT_IMG="/home/henrique/Projects/tcc/result-gathering/input/ufu.png"
WIDTH=1024
HEIGHT=1024
CFG=6
STEPS=20
SEED=100
SAMPLER="dpmpp_2m_sde"
SCHEDULER="karras"
BATCH_SIZE=4
BASE_STEPS=1.0
DENOISE=0.3
LORA_DIR="/home/henrique/comfy/ComfyUI/models/loras"

# Defina aqui suas LoRAs: categoria ou categoria:epoch
LORAS=(
  "iconic:1.0"
  "minimalistic:1.0"
)

PREFIX="ufu2"

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
  --input-image "$INPUT_IMG" \
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
