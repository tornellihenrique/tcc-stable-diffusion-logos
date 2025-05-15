#!/usr/bin/env bash

# ---------- Configurações (edite conforme necessário) ----------
SERVER="127.0.0.1:8188"
PROMPT_FILE="fix-text.json"
TEXTS_FILE="./prompts/aurora.txt"
OUTPUT_DIR="./output"
INPUT_DIR="/home/henrique/Projects/tcc/result-gathering/input/fix_text"
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
DENOISE=0.3
LORA_DIR="/home/henrique/comfy/ComfyUI/models/loras"

# Defina aqui suas LoRAs: categoria ou categoria:epoch:strength
LORAS=(
  "wordmark:1.0"
)

PREFIX_BASE="fix_text1_wordmark"

# ---------- Monta argumentos de LoRA ----------
LORA_ARGS=()
for L in "${LORAS[@]}"; do
  LORA_ARGS+=("--lora" "$L")
done

# ---------- Executa para cada imagem em INPUT_DIR ----------
for IMG_PATH in "$INPUT_DIR"/*.{png,jpg,jpeg}; do
  # Extrai nome-base sem extensão
  IMG_NAME=$(basename "$IMG_PATH")
  BASE_NAME="${IMG_NAME%.*}"

  # Define prefix único
  PREFIX="${PREFIX_BASE}_${BASE_NAME}"

  echo "[INFO] Processando imagem: $IMG_NAME"

  python execute-flow.py \
    --server      "$SERVER" \
    --prompt-file "$PROMPT_FILE" \
    --texts-file  "$TEXTS_FILE" \
    --output-dir  "$OUTPUT_DIR" \
    --input-image "$IMG_PATH" \
    --width       "$WIDTH" \
    --height      "$HEIGHT" \
    --cfg         "$CFG" \
    --steps       "$STEPS" \
    --seed        "$SEED" \
    --sampler     "$SAMPLER" \
    --scheduler   "$SCHEDULER" \
    --batch-size  "$BATCH_SIZE" \
    --base-steps  "$BASE_STEPS" \
    --denoise     "$DENOISE" \
    --lora-dir    "$LORA_DIR" \
    --prefix      "$PREFIX" \
    "${LORA_ARGS[@]}"

  echo "[INFO] Finalizado: $IMG_NAME -> $TARGET_DIR"
done
