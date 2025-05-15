#!/usr/bin/env python3
"""Compara saídas de *dois* re‑grids.

▪ **Batch grids**: empilha verticalmente (A em cima, B em baixo) com a etiqueta
  de cada diretório (LoRA) à esquerda.
▪ **Grid geral**: coloca lado‑a‑lado (A | B) com cabeçalhos no topo.
"""

import argparse, logging, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------- CLI ----------
cli = argparse.ArgumentParser(description="Compara dois re‑grids")
cli.add_argument("--dir-a", required=True, help="Regrid A (ex.: sem_estilo)")
cli.add_argument("--dir-b", required=True, help="Regrid B (ex.: com_estilo)")
cli.add_argument("--output", required=True, help="Destino das comparações")
cli.add_argument("--gap", type=int, default=40, help="Gap entre imagens")
cli.add_argument("--font", default="DejaVuSans-Bold.ttf", help="Fonte TTF")
args = cli.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

nameA = Path(args.dir_a).name
nameB = Path(args.dir_b).name
GAP   = args.gap

try:
    FONT = ImageFont.truetype(args.font, 32)
except IOError:
    FONT = ImageFont.load_default()

os.makedirs(args.output, exist_ok=True)

# ---------- helpers ----------

def _resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.width == target_w:
        return img
    ratio = target_w / img.width
    return img.resize((target_w, int(img.height*ratio)), Image.ANTIALIAS)


def combine_horizontal(path_a: str, path_b: str, out_path: str) -> None:
    """Lado‑a‑lado com cabeçalhos em cima (grid geral)."""
    imgA, imgB = Image.open(path_a), Image.open(path_b)
    max_h = max(imgA.height, imgB.height)
    imgA = imgA.resize((int(imgA.width*max_h/imgA.height), max_h), Image.ANTIALIAS) if imgA.height!=max_h else imgA
    imgB = imgB.resize((int(imgB.width*max_h/imgB.height), max_h), Image.ANTIALIAS) if imgB.height!=max_h else imgB
    header_h = FONT.getsize(nameA)[1] + 10
    total_w  = imgA.width + GAP + imgB.width
    canvas   = Image.new("RGB", (total_w, max_h + header_h), "white")
    draw     = ImageDraw.Draw(canvas)
    draw.text(((imgA.width-FONT.getsize(nameA)[0])//2, 5), nameA, font=FONT, fill="black")
    draw.text((imgA.width + GAP + (imgB.width-FONT.getsize(nameB)[0])//2, 5), nameB, font=FONT, fill="black")
    canvas.paste(imgA, (0, header_h))
    canvas.paste(imgB, (imgA.width + GAP, header_h))
    canvas.save(out_path)
    logging.info("Salvo %s", out_path)


def combine_vertical_left(path_a: str, path_b: str, out_path: str) -> None:
    """Empilha A∕B verticalmente; legenda (nome da pasta) na ESQUERDA de cada."""
    imgA, imgB = Image.open(path_a), Image.open(path_b)
    max_w = max(imgA.width, imgB.width)
    imgA, imgB = _resize_to_width(imgA, max_w), _resize_to_width(imgB, max_w)

    label_w = max(FONT.getsize(nameA)[0], FONT.getsize(nameB)[0]) + 20
    total_h = imgA.height + GAP + imgB.height
    canvas  = Image.new("RGB", (label_w + GAP + max_w, total_h), "white")
    draw    = ImageDraw.Draw(canvas)

    # texto à esquerda, centralizado verticalmente por bloco
    yA = (imgA.height - FONT.getsize(nameA)[1])//2
    yB = imgA.height + GAP + (imgB.height - FONT.getsize(nameB)[1])//2
    draw.text(( (label_w-FONT.getsize(nameA)[0])//2 , yA), nameA, font=FONT, fill="black")
    draw.text(( (label_w-FONT.getsize(nameB)[0])//2 , yB), nameB, font=FONT, fill="black")

    canvas.paste(imgA, (label_w + GAP, 0))
    canvas.paste(imgB, (label_w + GAP, imgA.height + GAP))
    canvas.save(out_path)
    logging.info("Salvo %s", out_path)

pa = args.dir_a
pb = args.dir_b

out_prompt = args.output
os.makedirs(args.output, exist_ok=True)

# --- batch grids (vertical empilhado) ---
for fn in os.listdir(pa):
    if fn.startswith("p") and "batch" in fn and fn.endswith(".png"):
        a_file = os.path.join(pa, fn)
        b_file = os.path.join(pb, fn)
        if not os.path.exists(b_file):
            logging.warning("%s inexistente em dir‑B", fn); continue
        combine_vertical_left(a_file, b_file, os.path.join(args.output, f"cmp_{fn}"))

# --- grid geral (horizontal cabeçalho) ---
grid_name = next((fn for fn in os.listdir(pa) if fn.startswith("grid_") and fn.endswith("_all_batches.png")), None)
if grid_name:
    a_master = os.path.join(pa, grid_name)
    b_master = os.path.join(pb, grid_name)
    if os.path.exists(b_master):
        combine_horizontal(a_master, b_master, os.path.join(args.output, "cmp_all_batches.png"))
