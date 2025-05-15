#!/usr/bin/env python3
"""
prepare_images_simple.py
------------------------
Prepara um diretório cheio de imagens para o formato básico do kohya_ss:
`output_dir/img/{repeats}_{instance_prompt} {class_prompt}/`.

• NÃO cria arquivos .txt individuais.
• Suporta redimensionamento, filtro por extensão e amostragem por porcentagem.

Uso:

python prepare_images_simple.py \
  --input_dir "/pasta/origem" \
  --output_dir "/pasta/saida" \
  --subset_percentage 25 \
  --resize 512 512 \
  --instance_prompt "logo" \
  --class_prompt "logos"
"""

import os
import argparse
import logging
import random
from pathlib import Path

from PIL import Image


# ---------- Argumentos ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preparar dataset de imagens para kohya_ss (sem .txt)"
    )
    p.add_argument("--input_dir", required=True, help="Pasta com as imagens originais")
    p.add_argument("--output_dir", required=True, help="Pasta de saída")
    p.add_argument("--instance_prompt", default="logo", help="Usado no nome da subpasta")
    p.add_argument("--class_prompt", default="logos", help="Usado no nome da subpasta")
    p.add_argument("--repeats", type=int, default=100, help="Prefixo repeats da subpasta")
    p.add_argument("--image_ext", default="png", help="Extensão final (png ou jpg)")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], help="Dimensões de saída, ex.: --resize 512 512")
    p.add_argument("--subset_percentage", type=float, default=100.0, help="Porcentagem de imagens a copiar (0‑100)")
    return p.parse_args()


# ---------- Logging ----------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler()]
    )


# ---------- Processamento ----------
def process_images(
    src_folder: Path,
    dst_folder: Path,
    image_ext: str,
    resize_dims: tuple[int, int],
    subset_pct: float
):
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".avif"}
    all_imgs = [p for p in src_folder.rglob("*") if p.suffix.lower() in allowed]
    logging.info("Imagens encontradas: %d", len(all_imgs))

    if subset_pct < 100.0:
        k = max(1, int(len(all_imgs) * subset_pct / 100.0))
        all_imgs = random.sample(all_imgs, k)
        logging.info("Após amostragem (%.1f%%): %d imagens", subset_pct, len(all_imgs))

    dst_folder.mkdir(parents=True, exist_ok=True)
    total = 0
    target_w, target_h = resize_dims

    for img_path in all_imgs:
        try:
            im = Image.open(img_path).convert("RGB")
            w, h = im.size

            # escala para preencher totalmente (aspect-fill)
            scale = max(target_w / w, target_h / h)
            new_size = (int(w * scale), int(h * scale))
            im_resized = im.resize(new_size, Image.Resampling.LANCZOS)

            # calcula box de recorte central
            left = (new_size[0] - target_w) // 2
            top  = (new_size[1] - target_h) // 2
            right  = left + target_w
            bottom = top  + target_h
            im_cropped = im_resized.crop((left, top, right, bottom))

            out_name = f"sample_{total:06d}.{image_ext}"
            im_cropped.save(dst_folder / out_name)

            total += 1
            if total % 100 == 0:
                logging.info("Copiadas %d imagens…", total)
        except Exception as e:
            logging.error("Erro em %s: %s", img_path, e)

    logging.info("Concluído — %d imagens processadas.", total)


# ---------- main ----------
def main():
    setup_logging()
    args = parse_args()

    src = Path(args.input_dir).expanduser()
    if not src.exists():
        logging.error("Pasta de entrada não encontrada: %s", src)
        return

    root_out = Path(args.output_dir).expanduser()
    (root_out / "log").mkdir(parents=True, exist_ok=True)
    (root_out / "model").mkdir(parents=True, exist_ok=True)
    images_dst = root_out / "img" / f"{args.repeats}_{args.instance_prompt} {args.class_prompt}"

    random.seed(42)
    process_images(
        src_folder=src,
        dst_folder=images_dst,
        image_ext=args.image_ext,
        resize_dims=tuple(args.resize),
        subset_pct=args.subset_percentage
    )


if __name__ == "__main__":
    main()
