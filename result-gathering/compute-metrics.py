#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
import pytesseract
import torch
import open_clip
from tqdm import tqdm

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

EXPECTED_RE = re.compile(r"text\s+([A-Za-z0-9\-_']+)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
MAX_WORDS = 3
MAX_CHARS = 30

# ───────────────────────── CLIP helpers ──────────────────────────

def load_clip(device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"CLIP device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, tokenizer, preprocess, device


def clip_similarity(model, tokenizer, preprocess, device, text: str, img_path: Path) -> float:
    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        tok = tokenizer([text]).to(device)
        with torch.no_grad():
            img_f = model.encode_image(img)
            txt_f = model.encode_text(tok)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            sim = (img_f @ txt_f.T).item()
        return (sim + 1) / 2
    except Exception as e:
        logging.warning(f"CLIP erro {img_path.name}: {e}")
        return float('nan')

# ───────────────────────── OCR helpers ───────────────────────────

def expected_from_prompt(prompt: str) -> str:
    m = EXPECTED_RE.search(prompt)
    return m.group(1) if m else prompt.split()[0]


def ocr_eval(path: Path, expected: str) -> Tuple[str, float, bool]:
    try:
        raw = pytesseract.image_to_string(Image.open(path))
        words = WORD_RE.findall(raw)
        clean_raw = re.sub(r"\s+", "", raw).lower()
        if len(words) == 0 or len(words) > MAX_WORDS or len(clean_raw) > MAX_CHARS:
            return "", float('nan'), True
        exp = re.sub(r"\s+", "", expected).lower()
        score = 1 - Levenshtein.distance(clean_raw, exp) / max(len(exp), 1) if exp else float('nan')
        return raw.strip(), score, False
    except Exception as e:
        logging.warning(f"OCR erro {path.name}: {e}")
        return "", float('nan'), True

# ───────────────────────── main ────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Calcula métricas e gera summary ordenado + agg")
    ap.add_argument("prompts_csv")
    ap.add_argument("summary_csv")
    args = ap.parse_args()

    with open(args.prompts_csv, newline='', encoding='utf-8') as fh:
        base_rows = list(csv.DictReader(fh))
    if not base_rows:
        logging.error("prompts.csv vazio"); sys.exit(1)

    logging.info(f"Processando {len(base_rows)} pares")
    model, tok, prep, device = load_clip()

    summaries: List[Dict[str, str]] = []
    for row in tqdm(base_rows, unit='pair'):
        img_no = Path(row.get('image_path_no', ''))
        img_lo = Path(row.get('image_path_lora', ''))
        prompt = row.get('prompt_text', '')
        expected = expected_from_prompt(prompt)

        clip_no = clip_similarity(model, tok, prep, device, prompt, img_no) if img_no.exists() else float('nan')
        clip_lo = clip_similarity(model, tok, prep, device, prompt, img_lo) if img_lo.exists() else float('nan')
        delta_clip = clip_lo - clip_no if not (np.isnan(clip_no) or np.isnan(clip_lo)) else float('nan')

        ocr_no, score_no, miss_no = ocr_eval(img_no, expected) if img_no.exists() else ('', float('nan'), True)
        ocr_lo, score_lo, miss_lo = ocr_eval(img_lo, expected) if img_lo.exists() else ('', float('nan'), True)
        delta_ocr = score_lo - score_no if not (np.isnan(score_no) or np.isnan(score_lo)) else float('nan')

        out = row.copy()
        out.update({
            'expected_text': expected,
            'clip_similarity_no': f"{clip_no:.4f}" if not np.isnan(clip_no) else '',
            'clip_similarity_lora': f"{clip_lo:.4f}" if not np.isnan(clip_lo) else '',
            'clip_delta': f"{delta_clip:.4f}" if not np.isnan(delta_clip) else '',
            'ocr_text_no': ocr_no,
            'ocr_text_lora': ocr_lo,
            'ocr_score_no': f"{score_no:.4f}" if not np.isnan(score_no) else '',
            'ocr_score_lora': f"{score_lo:.4f}" if not np.isnan(score_lo) else '',
            'ocr_delta': f"{delta_ocr:.4f}" if not np.isnan(delta_ocr) else '',
            'ocr_missing_no': int(miss_no),
            'ocr_missing_lora': int(miss_lo),
        })
        summaries.append(out)

    summaries.sort(key=lambda r: (
        int(r.get('prompt_idx', 0)),
        float(r.get('cfg', 0)),
        int(r.get('steps', 0)),
        int(r.get('batch_idx', 0))
    ))

    # grava summary.csv
    header = list(summaries[0].keys())
    with open(args.summary_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader(); writer.writerows(summaries)
    logging.info(f"summary.csv salvo ({len(summaries)} linhas)")

    # Agg path derivado do nome do summary
    base = Path(args.summary_csv)
    agg_path = base.parent / f"{base.stem}_agg{base.suffix}"

    groups: Dict[Tuple, List[Tuple[float,float,int,int]]] = {}
    for r in summaries:
        key = (r['flow'], r['loras'], r['cfg'], r['steps'])
        clip_d = float(r['clip_delta']) if r['clip_delta'] else np.nan
        ocr_d = float(r['ocr_delta']) if r['ocr_delta'] else np.nan
        groups.setdefault(key, []).append((clip_d, ocr_d, int(r['ocr_missing_no']), int(r['ocr_missing_lora'])))

    with open(agg_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['flow','loras','cfg','steps','n','clip_delta_mean','ocr_delta_mean','missing_no_%','missing_lora_%'])
        for key, vals in groups.items():
            arr = np.array(vals, dtype=float)
            clip_m = np.nanmean(arr[:,0])
            ocr_m = np.nanmean(arr[:,1])
            miss_no_pct = np.nanmean(arr[:,2])*100
            miss_lo_pct = np.nanmean(arr[:,3])*100
            w.writerow([*key, len(vals), f"{clip_m:.4f}", f"{ocr_m:.4f}", f"{miss_no_pct:.1f}", f"{miss_lo_pct:.1f}"])
    logging.info(f"summary_agg salvo em {agg_path}")

if __name__ == '__main__':
    main()
