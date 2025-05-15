#!/usr/bin/env python3
"""generate_prompts_csv.py  — versão *merge* no/LoRA

Cria **prompts.csv** onde cada linha agrega o par de imagens *sem* e *com*  LoRA
pertencentes ao mesmo (prompt_idx, cfg, steps, batch_idx).

Colunas-chave produzidas:
  flow, prompt_idx, cfg, steps, batch_idx,
  image_path_no, image_path_lora,
  width, height, seed, sampler, scheduler, loras, prompt_text

Uso:
    python generate_prompts_csv.py --flow text2img --output prompts.csv DIR1 DIR2 ...

Dependências: Pillow>=10.0
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image
except ImportError:
    sys.stderr.write("[ERRO] Pillow ausente. Instale com: pip install pillow\n")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Padrões de filename
# ──────────────────────────────────────────────────────────────────────────────
COMPARE_RE = re.compile(
    r"_p(?P<pidx>\d+)_cfg(?P<cfg>[0-9.]+)_st(?P<steps>\d+?)_compare\.png$",
    re.IGNORECASE,
)
SINGLE_RE = re.compile(
    r"^p(?P<pidx>\d+)_cfg(?P<cfg>[0-9.]+)_st(?P<steps>\d+?)_\d+?_"
    r"(?P<var>no|lora)_(?P<batch>\d+)\.png$",
    re.IGNORECASE,
)

# Campos numéricos a converter
NUM_FIELDS = {"cfg": float, "steps": int, "prompt_idx": int, "batch_idx": int, "seed": int}
# Metadados que mantemos
FILTER_KEYS = {"width", "height", "seed", "sampler", "scheduler", "loras", "prompt_text"}

# ──────────────────────────────────────────────────────────────────────────────
# Metadata extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_compare_meta(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    try:
        with Image.open(path) as im:
            for k, v in im.info.items():
                lk = k.lower().replace(" ", "_")
                if lk not in FILTER_KEYS:
                    continue
                if lk == "loras":
                    try:
                        lst = ast.literal_eval(v) if isinstance(v, str) else v
                        if isinstance(lst, list):
                            d[lk] = ";".join(map(str, lst))
                            continue
                    except Exception:
                        pass
                d[lk] = str(v)
    except Exception as exc:
        d["meta_error"] = str(exc)
    return d

# ──────────────────────────────────────────────────────────────────────────────
# Build maps: compare metadata per (pidx,cfg,steps) and singles aggregated pair
# ──────────────────────────────────────────────────────────────────────────────

def build_meta_map(root: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    m: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for p in root.rglob("*_compare.png"):
        mm = COMPARE_RE.search(p.name)
        if not mm:
            continue
        key = (mm["pidx"], mm["cfg"], mm["steps"])
        m[key] = extract_compare_meta(p)
    return m


def gather_rows(root: Path) -> List[Dict[str, str]]:
    # dict key=(pidx,cfg,steps,batch) : partial row
    aggregated: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    meta_map = build_meta_map(root)

    for img in root.rglob("singles/*.png"):
        sm = SINGLE_RE.match(img.name)
        if not sm:
            continue
        key4 = (sm["pidx"], sm["cfg"], sm["steps"], sm["batch"])
        row = aggregated.setdefault(key4, {
            "prompt_idx": sm["pidx"],
            "cfg": sm["cfg"],
            "steps": sm["steps"],
            "batch_idx": sm["batch"],
        })
        # add path variant
        if sm["var"].lower() == "no":
            row["image_path_no"] = str(img.resolve())
        else:
            row["image_path_lora"] = str(img.resolve())

    # merge compare meta
    merged_rows: List[Dict[str, str]] = []
    for (pidx,cfg,steps,batch), row in aggregated.items():
        row_key = (pidx,cfg,steps)
        row.update(meta_map.get(row_key, {}))
        merged_rows.append(cast_types(row))
    return merged_rows

# ──────────────────────────────────────────────────────────────────────────────
# Cast numeric helper
# ──────────────────────────────────────────────────────────────────────────────

def cast_types(r: Dict[str, str]) -> Dict[str, str]:
    for k, typ in NUM_FIELDS.items():
        if k in r:
            try:
                r[k] = typ(r[k])
            except Exception:
                pass
    return r

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gera prompts.csv (linha por par no/LoRA)")
    ap.add_argument("roots", nargs='+', help="Pastas raiz contendo os resultados")
    ap.add_argument("--flow", default="text2img", help="Valor da coluna flow")
    ap.add_argument("--output", default="prompts.csv", help="CSV de saída")
    args = ap.parse_args()

    all_rows: List[Dict[str, str]] = []
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            sys.stderr.write(f"[WARN] Caminho não existe: {root}\n")
            continue
        rows = gather_rows(root)
        for r in rows:
            r["flow"] = args.flow
        all_rows.extend(rows)

    if not all_rows:
        sys.stderr.write("[AVISO] Nenhum dado coletado.\n")
        return

    # header: flow + resto ordenado
    keys = set().union(*(r.keys() for r in all_rows))
    keys.discard("flow")
    header = ["flow"] + sorted(keys)

    with open(args.output, "w", newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[OK] {len(all_rows)} linhas escritas em {args.output}")


if __name__ == "__main__":
    main()
