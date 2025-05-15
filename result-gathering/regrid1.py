#!/usr/bin/env python3

import argparse, os, re, json, logging
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Re‑grid dos singles gerados")
ap.add_argument("--input-dir", required=True,  help="dir singles")
ap.add_argument("--output-dir", required=True, help="destino grids")
ap.add_argument("--prompts-file", required=True, help="TXT dos prompts")
ap.add_argument("--loras", default="", help="Lista de LoRAs (sep. por vírgula)")
ap.add_argument("--thumb", type=int, default=256)
ap.add_argument("--gap",   type=int, default=20)
ap.add_argument("--font",  default="DejaVuSans-Bold.ttf")
args = ap.parse_args()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s — %(levelname)s — %(message)s")

# ---------- flags ----------
COMPARE = bool(args.loras.strip())
LORAS_TXT = ", ".join([x.strip() for x in args.loras.split(",") if x.strip()])

# ---------- regex ----------
SINGLE_RE = re.compile(
    r"^p(?P<pidx>\d+)_cfg(?P<cfg>[0-9.]+)_st(?P<steps>\d+?)_\d+?_(?P<phase>no|lora)_(?P<batch>\d+)\.png$",
    re.IGNORECASE,
)

# ---------- prompts ----------
with open(args.prompts_file, encoding="utf-8") as f:
    PROMPTS = [ln.rstrip("\n") for ln in f]

# ---------- coleta ----------
Data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for fn in os.listdir(args.input_dir):
    m = SINGLE_RE.match(fn)
    if not m:
        continue
    g = m.groupdict()
    Data[int(g["pidx"])][int(g["batch"])][(g["cfg"], g["steps"])][g["phase"]] = \
        os.path.join(args.input_dir, fn)

if not Data:
    logging.error("Nada encontrado em %s", args.input_dir)
    exit(1)

# ---------- layout const ----------
try:
    FONT = ImageFont.truetype(args.font, 22)
except IOError:
    FONT = ImageFont.load_default()

TH   = args.thumb
GAP  = args.gap
LEGEND_W = FONT.getsize("LoRA")[0] + 20  # espaço barra esquerda

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for pidx, batches in sorted(Data.items()):
    prompt_txt = PROMPTS[pidx-1] if pidx-1 < len(PROMPTS) else "(prompt ausente)"
    batch_imgs = []

    cfgs  = sorted({cfg for b in batches.values() for (cfg, _) in b}, key=float)
    steps = sorted({st  for b in batches.values() for (_, st) in b}, key=int)
    combos = [(c, s) for c in cfgs for s in steps]

    row_cnt = 2 if COMPARE else 1
    row_h   = TH
    label_h = FONT.getsize("CFG 999 / S 999")[1]
    grid_base_h = row_h*row_cnt + (GAP if COMPARE else 0) + label_h
    col_w   = TH
    grid_w  = col_w*len(combos) + GAP*(len(combos)-1)

    for b_idx, mapping in sorted(batches.items()):
        logging.info("Prompt %d – Batch %d", pidx, b_idx)

        header_lines = [f"Prompt: {prompt_txt}"]
        if COMPARE:
            header_lines.append(f"LoRAs: {LORAS_TXT}")
        header = "\n".join(header_lines)
        hdr_w, hdr_h = ImageDraw.Draw(Image.new("RGB",(1,1))).multiline_textsize(header, FONT)

        left_w = LEGEND_W if COMPARE else 0
        canvas_w = left_w + (GAP if COMPARE else 0) + grid_w
        canvas_h = hdr_h + GAP + grid_base_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        d_canvas = ImageDraw.Draw(canvas)
        d_canvas.multiline_text(((canvas_w-hdr_w)//2, 0), header, font=FONT,
                                fill="black", align="center")

        base = Image.new("RGB", (grid_w, grid_base_h), "white")
        d_base = ImageDraw.Draw(base)

        for idx, (cfg, st) in enumerate(combos):
            pair = mapping.get((cfg, st), {})
            if "no" not in pair:
                continue
            x = idx*(col_w+GAP)
            y0 = 0
            no_img = Image.open(pair["no"]).resize((TH, TH))
            base.paste(no_img, (x, y0))
            if COMPARE and "lora" in pair:
                lora_img = Image.open(pair["lora"]).resize((TH, TH))
                base.paste(lora_img, (x, y0 + row_h + GAP))
            # legenda cfg/steps
            lbl = f"CFG {cfg} / S {st}"
            lw, lh = FONT.getsize(lbl)
            d_base.text((x + (col_w-lw)//2, row_h*row_cnt + (GAP if COMPARE else 0)),
                        lbl, font=FONT, fill="black")

        # barra lateral “Sem / LoRA”
        if COMPARE:
            bar = Image.new("RGB", (LEGEND_W, row_h*row_cnt + GAP), "white")
            db = ImageDraw.Draw(bar)
            db.text(((LEGEND_W-FONT.getsize("Sem")[0])//2,
                     (row_h-FONT.getsize("Sem")[1])//2),
                    "Sem", font=FONT, fill="black")
            db.text(((LEGEND_W-FONT.getsize("LoRA")[0])//2,
                     row_h + GAP + (row_h-FONT.getsize("LoRA")[1])//2),
                    "LoRA", font=FONT, fill="black")
            canvas.paste(bar, (0, hdr_h + GAP))
            canvas.paste(base, (LEGEND_W + GAP, hdr_h + GAP))
        else:
            canvas.paste(base, (0, hdr_h + GAP))

        out = os.path.join(output_dir, f"p{pidx}_batch{b_idx}.png")
        canvas.save(out)
        batch_imgs.append(canvas)
        logging.info("Salvo %s", out)

    # grid vertical de batches
    if not batch_imgs:
        continue
    W = max(im.width for im in batch_imgs)
    H = sum(im.height for im in batch_imgs) + GAP*(len(batch_imgs)-1) \
        + FONT.getsize(prompt_txt)[1] + GAP
    master = Image.new("RGB", (W, H), "white")
    d_master = ImageDraw.Draw(master)
    d_master.text(((W-FONT.getsize(prompt_txt)[0])//2, 0),
                  prompt_txt, font=FONT, fill="black")
    y = FONT.getsize(prompt_txt)[1] + GAP
    for im in batch_imgs:
        master.paste(im, ((W-im.width)//2, y))
        y += im.height + GAP
    master.save(os.path.join(output_dir, f"grid_p{pidx}_all_batches.png"))
    logging.info("Grid geral prompt %d salvo.", pidx)
