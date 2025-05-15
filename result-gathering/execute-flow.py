#!/usr/bin/env python3
import argparse
import uuid
import json
import os
import io
import sys
import re
import urllib.request
import logging
from math import ceil
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import websocket


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler()]
    )

def parse_int_list(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use números separados por vírgula, ex.: 4,6,8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cliente ComfyUI via WebSocket com argparse, seleção de LoRAs e comparativo em lote"
    )
    parser.add_argument("-s", "--server", default="127.0.0.1:8188",
                        help="Endereço do servidor (host:porta)")
    parser.add_argument("-p", "--prompt-file", required=True,
                        help="Caminho para o JSON do prompt")
    parser.add_argument("-t", "--texts-file", required=True,
                        help="Caminho para arquivo TXT com um prompt por linha")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="Diretório de saída para as imagens")
    parser.add_argument("-i", "--input-image", default="image.png",
                        help="Imagem de input para img2img")
    
    parser.add_argument("--width", type=int, default=1024, help="Largura")
    parser.add_argument("--height", type=int, default=1024, help="Altura")

    parser.add_argument(
        "--cfg", type=parse_int_list, default=[4],
        help="Um ou vários CFG separados por vírgula (ex.: 4,6,8)"
    )
    parser.add_argument(
        "--steps", type=parse_int_list, default=[20],
        help="Um ou vários Steps separados por vírgula (ex.: 15,30)"
    )

    parser.add_argument("--seed", type=int, default=100, help="Seed")
    parser.add_argument("--sampler", default="dpmpp_2m_sde", help="Sampler")
    parser.add_argument("--scheduler", default="karras", help="Scheduler")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--base-steps", type=float, default=1.0, help="Base steps portion")
    parser.add_argument("--denoise", type=float, default=0.8, help="img2img Denoise Strength")
    parser.add_argument("--lora-dir", default=".",
                        help="Diretório onde estão os arquivos .safetensors")
    parser.add_argument(
        "--lora", action="append", metavar="CATEGORIA[:EPOCH]",
        help=(
            "Escolha de LoRA por categoria e epoch, ex: cartoon:5 ou iconic:latest. "
            "Slots sequenciais (até 6)."
        )
    )
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefixo geral para todos os arquivos gerados")
    return parser.parse_args()


def discover_loras(lora_dir: str) -> dict[str, dict]:
    logging.info("Descobrindo LoRAs em: %s", lora_dir)
    groups: dict[str, dict] = {}
    for fname in os.listdir(lora_dir):
        if not fname.endswith(".safetensors"): continue
        name = fname[:-len(".safetensors")]
        m = re.match(r"^(.+)-(\d{6})$", name)
        parts = name.split("_")
        if len(parts) < 3:
            continue
        categoria = parts[2]
        if categoria not in groups:
            groups[categoria] = {"latest": None, "epochs": []}
        if m:
            epoch = int(m.group(2))
            groups[categoria]["epochs"].append((epoch, fname))
        else:
            groups[categoria]["latest"] = fname
    for cat in groups:
        groups[cat]["epochs"].sort(key=lambda x: x[0])
    logging.info("Categorias de LoRA encontradas: %s", list(groups.keys()))
    return groups


def choose_lora_file(groups: dict[str, dict], spec: str) -> tuple[str, float, str]:
    ep_str = "latest"
    strength = "1.0"
    spec_split = spec.split(":")
    if len(spec_split) > 2:
        [category, strength, ep_str] = spec_split
    elif len(spec_split) > 1:
        [category, strength] = spec_split
    else:
        [category] = spec_split
    strength_float = 1.0 if not strength else float(strength)
    logging.info("Spec: %s - Categoria: %s - Strength: %f - Epoch: %s", spec, category, strength_float, ep_str)
    if not ep_str:
        ep_str = "latest"
    if category not in groups:
        raise ValueError(f"Categoria desconhecida: {category}")
    info = groups[category]
    if ep_str in ("latest", "last"):
        fname = info.get("latest")
        if not fname:
            raise ValueError(f"Nenhum arquivo 'latest' encontrado para {category}")
        return fname, strength_float, "latest"
    try:
        target = int(ep_str)
    except ValueError:
        raise ValueError(f"Epoch inválido: {ep_str}")
    for epoch, fname in info["epochs"]:
        if epoch == target:
            return fname, strength_float, ep_str
    raise ValueError(f"Epoch {target} não encontrado para {category}")


def load_prompt(path: str) -> dict:
    logging.info("Carregando prompt JSON: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_texts(path: str) -> list[str]:
    logging.info("Carregando prompts de texto: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_images(images: dict[str, list[bytes]], output_dir: str,
                prefix: str = "", index: int = 0, phase: str = "no"):
    os.makedirs(output_dir, exist_ok=True)
    total = sum(len(v) for v in images.values())
    logging.info("Salvando %d imagens (%s) em %s", total, phase, output_dir)
    count = 0
    for img_list in images.values():
        for img_data in img_list:
            img = Image.open(io.BytesIO(img_data))
            parts = [prefix, str(index), phase, str(count)]
            name = "_".join([p for p in parts if p])
            filename = f"{name}.png"
            img.save(os.path.join(output_dir, filename))
            count += 1


def queue_prompt(prompt: dict, server: str, client_id: str) -> dict:
    logging.info("Enviando prompt ao servidor, client_id=%s", client_id)
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"http://{server}/prompt", data=data)
    resp = urllib.request.urlopen(req).read()
    logging.debug("Resposta /prompt: %s", resp)
    return json.loads(resp)


def get_images(ws: websocket.WebSocket, prompt: dict, server: str, client_id: str) -> dict[str, list[bytes]]:
    logging.info("Recebendo imagens via WebSocket, servidor=%s", server)
    prompt_id = queue_prompt(prompt, server, client_id)["prompt_id"]
    outputs: dict[str, list[bytes]] = {}
    current_node = None
    total_bytes = 0
    logging.info("Prompt_id recebido: %s", prompt_id)
    while True:
        msg = ws.recv()
        if isinstance(msg, str):
            j = json.loads(msg)
            logging.debug("WS msg: %s", j)
            if j.get("type") == "executing" and j["data"].get("prompt_id") == prompt_id:
                node = j["data"].get("node")
                logging.info("Executando nó: %s", node or "<fim>")
                if not node:
                    break
                current_node = node
        else:
            total_bytes += len(msg)
            logging.info("Bytes: %d", total_bytes)
            if current_node == "save_image_websocket_node":
                outputs.setdefault(current_node, []).append(msg[8:])
    logging.info("Recebimento completo: %d bytes", total_bytes)
    return outputs


def make_comparison(
    all_no: list[bytes], all_lora: list[bytes],
    prompt_text: str, loras: list[str],
    cfg: int, steps: int,  # <<< agora recebe cfg/steps
    output_dir: str, prefix: str = "", tag: str = "",
    show_prompt: bool = False, args=None,
) -> str:
    logging.info("Gerando comparação vertical para prompt %s", tag)

    # configurações de layout
    thumb_size = (256, 256)
    margin_x, margin_y = 20, 10
    label_width = 100  # espaço reservado à esquerda para legendas

    # carrega fonte
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    # texto informativo
    params = (
        f"CFG: {cfg} - Steps: {steps} - "
        f"Sampler: {args.sampler} - Scheduler: {args.scheduler}"
    )
    hdr_main = f"Prompt: {prompt_text}\nLoRAs: {', '.join(loras)}\n" if show_prompt else ""
    header = hdr_main + params

    # mede tamanho do header
    dummy = Image.new("RGB", (1, 1))
    ddraw = ImageDraw.Draw(dummy)
    text_w, text_h = ddraw.multiline_textsize(header, font=font)

    # número de imagens a mostrar (min(batch, batch))
    n = min(len(all_no), len(all_lora))
    row_w = thumb_size[0] * n + margin_x * (n - 1)

    # dimensões do canvas
    canvas_w = label_width + margin_x + max(text_w, row_w) + margin_x
    canvas_h = (
        text_h + margin_y +
        (thumb_size[1] + margin_y) * 2 +  # duas linhas de imagens
        margin_y +  # espaço antes das legendas
        font.getsize("Sem LoRA")[1] +
        margin_y
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # desenha header centralizado acima das imagens
    hx = (canvas_w - text_w) // 2
    hy = margin_y
    draw.multiline_text((hx, hy), header, font=font, fill="black", align="center")

    # posição inicial das duas linhas
    y0 = hy + text_h + margin_y

    # desenha primeira linha: Sem LoRA
    draw.text((margin_x, y0 + (thumb_size[1]//2) - font.getsize("Sem LoRA")[1]//2),
              "Sem\nLoRA", font=font, fill="black")
    for i in range(n):
        img = Image.open(io.BytesIO(all_no[i])).resize(thumb_size, Image.ANTIALIAS)
        x = label_width + margin_x + i * (thumb_size[0] + margin_x)
        canvas.paste(img, (x, y0))

    # desenha segunda linha: Com LoRA
    y1 = y0 + thumb_size[1] + margin_y
    draw.text((margin_x, y1 + (thumb_size[1]//2) - font.getsize("Com LoRA")[1]//2),
              "Com\nLoRA", font=font, fill="black")
    for i in range(n):
        img = Image.open(io.BytesIO(all_lora[i])).resize(thumb_size, Image.ANTIALIAS)
        x = label_width + margin_x + i * (thumb_size[0] + margin_x)
        canvas.paste(img, (x, y1))

    # adiciona metadados
    meta = PngImagePlugin.PngInfo()
    if args:
        for k, v in vars(args).items():
            meta.add_text(k, str(v))
    meta.add_text("prompt_text", prompt_text)
    meta.add_text("loras", json.dumps(loras))

    # salva
    filename = f"{prefix}_{tag}_compare.png" if prefix else f"{tag}_compare.png"
    out_path = os.path.join(output_dir, filename)
    canvas.save(out_path, pnginfo=meta)
    logging.info("Comparação salva em %s", out_path)
    return out_path


def assemble_grid(images: list[str], cfgs: list[int], steps: list[int],
                  output_dir: str, prefix: str, prompt: str, loras: list[str],
                  p_idx: int):
    thumbs = [Image.open(p) for p in images]
    cell_w, cell_h = thumbs[0].size

    grid_w = cell_w * len(steps)
    grid_h = cell_h * len(cfgs)

    # header comum
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    header = f"Prompt: {prompt}\nLoRAs: {', '.join(loras)}"
    dummy = Image.new("RGB", (1, 1))
    tw, th = ImageDraw.Draw(dummy).multiline_textsize(header, font=font)

    canvas = Image.new("RGB", (grid_w, grid_h + th + 20), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text(((grid_w - tw)//2, 10), header, font=font, fill="black", align="center")

    # cola células
    idx = 0
    for r, cfg in enumerate(cfgs):
        for c, st in enumerate(steps):
            x = c * cell_w
            y = th + 20 + r * cell_h
            canvas.paste(thumbs[idx], (x, y))
            idx += 1

    name = f"{prefix}_p{p_idx}_grid.png" if prefix else f"p{p_idx}_grid.png"
    out = os.path.join(output_dir, name)
    canvas.save(out)
    logging.info("Grid salvo em %s", out)


def apply_params(base_prompt: dict, *,
                 text: str,
                 cfg: int,
                 steps: int,
                 args) -> dict:
    p = json.loads(json.dumps(base_prompt))  # deep‑copy simples

    # resoluções
    p["35"]["inputs"]["width"]  = args.width
    p["35"]["inputs"]["height"] = args.height
    p["36"]["inputs"]["width"]  = args.width
    p["36"]["inputs"]["height"] = args.height
    p["108"]["inputs"]["value"] = str(args.width)
    p["110"]["inputs"]["value"] = str(args.height)

    # CFG / Steps / Seed
    p["111"]["inputs"]["value"] = str(cfg)
    p["112"]["inputs"]["value"] = str(steps)
    p["113"]["inputs"]["value"] = str(args.seed)

    # Sampler + Scheduler (nós 55 e 59 no seu fluxo)
    p["55"]["inputs"]["sampler_name"] = args.sampler
    p["55"]["inputs"]["scheduler"]    = args.scheduler
    p["59"]["inputs"]["sampler_name"] = args.sampler
    p["59"]["inputs"]["scheduler"]    = args.scheduler

    # batch e outras constantes
    p["114"]["inputs"]["value"] = str(args.batch_size)
    p["115"]["inputs"]["value"] = str(args.base_steps)

    # prompt de texto
    p["63"]["inputs"]["text"]   = text
    p["35"]["inputs"]["text_g"] = text
    p["35"]["inputs"]["text_l"] = text

    # img2img img input (opcional)
    if "89" in p and "117" in p and args.input_image:
        p["89"]["inputs"]["image"] = args.input_image
        p["117"]["inputs"]["value"] = args.denoise

    # fix-text img input (opcional)
    if "121" in p and args.input_image:
        p["121"]["inputs"]["image"] = args.input_image

    return p


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.info("Args: %s", args)

    # --- preparação inicial --------------------------------------------------
    client_id   = str(uuid.uuid4())
    lora_groups = discover_loras(args.lora_dir)

    base_prompt_json = load_prompt(args.prompt_file)
    all_prompts_txt  = load_texts(args.texts_file)

    base_out = os.path.join(args.output_dir, args.prefix) if args.prefix else args.output_dir
    singles  = os.path.join(base_out, "singles")   # PNGs individuais
    os.makedirs(singles, exist_ok=True)

    # -------------------------------------------------------------------------  
    for p_idx, text in enumerate(all_prompts_txt, start=1):
        logging.info("=== Prompt %d/%d ===", p_idx, len(all_prompts_txt))
        logging.info("Texto: %s", text)

        cmp_paths: list[str] = []          # guardará os caminhos de cada comparação
        cfg_list   = args.cfg              # listas vindas do argparse (já são list[int])
        steps_list = args.steps

        for cfg_val in cfg_list:
            for st_val in steps_list:
                tag = f"p{p_idx}_cfg{cfg_val}_st{st_val}"
                logging.info("→ CFG=%d | Steps=%d", cfg_val, st_val)

                # ---------- prompt SEM LoRA ----------
                prompt_no = apply_params(
                    base_prompt_json,
                    text=text,
                    cfg=cfg_val,
                    steps=st_val,
                    args=args
                )
                for nid, node in prompt_no.items():
                    if "lora_name" in node.get("inputs", {}):
                        node["inputs"]["switch"] = "Off"

                ws = websocket.WebSocket()
                ws.connect(f"ws://{args.server}/ws?clientId={client_id}")
                imgs_no = get_images(ws, prompt_no, args.server, client_id)
                ws.close()

                # ---------- prompt COM LoRA ----------
                if args.lora:
                    prompt_l = apply_params(
                        base_prompt_json,
                        text=text,
                        cfg=cfg_val,
                        steps=st_val,
                        args=args
                    )
                    lora_slots = [
                        nid for nid, node in prompt_l.items()
                        if "lora_name" in node.get("inputs", {})
                    ]
                    for nid in lora_slots:
                        prompt_l[nid]["inputs"]["switch"] = "Off"

                    for i, spec in enumerate(args.lora):
                        fname, strength, _ = choose_lora_file(lora_groups, spec)
                        nid = lora_slots[i]
                        prompt_l[nid]["inputs"]["lora_name"]      = fname
                        prompt_l[nid]["inputs"]["strength_model"] = strength
                        prompt_l[nid]["inputs"]["switch"]         = "On"

                    ws = websocket.WebSocket()
                    ws.connect(f"ws://{args.server}/ws?clientId={client_id}")
                    imgs_l = get_images(ws, prompt_l, args.server, client_id)
                    ws.close()
                else:
                    imgs_l = {k: v.copy() for k, v in imgs_no.items()}  # sem LoRA = igual

                # ---------- salvar PNGs individuais ----------
                save_images(imgs_no, singles, prefix=tag, index=p_idx, phase="no")
                save_images(imgs_l, singles, prefix=tag, index=p_idx, phase="lora")

                # ---------- gerar comparação (duas linhas) ----------
                cmp_file = make_comparison(
                    sum(imgs_no.values(), []),
                    sum(imgs_l.values(), []),
                    prompt_text=text,
                    loras=args.lora or ["(nenhuma)"],
                    cfg=cfg_val,
                    steps=st_val,
                    output_dir=base_out,
                    prefix=args.prefix,
                    tag=tag,
                    show_prompt=False,
                    args=args
                )
                cmp_paths.append(cmp_file)

        # ----------- grid final (linhas = CFG | colunas = Steps) --------------
        assemble_grid(
            images = cmp_paths,
            cfgs   = cfg_list,
            steps  = steps_list,
            output_dir = base_out,
            prefix     = args.prefix,
            prompt     = text,
            loras      = args.lora or ["(nenhuma)"],
            p_idx      = p_idx
        )

if __name__ == "__main__":
    main()
