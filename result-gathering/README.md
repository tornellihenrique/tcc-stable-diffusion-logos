# result-gathering

Scripts, fluxos ComfyUI e utilitários usados para produzir todas as imagens,
grades comparativas e métricas apresentadas na dissertação.

> **Requisito essencial:** o **ComfyUI** precisa estar rodando em segundo plano —
> consulte o repositório oficial <https://github.com/comfyanonymous/ComfyUI>
> para instalação. O script `execute-flow.py` comunica-se via API REST
> (`http://127.0.0.1:8188` por padrão).

## Estrutura

```
result-gathering/
├── execute-flow\.py
├── generate-prompts-csv.py
├── compute-metrics.py
├── regrid1.py
├── regrid2.py
├── requirements.txt
├── flows/
├── text2img.json
├── img2img.json
├── fix-text.json
├── prompts/                # .txt usados no execute-flow
├── metrics/                # CSVs gerados (summary, summary\_agg)
└── *.sh                    # scripts de conveniência (ver abaixo)
```

### Saída completa (imagens)

O conjunto completo de imagens geradas,
incluindo as pastas `output/` e `output-regrid/`, está disponível [neste link do Google Drive](https://drive.google.com/drive/folders/1Bp7RrKmlzmtDQTkTmp95nv4xNMyZqn5I?usp=sharing).

## Scripts principais

### `execute-flow.py`

Executa um fluxo ComfyUI parametrizado e salva resultados organizados.

| Parâmetro | Descrição |
|-----------|-----------|
| `--server`          | URL do servidor ComfyUI (default `http://127.0.0.1:8188`) |
| `--prompt-file`     | Caminho para o JSON do fluxo (ex.: `flows/text2img.json`) |
| `--texts-file`      | Arquivo `.txt` com instruções textuais, uma por linha |
| `--output-dir`      | Diretório raiz para salvar imagens |
| `--input-image`     | Imagem de entrada (para *img2img* ou *fix-text*) |
| `--width --height`  | Resolução de saída |
| `--cfg`             | Escalas CFG, separadas por vírgula |
| `--steps`           | Números de passos, separados por vírgula |
| `--seed`            | Semente inicial (int) |
| `--sampler`         | Nome do sampler (e.g., `dpmpp_2m`) |
| `--scheduler`       | Nome do scheduler (e.g., `karras`) |
| `--batch-size`      | Tamanho do lote |
| `--base-steps`      | Porcentagem usada para aplicar o Refiner |
| `--denoise`         | Fator de ruído para *img2img* |
| `--lora-dir`        | Pasta onde estão os LoRAs (ComfyUI) |
| `--lora`            | Lista de LoRAs no formato `nome:peso` ou `nome:latest` |
| `--prefix`          | Prefixo usado em nomes de arquivo e subpasta |

O script cria subpastas `singles/` (todas as imagens) e `grids/`
(comparações por CFG × Steps).

### `generate-prompts-csv.py`

Varre pastas de saída do `execute-flow.py` e gera um `prompts.csv`
consolidado, necessário para o cálculo de métricas.

| Parâmetro | Descrição |
|-----------|-----------|
| `--flow`   | Rótulo livre (ex.: `text2img`) |
| `--output` | Nome do arquivo CSV |
| diretórios | Um ou mais caminhos resultantes do `execute-flow.py` |

### `compute-metrics.py`

Calcula CLIP-Similarity e OCR-Accuracy lendo um `prompts.csv` e gera dois
arquivos:

* `summary.csv` — resultado por imagem  
* `summary_agg.csv` — médias agregadas

| Parâmetro | Descrição |
|-----------|-----------|
| `prompts.csv` | arquivo gerado pelo script anterior |
| `summary.csv` | caminho de saída para a tabela de métricas |

Internamente utiliza:

* `open_clip` — ViT-L/14  
* `rapidfuzz` — distância de string  
* `pytesseract` — OCR (requer Tesseract instalado)

### `regrid1.py`

Reconstrói grades de comparação (`grids/`) a partir das pastas `singles/`.

| Parâmetro | Descrição |
|-----------|-----------|
| `--input-dir`   | Diretório `singles/` |
| `--output-dir`  | Pasta destino para as grades |
| `--prompts-file`| `.txt` original com instruções textuais |

### `regrid2.py`

Recebe duas saídas já processadas por `regrid1.py` e monta grades
comparativas lado a lado (útil para confrontar “sem LoRA” × “com LoRA”).

---

## Shell scripts auxiliares

| Script | Ação |
|--------|------|
| `run-text2img.sh`              | executa fluxo text2img em lote |
| `run-img2img.sh`               | idem para img2img |
| `run-fix-text.sh`              | idem para fix-text |
| `run-text2img-wordmark{0..3}.sh` | variações wordmark |
| `run-text2img-iconic{0..3}.sh`   | variações iconic |

Abra cada `.sh` e ajuste caminhos de LoRA, saída e filtros conforme a
necessidade antes de rodar.

---

## Dependências

```bash
pip install -r requirements.txt
```

* Python ≥ 3.10
* Tesseract 5+ (para OCR)
* Servidor ComfyUI ativo
