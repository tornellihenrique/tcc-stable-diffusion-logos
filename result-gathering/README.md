# result-gathering

Ferramentas para geração de amostras, montagem de grids comparativos e cálculo de métricas.

| Script | Descrição |
|--------|-----------|
| execute-flow.py | Executa um fluxo ComfyUI via API REST. |
| generate-prompts-csv.py | Cria CSV com instruções textuais sistemáticas. |
| regrid1.py / regrid2.py | Organiza imagens em grades para comparação. |
| compute-metrics.py | Calcula CLIP-Similarity e OCR-Accuracy. |

Exemplo rápido:

```bash
python execute-flow.py \
  --flow flows/text2img.json \
  --prompts prompts.csv \
  --lora sdxl_logo_wordmark.safetensors
```

> Preencha cada bloco “Descrição” com instruções detalhadas de uso.
