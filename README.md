# TCC — Geração Automática de Logomarcas com Stable Diffusion XL

Repositório que acompanha a monografia **“Adaptação Leve do SDXL para Criação de Logomarcas”**.  
Aqui você encontra datasets, LoRAs, fluxos ComfyUI, scripts de avaliação e o texto completo da tese.

## Estrutura

| Pasta | Descrição |
|-------|-----------|
| **lora-datasets/** | Conjuntos de imagens usados nos cinco treinamentos. |
| **lora-finetuned-output/** | Arquivos `.safetensors` resultantes dos treinamentos. |
| **prepare-dataset/** | Scripts de curadoria, limpeza e anotação das imagens. |
| **train-dataset/** | Configurações JSON do kohya\_ss + submódulo oficial. |
| **result-gathering/** | Geração de amostras, grids comparativos e cálculo de métricas. |
| **thesis/** | Projeto LaTeX, figuras e PDF final da dissertação. |

## Requisitos rápidos

* Python ≥ 3.10
* PyTorch + CUDA 11
* GPU NVIDIA ≥ 12 GB para replicar os treinos

## Clonagem

```bash
git clone --recurse-submodules https://github.com/<user>/tcc-stable-diffusion-logos.git
```

## Citação

```
TODO: inserir referência BibTeX da dissertação
```