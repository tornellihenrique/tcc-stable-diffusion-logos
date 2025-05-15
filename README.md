# TCC — Geração e Aperfeiçoamento de Logotipos via Treinamento Adicional no Stable Diffusion XL

Repositório que acompanha a monografia de conclusão de curso  
**"Geração e Aperfeiçoamento de Logotipos via Treinamento Adicional no Stable Diffusion XL"**.  
Aqui estão todos os recursos necessários para reproduzir o estudo — datasets,
pesos LoRA, fluxos ComfyUI, scripts de geração e avaliação de métricas, além do
texto completo da tese em LaTeX.

> **PDF oficial da tese:**  
> 👉 **TODO — inserir link público do trabalho final** 👈

## Visão geral

| Pasta | Conteúdo |
|-------|----------|
| [**lora-datasets/**](/lora-datasets/) | Conjuntos de imagens e legendas prontos para treino (padrão `kohya_ss`). |
| [**lora-finetuned-output/**](/lora-finetuned-output/) | Todos os arquivos `.safetensors` e JSONs de configuração resultantes dos 5 treinamentos. |
| [**prepare-dataset/**](/prepare-dataset/) | Scripts que limpam, baixam ou convertem imagens para o layout de treino. |
| [**train-dataset/**](/train-dataset/) | Configurações JSON + submódulo oficial do **kohya_ss** para reproduzir os treinos. |
| [**result-gathering/**](/result-gathering/) | Execução de fluxos ComfyUI, criação de grades comparativas e cálculo de CLIP/OCR. |
| [**thesis/**](/thesis/) | Projeto LaTeX completo, figuras e `tcc_final.pdf`. |

## Requisitos globais

* Python ≥ 3.10  
* PyTorch + CUDA 11 (recomendado)  
* Tesseract 5+ (para OCR)  
* GPU NVIDIA ≥ 12 GB para reproduzir treinos LoRA  
* **ComfyUI** em execução para scripts de geração  
  <https://github.com/comfyanonymous/ComfyUI>

Instale todas as dependências:

```bash
python -m venv .venv            # (ou conda create -n sdxl-logos python=3.10)
source .venv/bin/activate
pip install -r requirements.txt
````

Submódulo `kohya_ss`:

```bash
git submodule update --init
```

## Passo a passo rápido

1. **Datasets**
   Ajuste ou crie novos conjuntos via `prepare-dataset/*.py`.

2. **Treinamento LoRA**
   Use as configurações em `train-dataset/` ou crie as suas:

   ```bash
   python train-dataset/kohya_ss/train_network.py \
     --config train-dataset/sdxl_logo_wordmark_0-0003_0-0001_0-0001.json
   ```

3. **Geração de amostras**
   Ver `result-gathering/`. Exemplo:

   ```bash
   cd result-gathering
   bash run-text2img-wordmark0.sh
   ```

4. **Métricas**

   ```bash
   python compute-metrics.py prompts.csv summary.csv
   ```

5. **Compilar a tese**

   ```bash
   cd thesis
   pdflatex  -synctex=1 -interaction=nonstopmode -file-line-error -recorder  "PC2_20202_11811BSI202.tex"
   ```

## Licença

Todo o material é disponibilizado para **uso acadêmico e de pesquisa**.
Ver detalhes individuais de licenças em cada subpasta quando aplicável.

---

Boa pesquisa e divirta-se experimentando com logomarcas geradas por IA!

