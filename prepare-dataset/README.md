# prepare-dataset

Coleção de utilitários que automatizam a curadoria, expansão e formatação de
conjuntos de imagens para treinamento adicional (LoRA) no **SDXL**.  
Todos os scripts convertem o material de entrada para o layout aceito pelo
**kohya_ss** :

```
\<output\_dir>/
├── img/                # imagens e descrições TXT
│   └── <repeats>\_<trigger> <tag> <tag>.../
│       ├── sample001.png
│       └── sample001.txt
├── log/                # reservado (opcional)
└── model/              # reservado (opcional)
```

> **Legenda de campos de nome de pasta**  
> • `repeats`   — número de repetições no arquivo de lista de treino  
> • `trigger`   — token exclusivo do conceito (ex.: `htd`)  
> • `tag`      — palavras-chave que descrevem cor, estilo ou objeto

## Scripts e parâmetros

### 1. `prepare-dataset-1.py`

*Gera descrições com MiniGPT-4 a partir de um diretório já existente.*

| Parâmetro CLI | Descrição | Padrão |
|---------------|-----------|--------|
| `--cfg-path`  | Caminho para o YAML de configuração do MiniGPT-4 | obrigatório |
| `--gpu-id`    | Índice da GPU a utilizar | `0` |

Variáveis internas (editar no cabeçalho do script):

* `dataset_root` — pasta com subpastas-categoria contendo `.png`  
* `output_dir`   — pasta destino no padrão kohya  
* `instance_prompt` — *trigger* (ex.: `"htd"`)  
* `class_prompt`    — lista de tags base  
* `repeats`         — valor de repetição  
* `images_per_category` — nº máximo de imagens por categoria  
* `image_size`     — redimensionamento square (px)

### 2. `prepare-dataset-2.py`

Versão refatorada do script 1 (mesma lógica e mesmos parâmetros), porém com
tratamento de erros e logs mais verbosos.

### 3. `prepare-dataset-3.py`

*Coleta imagens da web via iCrawler e legendas com MiniGPT-4.*

Não usa `dataset_root`. Espera uma lista `base_queries` dentro do script
(ex.: `["minimalist logo", "flat icon logo"]`).  
O crawler baixa imagens indefinidamente, gera descrições e grava direto em
`output_dir`.

Parâmetros CLI igual ao script 1 (`--cfg-path`, `--gpu-id`).

### 4. `prepare-dataset-4.py`

*Converte arquivos Parquet em layout kohya_ss.*

| Parâmetro CLI | Descrição |
|---------------|-----------|
| `--input_dir`        | diretório com `.parquet` |
| `--output_dir`       | destino no padrão kohya |
| `--instance_prompt`  | trigger |
| `--class_prompt`     | tags base |
| `--repeats`          | número de repetições |
| `--image_ext`        | extensão de saída (`png` ou `jpg`) |
| `--resize`           | resolução quadrada (px) |
| `--subset_percentage`| porcentagem aleatória do dataset a usar (0‒100) |

### 5. `prepare-images-simple.py`

Variante enxuta do script 4.  
Em vez de Parquet, lê imagens planas em `input_dir` e gera o mesmo layout.

## Utilização

Esta seção descreve os passos mínimos para executar qualquer um dos scripts da
pasta **prepare-dataset**.

### 1. Criar ambiente Python

```bash
cd prepare-dataset

# venv (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Baixar pesos do MiniGPT-4 (scripts 1 – 3)

Os scripts `prepare-dataset-{1,2,3}.py` usam o **MiniGPT-4** para gerar
descrições automáticas.

1. Clone o repositório oficial e baixe os checkpoints:
   [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
2. Copie o arquivo de configuração de inferência (ex.: `eval_configs/minigpt4_eval.yaml`) para um local acessível.
3. Ajuste o caminho em `--cfg-path`.

### 3. Estrutura de entrada

* Scripts 1 e 2 esperam um diretório raiz com subpastas-categoria contendo `.png`.
* Script 3 não requer dados locais; ele baixa imagens via **icrawler** a partir
  da lista `base_queries`.
* Scripts 4 e `prepare-images-simple.py` usam, respectivamente, arquivos
  Parquet ou imagens soltas em `input_dir`.

### 4. Exemplos rápidos

| Objetivo                                          | Comando                                                                                                                                                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gerar descrições com MiniGPT-4 (script 1)         | `python prepare-dataset-1.py --cfg-path ../minigpt4_eval.yaml --gpu-id 0`                                                                                                                                |
| Versão refatorada (script 2)                      | `python prepare-dataset-2.py --cfg-path ../minigpt4_eval.yaml --gpu-id 0`                                                                                                                                |
| Coletar imagens da web (script 3)                 | `python prepare-dataset-3.py --cfg-path ../minigpt4_eval.yaml --gpu-id 0`                                                                                                                                |
| Converter Parquet para layout kohya (script 4)    | `python prepare-dataset-4.py --input_dir parquet_raw --output_dir minimalistic_4 --instance_prompt htd --class_prompt "minimalist logo" --repeats 5 --image_ext png --resize 768 --subset_percentage 20` |
| Converter imagens simples (prepare-images-simple) | `python prepare-images-simple.py --input_dir raw_png --output_dir cartoon_2 --instance_prompt htd --class_prompt "cartoon logo" --repeats 5 --image_ext png --resize 768 --subset_percentage 100`        |

### 5. Saída

Cada script gera um diretório no padrão **kohya\_ss**:

```
<output_dir>/
└── img/
    └── <repeats>_<trigger> <tag> <tag>.../
        ├── example1.png
        └── example1.txt
```

As pastas `log/` e `model/` são criadas (quando necessário) apenas para
compatibilidade com fluxos de treinamento, mas podem ser ignoradas se não
forem usadas.

---

Após esses passos você terá datasets limpos, anotados e no formato
exigido pelos treinos LoRA descritos na dissertação.
