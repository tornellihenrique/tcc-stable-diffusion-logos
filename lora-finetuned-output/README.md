# lora-finetuned-output

Esta pasta reúne **todos** os arquivos gerados nos cinco treinamentos adicionais de LoRA.  
Para organização, os arquivos foram separados em duas subpastas:

```

lora-finetuned-output/
├── models/      ← arquivos .safetensors de cada época
└── configs/     ← arquivos .json usados pelo kohya\_ss

```

## Convenção de nomes

```
sdxl_logo_<target>_<repeat?>_<LR>_<TextLR>_<UnetLR>-<epoch>.safetensors
```

| Campo | Significado |
|-------|-------------|
| `<target>`  | categoria do LoRA (`wordmark`, `iconic`, `minimalistic`, `vintage`, `cartoon`, `ufu`) |
| `<repeat?>` | opcional (valor que indica repetição do dataset, ex.: `1`) |
| `<LR>`      | learning rate geral (ponto decimal representado por hífen: `0-001` → 0.001) |
| `<TextLR>`  | learning rate do encoder de texto |
| `<UnetLR>`  | learning rate da U-Net |
| `<epoch>`   | número da época salvo (000001 .. 000009). Ausente no arquivo final (última época). |

> Exemplo  
> `sdxl_logo_vintage_1_0-001_0-00005_0-0001-000004.safetensors`  
> → LoRA “vintage” • dataset repetido 1× • LR 0.001 • Text LR 0.00005 • UNet LR 0.0001 • checkpoint da 4ª época.

Os arquivos `*.json` em **configs/** replicam os hiperparâmetros usados no treinamento kohya_ss para rastreabilidade.

## Inventário rápido

| Categoria | Safetensors (épocas 1‒9) | Safetensor “final” | Config JSON |
|-----------|--------------------------|--------------------|-------------|
| cartoon   | 9                        | ✔                 | ✔ |
| iconic    | 4 (\*épocas pares)       | ✔                 | ✔ |
| minimalistic | 9                      | ✔                 | ✔ |
| vintage   | 9                        | ✔                 | ✔ |
| ufu       | 9                        | ✔                 | ✔ |
| wordmark  | 4 (\*épocas pares)       | ✔                 | ✔ |

\*Para `iconic` e `wordmark` apenas checkpoints a cada 2 épocas foram salvos.

## Como carregar no ComfyUI

1. Copie o(s) arquivo(s) `.safetensors` desejado(s) para a pasta `models/loras` ou mantenha o caminho completo.
2. No fluxo, adicione o nó **LoRA Loader** e selecione o arquivo.
3. Ajuste o peso (`strength / alpha`) conforme a tabela sugerida no capítulo de metodologia.

## Checksums

Verifique integridade:

```bash
sha256sum -c checksums.txt
```
