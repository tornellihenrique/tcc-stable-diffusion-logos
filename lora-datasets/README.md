# lora-datasets

Coleção completa dos conjuntos de imagens utilizados nos treinamentos adicionais desta tese.

## Lista de datasets

| Pasta | Foco visual | Nº de imagens | Foi aplicado? |
|-------|-------------|---------------|-----------|
| `wordmark_1/`      | logos centrados em tipografia (wordmarks)    | 100 | Sim |
| `iconic_1/`        | logos baseados em símbolos ou ícones         | 100 | Sim |
| `minimalist_1/`    | primeiras referências de estilo minimalista  | 19 | Não |
| `minimalistic_2/`  | minimalista — lote complementar              | 20 | Não |
| `minimalistic_3/`  | minimalista — lote complementar              | 85 | Sim |
| `vintage_1/`       | referências de estilo vintage                | 9 | Não |
| `vintage_3/`       | vintage — lote complementar                  | 32 | Sim |
| `cartoon_1/`       | referências de estilo cartoon                | 63 | Sim |
| `ufu_1/`           | logomarcas da UFU para teste de mock-ups     | 10 | Não |

> **Observação:** Os datasets aplicados foram aqueles utilizados no treinamento dos LoRAs utilizados para gerar os resultados finais da tese.

## Estrutura de cada dataset

```
\<dataset\_name>/
├── img/              ← imagens + descrições
│   └── <repeats>\_<trigger> <tag> <tag>.../
│       ├── sample1.png
│       ├── sample1.txt
│       ├── sample2.png
│       └── sample2.txt
├── log/              ← diretórios reservados ao kohya\_ss (não usados)
└── model/            ← idem
```

* **img/** contém a pasta de imagens propriamente dita.  
* O nome da subpasta segue o padrão `<repeats>_<trigger> <tag> <tag>...`  
  * `repeats` — quantas vezes a imagem deve ser repetida na lista de treino (balanceamento).  
  * `trigger` — sempre `htd`; serve apenas como token de referência e **não** deve aparecer nos prompts dos fluxos ComfyUI.  
  * `<tag>` — palavras-chave em inglês que descrevem cor, estilo ou objeto principal.
* Para cada `sampleN.png` existe um `sampleN.txt` cuja linha contém
  * uma lista de tags separadas por vírgula **ou**  
  * uma frase descritiva curta.

Esses pares já estão prontos para serem lidos pela pipeline de treinamento adicional do **kohya_ss** ou por qualquer script customizado de LoRA.

## Como usar com kohya\_ss

1. Clone o submódulo (ver `train-dataset/README.md`).  
2. Aponte o parâmetro `--train_data_dir` para `lora-datasets/<dataset>/img`.  
3. Ajuste o campo `repeat` do nome da subpasta para balancear a frequência dessa classe no treinamento.  

Exemplo:

```bash
python train_network.py \
  --config train-dataset/sdxl_logo_wordmark_0-0003_0-0001_0-0001.json \
  --train_data_dir lora-datasets/wordmark_1/img
```

> As pastas `log/` e `model/` foram incluídas apenas para compatibilidade com o layout padrão do kohya\_ss; nesta tese não foram utilizadas.

## Licença e uso dos dados

Todo o material foi coletado exclusivamente para fins acadêmicos e deve ser utilizado seguindo os mesmos termos.
Consulte o arquivo `LICENSE.txt` (TODO) para detalhes de atribuição quando necessário.
