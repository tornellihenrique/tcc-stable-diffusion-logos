# train-dataset

Configurações JSON usadas no **kohya_ss** (incluído como submódulo).

Clone com:

```bash
git submodule update --init
```

| Config JSON                                           | LR (η) | Text LR | UNet LR |
| ----------------------------------------------------- | ------ | ------- | ------- |
| sdxl\_logo\_wordmark\_0-0003\_0-0001\_0-0001.json     | 0.0003 | 0.0001  | 0.0001  |
| sdxl\_logo\_iconic\_0-0003\_0-0001\_0-0001.json       | 0.0003 | 0.0001  | 0.0001  |
| sdxl\_logo\_minimalistic\_0-001\_0-00005\_0-0001.json | 0.001  | 0.00005 | 0.0001  |
| …                                                     | …      | …       | …       |

Treinamento exemplo:

```bash
python train-dataset/kohya_ss/train_network.py \
  --config train-dataset/sdxl_logo_wordmark_0-0003_0-0001_0-0001.json
```