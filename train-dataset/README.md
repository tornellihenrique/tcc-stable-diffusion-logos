# train-dataset

Configurações JSON usadas no **kohya_ss** (incluído como submódulo).

Clone com:

```bash
git submodule update --init
```

| Config JSON                                           | LR (η) | Text LR | UNet LR |
| ----------------------------------------------------- | ------ | ------- | ------- |
| `sdxl_logo_wordmark_0-0003_0-0001_0-0001.json`      | 0.0003 | 0.0001  | 0.0001  |
| `sdxl_logo_iconic_0-0003_0-0001_0-0001.json`        | 0.0003 | 0.0001  | 0.0001  |
| `sdxl_logo_minimalistic_0-001_0-00005_0-0001.json`  | 0.001  | 0.00005 | 0.0001  |
| `sdxl_logo_vintage_1_0-001_0-00005_0-0001.json`     | 0.001  | 0.00005 | 0.0001  |
| `sdxl_logo_cartoon_1_0-001_0-00005_0-0001.json`     | 0.001  | 0.00005 | 0.0001  |

Treinamento exemplo:

```bash
python train-dataset/kohya_ss/train_network.py \
  --config train-dataset/sdxl_logo_wordmark_0-0003_0-0001_0-0001.json
```