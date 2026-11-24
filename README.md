# Verificação de Locutor com ResNet-34 e Fine-Tuning

Projeto final da disciplina de Biometria da UTFPR, Câmpus Ponta Grossa (2026/2), por Raul Rodrigues e Paulo Henrique.

O sistema realiza **verificação de locutor** 1:1: dadas duas gravações de voz, decide se elas pertencem à mesma pessoa. Uma CNN mapeia cada gravação para um embedding (representação vetorial), e dois embeddings são comparados por similaridade de cosseno.

| | |
|---|---|
| Modalidade biométrica | Voz |
| Base principal | LibriSpeech `train-clean-100` (251 locutores) |
| Segunda base | VCTK 0.92, somente `mic1` (110 locutores) |
| CNN | ResNet-34 (arquitetura thin ResNet do WeSpeaker, 32 canais base) |
| Pré-treinamento | VoxCeleb2 dev (5.994 locutores), com ArcFace |
| Embedding | 256 dimensões, normalizado por L2 |
| Comparação | Similaridade de cosseno |
| Seeds | 42 para os protocolos; 42, 123 e 2026 para o treinamento |

## Resultados

Os resultados agregados estão em [`results/summary/`](results/summary/):

- `metrics.csv`: EER, AUC, FMR, FNMR, TAR@FAR=1%, FTA e limiar de cada condição
- `efficiency.csv`: parâmetros, tamanho do modelo e tempo de inferência
- `datasets.csv`: locutores, arquivos, taxa de amostragem e durações de cada base
- `tables.typ`: tabelas formatadas para o artigo
- `robustness.png`: EER e FMR sob ruído aditivo

Cada avaliação tem seu próprio diretório, `results/<run>/<dataset>_<split>/`, com `scores.csv`, `metrics.json`, `roc.png` e `distributions.png`. Nomes das execuções:

| Execução | Significado |
|---|---|
| `pretrained` | Pesos da VoxCeleb, sem fine-tuning (referência) |
| `finetuned_seed<S>` | Ajustado na LibriSpeech com a seed `S` |
| `<run>_snr<L>` | Mesmo modelo com ruído branco gaussiano a `L` dB de SNR |
| `<run>_novad` | Mesmo modelo sem detecção de atividade de voz (ablação) |

A análise de erros está em `results/error_analysis/`.

## Método

### Pré-processamento e formato de entrada

1. Carrega o áudio em mono e o reamostra para 16 kHz.
2. Aplica um VAD (detecção de atividade de voz) baseado em energia, com quadros de 25 ms e deslocamento de 10 ms. Quadros mais de 40 dB abaixo do quadro mais forte são silêncio, e as regiões de fala são estendidas em 100 ms para cada lado.
3. Um arquivo com menos de 0,5 s de fala após o VAD conta como **falha de aquisição** (_Failure to Acquire_, FTA).
4. Calcula 80 bandas log-Mel no padrão Kaldi (janela de Hamming de 25 ms, deslocamento de 10 ms, sem dither, forma de onda multiplicada por 2^15).
5. Subtrai a média de cada banda ao longo do áudio.

A entrada do modelo é uma matriz `(quadros, 80)`. Todos os parâmetros estão em [`config.yaml`](config.yaml).

### Modelo e pesos

A ResNet-34 tem 6.634.336 parâmetros (25,4 MB) e usa agregação estatística temporal (média e desvio padrão), seguida de uma camada linear que produz o embedding.

Os pesos iniciais são os da [ResNet-34 do WeSpeaker treinada na VoxCeleb](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34) (`avg_model.pt`). O comando `python model.py` os baixa e confere o SHA-256. O repositório no Hugging Face é marcado com a licença Apache-2.0, e o WeSpeaker informa que seus modelos treinados na VoxCeleb seguem a licença CC BY 4.0 da base.

### Treinamento

O fine-tuning usa os locutores de treino da LibriSpeech (175 locutores, 19.720 amostras):

- **Perda:** AAM-Softmax (ArcFace), margem 0,2 e escala 32. A camada de classificação é descartada após o treinamento.
- **Cronograma:** 15 épocas. A rede fica congelada na primeira época, enquanto a nova camada de classificação se ajusta.
- **Otimizador:** SGD com momento de Nesterov de 0,9, decaimento de pesos de 1e-4, redução cossenoidal e taxa de aprendizado de 1e-3 (rede) e 1e-2 (camada de classificação).
- **Lotes:** 128 recortes aleatórios de 2 s, extraídos após o VAD.
- **Seleção do modelo:** menor EER de validação, com desempate pela maior AUC de validação.

### Protocolos

Nenhuma das bases tem um protocolo oficial de verificação, por isso [`make_protocols.py`](make_protocols.py) gera um com a seed 42. Os arquivos estão versionados em [`protocols/`](protocols/).

- Os locutores são divididos em 70/15/15 entre treino, validação e teste, com estratificação por gênero. Nenhum locutor aparece em mais de um conjunto.
- Cada conjunto avaliado tem 100 pares genuínos por locutor e 10 pares impostores para cada par genuíno.

| Base | Locutores de treino / validação / teste | Pares genuínos / impostores por conjunto avaliado |
|---|---|---|
| LibriSpeech | 175 / 38 / 38 | 3.800 / 38.000 |
| VCTK | 78 / 16 / 16 | 1.600 / 16.000 |

### Comparação e limiar

O escore é a similaridade de cosseno entre os embeddings normalizados por L2. O limiar de decisão é o **limiar de EER da validação da LibriSpeech**. Ele é congelado antes da avaliação de qualquer conjunto de teste, e o teste nunca escolhe limiar, modelo ou hiperparâmetro.

- **Avaliação entre bases (VCTK) e ruído:** o modelo e o limiar da validação limpa da LibriSpeech são usados sem alteração, sem fine-tuning na base de destino.
- **Ablação sem VAD:** como o pipeline muda, o limiar vem de uma execução sem VAD no mesmo conjunto de validação.
- **Ruído:** ruído branco gaussiano a 20, 10 e 0 dB de SNR, adicionado à forma de onda antes do VAD, com seed definida por arquivo.

## Ambiente

- Python 3.12, PyTorch 2.11.0 e torchaudio 2.11.0. As versões exatas de todas as bibliotecas estão em [`requirements.txt`](requirements.txt) e `uv.lock`.
- **O fine-tuning e as avaliações** foram executados no Kaggle: GPU Tesla T4, Intel Xeon @ 2,00 GHz (4 vCPUs), Python 3.13, PyTorch 2.11.0+cu130 (CUDA 13.0).
- **A validação do modelo `pretrained` e os scripts de análise** foram executados localmente: AMD Ryzen 9 3900X, 16 GB de RAM, AMD Radeon RX 6700 XT, PyTorch 2.11.0+rocm7.2.

O hardware usado em cada avaliação está registrado no respectivo `metrics.json`.

## Instalação

Com Nix (o flake fornece Python, uv e as bibliotecas do sistema):

```bash
nix develop
uv sync --frozen
```

Sem Nix, em uma GPU AMD com ROCm:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A RX 6700 XT também precisa de `export HSA_OVERRIDE_GFX_VERSION=10.3.0`, que o flake define automaticamente.

Em uma GPU NVIDIA:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
grep -vE '^(--extra-index-url|torch|torchaudio|triton-rocm)' requirements.txt > requirements-cuda.txt
pip install -r requirements-cuda.txt
```

## Dados

As bases não são redistribuídas neste repositório. Ambas têm licença CC BY 4.0 e estão disponíveis nas fontes oficiais:

- [LibriSpeech](https://www.openslr.org/12) (`train-clean-100.tar.gz`, MD5 `2a93770f6d5c6c964bc36631d331a522`)
- [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443)

```bash
mkdir -p data/downloads

curl -L -o data/downloads/train-clean-100.tar.gz https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf data/downloads/train-clean-100.tar.gz -C data

curl -L -o data/downloads/DS_10283_3443.zip https://datashare.ed.ac.uk/download/DS_10283_3443.zip
python -m zipfile -e data/downloads/DS_10283_3443.zip data/downloads/vctk
python -m zipfile -e data/downloads/vctk/VCTK-Corpus-0.92.zip data/VCTK-Corpus-0.92
```

Estrutura esperada:

```
data/
├── LibriSpeech/
│   ├── SPEAKERS.TXT
│   └── train-clean-100/<speaker>/<chapter>/*.flac
└── VCTK-Corpus-0.92/
    ├── speaker-info.txt
    └── wav48_silence_trimmed/<speaker>/*_mic1.flac
```

## Reprodução dos resultados

Os protocolos versionados são os usados no artigo. Para regenerá-los de forma determinística:

```bash
python make_protocols.py
```

Baixe os pesos pré-treinados:

```bash
python model.py
```

### 1. Fine-tuning

```bash
for seed in 42 123 2026; do
  python train.py --seed "$seed"
done
```

Isso gera `checkpoints/finetuned_seed<S>/best.pt`. Os pesos usados no artigo estão anexados à Release `final-v1.0` do GitHub. Para usá-los sem treinar de novo, extraia `finetune_outputs_v2.tar.gz` na raiz do repositório.

### 2. Todas as avaliações

```bash
for run in pretrained finetuned_seed42 finetuned_seed123 finetuned_seed2026; do
  weights="checkpoints/$run/best.pt"
  [ "$run" = pretrained ] && weights=checkpoints/wespeaker-voxceleb-resnet34/avg_model.pt
  frozen="results/$run/librispeech_validation/metrics.json"

  # Validação: define o limiar
  python extract_embeddings.py --run "$run" --dataset librispeech --split validation --weights "$weights"
  python evaluate.py --run "$run" --dataset librispeech --split validation

  # Teste na LibriSpeech e avaliação entre bases na VCTK
  for dataset in librispeech vctk; do
    python extract_embeddings.py --run "$run" --dataset "$dataset" --split test --weights "$weights"
    python evaluate.py --run "$run" --dataset "$dataset" --split test --threshold-from "$frozen"
  done

  # Robustez a ruído aditivo
  python robustness.py --run "$run" --weights "$weights" --dataset librispeech --split test --threshold-from "$frozen"

  # Ablação do VAD, com limiar de validação próprio
  python extract_embeddings.py --run "${run}_novad" --dataset librispeech --split validation --weights "$weights" --no-vad
  python evaluate.py --run "${run}_novad" --dataset librispeech --split validation
  python extract_embeddings.py --run "${run}_novad" --dataset librispeech --split test --weights "$weights" --no-vad
  python evaluate.py --run "${run}_novad" --dataset librispeech --split test \
    --threshold-from "results/${run}_novad/librispeech_validation/metrics.json"
done
```

### 3. Tabelas, figuras e análise de erros

```bash
python dataset_stats.py
python summarize.py
python error_analysis.py --run finetuned_seed42 --dataset librispeech --split test
python error_analysis.py --run finetuned_seed42 --dataset librispeech --split validation
python preprocess.py data/LibriSpeech/train-clean-100/7190/90543/7190-90543-0043.flac --out results/pipeline.png
```

### Execução no Kaggle

As execuções publicadas de fine-tuning e avaliação usaram estes scripts em um notebook com GPU T4 e internet habilitada:

```bash
git clone https://github.com/SandWoodJones/Biometria.git repo
cd repo
python kaggle/finetune.py       # fine-tuning, commit 18aec61
python kaggle/evaluate_all.py   # avaliações, commit 9ca912d
python kaggle/evaluate_dupla7.py  # pares da Dupla 7 (experimento B), commit 3d4e1ca
```

O `evaluate_all.py` precisa da saída do notebook de fine-tuning anexada como entrada.

O `evaluate_dupla7.py` também precisa de um dataset com os áudios de `protocols/vctk_dupla7/test.csv` (arquivo `.tar` conferido contra `sha256.csv`), porque o servidor do VCTK bloqueou o download a partir do Kaggle. Os pares foram importados com `python import_external_pairs.py`.

## Estrutura do repositório

```
config.yaml             parâmetros do pipeline, do treinamento e dos caminhos
make_protocols.py       divisão de locutores e pares de verificação
dataset_stats.py        estatísticas das bases, lidas dos cabeçalhos dos áudios
preprocess.py           leitura do áudio, ruído, VAD e atributos log-Mel
model.py                ResNet-34 e pesos pré-treinados
train.py                fine-tuning com AAM-Softmax
extract_embeddings.py   embeddings, tempo de inferência e dados de hardware
evaluate.py             escores, métricas, ROC e distribuições de escores
robustness.py           avaliação sob ruído aditivo
summarize.py            tabelas agregadas e figura de robustez
error_analysis.py       piores erros, com figuras e medidas acústicas
kaggle/                 execução do fine-tuning e das avaliações no Kaggle
protocols/              divisões e pares versionados
results/                escores, métricas e figuras
artigo/                 artigo em Typst
```
