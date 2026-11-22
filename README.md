# Speaker Verification with a Fine-Tuned ResNet-34

Final project for the Biometrics course at UTFPR Ponta Grossa (2026/2), by Paulo Henrique and Raul Rodrigues.

The system performs 1:1 **speaker verification**: given two voice recordings, it decides whether they come from the same person. A CNN maps each recording to an embedding, and two embeddings are compared with cosine similarity.

| | |
|---|---|
| Biometric modality | Voice |
| Primary dataset | LibriSpeech `train-clean-100` (251 speakers) |
| Second dataset | VCTK 0.92, `mic1` only (110 speakers) |
| CNN | ResNet-34 (WeSpeaker thin ResNet layout, 32 base channels) |
| Pretraining | VoxCeleb2 dev (5,994 speakers), with ArcFace |
| Embedding | 256 dimensions, L2-normalized |
| Matching | Cosine similarity |
| Seeds | 42 for protocols; 42, 123 and 2026 for training |

## Results

Aggregated results are in [`results/summary/`](results/summary/):

- `metrics.csv`: EER, AUC, FMR, FNMR, TAR@FAR=1%, FTA and threshold for every condition
- `efficiency.csv`: parameters, model size and inference time
- `tables.typ`: the same tables, formatted for the article
- `robustness.png`: EER and FMR under additive noise

Each evaluation has its own directory, `results/<run>/<dataset>_<split>/`, containing `scores.csv`, `metrics.json`, `roc.png` and `distributions.png`. Run names:

| Run | Meaning |
|---|---|
| `pretrained` | VoxCeleb weights, no fine-tuning (baseline) |
| `finetuned_seed<S>` | Fine-tuned on LibriSpeech with seed `S` |
| `<run>_snr<L>` | Same model with white Gaussian noise at `L` dB SNR |
| `<run>_novad` | Same model without voice activity detection (ablation) |

The error analysis is in `results/error_analysis/`.

## Method

### Preprocessing and input format

1. Load the audio as mono and resample it to 16 kHz.
2. Apply an energy-based VAD: 25 ms frames with a 10 ms hop. Frames more than 40 dB below the loudest frame are silence, and speech regions are widened by 100 ms on each side.
3. A file with less than 0.5 s of speech after VAD counts as a **Failure to Acquire** (FTA).
4. Compute 80-band Kaldi log-Mel filterbanks (25 ms window, 10 ms shift, Hamming window, no dither, waveform scaled by 2^15).
5. Subtract the per-utterance mean.

The model input is a `(frames, 80)` matrix. All parameters are in [`config.yaml`](config.yaml).

### Model and weights

The ResNet-34 has 6,634,336 parameters (25.4 MB) and uses temporal statistics pooling (mean and standard deviation) followed by a linear layer that produces the embedding.

Initial weights are the [WeSpeaker VoxCeleb ResNet-34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34) (`avg_model.pt`). `python model.py` downloads them and verifies their SHA-256. The Hugging Face repository is tagged Apache-2.0, and WeSpeaker states that its VoxCeleb models follow the dataset's CC BY 4.0 license.

### Training

Fine-tuning uses the LibriSpeech training speakers (175 speakers, 19,720 utterances):

- **Loss:** AAM-Softmax (ArcFace), margin 0.2, scale 32. The classification head is discarded after training.
- **Schedule:** 15 epochs. The backbone is frozen for the first epoch while the new head settles.
- **Optimizer:** SGD with Nesterov momentum 0.9, weight decay 1e-4, cosine annealing, learning rate 1e-3 (backbone) and 1e-2 (head).
- **Batches:** 128 random 2 s crops taken after VAD.
- **Model selection:** lowest validation EER, with ties broken by highest validation AUC.

### Protocols

Neither dataset has an official verification protocol, so [`make_protocols.py`](make_protocols.py) generates one with seed 42. The files are versioned in [`protocols/`](protocols/).

- Speakers are split 70/15/15 into train, validation and test, stratified by gender. No speaker appears in more than one split.
- Each evaluated split has 100 genuine pairs per speaker and 10 impostor pairs per genuine pair.

| Dataset | Train / validation / test speakers | Genuine / impostor pairs per evaluated split |
|---|---|---|
| LibriSpeech | 175 / 38 / 38 | 3,800 / 38,000 |
| VCTK | 78 / 16 / 16 | 1,600 / 16,000 |

### Matching and threshold

The score is the cosine similarity of the L2-normalized embeddings. The decision threshold is the **EER threshold of the LibriSpeech validation split**. It is frozen before evaluating any test set, and the test set never selects a threshold, model or hyperparameter.

- **Cross-dataset (VCTK) and noise:** the model and the clean LibriSpeech validation threshold are used unchanged, with no fine-tuning on the target.
- **No-VAD ablation:** the pipeline changes, so its threshold comes from a no-VAD run on the same validation split.
- **Noise:** white Gaussian noise at 20, 10 and 0 dB SNR, added to the waveform before VAD and seeded per file.

## Environment

- Python 3.12, PyTorch 2.11.0, torchaudio 2.11.0. Exact versions of all libraries are in [`requirements.txt`](requirements.txt) and `uv.lock`.
- **Fine-tuning and evaluation** ran on Kaggle: Tesla T4 GPU, Intel Xeon @ 2.00 GHz (4 vCPUs), PyTorch 2.11.0+cu130 (CUDA 13.0).
- **The `pretrained` validation run and the analysis scripts** ran locally: AMD Ryzen 9 3900X, 16 GB RAM, AMD Radeon RX 6700 XT, PyTorch 2.11.0+rocm7.2.

The hardware used for each evaluation is recorded in its `metrics.json`.

## Setup

With Nix (the flake provides Python, uv and system libraries):

```bash
nix develop
uv sync --frozen
```

Without Nix, on an AMD GPU with ROCm:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The RX 6700 XT also needs `export HSA_OVERRIDE_GFX_VERSION=10.3.0`, which the flake sets automatically.

On an NVIDIA GPU:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
grep -vE '^(--extra-index-url|torch|torchaudio|triton-rocm)' requirements.txt > requirements-cuda.txt
pip install -r requirements-cuda.txt
```

## Data

The datasets are not redistributed here. Both are licensed under CC BY 4.0 and are available from their official sources:

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

Expected layout:

```
data/
├── LibriSpeech/
│   ├── SPEAKERS.TXT
│   └── train-clean-100/<speaker>/<chapter>/*.flac
└── VCTK-Corpus-0.92/
    ├── speaker-info.txt
    └── wav48_silence_trimmed/<speaker>/*_mic1.flac
```

## Reproducing the results

The versioned protocols are the ones used in the article. To regenerate them deterministically:

```bash
python make_protocols.py
```

Download the pretrained weights:

```bash
python model.py
```

### 1. Fine-tuning

```bash
for seed in 42 123 2026; do
  python train.py --seed "$seed"
done
```

This writes `checkpoints/finetuned_seed<S>/best.pt`. The weights used in the article are attached to the `final-v1.0` GitHub Release. Extract `finetune_outputs_v2.tar.gz` at the repository root to use them without retraining.

### 2. All evaluations

```bash
for run in pretrained finetuned_seed42 finetuned_seed123 finetuned_seed2026; do
  weights="checkpoints/$run/best.pt"
  [ "$run" = pretrained ] && weights=checkpoints/wespeaker-voxceleb-resnet34/avg_model.pt
  frozen="results/$run/librispeech_validation/metrics.json"

  # Validation: sets the threshold
  python extract_embeddings.py --run "$run" --dataset librispeech --split validation --weights "$weights"
  python evaluate.py --run "$run" --dataset librispeech --split validation

  # LibriSpeech test and VCTK cross-dataset test
  for dataset in librispeech vctk; do
    python extract_embeddings.py --run "$run" --dataset "$dataset" --split test --weights "$weights"
    python evaluate.py --run "$run" --dataset "$dataset" --split test --threshold-from "$frozen"
  done

  # Robustness to additive noise
  python robustness.py --run "$run" --weights "$weights" --dataset librispeech --split test --threshold-from "$frozen"

  # VAD ablation, with its own validation threshold
  python extract_embeddings.py --run "${run}_novad" --dataset librispeech --split validation --weights "$weights" --no-vad
  python evaluate.py --run "${run}_novad" --dataset librispeech --split validation
  python extract_embeddings.py --run "${run}_novad" --dataset librispeech --split test --weights "$weights" --no-vad
  python evaluate.py --run "${run}_novad" --dataset librispeech --split test \
    --threshold-from "results/${run}_novad/librispeech_validation/metrics.json"
done
```

### 3. Tables, figures and error analysis

```bash
python summarize.py
python error_analysis.py --run finetuned_seed42 --dataset librispeech --split test
python error_analysis.py --run finetuned_seed42 --dataset librispeech --split validation
python preprocess.py data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac --out results/pipeline.png
```

### Running on Kaggle

The published fine-tuning and evaluation runs used these scripts in a notebook with a T4 GPU and internet enabled:

```bash
git clone https://github.com/SandWoodJones/Biometria.git repo
cd repo
python kaggle/finetune.py       # fine-tuning, commit 18aec61
python kaggle/evaluate_all.py   # evaluations, commit 9ca912d
```

`evaluate_all.py` needs the output of the fine-tuning notebook attached as input.

## Repository structure

```
config.yaml             all pipeline, training and path parameters
make_protocols.py       speaker splits and verification pairs
preprocess.py           audio loading, noise, VAD and log-Mel features
model.py                ResNet-34 and pretrained weights
train.py                AAM-Softmax fine-tuning
extract_embeddings.py   embeddings, inference time and hardware info
evaluate.py             scores, metrics, ROC and score distributions
robustness.py           evaluation under additive noise
summarize.py            aggregated tables and robustness figure
error_analysis.py       worst errors, with figures and acoustic measurements
kaggle/                 Kaggle runners for fine-tuning and evaluation
protocols/              versioned splits and pairs
results/                scores, metrics and figures
```
