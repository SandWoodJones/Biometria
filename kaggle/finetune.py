"""Run the fine-tuning experiment (all seeds) on a Kaggle GPU notebook.

In a Kaggle notebook with a GPU (T4) and internet enabled, run one cell:

    !git clone <repository URL> repo && cd repo && git checkout <commit>
    !cd repo && python kaggle/finetune.py

It downloads LibriSpeech train-clean-100 (MD5-checked) and the pretrained weights, trains every seed in
config.yaml, evaluates each best checkpoint on the LibriSpeech validation pairs and packs checkpoints
and results into /kaggle/working/finetune_outputs.tar.gz. Seeds that already finished are skipped.
"""

import hashlib
import importlib.util
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_MD5 = "2a93770f6d5c6c964bc36631d331a522"
DATA_DIR = Path("/kaggle/temp/data")
OUTPUT = Path("/kaggle/working/finetune_outputs.tar.gz")
REQUIRED = {"soundfile": "soundfile", "yaml": "pyyaml", "sklearn": "scikit-learn", "torchaudio": "torchaudio"}


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_librispeech():
    """Stream the archive through MD5 and tar so it is never stored twice on disk."""
    if (DATA_DIR / "LibriSpeech" / "train-clean-100").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()

    class Hashing:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size=-1):
            chunk = self.stream.read(size)
            md5.update(chunk)
            return chunk

    with urllib.request.urlopen(LIBRISPEECH_URL) as response:
        stream = Hashing(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            archive.extractall(DATA_DIR, filter="data")
        while stream.read(1 << 20):
            pass
    if md5.hexdigest() != LIBRISPEECH_MD5:
        raise RuntimeError(f"train-clean-100.tar.gz MD5 {md5.hexdigest()} != {LIBRISPEECH_MD5}")


def main():
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        run(sys.executable, "-m", "pip", "install", *missing)
    import torch

    if not torch.cuda.is_available():
        sys.exit("No GPU: enable a GPU accelerator in the notebook settings.")
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, {os.cpu_count()} CPUs", flush=True)

    download_librispeech()
    if not Path("data").exists():
        Path("data").symlink_to(DATA_DIR)
    run(sys.executable, "model.py")

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    workers = os.cpu_count()
    cfg["training"]["workers"] = workers
    Path("config_kaggle.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    for seed in cfg["training"]["seeds"]:
        run_name = f"finetuned_seed{seed}"
        weights = Path(cfg["paths"]["checkpoints"]) / run_name / "best.pt"
        if not weights.with_suffix(".json").exists():
            run(sys.executable, "train.py", "--config", "config_kaggle.yaml", "--seed", str(seed))
        run(
            sys.executable, "extract_embeddings.py", "--config", "config_kaggle.yaml", "--run", run_name,
            "--dataset", "librispeech", "--split", "validation", "--weights", str(weights), "--workers", str(workers),
        )
        run(sys.executable, "evaluate.py", "--config", "config_kaggle.yaml", "--run", run_name, "--dataset", "librispeech", "--split", "validation")

    with tarfile.open(OUTPUT, "w:gz") as archive:
        for seed in cfg["training"]["seeds"]:
            for root in (cfg["paths"]["checkpoints"], cfg["paths"]["results"]):
                archive.add(Path(root) / f"finetuned_seed{seed}")
    print(f"outputs -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
