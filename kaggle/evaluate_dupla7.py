"""Evaluate every model on the VCTK test pairs of the other voice pair (Dupla 7), for experiment B.

In a Kaggle notebook with a GPU (T4), internet enabled and the output of the fine-tuning notebook
attached as input, run one cell:

    !git clone <repository URL> repo && cd repo && python kaggle/evaluate_dupla7.py

The pairs come from protocols/vctk_dupla7/ (see import_external_pairs.py) and every model keeps the
threshold frozen on the LibriSpeech validation split. Results are packed into
/kaggle/working/dupla7_outputs.tar.gz. Only VCTK is needed: attach a dataset with the tar of the files
in protocols/vctk_dupla7/test.csv (checked against sha256.csv there), otherwise it is downloaded.
"""

import importlib.util
import os
import sys
import tarfile
from pathlib import Path

import yaml

from evaluate_all import download_vctk, evaluate, install_checkpoints
from finetune import DATA_DIR, REQUIRED, run

PROTOCOL = "vctk_dupla7"
OUTPUT = Path("/kaggle/working/dupla7_outputs.tar.gz")


def main():
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        run(sys.executable, "-m", "pip", "install", *missing)
    import torch

    if not torch.cuda.is_available():
        sys.exit("No GPU: enable a GPU accelerator in the notebook settings.")
    workers = os.cpu_count()
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, Python {sys.version.split()[0]}, {workers} CPUs", flush=True)

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    checkpoints = Path(cfg["paths"]["checkpoints"])
    install_checkpoints(checkpoints)
    print("fine-tuned checkpoints verified", flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not Path("data").exists():
        Path("data").symlink_to(DATA_DIR)
    download_vctk(cfg, (PROTOCOL,))
    run(sys.executable, "model.py")

    models = {"pretrained": Path(cfg["model"]["pretrained_weights"])}
    models.update({f"finetuned_seed{s}": checkpoints / f"finetuned_seed{s}" / "best.pt" for s in cfg["training"]["seeds"]})
    for name, weights in models.items():
        evaluate(name, weights, PROTOCOL, "test", Path("results") / name / "librispeech_validation" / "metrics.json", workers)

    with tarfile.open(OUTPUT, "w:gz") as archive:
        for name in models:
            archive.add(Path("results") / name / f"{PROTOCOL}_test")
    print(f"outputs -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
