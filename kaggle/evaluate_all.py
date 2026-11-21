"""Run every evaluation that needs mass embedding extraction on a Kaggle GPU notebook.

In a Kaggle notebook with a GPU (T4), internet enabled and the output of the fine-tuning notebook
attached as input, run one cell:

    !git clone <repository URL> repo && cd repo && git checkout <commit>
    !cd repo && python kaggle/evaluate_all.py

For the pretrained model and each fine-tuned seed it evaluates, always with thresholds frozen on validation:
LibriSpeech test, VCTK test (cross-dataset), LibriSpeech test under noise, and the no-VAD ablation.
Results are packed into /kaggle/working/evaluation_outputs.tar.gz. Finished evaluations are skipped.
"""

import hashlib
import http.client
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from finetune import DATA_DIR, REQUIRED, download_librispeech, run

VCTK_URL = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
VCTK_SIZE = 11749118645
VCTK_INNER = "VCTK-Corpus-0.92.zip"
MIN_FREE_GB = 35
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/evaluation_outputs.tar.gz")
CHECKPOINTS = {
    42: "65952608462365da6d31d187d4ab3d09549119b9b2e2a16d4e625c854358d3d8",
    123: "3a361cb08a20fe5a36846ddbc4b3eb43fddce39e2e3bd464a7bbf944aa89ca43",
    2026: "117ad3149363e527a37732b237c96601fdfa8364ab7294d7c7f36760db2fd853",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_checkpoints(root):
    """Copy the fine-tuned weights from the attached input, unpacked or as the original archive."""
    archives = sorted(INPUT.rglob("finetune_outputs*.tar.gz"))
    for seed, digest in CHECKPOINTS.items():
        target = root / f"finetuned_seed{seed}"
        if (target / "best.pt").exists() and sha256(target / "best.pt") == digest:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for found in INPUT.rglob(f"finetuned_seed{seed}/best.pt"):
            if sha256(found) == digest:
                shutil.copy(found, target / "best.pt")
                shutil.copy(found.with_suffix(".json"), target / "best.json")
                break
        else:
            for path in archives:
                with tarfile.open(path) as archive:
                    for suffix in ("pt", "json"):
                        member = archive.extractfile(f"checkpoints/finetuned_seed{seed}/best.{suffix}")
                        (target / f"best.{suffix}").write_bytes(member.read())
                if sha256(target / "best.pt") == digest:
                    break
        if not (target / "best.pt").exists() or sha256(target / "best.pt") != digest:
            sys.exit(f"Checkpoint of seed {seed} not found: attach version 2 of the fine-tuning notebook as input.")


def download(url, path, size):
    """Download with resume, printing progress every GB."""
    for _ in range(20):
        done = path.stat().st_size if path.exists() else 0
        if done >= size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, path.open("ab") as out:
                if response.status != 206:
                    out.truncate(0)
                    done = 0
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    if (done + len(chunk)) // 2**30 > done // 2**30:
                        print(f"VCTK: {(done + len(chunk)) // 2**30} of {size // 2**30} GB", flush=True)
                    done += len(chunk)
        except (OSError, http.client.HTTPException) as error:
            print(f"VCTK download interrupted at {done} bytes ({error}), resuming", flush=True)
    if not path.exists() or path.stat().st_size != size:
        raise RuntimeError("VCTK download did not complete")


def download_vctk(cfg):
    """Fetch VCTK and keep only the test files; zipfile checks the CRC of everything it extracts."""
    root = Path(cfg["datasets"]["vctk"]["root"])
    paths = pd.read_csv(Path(cfg["protocols"]["dir"]) / "vctk" / "test.csv")["path"].tolist()
    target = DATA_DIR / root.relative_to("data")
    if all((target / p).exists() for p in paths):
        return
    outer_path = DATA_DIR / "DS_10283_3443.zip"
    print("downloading VCTK", flush=True)
    download(VCTK_URL, outer_path, VCTK_SIZE)
    print("unpacking VCTK", flush=True)
    with zipfile.ZipFile(outer_path) as outer:
        outer.extract(VCTK_INNER, DATA_DIR)
    outer_path.unlink()
    inner_path = DATA_DIR / VCTK_INNER
    with zipfile.ZipFile(inner_path) as inner:
        for p in paths:
            inner.extract(f"{root.name}/{p}", target.parent)
    inner_path.unlink()
    print(f"VCTK: {len(paths)} test files ready", flush=True)


def evaluate(run_name, weights, dataset, split, threshold, workers, *extract_args):
    if (Path("results") / run_name / f"{dataset}_{split}" / "metrics.json").exists():
        return
    common = ["--dataset", dataset, "--split", split, "--run", run_name]
    run(sys.executable, "extract_embeddings.py", *common, "--weights", str(weights), "--workers", str(workers), *extract_args)
    run(sys.executable, "evaluate.py", *common, *(["--threshold-from", str(threshold)] if threshold else []))


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
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, {workers} CPUs", flush=True)

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    checkpoints = Path(cfg["paths"]["checkpoints"])
    install_checkpoints(checkpoints)
    print("fine-tuned checkpoints verified", flush=True)

    models = {"pretrained": Path(cfg["model"]["pretrained_weights"])}
    models.update({f"finetuned_seed{s}": checkpoints / f"finetuned_seed{s}" / "best.pt" for s in cfg["training"]["seeds"]})
    for name in models:
        if not (Path("results") / name / "librispeech_validation" / "metrics.json").exists():
            sys.exit(f"results/{name}/librispeech_validation/metrics.json is not in this commit.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not Path("data").exists():
        Path("data").symlink_to(DATA_DIR)
    if not (DATA_DIR / "LibriSpeech" / "train-clean-100").exists() and shutil.disk_usage(DATA_DIR).free < MIN_FREE_GB * 2**30:
        sys.exit(f"Need {MIN_FREE_GB} GB free in {DATA_DIR}.")
    print("downloading LibriSpeech", flush=True)
    download_librispeech()
    download_vctk(cfg)
    run(sys.executable, "model.py")

    for name, weights in models.items():
        threshold = Path("results") / name / "librispeech_validation" / "metrics.json"
        evaluate(name, weights, "librispeech", "test", threshold, workers)
        evaluate(name, weights, "vctk", "test", threshold, workers)
        run(
            sys.executable, "robustness.py", "--run", name, "--weights", str(weights), "--dataset", "librispeech",
            "--split", "test", "--threshold-from", str(threshold), "--workers", str(workers),
        )
        ablation = f"{name}_novad"
        evaluate(ablation, weights, "librispeech", "validation", None, workers, "--no-vad")
        evaluate(ablation, weights, "librispeech", "test", Path("results") / ablation / "librispeech_validation" / "metrics.json", workers, "--no-vad")

    with tarfile.open(OUTPUT, "w:gz") as archive:
        for name in models:
            for path in sorted(Path("results").glob(f"{name}*")):
                archive.add(path)
    print(f"outputs -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
