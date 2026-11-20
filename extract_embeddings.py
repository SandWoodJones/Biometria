"""Compute L2-normalised embeddings for every file of one protocol split."""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import count_parameters, load_model, model_size_mb
from preprocess import preprocess

WARMUP = 10


class SplitFiles(Dataset):
    def __init__(self, paths, root, cfg):
        self.paths, self.root, self.cfg = paths, Path(root), cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        start = time.perf_counter()
        feats = preprocess(self.root / self.paths[i], self.cfg)
        return self.paths[i], feats, time.perf_counter() - start


def embed_files(model, paths, root, cfg, device, workers, desc=None):
    """Embed whole utterances one at a time. Returns a store dict (see main) without metadata."""
    loader = DataLoader(
        SplitFiles(paths, root, cfg),
        batch_size=None,
        num_workers=workers,
        worker_init_fn=lambda _: torch.set_num_threads(1),
    )
    done, embeddings, failed = [], [], []
    preprocess_s, forward_s = [], []
    model.eval()
    with torch.inference_mode():
        for i, (path, feats, prep_time) in enumerate(tqdm(loader, desc=desc, leave=False)):
            preprocess_s.append(prep_time)
            if feats is None:
                failed.append(path)
                continue
            feats = feats.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            embedding = model(feats[None])
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= WARMUP:
                forward_s.append(time.perf_counter() - start)
            done.append(path)
            embeddings.append(F.normalize(embedding, dim=1).cpu())
    return {
        "paths": done,
        "embeddings": torch.cat(embeddings),
        "failed": failed,
        "num_files": len(paths),
        "timing_ms": {
            "preprocess_mean": 1000 * float(np.mean(preprocess_s)),
            "embedding_mean": 1000 * float(np.mean(forward_s)),
            "embedding_device": device.type,
        },
    }


def hardware():
    cpu = next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        platform.processor(),
    )
    info = {"cpu": cpu, "torch": str(torch.__version__)}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_arch=getattr(props, "gcnArchName", ""), gpu_memory_gb=round(props.total_memory / 2**30, 1))
        info["gpu_runtime"] = f"HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True, choices=["librispeech", "vctk"])
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--run", required=True, help="name for this model/pipeline variant, e.g. pretrained")
    parser.add_argument("--weights", help="model weights (default: the pretrained checkpoint from the config)")
    parser.add_argument("--no-vad", action="store_true", help="skip voice activity detection (ablation)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_cfg = copy.deepcopy(cfg)
    run_cfg["vad"]["enabled"] = not args.no_vad
    weights = Path(args.weights or cfg["model"]["pretrained_weights"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weights, cfg["model"]["embed_dim"]).to(device)
    paths = pd.read_csv(Path(cfg["protocols"]["dir"]) / args.dataset / f"{args.split}.csv")["path"].tolist()
    store = embed_files(model, paths, cfg["datasets"][args.dataset]["root"], run_cfg, device, args.workers, f"{args.dataset}/{args.split}")

    store["model"] = {
        "architecture": cfg["model"]["architecture"],
        "weights": str(weights),
        "pretraining": cfg["model"]["pretraining"],
        "num_parameters": count_parameters(model),
        "embedding_dim": cfg["model"]["embed_dim"],
        "model_size_mb": round(model_size_mb(model), 2),
    }
    # Weights produced by train.py have a sidecar describing how they were trained.
    sidecar = weights.with_suffix(".json")
    if sidecar.exists():
        store["model"]["training"] = json.loads(sidecar.read_text())
    store["pipeline"] = {k: run_cfg[k] for k in ("audio", "vad", "features")}
    store["hardware"] = hardware()

    out = Path(cfg["paths"]["embeddings"]) / args.run / args.dataset / f"{args.split}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out)
    print(f"{len(store['paths'])} embeddings, {len(store['failed'])} failures to acquire -> {out}")


if __name__ == "__main__":
    main()
