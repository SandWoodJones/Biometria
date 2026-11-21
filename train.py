"""Fine-tune the pretrained ResNet-34 on the training speakers with AAM-Softmax (ArcFace).

The classification head only exists during training; the saved weights are the embedding extractor.
The best epoch is chosen by EER on the validation pairs. The test split is never read.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate import roc_and_eer, score_pairs
from extract_embeddings import embed_files
from model import load_model
from preprocess import fbank, load_audio, speech_mask


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax: logits s*cos(theta + m) for the target class."""

    def __init__(self, embed_dim, num_classes, margin, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        # Past theta = pi - m, cos(theta + m) stops decreasing; fall back to a linear penalty there.
        self.threshold, self.penalty = math.cos(math.pi - margin), math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1 - cosine.pow(2)).clamp(min=1e-7))
        target = cosine * self.cos_m - sine * self.sin_m
        target = torch.where(cosine > self.threshold, target, cosine - self.penalty)
        one_hot = F.one_hot(labels, cosine.size(1)).bool()
        logits = torch.where(one_hot, target, cosine) * self.scale
        return F.cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).float().mean()


class TrainCrops(Dataset):
    """One random fixed-length crop of each training utterance, after VAD."""

    def __init__(self, files, root, cfg, crop_s):
        self.paths = files["path"].tolist()
        speakers = sorted(files["speaker"].astype(str).unique())
        label_of = {s: i for i, s in enumerate(speakers)}
        self.labels = [label_of[s] for s in files["speaker"].astype(str)]
        self.num_classes = len(speakers)
        self.root, self.cfg = Path(root), cfg
        self.crop = int(crop_s * cfg["audio"]["sample_rate"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        sr = self.cfg["audio"]["sample_rate"]
        wave = load_audio(self.root / self.paths[i], sr)
        if self.cfg["vad"]["enabled"]:
            speech = wave[speech_mask(wave, sr, self.cfg["vad"])]
            if len(speech) >= self.cfg["audio"]["min_speech_s"] * sr:
                wave = speech
        if len(wave) < self.crop:
            wave = wave.repeat(math.ceil(self.crop / len(wave)))
        start = int(torch.randint(0, len(wave) - self.crop + 1, ()))
        return fbank(wave[start : start + self.crop], sr, self.cfg["features"]), self.labels[i]


def validation_metrics(model, cfg, dataset, device, workers):
    """Validation EER and AUC; the AUC breaks ties, since the EER moves in steps of one pair."""
    paths = pd.read_csv(Path(cfg["protocols"]["dir"]) / dataset / "validation.csv")["path"].tolist()
    store = embed_files(model, paths, cfg["datasets"][dataset]["root"], cfg, device, workers, "validation")
    scores, _ = score_pairs(store, Path(cfg["protocols"]["dir"]) / dataset, "validation")
    return roc_and_eer(scores)[2], float(roc_auc_score(scores["tipo"] == "genuine", scores["score"]))


def selection_key(entry):
    return round(entry["validation_eer"], 9), -entry["validation_auc"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_history(history, path):
    fig, (loss_ax, eer_ax) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [h["epoch"] for h in history]
    loss_ax.plot(epochs[1:], [h["train_loss"] for h in history[1:]], marker="o")
    loss_ax.set(xlabel="epoch", ylabel="AAM-Softmax loss", title="Training loss")
    eer_ax.plot(epochs, [100 * h["validation_eer"] for h in history], marker="o")
    eer_ax.set(xlabel="epoch (0 = pretrained, no fine-tuning)", ylabel="EER (%)", title="Validation EER")
    for ax in (loss_ax, eer_ax):
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg = cfg["training"]
    dataset = tcfg["dataset"]
    run = f"finetuned_seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    files = pd.read_csv(Path(cfg["protocols"]["dir"]) / dataset / "train.csv", dtype={"speaker": str})
    data = TrainCrops(files, cfg["datasets"][dataset]["root"], cfg, tcfg["crop_s"])
    loader = DataLoader(
        data,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=tcfg["workers"],
        worker_init_fn=lambda _: torch.set_num_threads(1),
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=True,
    )

    model = load_model(cfg["model"]["pretrained_weights"], cfg["model"]["embed_dim"]).to(device)
    head = AAMSoftmax(cfg["model"]["embed_dim"], data.num_classes, tcfg["margin"], tcfg["scale"]).to(device)
    optimizer = torch.optim.SGD(
        [{"params": model.parameters(), "lr": tcfg["lr"]}, {"params": head.parameters(), "lr": tcfg["head_lr"]}],
        momentum=tcfg["momentum"],
        nesterov=True,
        weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"] * len(loader))

    out_dir = Path(cfg["paths"]["checkpoints"]) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    eer, auc = validation_metrics(model, cfg, dataset, device, tcfg["workers"])
    history = [{"epoch": 0, "validation_eer": eer, "validation_auc": auc}]
    print(f"epoch 0 (pretrained): validation EER {100 * eer:.3f}%, AUC {auc:.5f}")
    best = None

    for epoch in range(1, tcfg["epochs"] + 1):
        frozen = epoch <= tcfg["head_warmup_epochs"]
        model.train(not frozen)
        head.train()
        start, losses, accuracies = time.perf_counter(), [], []
        for feats, labels in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            with torch.set_grad_enabled(not frozen):
                embeddings = model(feats)
            loss, accuracy = head(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            accuracies.append(accuracy.item())

        eer, auc = validation_metrics(model, cfg, dataset, device, tcfg["workers"])
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "train_accuracy": float(np.mean(accuracies)),
                "validation_eer": eer,
                "validation_auc": auc,
                "epoch_minutes": (time.perf_counter() - start) / 60,
            }
        )
        print(
            f"epoch {epoch}{' (head only)' if frozen else ''}: loss {history[-1]['train_loss']:.3f}, "
            f"acc {100 * history[-1]['train_accuracy']:.1f}%, validation EER {100 * eer:.3f}%, AUC {auc:.5f}, "
            f"{history[-1]['epoch_minutes']:.1f} min"
        )
        if not frozen and (best is None or selection_key(history[-1]) < selection_key(best)):
            best = history[-1]
            torch.save(model.state_dict(), out_dir / "best.pt")

    description = {
        "base": cfg["model"]["pretraining"],
        "fine_tuning": f"{dataset} train split ({data.num_classes} speakers, {len(data)} utterances), AAM-Softmax",
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_validation_eer": best["validation_eer"],
        "best_validation_auc": best["validation_auc"],
        "selection": "lowest validation EER, ties broken by highest validation AUC",
        "hyperparameters": tcfg,
        "trainable_layers": "all (backbone frozen during head warm-up epochs)",
    }
    (out_dir / "best.json").write_text(json.dumps(description, indent=2) + "\n")

    results = Path(cfg["paths"]["results"]) / run
    results.mkdir(parents=True, exist_ok=True)
    (results / "training.json").write_text(json.dumps({"description": description, "history": history}, indent=2) + "\n")
    plot_history(history, results / "training.png")
    print(f"best epoch {best['epoch']}: validation EER {100 * best['validation_eer']:.3f}% -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
