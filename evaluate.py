"""Score verification pairs with cosine similarity and compute the biometric metrics.

The decision threshold is the EER threshold of a validation split. Evaluating a test split requires
--threshold-from pointing at that validation metrics.json, so the test set never sets the threshold.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve


def score_pairs(store, pairs_dir, split):
    index = {path: i for i, path in enumerate(store["paths"])}
    embeddings = store["embeddings"]
    frames, excluded = [], 0
    for kind in ("genuine", "impostor"):
        pairs = pd.read_csv(pairs_dir / f"{split}_{kind}_pairs.csv")
        usable = pairs["amostra_1"].isin(index) & pairs["amostra_2"].isin(index)
        excluded += int((~usable).sum())
        pairs = pairs[usable].copy()
        a = embeddings[pairs["amostra_1"].map(index).to_numpy(copy=True)]
        b = embeddings[pairs["amostra_2"].map(index).to_numpy(copy=True)]
        pairs["tipo"] = kind
        pairs["score"] = (a * b).sum(dim=1).numpy()
        frames.append(pairs)
    return pd.concat(frames, ignore_index=True), excluded


def roc_and_eer(scores):
    """ROC (FMR, TAR) plus the equal error rate and its threshold."""
    labels = (scores["tipo"] == "genuine").to_numpy()
    fmr, tar, thresholds = roc_curve(labels, scores["score"], drop_intermediate=False)
    k = np.nanargmin(np.abs((1 - tar) - fmr))
    return fmr, tar, float((fmr[k] + 1 - tar[k]) / 2), float(thresholds[k])


def error_rates(scores, threshold):
    genuine = scores.loc[scores["tipo"] == "genuine", "score"]
    impostor = scores.loc[scores["tipo"] == "impostor", "score"]
    return float((impostor >= threshold).mean()), float((genuine < threshold).mean())


def decimal_comma(value, _):
    return f"{value:g}".replace(".", ",")


# Figure labels are in Portuguese because the plots go straight into the article.
def plot_roc(fmr, tar, eer, path):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.plot(fmr, tar)
    ax.plot([eer], [1 - eer], "o", label=f"EER = {100 * eer:.2f}%".replace(".", ","))
    # The curve hugs the top of the plot, so the vertical axis starts just below its lowest visible point.
    bottom = min(0.98, np.floor(100 * np.interp(1e-4, fmr, tar)) / 100)
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(bottom, 1.001), xlabel="FMR (escala log)", ylabel="TAR = 1 - FNMR", title="Curva ROC")
    ax.yaxis.set_major_formatter(decimal_comma)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(scores, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(scores["score"].min(), scores["score"].max(), 100)
    for kind, label in (("impostor", "impostor"), ("genuine", "genuíno")):
        ax.hist(scores.loc[scores["tipo"] == kind, "score"], bins=bins, density=True, alpha=0.6, label=label)
    ax.axvline(threshold, color="k", linestyle="--", label=f"limiar = {threshold:.4f}".replace(".", ","))
    ax.set(xlabel="similaridade de cosseno", ylabel="densidade", title="Distribuição de escores genuínos e impostores")
    ax.xaxis.set_major_formatter(decimal_comma)
    ax.yaxis.set_major_formatter(decimal_comma)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", required=True, choices=["librispeech", "vctk", "vctk_dupla7"])
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--threshold-from", type=Path, help="validation metrics.json whose threshold is applied")
    args = parser.parse_args()
    if args.split == "test" and args.threshold_from is None:
        parser.error("a test split must use a threshold frozen from validation (--threshold-from)")

    cfg = yaml.safe_load(Path(args.config).read_text())
    store = torch.load(Path(cfg["paths"]["embeddings"]) / args.run / args.dataset / f"{args.split}.pt", weights_only=True)
    scores, excluded = score_pairs(store, Path(cfg["protocols"]["dir"]) / args.dataset, args.split)

    labels = (scores["tipo"] == "genuine").to_numpy()
    fmr, tar, eer, eer_threshold = roc_and_eer(scores)

    if args.threshold_from:
        source = json.loads(args.threshold_from.read_text())
        threshold, threshold_source = source["threshold"], str(args.threshold_from)
    else:
        threshold, threshold_source = eer_threshold, "EER threshold of this validation split"
    fmr_at, fnmr_at = error_rates(scores, threshold)

    model = store["model"]
    metrics = {
        "modality": "voice",
        "dataset": args.dataset,
        "split": args.split,
        "run": args.run,
        "cnn": model["architecture"],
        "pretraining": model["pretraining"],
        "weights": model["weights"],
        "training": model.get("training"),
        "num_parameters": model["num_parameters"],
        "embedding_dim": model["embedding_dim"],
        "matching": "cosine similarity of L2-normalised embeddings",
        "threshold": threshold,
        "threshold_source": threshold_source,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": float(roc_auc_score(labels, scores["score"])),
        "fmr": fmr_at,
        "fnmr": fnmr_at,
        "tar_at_far_1pct": float(np.interp(0.01, fmr, tar)),
        "fta": len(store["failed"]) / store["num_files"],
        "num_genuine_pairs": int(labels.sum()),
        "num_impostor_pairs": int((~labels).sum()),
        "excluded_pairs_fta": excluded,
        "inference_time_ms": store["timing_ms"],
        "model_size_mb": model["model_size_mb"],
        "hardware": store["hardware"],
        "pipeline": store["pipeline"],
    }

    out = Path(cfg["paths"]["results"]) / args.run / f"{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    scores[["amostra_1", "amostra_2", "tipo", "score"]].to_csv(out / "scores.csv", index=False, float_format="%.6f")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_roc(fmr, tar, eer, out / "roc.png")
    plot_distributions(scores, threshold, out / "distributions.png")
    print(
        f"{args.run} {args.dataset}/{args.split}: EER {100 * eer:.2f}%  AUC {metrics['auc']:.4f}  "
        f"FMR {100 * fmr_at:.2f}%  FNMR {100 * fnmr_at:.2f}% @ {threshold:.4f}  "
        f"TAR@FAR=1% {100 * metrics['tar_at_far_1pct']:.2f}%  FTA {100 * metrics['fta']:.2f}%"
    )


if __name__ == "__main__":
    main()
