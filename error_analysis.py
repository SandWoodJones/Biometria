"""Inspect the worst verification errors of one evaluated run at its frozen threshold.

Selects the highest-scoring false matches (one per speaker pair) and the lowest-scoring false non-matches
(no file repeated), then records acoustic measurements for each file and whether the other models make
the same error. Writes cases.csv, chapter_scores.csv, summary.json and one figure per case to
results/error_analysis/<run>_<dataset>_<split>/. Figure labels are in Portuguese for the article.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import yaml

from preprocess import fbank, load_audio, speech_mask

KINDS = {"false_match": "Falso match", "false_non_match": "Falso non-match"}
RAMP = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])
SPEECH, SILENCE, TEXT, MUTED = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e"


def load_run(results, run, dataset, split):
    directory = results / run / f"{dataset}_{split}"
    metrics = json.loads((directory / "metrics.json").read_text())
    return metrics["threshold"], pd.read_csv(directory / "scores.csv")


def annotate(scores, files):
    """Add speaker, gender and recording session (LibriSpeech chapter) of both samples."""
    scores = scores.copy()
    for column, side in (("amostra_1", "a"), ("amostra_2", "b")):
        scores[f"{side}_speaker"] = scores[column].map(files["speaker"])
        scores[f"{side}_gender"] = scores[column].map(files["gender"])
        scores[f"{side}_chapter"] = scores[column].map(lambda p: p.split("/")[1] if p.count("/") == 2 else "")
    return scores


def select_false_matches(errors, n):
    ordered = errors.sort_values("score", ascending=False)
    keys = [tuple(sorted(pair)) for pair in zip(ordered["a_speaker"], ordered["b_speaker"])]
    return ordered[~pd.Series(keys, index=ordered.index).duplicated()].head(n)


def select_false_non_matches(errors, n):
    chosen, seen = [], set()
    for index, row in errors.sort_values("score").iterrows():
        if len(chosen) < n and not {row["amostra_1"], row["amostra_2"]} & seen:
            chosen.append(index)
            seen |= {row["amostra_1"], row["amostra_2"]}
    return errors.loc[chosen]


def acoustics(path, cfg):
    """Duration, VAD speech fraction, speech-to-floor level and 95% spectral roll-off of one file."""
    sr = cfg["audio"]["sample_rate"]
    wave = load_audio(path, sr)
    frame = int(sr * cfg["vad"]["frame_ms"] / 1000)
    hop = int(sr * cfg["vad"]["hop_ms"] / 1000)
    frames = wave.unfold(0, frame, hop)
    energy_db = 10 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)
    power = torch.fft.rfft(frames * torch.hann_window(frame)).abs().pow(2).mean(dim=0)
    cumulative = power.cumsum(0) / power.sum()
    rolloff = int(torch.searchsorted(cumulative, 0.95)) * sr / frame
    mask = speech_mask(wave, sr, cfg["vad"])
    floor = float(torch.quantile(energy_db, 0.05))
    level = float(torch.quantile(energy_db, 0.95))
    return {
        "duration_s": len(wave) / sr,
        "speech_fraction": float(mask.float().mean()),
        "level_above_floor_db": level - floor,
        "rolloff_hz": rolloff,
    }


def plot_case(case, root, cfg, path):
    sr = cfg["audio"]["sample_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), constrained_layout=True)
    for column, sample, side in ((0, case["amostra_1"], "a"), (1, case["amostra_2"], "b")):
        wave = load_audio(root / sample, sr)
        mask = speech_mask(wave, sr, cfg["vad"])
        t = torch.arange(len(wave)) / sr
        axes[0, column].plot(t, wave.where(mask, torch.tensor(float("nan"))), color=SPEECH, linewidth=0.4, label="fala")
        axes[0, column].plot(t, wave.where(~mask, torch.tensor(float("nan"))), color=SILENCE, linewidth=0.4, label="silêncio")
        axes[0, column].set_title(
            f"{Path(sample).stem}  (locutor {case[f'{side}_speaker']}, {case[f'{side}_gender']})", loc="left", fontsize=9, color=TEXT
        )
        axes[0, column].set_xlabel("tempo (s)", color=MUTED, fontsize=8)
        axes[0, column].set_xlim(0, len(wave) / sr)
        feats = fbank(wave[mask], sr, cfg["features"])
        axes[1, column].imshow(feats.T, origin="lower", aspect="auto", interpolation="nearest", cmap=RAMP)
        axes[1, column].set_xlabel("quadro (10 ms), após VAD", color=MUTED, fontsize=8)
        axes[1, column].set_ylabel("banda Mel", color=MUTED, fontsize=8)
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8, labelcolor=TEXT)
    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    score, threshold = (f"{case[k]:.3f}".replace(".", ",") for k in ("score", "threshold"))
    fig.suptitle(f"{KINDS[case['kind']]}: score {score}, limiar {threshold}", x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def chapter_scores(scores, speakers):
    genuine = scores[(scores["tipo"] == "genuine") & scores["a_speaker"].isin(speakers)].copy()
    first, second = genuine[["a_chapter", "b_chapter"]].min(axis=1), genuine[["a_chapter", "b_chapter"]].max(axis=1)
    genuine["chapter_1"], genuine["chapter_2"] = first, second
    table = genuine.groupby(["a_speaker", "chapter_1", "chapter_2"])["score"].agg(["count", "mean", "min", "max"]).reset_index()
    return table.rename(columns={"a_speaker": "speaker"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", required=True, choices=["librispeech", "vctk"])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--cases", type=int, default=3, help="number of cases of each error type")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results = Path(cfg["paths"]["results"])
    root = Path(cfg["datasets"][args.dataset]["root"])
    files = pd.read_csv(Path(cfg["protocols"]["dir"]) / args.dataset / f"{args.split}.csv", dtype={"speaker": str}).set_index("path")

    threshold, scores = load_run(results, args.run, args.dataset, args.split)
    scores = annotate(scores, files)
    false_matches = scores[(scores["tipo"] == "impostor") & (scores["score"] >= threshold)]
    false_non_matches = scores[(scores["tipo"] == "genuine") & (scores["score"] < threshold)]
    cases = pd.concat(
        [
            select_false_matches(false_matches, args.cases).assign(kind="false_match"),
            select_false_non_matches(false_non_matches, args.cases).assign(kind="false_non_match"),
        ],
        ignore_index=True,
    )
    cases["threshold"] = threshold

    models = ["pretrained", *(f"finetuned_seed{s}" for s in cfg["training"]["seeds"])]
    repeated = np.zeros(len(cases), dtype=int)
    for model in models:
        other_threshold, other = load_run(results, model, args.dataset, args.split)
        indexed = other.set_index(["amostra_1", "amostra_2"])["score"]
        values = indexed.loc[list(zip(cases["amostra_1"], cases["amostra_2"]))].to_numpy()
        repeated += np.where(cases["kind"] == "false_match", values >= other_threshold, values < other_threshold)
    cases["models_with_same_error"] = repeated
    cases["models_compared"] = len(models)

    for side, column in (("a", "amostra_1"), ("b", "amostra_2")):
        measured = pd.DataFrame([acoustics(root / path, cfg) for path in cases[column]]).add_prefix(f"{side}_")
        cases = pd.concat([cases, measured], axis=1)

    out = results / "error_analysis" / f"{args.run}_{args.dataset}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    cases.drop(columns="tipo").to_csv(out / "cases.csv", index=False, float_format="%.4f")
    speakers = sorted(set(cases["a_speaker"]) | set(cases["b_speaker"]))
    chapter_scores(scores, speakers).to_csv(out / "chapter_scores.csv", index=False, float_format="%.4f")
    for kind, group in cases.groupby("kind"):
        for i, (_, case) in enumerate(group.iterrows(), 1):
            plot_case(case, root, cfg, out / f"{kind}_{i}.png")

    speaker_pairs = false_matches.apply(lambda r: "-".join(sorted((r["a_speaker"], r["b_speaker"]))), axis=1)
    summary = {
        "run": args.run,
        "dataset": args.dataset,
        "split": args.split,
        "threshold": threshold,
        "num_false_matches": len(false_matches),
        "num_false_non_matches": len(false_non_matches),
        "false_matches_by_gender_pair": (false_matches["a_gender"] + false_matches["b_gender"]).value_counts().to_dict(),
        "false_matches_by_speaker_pair": speaker_pairs.value_counts().head(10).to_dict(),
        "false_non_matches_by_speaker": false_non_matches["a_speaker"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(false_matches)} false matches, {len(false_non_matches)} false non-matches -> {out}")


if __name__ == "__main__":
    main()
