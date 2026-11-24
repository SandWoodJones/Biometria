"""Aggregate the evaluation runs into the tables and the robustness figure used in the article.

The fine-tuned model is reported as mean and sample standard deviation over the seeds in the config.
Writes metrics.csv, efficiency.csv, tables.typ, robustness.png and robustness_vertical.png (for the
slides) to results/summary/. Labels of the Typst tables and of the figures are in Portuguese because
they go straight into the article.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# name, Portuguese label, run suffix, result directory
CONDITIONS = [
    ("librispeech_validation", "LibriSpeech, validação", "", "librispeech_validation"),
    ("librispeech_test", "LibriSpeech, teste", "", "librispeech_test"),
    ("vctk_test", "VCTK, teste (cross-dataset)", "", "vctk_test"),
    ("vctk_dupla7_test", "VCTK, pares da Dupla 7", "", "vctk_dupla7_test"),
    ("snr20", "Ruído, SNR 20 dB", "_snr20", "librispeech_test"),
    ("snr10", "Ruído, SNR 10 dB", "_snr10", "librispeech_test"),
    ("snr0", "Ruído, SNR 0 dB", "_snr0", "librispeech_test"),
    ("no_vad", "Sem VAD, teste", "_novad", "librispeech_test"),
]
ROBUSTNESS = [("librispeech_test", "Limpo"), ("snr20", "20 dB"), ("snr10", "10 dB"), ("snr0", "0 dB")]
PERCENT = {"eer": "EER", "fmr": "FMR", "fnmr": "FNMR", "tar_at_far_1pct": "TAR@FAR=1%", "fta": "FTA"}
MODEL_LABELS = {"pretrained": "Pré-treinado", "finetuned": "Ajustado"}
# Conditions in the per-condition table; the clean test sets are in the standardized table
RESULT_CONDITIONS = ("vctk_dupla7_test", "snr20", "snr10", "snr0", "no_vad")
# Condition and dataset label of the standardized table (section 26 of the assignment)
STANDARD = [
    ("librispeech_test", "LibriSpeech train-clean-100, teste"),
    ("vctk_test", "VCTK 0.92 (mic1), teste"),
]
STANDARD_MEASURED = [
    "Threshold", "EER (%)", "AUC", "FMR (%)", "FNMR (%)", "TAR\\@FAR=1% (%)", "FTA (%)",
    "Pré-processamento com leitura (ms/amostra)", "Embedding (ms/amostra)",
]
COLORS = {"pretrained": "#2a78d6", "finetuned": "#eb6834"}
MARKERS = {"pretrained": "o", "finetuned": "s"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_metrics(results, run, directory):
    return json.loads((results / run / directory / "metrics.json").read_text())


def aggregate(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else np.nan


def metrics_table(results, seeds):
    rows = []
    for condition, label, suffix, directory in CONDITIONS:
        for model, runs in (("pretrained", ["pretrained"]), ("finetuned", [f"finetuned_seed{s}" for s in seeds])):
            metrics = [read_metrics(results, run + suffix, directory) for run in runs]
            row = {
                "condition": condition,
                "label": label,
                "model": model,
                "num_runs": len(runs),
                "num_genuine_pairs": metrics[0]["num_genuine_pairs"],
                "num_impostor_pairs": metrics[0]["num_impostor_pairs"],
            }
            for field in (*PERCENT, "auc", "threshold"):
                scale = 100 if field in PERCENT else 1
                row[f"{field}_mean"], row[f"{field}_std"] = aggregate([scale * m[field] for m in metrics])
            rows.append(row)
    return pd.DataFrame(rows)


def efficiency_table(results, seeds):
    rows = []
    for model, runs in (("pretrained", ["pretrained"]), ("finetuned", [f"finetuned_seed{s}" for s in seeds])):
        metrics = [read_metrics(results, run, "librispeech_test") for run in runs]
        preprocess = aggregate([m["inference_time_ms"]["preprocess_mean"] for m in metrics])
        embedding = aggregate([m["inference_time_ms"]["embedding_mean"] for m in metrics])
        rows.append(
            {
                "model": model,
                "num_parameters": metrics[0]["num_parameters"],
                "embedding_dim": metrics[0]["embedding_dim"],
                "model_size_mb": metrics[0]["model_size_mb"],
                "preprocess_ms_mean": preprocess[0],
                "preprocess_ms_std": preprocess[1],
                "embedding_ms_mean": embedding[0],
                "embedding_ms_std": embedding[1],
                "cpu": metrics[0]["hardware"]["cpu"],
                "gpu": metrics[0]["hardware"]["gpu"],
            }
        )
    return pd.DataFrame(rows)


def number(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def with_std(mean, std, digits):
    return number(mean, digits) if np.isnan(std) else f"{number(mean, digits)} ± {number(std, digits)}"


def typst_table(name, columns, rows, text_columns=2):
    cells = ",\n".join("  " + ", ".join(f"[{cell}]" for cell in row) for row in rows)
    header = ", ".join(f"[*{column}*]" for column in columns)
    align = ["left"] * text_columns + ["right"] * (len(columns) - text_columns)
    return (
        f"#let {name} = table(\n  columns: {len(columns)},\n  align: ({', '.join(align)}),\n"
        f"  table.header({header}),\n{cells},\n)\n"
    )


def standard_table(metrics, results, seeds):
    """Section 26 fields for both datasets; values shared by every column span the whole row."""
    columns = []
    for condition, _ in STANDARD:
        directory = next(c[3] for c in CONDITIONS if c[0] == condition)
        summary = metrics[metrics["condition"] == condition].set_index("model")
        for model, runs in (("pretrained", ["pretrained"]), ("finetuned", [f"finetuned_seed{s}" for s in seeds])):
            loaded = [read_metrics(results, run, directory) for run in runs]
            row = summary.loc[model]
            stat = lambda field, digits: with_std(row[f"{field}_mean"], row[f"{field}_std"], digits)
            time = lambda key: with_std(*aggregate([m["inference_time_ms"][key] for m in loaded]), 1)
            columns.append(
                [
                    stat("threshold", 4),
                    stat("eer", 3),
                    stat("auc", 6),
                    stat("fmr", 3),
                    stat("fnmr", 3),
                    stat("tar_at_far_1pct", 3),
                    stat("fta", 3),
                    time("preprocess_mean"),
                    time("embedding_mean"),
                ]
            )
    first = read_metrics(results, "pretrained", "librispeech_test")
    hardware = f"{first['hardware']['gpu']}, {first['hardware']['cpu']}".replace("@", "\\@")
    spanned = lambda label, value: f"  [{label}], table.cell(colspan: 4, align: center)[{value}]"
    measured = ["  " + ", ".join(f"[{cell}]" for cell in row) for row in zip(STANDARD_MEASURED, *columns)]
    rows = [
        spanned("Modalidade", "Voz"),
        spanned("CNN", "ResNet-34 (WeSpeaker)"),
        spanned("Pré-treinamento", "VoxCeleb2 dev (ajustado: + LibriSpeech)"),
        spanned("Nº de parâmetros", f"{first['num_parameters']:,}".replace(",", ".")),
        spanned("Dimensão do embedding", first["embedding_dim"]),
        spanned("Matching", "Similaridade de cosseno"),
        *measured,
        spanned("Tamanho do modelo (MB)", number(first["model_size_mb"], 1)),
        spanned("Hardware", hardware),
    ]
    datasets = ", ".join(f"table.cell(colspan: 2, align: center)[*{label}*]" for _, label in STANDARD)
    models = f"[*{MODEL_LABELS['pretrained']}*], [*{MODEL_LABELS['finetuned']} ({len(seeds)} seeds)*]"
    return (
        "#let tabela_padronizada = table(\n  columns: 5,\n  align: (left, right, right, right, right),\n"
        f"  table.header([*Dataset*], {datasets},\n    [*Modelo*], {models}, {models}),\n" + ",\n".join(rows) + ",\n)\n"
    )


def write_typst(metrics, efficiency, standard, path):
    fields = ("eer", "fmr", "fnmr", "tar_at_far_1pct")
    result_rows = [
        [r.label, MODEL_LABELS[r.model], *(with_std(getattr(r, f"{f}_mean"), getattr(r, f"{f}_std"), 3) for f in fields)]
        for r in metrics[metrics["condition"].isin(RESULT_CONDITIONS)].itertuples()
    ]
    efficiency_rows = [
        [
            MODEL_LABELS[r.model],
            f"{r.num_parameters:,}".replace(",", "."),
            r.embedding_dim,
            number(r.model_size_mb, 1),
            with_std(r.preprocess_ms_mean, r.preprocess_ms_std, 1),
            with_std(r.embedding_ms_mean, r.embedding_ms_std, 1),
        ]
        for r in efficiency.itertuples()
    ]
    path.write_text(
        "// Generated by summarize.py. Do not edit by hand.\n"
        + typst_table("tabela_resultados", ["Condição", "Modelo", "EER (%)", "FMR (%)", "FNMR (%)", "TAR\\@FAR=1% (%)"], result_rows)
        + "\n"
        + typst_table("tabela_eficiencia", ["Modelo", "Parâmetros", "Embedding", "Tamanho (MB)", "Pré-proc. (ms)", "Embedding (ms)"], efficiency_rows)
        + "\n"
        + standard
    )


def plot_robustness(metrics, path, vertical=False):
    x = np.arange(len(ROBUSTNESS))
    shape, size = ((2, 1), (5.5, 6.5)) if vertical else ((1, 2), (9, 3.8))
    fig, axes = plt.subplots(*shape, figsize=size, constrained_layout=True)
    for ax, field, title in zip(axes, ("eer", "fmr"), ("EER", "FMR no limiar congelado da validação")):
        for model in ("pretrained", "finetuned"):
            rows = metrics[metrics["model"] == model].set_index("condition").loc[[c for c, _ in ROBUSTNESS]]
            means, stds = rows[f"{field}_mean"].to_numpy(), rows[f"{field}_std"].fillna(0).to_numpy()
            ax.errorbar(
                x, means, yerr=stds, color=COLORS[model], marker=MARKERS[model], markersize=7, linewidth=2,
                capsize=3, markeredgecolor="white", markeredgewidth=1.5, label=MODEL_LABELS[model], clip_on=False, zorder=3,
            )
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.set_xticks(x, [label for _, label in ROBUSTNESS])
        ax.set_xlabel("Nível de ruído branco", color=MUTED)
        ax.set_ylabel("%", color=MUTED)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}".replace(".", ","))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].legend(frameon=False, loc="upper left", labelcolor=TEXT)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results = Path(cfg["paths"]["results"])
    seeds = cfg["training"]["seeds"]
    out = results / "summary"
    out.mkdir(parents=True, exist_ok=True)

    metrics = metrics_table(results, seeds)
    efficiency = efficiency_table(results, seeds)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    efficiency.to_csv(out / "efficiency.csv", index=False, float_format="%.6f")
    write_typst(metrics, efficiency, standard_table(metrics, results, seeds), out / "tables.typ")
    plot_robustness(metrics, out / "robustness.png")
    plot_robustness(metrics, out / "robustness_vertical.png", vertical=True)
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
