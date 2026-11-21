"""Evaluate one model under additive white Gaussian noise at the SNR levels in the config.

The noise is added to the waveform before VAD. The decision threshold stays frozen at the clean
validation threshold given by --threshold-from, and the clean evaluation of the same split must exist.
Each level is saved as the run <run>_snr<level>; the summary goes to results/<run>/robustness_<dataset>_<split>.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

FIELDS = ("eer", "auc", "fmr", "fnmr", "tar_at_far_1pct", "fta")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True, choices=["librispeech", "vctk"])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--threshold-from", type=Path, required=True, help="clean validation metrics.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    results = Path(cfg["paths"]["results"])
    name = f"{args.dataset}_{args.split}"
    clean = results / args.run / name / "metrics.json"
    if not clean.exists():
        parser.error(f"run the clean evaluation first: {clean} not found")

    levels = [{"snr_db": None, **{k: json.loads(clean.read_text())[k] for k in FIELDS}}]
    common = ["--config", args.config, "--dataset", args.dataset, "--split", args.split]
    for snr in cfg["robustness"]["snr_db"]:
        run = f"{args.run}_snr{snr}"
        metrics = results / run / name / "metrics.json"
        if not metrics.exists():
            subprocess.run(
                [sys.executable, "extract_embeddings.py", *common, "--run", run, "--weights", args.weights,
                 "--snr-db", str(snr), "--workers", str(args.workers)],
                check=True,
            )
            subprocess.run(
                [sys.executable, "evaluate.py", *common, "--run", run, "--threshold-from", str(args.threshold_from)],
                check=True,
            )
        levels.append({"snr_db": snr, **{k: json.loads(metrics.read_text())[k] for k in FIELDS}})

    summary = {
        "run": args.run,
        "dataset": args.dataset,
        "split": args.split,
        "noise": cfg["robustness"]["noise"],
        "threshold_source": str(args.threshold_from),
        "levels": levels,
    }
    out = results / args.run / f"robustness_{name}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"robustness summary -> {out}")


if __name__ == "__main__":
    main()
