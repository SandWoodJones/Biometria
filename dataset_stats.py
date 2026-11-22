"""Describe the audio files of each dataset protocol, reading only the file headers.

Writes speakers, files, sample rate and duration statistics to results/summary/datasets.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

SPLITS = ("train", "validation", "test")


def describe(name, root, protocol_dir):
    files = pd.concat([pd.read_csv(protocol_dir / name / f"{split}.csv", dtype={"speaker": str}) for split in SPLITS])
    headers = [sf.info(root / path) for path in files["path"]]
    durations = pd.Series([header.duration for header in headers])
    genders = files.drop_duplicates("speaker")["gender"].value_counts()
    return {
        "dataset": name,
        "speakers": files["speaker"].nunique(),
        "female_speakers": int(genders["F"]),
        "male_speakers": int(genders["M"]),
        "files": len(files),
        "sample_rates_hz": " ".join(str(rate) for rate in sorted({header.samplerate for header in headers})),
        "total_hours": durations.sum() / 3600,
        "median_duration_s": durations.median(),
        "min_duration_s": durations.min(),
        "max_duration_s": durations.max(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    protocol_dir = Path(cfg["protocols"]["dir"])
    rows = [describe(name, Path(dataset["root"]), protocol_dir) for name, dataset in cfg["datasets"].items()]
    out = Path(cfg["paths"]["results"]) / "summary" / "datasets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f")
    print(f"dataset statistics -> {out}")


if __name__ == "__main__":
    main()
