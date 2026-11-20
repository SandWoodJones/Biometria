"""Build speaker-disjoint splits and verification trial pairs for each dataset."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "validation", "test")


def list_librispeech(cfg):
    root = Path(cfg["root"])
    genders = {}
    for line in Path(cfg["speakers_file"]).read_text().splitlines():
        fields = [f.strip() for f in line.split("|")]
        if not line.startswith(";") and len(fields) >= 3 and fields[2] == "train-clean-100":
            genders[fields[0]] = fields[1]
    rows = [
        {"path": str(p.relative_to(root)), "speaker": p.parts[-3], "gender": genders[p.parts[-3]]}
        for p in sorted(root.glob("*/*/*.flac"))
    ]
    return pd.DataFrame(rows)


def list_vctk(cfg):
    root = Path(cfg["root"])
    genders = {}
    for line in Path(cfg["speakers_file"]).read_text().splitlines()[1:]:
        fields = line.split()
        if fields:
            genders[fields[0]] = fields[2]
    rows = [
        {"path": str(p.relative_to(root)), "speaker": p.parent.name, "gender": genders[p.parent.name]}
        for p in sorted(root.glob(f"*/*_{cfg['mic']}.flac"))
    ]
    return pd.DataFrame(rows)


LISTERS = {"librispeech": list_librispeech, "vctk": list_vctk}


def split_speakers(files, ratios, rng):
    """Assign whole speakers to splits, stratified by gender."""
    split_of = {}
    for _, group in files.groupby("gender"):
        speakers = sorted(group["speaker"].unique())
        rng.shuffle(speakers)
        n_val = round(len(speakers) * ratios["validation"])
        n_test = round(len(speakers) * ratios["test"])
        for i, speaker in enumerate(speakers):
            split_of[speaker] = "validation" if i < n_val else "test" if i < n_val + n_test else "train"
    return files["speaker"].map(split_of)


def genuine_pairs(files, per_speaker, rng):
    pairs = []
    for _, group in files.groupby("speaker"):
        paths = group["path"].to_numpy()
        candidates = list(itertools.combinations(range(len(paths)), 2))
        chosen = rng.choice(len(candidates), size=min(per_speaker, len(candidates)), replace=False)
        pairs += [(paths[candidates[k][0]], paths[candidates[k][1]]) for k in sorted(chosen)]
    return pairs


def impostor_pairs(files, n, rng):
    paths = files["path"].to_numpy()
    speakers = files["speaker"].to_numpy()
    pairs = set()
    while len(pairs) < n:
        i, j = rng.integers(len(paths), size=2)
        if speakers[i] != speakers[j]:
            pairs.add(tuple(sorted((paths[i], paths[j]))))
    return sorted(pairs)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = cfg["protocols"]

    for name, lister in LISTERS.items():
        rng = np.random.default_rng(cfg["seed"])
        files = lister(cfg["datasets"][name])
        files["split"] = split_speakers(files, pcfg["split"], rng)
        out = Path(pcfg["dir"]) / name

        speakers = {s: set(files.loc[files["split"] == s, "speaker"]) for s in SPLITS}
        for a, b in itertools.combinations(SPLITS, 2):
            assert not speakers[a] & speakers[b], f"{name}: speakers shared between {a} and {b}"

        for split in SPLITS:
            write(files[files["split"] == split].drop(columns="split"), out / f"{split}.csv")

        summary = [f"{s}={len(speakers[s])} spk" for s in SPLITS]
        for split in ("validation", "test"):
            subset = files[files["split"] == split]
            genuine = genuine_pairs(subset, pcfg["genuine_pairs_per_speaker"], rng)
            impostor = impostor_pairs(subset, len(genuine) * pcfg["impostor_to_genuine_ratio"], rng)
            columns = ["amostra_1", "amostra_2"]
            write(pd.DataFrame(genuine, columns=columns), out / f"{split}_genuine_pairs.csv")
            write(pd.DataFrame(impostor, columns=columns), out / f"{split}_impostor_pairs.csv")
            summary.append(f"{split} pairs: {len(genuine)} genuine / {len(impostor)} impostor")
        print(f"{name}: " + ", ".join(summary))


if __name__ == "__main__":
    main()
