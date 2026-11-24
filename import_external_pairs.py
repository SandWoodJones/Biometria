"""Import test pairs made by another group into this repository's protocol format.

Downloads the genuine and impostor test pairs listed under `external_protocols` in the config, at the
pinned commit, keeps the original files in protocols/<name>/original/ and writes test.csv,
test_genuine_pairs.csv and test_impostor_pairs.csv with paths relative to the dataset root.
"""

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from make_protocols import LISTERS

FILES = {"genuine": "test_genuine_pairs.csv", "impostor": "test_impostor_pairs.csv"}


def relative(path, root_name):
    """Strip everything up to the dataset root folder, whatever the path separator."""
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[parts.index(root_name) + 1 :])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", default="vctk_dupla7")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    source = cfg["external_protocols"][args.name]
    dataset = cfg["datasets"][source["dataset"]]
    root = Path(dataset["root"])
    out = Path(cfg["protocols"]["dir"]) / args.name
    (out / "original").mkdir(parents=True, exist_ok=True)

    pairs = {}
    for kind, name in FILES.items():
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{source['path']}/{name}"
        original = out / "original" / name
        original.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        table = pd.read_csv(original)
        assert (table["tipo"] == kind).all(), f"{name}: unexpected tipo values"
        pairs[kind] = pd.DataFrame({column: table[column].map(lambda p: relative(p, root.name)) for column in ("amostra_1", "amostra_2")})

    genders = LISTERS[source["dataset"]](dataset).drop_duplicates("speaker").set_index("speaker")["gender"]
    paths = sorted(set(pd.concat([p[c] for p in pairs.values() for c in ("amostra_1", "amostra_2")])))
    files = pd.DataFrame({"path": paths, "speaker": [Path(p).parent.name for p in paths]})
    files["gender"] = files["speaker"].map(genders)

    speaker = files.set_index("path")["speaker"]
    for kind, table in pairs.items():
        same = table["amostra_1"].map(speaker) == table["amostra_2"].map(speaker)
        assert same.all() if kind == "genuine" else not same.any(), f"{kind} pairs with the wrong speaker relation"
    missing = [p for p in paths if not (root / p).exists()]
    assert not missing, f"{len(missing)} files missing locally, e.g. {missing[0]}"
    assert files["gender"].notna().all(), "speaker without gender metadata"

    files.to_csv(out / "test.csv", index=False)
    for kind, name in FILES.items():
        pairs[kind].to_csv(out / name, index=False)
    print(f"{args.name}: {len(files)} files, {len(pairs['genuine'])} genuine / {len(pairs['impostor'])} impostor pairs -> {out}")


if __name__ == "__main__":
    main()
