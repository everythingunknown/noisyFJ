from pathlib import Path
import gzip
import hashlib
import sys
import tomllib
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = tomllib.loads((ROOT / "datasets.toml").read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(item):
    target = ROOT / "data" / item["id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "complete").exists():
        return
    raw = ROOT / "data" / "raw" / item["url"].rsplit("/", 1)[1]
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print("Download", item["id"], flush=True)
        part = raw.with_suffix(raw.suffix + ".part")
        urllib.request.urlretrieve(item["url"], part)
        part.replace(raw)
    assert digest(raw) == item["sha256"], f"Checksum: {item['id']}"
    print("Prepare", item["id"], flush=True)
    fmt = item["format"]
    if fmt in ("csv", "edges"):
        with zipfile.ZipFile(raw) as archive:
            member = "edges.csv" if fmt == "csv" else next(
                name for name in archive.namelist() if name.endswith(".edges"))
            with archive.open(member) as stream:
                edges = np.loadtxt(stream, dtype=np.int32, usecols=(0, 1),
                                   delimiter="," if fmt == "csv" else None,
                                   comments="#" if fmt == "csv" else "%")
            if item["catalog"]:
                with archive.open("nodes.csv") as stream:
                    nodes = np.loadtxt(stream, dtype=np.int32, delimiter=",", usecols=(0,))
                assert np.array_equal(nodes, np.arange(item["n"]))
    else:
        with gzip.open(raw, "rt") as stream:
            edges = np.loadtxt(stream, dtype=np.int32,
                               usecols=(0, 1, 2) if fmt == "ratings" else (0, 1),
                               delimiter="," if fmt == "ratings" else None)
    assert len(edges) == item["records"]
    if item["catalog"]:
        assert edges.min() >= 0 and edges.max() < item["n"]
        edges += 1
    else:
        nodes = np.unique(edges[:, :2])
        assert len(nodes) == item["n"]
        if fmt == "ratings":
            edges = edges[edges[:, 2] > 0, :2]
        edges = np.searchsorted(nodes, edges).astype(np.int32) + 1
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    edges = edges[np.r_[True, np.any(edges[1:] != edges[:-1], axis=1)]]
    assert len(edges) == item["m"], item["id"]
    edges[:, 0].astype("<i4").tofile(target / "sources.i32")
    edges[:, 1].astype("<i4").tofile(target / "targets.i32")
    nodes.astype("<i4").tofile(target / "node_ids.i32")
    (target / "complete").write_text(item["sha256"] + "\n")


if __name__ == "__main__":
    requested = set(sys.argv[1:])
    assert requested <= {d["id"] for d in CONFIG["datasets"]}
    for dataset in CONFIG["datasets"]:
        if not requested or dataset["id"] in requested:
            prepare(dataset)
