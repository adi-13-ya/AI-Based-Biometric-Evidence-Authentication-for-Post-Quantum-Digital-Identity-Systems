"""Download the exact public-data subsets used in the report, without
downloading whole archives.

Both sources support random access over HTTP range requests: parquet files
are read row group by row group (100 images each), and zip archives are
read through their central directory so only the selected members are
fetched. Selections are deterministic (fixed seed), so re-running this
script reproduces the same subsets.

    CelebA-Spoof (Ar4ikov/celebA_spoof on Hugging Face)
        datasets/celeba_spoof/train-subset.parquet   4 row groups x 10 train shards
        datasets/celeba_spoof/valid-subset.parquet   2 row groups x 5 valid shards
        datasets/celeba_spoof/test-0000{0,1}.parquet whole test shards (fetched directly)
    DeepFakeFace (OpenRL/DeepFakeFace)
        datasets/dff/{wiki,insight,inpainting,text2img}.zip  members chosen by
        evaluation/train_deepfake.py's identity split and PER_SPLIT sizes
    CASIA-FASD test frames and the UniqueData webcam sample are small and are
    fetched whole (see datasets/download_samples.md).

Run: python evaluation/fetch_data.py [celeba] [dff]
"""
from __future__ import annotations

import io
import json
import struct
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HF = "https://huggingface.co/datasets"
SEED = 2026


def _curl(args: list[str], timeout: int = 300) -> bytes:
    """Network I/O goes through curl: on some networks Python's own HTTP
    stack stalls (address-family selection) where curl does not."""
    import subprocess

    for attempt in range(8):
        r = subprocess.run(["curl", "-sSL", "--fail", "-m", str(timeout), *args], capture_output=True)
        if r.returncode == 0:
            return r.stdout
    raise IOError(f"curl {' '.join(args)} failed: {r.stderr.decode(errors='replace')[:200]}")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str):
        headers = _curl(["-I", "-w", "%{url_effective}", url], timeout=60).decode(errors="replace")
        self.url = headers.strip().splitlines()[-1].strip()  # resolved CDN URL, so later requests skip the redirect
        lengths = [line.split(":", 1)[1].strip() for line in headers.splitlines() if line.lower().startswith("content-length:")]
        self.size = int(lengths[-1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read_range(self, start: int, length: int) -> bytes:
        for _ in range(5):
            data = _curl(["-r", f"{start}-{start + length - 1}", self.url])
            if len(data) == length:
                return data
        raise IOError(f"short read at {start}+{length}")

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.read_range(self.pos, n)
        b[:n] = data
        self.pos += n
        return n


# --------------------------------------------------------------------------
# CelebA-Spoof
# --------------------------------------------------------------------------
def _celeba_shards(prefix: str) -> list[str]:
    tree = json.loads(_curl(["https://huggingface.co/api/datasets/Ar4ikov/celebA_spoof/tree/main/data"]))
    return sorted(x["path"] for x in tree if x["path"].startswith(f"data/{prefix}-"))


def _celeba_subset(prefix: str, n_shards: int, groups_per_shard: int, out: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists():
        print(f"{out} exists")
        return
    shards = _celeba_shards(prefix)
    rng = np.random.default_rng(SEED)
    chosen = [shards[i] for i in np.linspace(0, len(shards) - 1, n_shards).round().astype(int)]
    tables = []
    for shard in chosen:
        f = pq.ParquetFile(io.BufferedReader(HttpRangeFile(f"{HF}/Ar4ikov/celebA_spoof/resolve/main/{shard}"), 1 << 22))
        groups = sorted(rng.choice(f.metadata.num_row_groups, size=groups_per_shard, replace=False).tolist())
        t = f.read_row_groups(groups)
        tables.append(t)
        print(f"  {shard}: row groups {groups} -> {t.num_rows} rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables), out)
    print(f"wrote {out}")


def fetch_celeba() -> None:
    root = REPO / "datasets" / "celeba_spoof"
    _celeba_subset("train", n_shards=10, groups_per_shard=4, out=root / "train-subset.parquet")
    _celeba_subset("valid", n_shards=5, groups_per_shard=2, out=root / "valid-subset.parquet")


# --------------------------------------------------------------------------
# DeepFakeFace
# --------------------------------------------------------------------------
def _member_bytes(remote: HttpRangeFile, info: zipfile.ZipInfo) -> bytes:
    # one request: local header (30 B + name + extra, extra assumed <= 512 B) and the data
    guess = 30 + len(info.orig_filename.encode()) + 512 + info.compress_size
    blob = remote.read_range(info.header_offset, min(guess, remote.size - info.header_offset))
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    data = blob[start: start + info.compress_size]
    if len(data) < info.compress_size:
        data = remote.read_range(info.header_offset + start, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        return data
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")


def fetch_dff() -> None:
    from evaluation.train_deepfake import DFF, GENERATORS, PER_SPLIT, person_split

    DFF.mkdir(parents=True, exist_ok=True)
    remotes = {n: HttpRangeFile(f"{HF}/OpenRL/DeepFakeFace/resolve/main/{n}.zip") for n in ["wiki", *GENERATORS]}
    infos = {}
    for name, remote in remotes.items():
        zf = zipfile.ZipFile(io.BufferedReader(remote, 1 << 22))
        infos[name] = {"/".join(Path(i.filename).parts[-2:]): i for i in zf.infolist() if i.filename.lower().endswith((".jpg", ".png"))}
        print(f"  {name}.zip: {len(infos[name])} images listed")

    rng = np.random.default_rng(SEED)
    wanted: dict[str, list[str]] = {n: [] for n in remotes}
    for split, (n_real, n_fake) in PER_SPLIT.items():
        real_keys = sorted(k for k in infos["wiki"] if person_split(k) == split)
        wanted["wiki"] += rng.choice(real_keys, size=min(n_real, len(real_keys)), replace=False).tolist()
        for gen, n in n_fake.items():
            keys = sorted(k for k in infos[gen] if person_split(k) == split)
            wanted[gen] += rng.choice(keys, size=min(n, len(keys)), replace=False).tolist()

    for name, keys in wanted.items():
        out = DFF / f"{name}.zip"
        if out.exists():
            print(f"{out} exists")
            continue
        remote = remotes[name]
        with ThreadPoolExecutor(max_workers=12) as pool:
            blobs = list(pool.map(lambda k: _member_bytes(remote, infos[name][k]), keys))
        tmp = out.with_suffix(".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for key, blob in zip(keys, blobs):
                zf.writestr(f"{name}/{key}", blob)
        tmp.rename(out)
        print(f"wrote {out}: {len(keys)} images")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["celeba", "dff"]
    if "celeba" in targets:
        fetch_celeba()
    if "dff" in targets:
        fetch_dff()
