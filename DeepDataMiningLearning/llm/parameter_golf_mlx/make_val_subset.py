"""Make a data dir whose val shard is the first N+1 tokens of the official val split.

train_gpt_mlx.py always scores the *whole* fineweb_val split (62M tokens, ~2 evals per run). nanolm's
laptop configs score the first 2,097,152 tokens. With a val shard of exactly 2,097,153 tokens both
score the same targets with the same byte counting, so their val_bpb is directly comparable, and
evaluation takes seconds instead of minutes. The train shards are symlinked, not copied.

    python parameter_golf_mlx/make_val_subset.py                      # -> data/golf_val2m/fineweb10B_sp1024
    python parameter_golf_mlx/make_val_subset.py --tokens 1048576 --dest data/golf_val1m
"""
import argparse
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20240520, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/golf/fineweb10B_sp1024")
    ap.add_argument("--dest", default="data/golf_val2m")
    ap.add_argument("--tokens", type=int, default=2_097_152, help="number of scored targets")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dest) / Path(a.src).name
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("fineweb_train_*.bin")):
        link = dst / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    val = sorted(src.glob("fineweb_val_*.bin"))[0]
    header = np.fromfile(val, dtype="<i4", count=256)
    assert header[0] == MAGIC and header[1] == VERSION, f"unexpected shard header in {val}"
    n = a.tokens + 1                                         # inputs + the final target
    toks = np.fromfile(val, dtype="<u2", count=n, offset=256 * 4)
    assert toks.size == n, f"val split has only {toks.size} tokens"
    header = header.copy()
    header[2] = n
    with open(dst / "fineweb_val_000000.bin", "wb") as f:
        f.write(header.astype("<i4").tobytes())
        f.write(toks.astype("<u2").tobytes())
    print(f"{dst}: {len(list(dst.glob('fineweb_train_*.bin')))} train shard(s) linked, val = first {n:,} tokens")


if __name__ == "__main__":
    main()
