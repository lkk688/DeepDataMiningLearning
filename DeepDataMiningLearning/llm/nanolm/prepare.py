"""Download / tokenize data ahead of training (train.py also does this lazily).

    python -m nanolm.prepare --config nanolm/configs/02_char_transformer.yaml   # tokenize a text source
    python -m nanolm.prepare --golf-shards 1 --dest data/golf                  # parameter-golf FineWeb sp1024
"""
from __future__ import annotations

import argparse

from .config import load_config
from .data import download_golf_fineweb, prepare_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--golf-shards", type=int, default=-1,
                    help="download N FineWeb train shards (100M tokens, ~200 MB each) + val + tokenizer")
    ap.add_argument("--variant", default="sp1024", help="sp1024 | sp4096 | sp8192 (if hosted)")
    ap.add_argument("--dest", default="data/golf")
    args = ap.parse_args(argv)
    if args.golf_shards >= 0:
        download_golf_fineweb(args.dest, args.golf_shards, args.variant)
        print(f"done -> {args.dest}/fineweb10B_{args.variant}")
    if args.config:
        meta = prepare_dataset(load_config(args.config, args.overrides).data)
        print({k: v for k, v in meta.items() if k != "chars"})


if __name__ == "__main__":
    main()
