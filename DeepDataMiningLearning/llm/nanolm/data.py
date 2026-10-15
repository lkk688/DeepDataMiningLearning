"""Data pipeline: raw text -> tokenizer -> uint16 token shards on disk -> random/sequential batches.

Lessons taken from parameter-golf (vs. the old lm_dataset.py):
  1. Tokenize ONCE, store tokens as a flat uint16 array (golf .bin shard format), and memory-map
     it. No Python list-of-lists, no 13M overlapping "sequences" materialised in RAM.
  2. A training "epoch" over stride-1 windows is meaningless (each token is seen seq_len times);
     we count *tokens seen* and *optimizer steps* instead.
  3. The validation split must be fixed by the DATA, never by the run seed, and must not be a
     distribution-shifted tail (e.g. the last 10% of an alphabetically sorted dictionary).
  4. Report bits-per-byte (bpb), not per-token loss/perplexity: per-token numbers are not
     comparable across tokenizers (a char model and a BPE model predict different units).
     Every tokenizer therefore exposes a per-token byte-count lookup table.
"""
from __future__ import annotations

import glob
import json
import os
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

SHARD_MAGIC = 20240520          # parameter-golf / modded-nanogpt shard header
SHARD_VERSION = 1
HEADER_INTS = 256
TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
GOLF_HF_REPO = "willdepueoai/parameter-golf"


# =============================================================================== tokenizers
class Tokenizer:
    """Common interface. `byte_luts()` returns three arrays indexed by token id:
        base_bytes[i]        -- UTF-8 bytes the token contributes
        has_leading_space[i] -- SentencePiece '▁' prefix (counts 1 byte unless after a boundary)
        is_boundary[i]       -- control/BOS/EOS tokens (0 bytes, and reset the leading space rule)
    so that   bytes(target) = base_bytes[y] + (has_leading_space[y] & ~is_boundary[x_prev]).
    This is exactly the scoring rule of the parameter-golf challenge."""
    spec: str
    vocab_size: int
    eot_id: int | None = None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids) -> str: ...
    def byte_luts(self): ...
    def meta(self) -> dict:
        return {"spec": self.spec, "vocab_size": self.vocab_size}


class CharTokenizer(Tokenizer):
    """One token per Unicode character; vocabulary = characters present in the corpus."""

    def __init__(self, chars: list[str]):
        self.spec = "char"
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unk = self.stoi.get("?", 0)
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids if 0 <= int(i) < self.vocab_size)

    def byte_luts(self):
        base = np.array([len(c.encode("utf-8")) for c in self.chars], dtype=np.int16)
        z = np.zeros(self.vocab_size, dtype=bool)
        return base, z, z.copy()

    def meta(self) -> dict:
        return {**super().meta(), "chars": self.chars}


class ByteTokenizer(Tokenizer):
    """Raw UTF-8 bytes (vocab 256). Loss in bits/token == bits-per-byte."""

    def __init__(self):
        self.spec, self.vocab_size = "byte", 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def byte_luts(self):
        z = np.zeros(256, dtype=bool)
        return np.ones(256, dtype=np.int16), z, z.copy()


class TiktokenTokenizer(Tokenizer):
    """GPT-2 BPE via tiktoken (vocab 50257, fits in uint16)."""

    def __init__(self, name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(name)
        self.spec, self.vocab_size, self.eot_id = name, self.enc.n_vocab, self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])

    def byte_luts(self):
        base = np.zeros(self.vocab_size, dtype=np.int16)
        boundary = np.zeros(self.vocab_size, dtype=bool)
        for i in range(self.vocab_size):
            try:
                base[i] = len(self.enc.decode_single_token_bytes(i))
            except KeyError:
                boundary[i] = True
        boundary[self.eot_id] = True
        base[self.eot_id] = 0
        return base, np.zeros(self.vocab_size, dtype=bool), boundary


class SentencePieceTokenizer(Tokenizer):
    """SentencePiece model, e.g. the golf `fineweb_1024_bpe.model` (sp1024)."""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.spec, self.vocab_size = f"sp:{model_path}", int(self.sp.vocab_size())
        self.eot_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else None

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])

    def byte_luts(self):  # identical to parameter-golf build_sentencepiece_luts()
        n = self.vocab_size
        base = np.zeros(n, dtype=np.int16)
        lead = np.zeros(n, dtype=bool)
        boundary = np.ones(n, dtype=bool)
        for i in range(n):
            if self.sp.is_control(i) or self.sp.is_unknown(i) or self.sp.is_unused(i):
                continue
            boundary[i] = False
            if self.sp.is_byte(i):
                base[i] = 1
                continue
            piece = self.sp.id_to_piece(i)
            if piece.startswith("▁"):
                lead[i] = True
                piece = piece[1:]
            base[i] = len(piece.encode("utf-8"))
        return base, lead, boundary


def build_tokenizer(spec: str, text_for_vocab: str | None = None, meta: dict | None = None) -> Tokenizer:
    if spec == "char":
        if meta and "chars" in meta:
            return CharTokenizer(meta["chars"])
        if text_for_vocab is None:
            raise ValueError("char tokenizer needs the corpus (or saved meta) to build its vocabulary")
        return CharTokenizer.from_text(text_for_vocab)
    if spec == "byte":
        return ByteTokenizer()
    if spec in ("gpt2", "cl100k_base", "r50k_base"):
        return TiktokenTokenizer(spec)
    if spec.startswith("sp:"):
        return SentencePieceTokenizer(os.path.expanduser(spec[3:]))
    raise ValueError(f"unknown tokenizer spec {spec!r} (char | byte | gpt2 | sp:<model>)")


# ============================================================================ shard I/O
def write_shard(path: str | Path, tokens) -> None:
    arr = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0], header[1], header[2] = SHARD_MAGIC, SHARD_VERSION, arr.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.astype("<u2").tobytes())


def read_shard(path: str | Path) -> np.ndarray:
    """Memory-map a shard: costs no RAM until the pages are touched."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
        raise ValueError(f"{path}: not a nanolm/parameter-golf token shard")
    n = int(header[2])
    expected = HEADER_INTS * 4 + 2 * n
    if os.path.getsize(path) != expected:
        raise ValueError(f"{path}: size mismatch (expected {expected} bytes)")
    return np.memmap(path, dtype="<u2", mode="r", offset=HEADER_INTS * 4, shape=(n,))


# ======================================================================= dataset prepare
def _dataset_name(cfg) -> str:
    if cfg.source == "text":
        return "text_" + "_".join(Path(f).stem for f in cfg.files)[:60]
    if cfg.source == "hf":
        name = cfg.hf_name.replace("/", "__") + (f"_{cfg.hf_config}" if cfg.hf_config else "")
        return name + (f"_n{cfg.hf_max_docs}" if cfg.hf_max_docs else "")
    return cfg.source


def _load_raw_docs(cfg) -> list[str]:
    """Return the list of documents. A single-document corpus is fine (Shakespeare)."""
    if cfg.source == "tinyshakespeare":
        raw = Path(cfg.cache_dir) / "raw" / "tinyshakespeare.txt"
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            print(f"[data] downloading Tiny Shakespeare -> {raw}")
            urllib.request.urlretrieve(TINYSHAKESPEARE_URL, raw)
        return [raw.read_text(encoding="utf-8")]
    if cfg.source == "text":
        if not cfg.files:
            raise ValueError("data.source=text needs data.files=[...]")
        return [Path(f).read_text(encoding="utf-8") for f in cfg.files]
    if cfg.source == "hf":
        from datasets import load_dataset
        ds = load_dataset(cfg.hf_name, cfg.hf_config or None, split=cfg.hf_split)
        if cfg.hf_max_docs:
            ds = ds.select(range(min(cfg.hf_max_docs, len(ds))))
        docs = []
        for row in ds:
            parts = [str(row[k]) for k in cfg.hf_text_fields if row.get(k)]
            if parts:
                docs.append(" ".join(parts))
        return docs
    raise ValueError(f"unknown data.source {cfg.source!r}")


def _tok_name(spec: str) -> str:
    return Path(spec[3:]).stem if spec.startswith("sp:") else spec


def prepare_dataset(cfg) -> dict:
    """Tokenize text sources into <cache_dir>/<name>_<tok>/{train,val}.bin + meta.json (cached)."""
    if cfg.source == "shards":
        return _shard_dataset_info(cfg)
    tag = f"{_dataset_name(cfg)}_{_tok_name(cfg.tokenizer)}_{cfg.split_by}{int(cfg.val_fraction * 1000)}"
    out = Path(cfg.cache_dir) / tag
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["train_files"] = [str(out / "train.bin")]
        meta["val_files"] = [str(out / "val.bin")]
        return meta

    docs = _load_raw_docs(cfg)
    # ---- split BEFORE tokenizing, by data-defined rule (independent of the run seed)
    if cfg.split_by == "doc" and len(docs) > 1:
        idx = list(range(len(docs)))
        random.Random(0).shuffle(idx)         # fixed data seed, NOT the run seed
        n_val = max(1, int(len(docs) * cfg.val_fraction))
        val_docs = [docs[i] for i in idx[:n_val]]
        train_docs = [docs[i] for i in idx[n_val:]]
    else:                                     # contiguous tail of the concatenated corpus
        full = "\n\n".join(docs)
        cut = int(len(full) * (1 - cfg.val_fraction))
        train_docs, val_docs = [full[:cut]], [full[cut:]]

    tok = build_tokenizer(cfg.tokenizer, text_for_vocab="".join(train_docs + val_docs))
    if tok.vocab_size > 65535:
        raise ValueError("vocab does not fit in uint16 shards")

    def encode_docs(ds: list[str]) -> np.ndarray:
        ids: list[int] = []
        for i, d in enumerate(ds):
            if i > 0:
                if tok.eot_id is not None:
                    ids.append(tok.eot_id)
                else:
                    ids.extend(tok.encode("\n\n"))
            ids.extend(tok.encode(d))
        return np.asarray(ids, dtype=np.uint16)

    out.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = encode_docs(train_docs), encode_docs(val_docs)
    write_shard(out / "train.bin", train_ids)
    write_shard(out / "val.bin", val_ids)
    meta = {
        **tok.meta(),
        "dataset": tag,
        "n_train_tokens": int(train_ids.size),
        "n_val_tokens": int(val_ids.size),
        "n_val_bytes_utf8": sum(len(d.encode("utf-8")) for d in val_docs),
        "n_train_docs": len(train_docs),
        "n_val_docs": len(val_docs),
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[data] wrote {out}: {meta['n_train_tokens']:,} train / {meta['n_val_tokens']:,} val tokens "
          f"(vocab {tok.vocab_size})")
    meta["train_files"] = [str(out / "train.bin")]
    meta["val_files"] = [str(out / "val.bin")]
    return meta


def _shard_dataset_info(cfg) -> dict:
    d = os.path.expanduser(cfg.shard_dir)
    train = sorted(glob.glob(os.path.join(d, cfg.train_pattern)))
    val = sorted(glob.glob(os.path.join(d, cfg.val_pattern)))
    if cfg.max_train_shards:
        train = train[: cfg.max_train_shards]
    if not train or not val:
        raise FileNotFoundError(f"no shards in {d!r} matching {cfg.train_pattern} / {cfg.val_pattern}. "
                                f"Run: python -m nanolm.prepare --golf-shards 1")
    if not cfg.tokenizer.startswith("sp:"):
        raise ValueError("data.source=shards needs data.tokenizer=sp:<model> for bpb byte counting")
    tok = build_tokenizer(cfg.tokenizer)
    return {**tok.meta(), "dataset": Path(d).name, "train_files": train, "val_files": val}


def download_golf_fineweb(dest: str, train_shards: int = 1, variant: str = "sp1024") -> None:
    """Fetch the parameter-golf FineWeb shards (100M tokens each) + tokenizer from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import shutil
    dest = Path(dest)
    ds = f"fineweb10B_{variant}"
    vocab = variant[2:]
    files = [(f"datasets/datasets/{ds}", "fineweb_val_000000.bin", dest / ds)]
    files += [(f"datasets/datasets/{ds}", f"fineweb_train_{i:06d}.bin", dest / ds) for i in range(train_shards)]
    files += [("datasets/tokenizers", f"fineweb_{vocab}_bpe.model", dest / "tokenizers")]
    for sub, name, target_dir in files:
        target = target_dir / name
        if target.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading {sub}/{name}")
        p = hf_hub_download(repo_id=GOLF_HF_REPO, filename=name, subfolder=sub, repo_type="dataset")
        shutil.copy2(Path(p).resolve(), target)


# ============================================================================= batching
class TokenData:
    """Holds memory-mapped train/val tokens and hands out (x, y) batches."""

    def __init__(self, meta: dict, sampler: str = "random", seed: int = 0, rank: int = 0, world: int = 1):
        self.meta = meta
        self.train = [read_shard(f) for f in meta["train_files"]]
        self.val = np.concatenate([np.asarray(read_shard(f)) for f in meta["val_files"]])
        self.sampler = sampler
        self.rank, self.world = rank, world
        self.rng = np.random.default_rng(seed + 1000 * rank)
        self._sizes = np.array([s.size for s in self.train], dtype=np.float64)
        self._shard, self._pos = 0, 0         # sequential stream state

    @property
    def n_train_tokens(self) -> int:
        return int(self._sizes.sum())

    def _random_batch(self, B: int, T: int) -> np.ndarray:
        out = np.empty((B, T + 1), dtype=np.int64)
        shard_ids = self.rng.choice(len(self.train), size=B, p=self._sizes / self._sizes.sum())
        for b, s in enumerate(shard_ids):
            arr = self.train[s]
            i = int(self.rng.integers(0, arr.size - T - 1))
            out[b] = arr[i: i + T + 1]
        return out

    def _take(self, n: int) -> np.ndarray:
        chunks, need = [], n
        while need > 0:
            arr = self.train[self._shard]
            k = min(need, arr.size - self._pos)
            if k <= 0:
                self._shard, self._pos = (self._shard + 1) % len(self.train), 0
                continue
            chunks.append(np.asarray(arr[self._pos: self._pos + k]))
            self._pos += k
            need -= k
        return np.concatenate(chunks).astype(np.int64)

    def _sequential_batch(self, B: int, T: int) -> np.ndarray:
        # golf DistributedTokenLoader: one contiguous chunk per rank, deterministic, no repeats
        span = B * T + 1
        chunk = self._take(span * self.world)
        local = chunk[self.rank * span: (self.rank + 1) * span]
        x = local[:-1].reshape(B, T)
        y = local[1:].reshape(B, T)
        return np.concatenate([x, y[:, -1:]], axis=1)

    def train_batch(self, B: int, T: int, device: torch.device):
        arr = self._random_batch(B, T) if self.sampler == "random" else self._sequential_batch(B, T)
        t = torch.from_numpy(arr)
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        return t[:, :-1], t[:, 1:]

    def val_windows(self, T: int, max_tokens: int = 0):
        """Non-overlapping eval windows over the (deterministic) start of the val split."""
        n = self.val.size - 1
        if max_tokens:
            n = min(n, max_tokens)
        n_win = n // T
        if n_win == 0:
            raise ValueError(f"val split ({self.val.size} tokens) shorter than seq_len={T}")
        return self.val[: n_win * T + 1]


def byte_lut_tensors(meta: dict, device: torch.device):
    tok = build_tokenizer(meta["spec"], meta=meta)
    base, lead, boundary = tok.byte_luts()
    V = max(meta["vocab_size"], base.size)
    pad = V - base.size
    if pad:
        base = np.pad(base, (0, pad))
        lead = np.pad(lead, (0, pad))
        boundary = np.pad(boundary, (0, pad), constant_values=True)
    return (torch.tensor(base, dtype=torch.int32, device=device),
            torch.tensor(lead, dtype=torch.bool, device=device),
            torch.tensor(boundary, dtype=torch.bool, device=device))


def count_bytes(x_prev: torch.Tensor, y: torch.Tensor, luts) -> torch.Tensor:
    base, lead, boundary = luts
    return base[y] + (lead[y] & ~boundary[x_prev]).to(torch.int32)
