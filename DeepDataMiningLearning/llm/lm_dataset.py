import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
# ============================================================
# 2) DATA MODULE — files / HF datasets; char/word/HF tokenizers
#     supports teacher-forced (per-token) and final-token modes
# ============================================================
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# --- Tokenizers ---
class CharTokenizer:
    def __init__(self, texts: List[str]):
        vocab = sorted(set("".join(texts)))
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    def encode(self, s: str) -> List[int]: return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids: List[int]) -> str: return "".join(self.itos[i] for i in ids if i in self.itos)

class WordTokenizer:
    def __init__(self, texts: List[str]):
        import re
        words = []
        for t in texts: words += re.findall(r"\b\w+\b", t.lower())
        vocab = sorted(set(words))
        self.stoi = {w:i for i,w in enumerate(vocab)}
        self.itos = {i:w for w,i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    def encode(self, s: str) -> List[int]:
        import re; ws = re.findall(r"\b\w+\b", s.lower())
        return [self.stoi[w] for w in ws if w in self.stoi]
    def decode(self, ids: List[int]) -> str: return " ".join(self.itos[i] for i in ids if i in self.itos)

class HFTokenizerWrapper:
    """Wrap any HF tokenizer (BPE/WordPiece)."""
    def __init__(self, name_or_path: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.sep_token or "[PAD]"
        self.vocab_size = self.tok.vocab_size
    def encode(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"]
    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
    @property
    def pad_id(self): return self.tok.pad_token_id
    @property
    def bos_id(self): return self.tok.bos_token_id
    @property
    def eos_id(self): return self.tok.eos_token_id

#pip install datasets
# --- Dataset builders ---
class SequenceDataset(Dataset):
    """
    Teacher-forced LM: return sequences x[0:T], y[1:T+1]
    Final-token: sample a random prefix -> next token target
    """
    def __init__(self, ids: List[int], seq_len: int, mode: str = "teacher-forced"):
        super().__init__()
        self.ids = ids
        self.seq_len = seq_len
        assert mode in ("teacher-forced", "final-token")
        self.mode = mode

    def __len__(self):
        if self.mode == "teacher-forced":
            return max(0, len(self.ids) - self.seq_len - 1)
        # final-token: one sample per position (except first)
        return max(0, len(self.ids) - 1)

    def __getitem__(self, idx):
        if self.mode == "teacher-forced":
            x = torch.tensor(self.ids[idx: idx+self.seq_len], dtype=torch.long)
            y = torch.tensor(self.ids[idx+1: idx+self.seq_len+1], dtype=torch.long)
            return x, y
        else:  # final-token: random prefix ending at idx+1
            # choose a prefix length in [1, seq_len]
            L = random.randint(1, min(self.seq_len, idx+1))
            start = idx + 1 - L
            prefix = torch.tensor(self.ids[start: start+L], dtype=torch.long)
            target = torch.tensor(self.ids[start+L], dtype=torch.long)
            return prefix, target

def collate_teacher(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def collate_final(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    maxT = max(lengths).item()
    xpad = torch.stack([F.pad(x, (0, maxT-len(x)), value=0) for x in xs])
    y = torch.stack(ys)
    return xpad, y, lengths

# --- High-level DataModule ---
@dataclass
class DataConfig:
    # source: either list of text files OR HF dataset name
    files: Optional[List[str]] = None
    hf_name: Optional[str] = None
    hf_split: str = "train"
    tokenizer: str = "char"        # 'char' | 'word' | 'hf:<name>'
    seq_len: int = 256
    batch_size: int = 64
    split_ratio: float = 0.9
    lowercase: bool = False
    mode: str = "teacher-forced"   # or "final-token"

class DataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        raw_text = self._load_text()
        if cfg.lowercase: raw_text = raw_text.lower()
        self.text = raw_text

        # tokenizer
        if cfg.tokenizer == "char":
            self.tok = CharTokenizer([raw_text])
            self.pad_id = 0
        elif cfg.tokenizer == "word":
            self.tok = WordTokenizer([raw_text])
            self.pad_id = 0
        elif cfg.tokenizer.startswith("hf:"):
            name = cfg.tokenizer.split("hf:")[1]
            self.tok = HFTokenizerWrapper(name)
            self.pad_id = self.tok.pad_id or 0
        else:
            raise ValueError("Unknown tokenizer.")

        # ids
        ids = self.tok.encode(raw_text)
        split = int(cfg.split_ratio * len(ids))
        self.train_ids, self.valid_ids = ids[:split], ids[split:]
        self.vocab_size = self.tok.vocab_size

    def _load_text(self) -> str:
        if self.cfg.files:
            text = ""
            for p in self.cfg.files:
                with open(p, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"
            return text

        elif self.cfg.hf_name:
            if load_dataset is None:
                raise ImportError("Install `datasets` to use hf_name.")

            # 👇 Split dataset and config
            parts = self.cfg.hf_name.split(",")
            name = parts[0]
            config = parts[1] if len(parts) > 1 else "wikitext-2-raw-v1"

            print(f"📚 Loading Hugging Face dataset: {name}, config: {config}")
            ds = load_dataset(name, config)

            split = self.cfg.hf_split if self.cfg.hf_split in ds else list(ds.keys())[0]
            text = "\n".join(ds[split]["text"])
            return text

        else:
            raise ValueError("Provide either `files` or `hf_name`.")

    def loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds = SequenceDataset(self.train_ids, self.cfg.seq_len, self.cfg.mode)
        valid_ds = SequenceDataset(self.valid_ids, self.cfg.seq_len, self.cfg.mode)
        if self.cfg.mode == "teacher-forced":
            collate = collate_teacher
        else:
            collate = collate_final
        dl_train = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
        dl_valid = DataLoader(valid_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False, collate_fn=collate)
        return dl_train, dl_valid

