import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import random
import os
from datasets import load_dataset, get_dataset_config_names, Value
import os

# ============================================================
# 2) DATA MODULE — files / HF datasets; char/word/HF tokenizers
#     supports teacher-forced (per-token) and final-token modes
# ============================================================
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
import re
import unicodedata
import nltk
from nltk.corpus import words
from typing import List

# Download once if needed
# nltk.download('words')

EN_WORDS = set(w.lower() for w in words.words())

def clean_texts(
    texts: List[str],
    keep_lang: str = "ascii",      # 'ascii' | 'en' | 'en_zh' | 'all'
    keep_emojis_math: bool = True,
    lowercase: bool = True,
    normalize_unicode: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    collapse_spaces: bool = True,
    collapse_newlines: bool = True,
    strip: bool = True,
    keep_space: bool = True,
    min_len: int = 3,
    english_ratio_threshold: float = 0.4,   # used only if keep_lang='en'
    min_word_count: int = 2,                # used only if keep_lang='en'
    verbose: bool = True,
) -> List[str]:
    """
    Clean and normalize raw text with optional language-specific filtering.

    keep_lang options:
      - 'ascii' : English letters, digits, punctuation
      - 'en'    : English-only, validated by dictionary match
      - 'en_zh' : English + Chinese
      - 'all'   : Keep everything

    Returns:
        List[str]: cleaned and filtered text lines.
    """

    cleaned = []
    total = len(texts)

    # ------------------------------------------------------------
    # 1️⃣  Regex setup
    # ------------------------------------------------------------
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    email_pattern = re.compile(r"\S+@\S+\.\S+")
    ascii_pattern = r"A-Za-z0-9\s.,!?;:'\"()\-_\[\]{}@#$%^&*/+=<>~|`\\"
    chinese_pattern = r"\u4e00-\u9fff"
    emoji_pattern = (
        r"\U0001F300-\U0001F5FF"
        r"\U0001F600-\U0001F64F"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F700-\U0001F77F"
        r"\U0001F900-\U0001F9FF"
        r"\U0001FA70-\U0001FAFF"
    )
    math_pattern = r"\u00B1\u00D7\u00F7\u2211\u221A\u03C0\u03A0\u2260\u2264\u2265\u00B0\u222B"

    # Choose allowed character pattern
    if keep_lang in ("ascii", "en"):
        allowed = ascii_pattern
    elif keep_lang == "en_zh":
        allowed = ascii_pattern + chinese_pattern
    else:  # 'all'
        allowed = r"\s\S"  # everything

    if keep_emojis_math:
        allowed += emoji_pattern + math_pattern
    allowed_re = re.compile(f"[{allowed}]+")

    word_pattern = re.compile(r"[A-Za-z]+")  # used for English filtering

    # ------------------------------------------------------------
    # 2️⃣  Main cleaning loop
    # ------------------------------------------------------------
    for line in texts:
        if not isinstance(line, str):
            continue
        text = line

        # Unicode normalization
        if normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Remove URLs/emails
        if remove_urls:
            text = url_pattern.sub("", text)
        if remove_emails:
            text = email_pattern.sub("", text)

        # Character filtering
        text = "".join(allowed_re.findall(text))

        # Lowercase (English)
        if lowercase:
            text = text.lower()

        # Collapse spaces / newlines
        if collapse_spaces:
            text = re.sub(r"[ \t]+", " ", text)
        if collapse_newlines:
            text = re.sub(r"[\r\n]+", "\n", text)

        # Keep or remove spaces
        if not keep_space:
            text = text.replace(" ", "")

        # Trim
        if strip:
            text = text.strip()
        if len(text) < min_len:
            continue

        # --------------------------------------------------------
        # 🧠  English-only dictionary filtering
        # --------------------------------------------------------
        if keep_lang == "en":
            tokens = word_pattern.findall(text)
            if len(tokens) < min_word_count:
                continue
            english_count = sum(1 for t in tokens if t.lower() in EN_WORDS)
            ratio = english_count / len(tokens)
            if ratio < english_ratio_threshold:
                continue

        cleaned.append(text)

    if verbose:
        print(f"✅ clean_texts: kept {len(cleaned):,}/{total:,} lines "
              f"({len(cleaned)/max(total,1)*100:.2f}% retained)")

    return cleaned

# --- Tokenizers ---
#add_eos=True: Keeps each training word independent (great for word lists).
#You’re training on a continuous text corpus (like WikiText, Tiny Shakespeare, or BooksCorpus).
#→ In this case, the model should not reset between words — it should learn continuous dependencies.
class CharTokenizer:
    def __init__(self, texts: List[str], add_eos=True, add_pad=True):
        # Build vocabulary from characters
        vocab = sorted(set("".join(texts)))
        if add_eos and "<eos>" not in vocab:
            vocab.append("<eos>")
        if add_pad and "<pad>" not in vocab:
            vocab.append("<pad>")

        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.pad_token = "<pad>" if add_pad else None
        self.pad_idx = self.stoi.get("<pad>", None)
        self.eos_idx = self.stoi.get("<eos>", None)

    def encode(self, s: str) -> List[int]:
        ids = [self.stoi[c] for c in s if c in self.stoi]
        if self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(
            self.itos[i]
            for i in ids
            if self.itos[i] not in ("<pad>", "<eos>")
        )
        # chars = []
        # for i in ids:
        #     ch = self.itos[i]
        #     if ch == "<eos>": break
        #     chars.append(ch)
        # return "".join(chars)
    
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

    # expose alias
    @property
    def tokenizer(self):
        """Return the underlying HF tokenizer (for compatibility)."""
        return self.tok

    def encode(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
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
    Simple sequential dataset for LM and typing tasks.

    * teacher-forced mode → standard language model training.
        x = ids[i:i+T]
        y = ids[i+1:i+T+1]
        ⇒ shapes [T]

    * final-token mode → random prefix → single next-token target.
    """
    def __init__(self, ids, seq_len, mode="teacher-forced"):
        super().__init__()
        assert mode in ("teacher-forced", "final-token")
        self.mode = mode
        self.seq_len = seq_len

        # Detect dataset shape
        if len(ids) == 0:
            raise ValueError("Empty id list.")
        if isinstance(ids[0], list):
            # already chunked sequences: List[List[int]]
            self.chunked = True
        else:
            # flat stream: List[int]
            self.chunked = False
        self.ids = ids

    def __len__(self):
        if self.chunked:
            return len(self.ids)
        if self.mode == "teacher-forced":
            return max(0, len(self.ids) - self.seq_len - 1)
        return max(0, len(self.ids) - 1)

    def __getitem__(self, idx):
        # --- pre-chunked sequences ---
        if self.chunked:
            seq = self.ids[idx]
            if not all(isinstance(x, int) for x in seq):
                raise TypeError(f"Nested list detected at idx={idx}")
            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            return x, y

        # --- flat token stream ---
        if self.mode == "teacher-forced":
            x = torch.tensor(self.ids[idx: idx+self.seq_len], dtype=torch.long)
            y = torch.tensor(self.ids[idx+1: idx+self.seq_len+1], dtype=torch.long)
            return x, y
        else:
            L = random.randint(1, min(self.seq_len, idx+1))
            start = idx + 1 - L
            prefix = torch.tensor(self.ids[start:start+L], dtype=torch.long)
            target = torch.tensor([self.ids[start+L]], dtype=torch.long)
            return prefix, target

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def collate_teacher(batch, pad_token_id: int = 0):
    """
    Collate (x,y) pairs for teacher-forced or typing datasets.

    Each x,y : 1-D LongTensor [L]
    Returns:
        x_batch [B,Lmax], y_batch [B,Lmax], lengths [B]
    """
    xs, ys = zip(*batch)
    assert all(x.ndim == 1 for x in xs), "inputs must be 1-D"
    assert all(y.ndim == 1 for y in ys), "targets must be 1-D"

    variable_len = len(set(len(x) for x in xs)) > 1
    if not variable_len:
        x_batch = torch.stack(xs)              # [B,T]
        y_batch = torch.stack(ys)
        lengths = torch.full((len(xs),), x_batch.size(1), dtype=torch.long)
    else:
        lengths = torch.tensor([len(x) for x in xs])
        x_batch = pad_sequence(xs, batch_first=True, padding_value=pad_token_id)
        y_batch = pad_sequence(ys, batch_first=True, padding_value=pad_token_id)
        # truncate to same max_len
        max_len = lengths.max().item()
        x_batch, y_batch = x_batch[:, :max_len], y_batch[:, :max_len]

    return x_batch, y_batch, lengths
def collate_final(batch, pad_token_id: int = 0):
    """
    Collate for final-token / typing-prefix prediction.
    Each element (x,y) → x:[L], y:[K]
    Returns x_padded[B,Lmax], y_padded[B,Kmax], x_lengths[B]
    """
    xs, ys = zip(*batch)
    assert all(x.ndim == 1 for x in xs), "inputs must be 1-D"
    assert all(y.ndim == 1 for y in ys), "targets must be 1-D"

    x_lengths = torch.tensor([len(x) for x in xs])
    y_lengths = torch.tensor([len(y) for y in ys])
    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_token_id)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_token_id)
    return x_padded[:, :x_lengths.max()], y_padded[:, :y_lengths.max()], x_lengths


# ============================================================
# TypingDataset: Reuses SequenceDataset for predictive typing
# ============================================================
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

class TypingDataset(Dataset):
    """
    Predictive typing dataset.
      • loads text from HF dataset or text files
      • trains / loads byte-level BPE tokenizer
      • returns random prefix  → next N tokens target

    Returns:
        x : 1-D LongTensor [L≤seq_len]
        y : 1-D LongTensor [K≤next_window]
    """
    def __init__(
        self,
        hf_name: str | None = None,
        txt_files: list[str] | None = None,
        seq_len: int = 128,
        vocab_size: int = 8000,
        min_freq: int = 2,
        cache_dir: str = "tokenizer_cache",
        num_prefixes_per_sentence: int = 3,
        next_window: int = 5,
    ):
        super().__init__()
        import os
        self.seq_len, self.next_window = seq_len, next_window
        os.makedirs(cache_dir, exist_ok=True)
        self.tokenizer_path = f"{cache_dir}/byte_bpe_tokenizer.json"

        # --- Load text ---
        texts = []
        if hf_name:
            print(f"📚 Loading HF dataset: {hf_name}")
            ds = load_dataset(hf_name, split="train")
            if "text" in ds.features:
                texts = ds["text"]
            elif "dialog" in ds.features:
                for d in ds["dialog"]:
                    texts.extend(turn for turn in d)
        if txt_files:
            for fpath in txt_files:
                with open(fpath, "r", encoding="utf-8") as f:
                    texts.extend(f.readlines())
        print(f"✅ Loaded {len(texts):,} total sentences")

        # --- Tokenizer ---
        from tokenizers import Tokenizer, ByteLevelBPETokenizer
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            tok = ByteLevelBPETokenizer()
            tok.train_from_iterator(
                texts, vocab_size=vocab_size, min_frequency=min_freq,
                special_tokens=["<pad>", "<bos>", "<eos>"]
            )
            tok.save(self.tokenizer_path)
            self.tokenizer = tok
            print(f"💾 Saved tokenizer → {self.tokenizer_path}")

        # --- Encode corpus ---
        self.encoded_data = []
        for t in texts:
            ids = self.tokenizer.encode(t.strip()).ids
            if len(ids) >= 3:
                self.encoded_data.extend(ids + [self.tokenizer.token_to_id("<eos>")])
        print(f"🧠 Total encoded tokens: {len(self.encoded_data):,}")

        # --- Underlying sequence dataset ---
        self.dataset = SequenceDataset(self.encoded_data, seq_len=seq_len, mode="teacher-forced")

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        # base x,y of shape [T]
        x, y = self.dataset[idx]
        # simulate partial prefix typing
        prefix_len = torch.randint(low=5, high=min(len(x), self.seq_len), size=(1,)).item()
        x = x[:prefix_len]
        y = y[prefix_len : prefix_len + self.next_window]
        # ensure both 1-D
        assert x.ndim == 1 and y.ndim == 1, "TypingDataset must return 1-D tensors"
        return x, y
    
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import random
import re

class DictionaryTypingDataset(Dataset):
    """
    Typing dataset built from English dictionary (word + definition).

    Each sample is a (prefix → next few chars) pair, mimicking
    partial typing across a word or phrase boundary.

    Example:
        x='progr'  →  y='amming is the process of writing...'
    """

    def __init__(
        self,
        hf_name: str = "npvinHnivqn/EnglishDictionary",
        seq_len: int = 64,
        next_window: int = 5,
        max_prefix_len: int = 12,
        lowercase: bool = True,
        vocab_chars: str | None = None,
        min_word_len: int = 3,
        min_def_len: int = 10,
    ):
        super().__init__()

        print(f"📚 Loading dictionary dataset: {hf_name}")
        ds = load_dataset(hf_name, split="train")

        # detect columns dynamically
        if "word" in ds.features:
            word_field = "word"
        else:
            word_field = list(ds.features.keys())[0]
        definition_field = "definition" if "definition" in ds.features else None

        # --- Build combined corpus ---
        entries = []
        for ex in ds:
            word = str(ex.get(word_field, "")).strip()
            definition = str(ex.get(definition_field, "")).strip() if definition_field else ""
            if len(word) < min_word_len or len(definition) < min_def_len:
                continue
            combined = f"{word}: {definition}"
            entries.append(combined.lower() if lowercase else combined)

        print(f"✅ Loaded {len(entries):,} dictionary entries.")

        # --- Build character vocabulary ---
        if vocab_chars is None:
            vocab_chars = sorted(set("".join(entries)))
        self.vocab = {ch: i + 1 for i, ch in enumerate(vocab_chars)}  # 0 = pad
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        self.pad_id = 0
        self.vocab_size = len(self.vocab) + 1
        print(f"🔡 Vocab size: {self.vocab_size}")

        # --- Generate (prefix, target) examples ---
        examples = []
        for text in entries:
            if len(text) < next_window + 2:
                continue
            # choose random prefix within limits
            prefix_len = random.randint(1, min(max_prefix_len, len(text) - next_window))
            prefix = text[:prefix_len]
            target = text[prefix_len: prefix_len + next_window]
            examples.append((prefix, target))

        self.examples = examples
        print(f"✅ Built {len(self.examples):,} typing pairs (prefix → next).")

        self.seq_len = seq_len
        self.next_window = next_window

    def encode(self, text: str):
        """Convert text string to list[int]."""
        return [self.vocab.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]):
        """Convert list[int] to text."""
        return "".join(self.inv_vocab.get(i, "") for i in ids if i != self.pad_id)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prefix, target = self.examples[idx]
        x = torch.tensor(self.encode(prefix), dtype=torch.long)
        y = torch.tensor(self.encode(target), dtype=torch.long)
        return x, y
    
# ============================================================
# Hugging Face Dataset Wrapper for Encoder–Decoder Training
# ============================================================
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ============================================================
# Seq2Seq Dataset (for encoder–decoder Transformers)
# ============================================================
class Seq2SeqDataset(Dataset):
    """
    Dataset for encoder–decoder Transformer training.
    Each item is (src_input, tgt_input, tgt_output):
        - src_input: tokenized source text (input to encoder)
        - tgt_input: tokenized target text, shifted right (input to decoder)
        - tgt_output: tokenized target text, no shift (training labels)
    """

    def __init__(self, src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, seq_len=128):
        assert len(src_texts) == len(tgt_texts), "Source and target sizes must match."
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # Tokenize source and target
        src = self.tokenizer_src.encode(self.src_texts[idx])[: self.seq_len]
        tgt = self.tokenizer_tgt.encode(self.tgt_texts[idx])[: self.seq_len]

        # Decoder input is target shifted right, output is original target
        tgt_input = [self.tokenizer_tgt.stoi["<eos>"]] + tgt[:-1] if "<eos>" in self.tokenizer_tgt.stoi else tgt[:-1]
        tgt_output = tgt

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long),
        )
        
# ============================================================
# Modern Seq2Seq DataModule for Hugging Face translation datasets
# ============================================================

class Seq2SeqDataModuleHF:
    """
    Modern DataModule for encoder–decoder (Seq2Seq) datasets.

    Supports multilingual datasets (e.g. WMT, OPUS, etc.)
    and automatic config detection with optional user override.
    """

    def __init__(
        self,
        dataset_repo: str,
        seq_len: int = 128,
        batch_size: int = 16,
        hf_split: str = "train",
        hf_config: str = None,          # ✅ new optional config
        src_lang: str = "en",
        tgt_lang: str = "zh",
    ):
        self.dataset_repo = dataset_repo
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.hf_config = hf_config

        print(f"📚 Loading HF dataset: {dataset_repo}")

        # 1️⃣ Detect available configs
        configs = []
        try:
            configs = get_dataset_config_names(dataset_repo)
        except Exception:
            configs = []

        if configs:
            print(f"🧩 Found configs: {configs}")

        # 2️⃣ Determine config to use
        cfg_name = hf_config if hf_config else (configs[0] if configs else None)
        print(f"✅ Using dataset config: '{cfg_name or 'default'}'")

        # 3️⃣ Load dataset
        try:
            self.dataset = load_dataset(dataset_repo, cfg_name)
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load dataset '{dataset_repo}' (config={cfg_name}).\nReason: {e}"
            )

        # 4️⃣ Select split
        split = hf_split if hf_split in self.dataset else list(self.dataset.keys())[0]
        print(f"📗 Using split: '{split}'")
        self.ds_split = self.dataset[split]

        # 5️⃣ Detect translation field
        self.translation_field = None
        if "translation" in self.ds_split.features:
            self.translation_field = "translation"
        else:
            # find nested translation-like field
            for k, feat in self.ds_split.features.items():
                if isinstance(feat, dict) or (
                    hasattr(feat, "feature") and "string" in str(feat.feature)
                ):
                    self.translation_field = k
                    break

        if not self.translation_field:
            raise ValueError(
                f"❌ Could not find translation field in dataset: {dataset_repo}"
            )

        print(f"🔤 Using translation field: '{self.translation_field}'")

        # 6️⃣ Initialize tokenizers
        self._prepare_tokenizers()

    # ------------------------------------------------------------
    def _prepare_tokenizers(self):
        """Simple byte-level BPE tokenizer (can be replaced with HF tokenizers)."""
        from tokenizers import ByteLevelBPETokenizer
        print("🧠 Preparing tokenizers (Byte-Level BPE shared)...")
        self.src_tok = ByteLevelBPETokenizer()
        self.tgt_tok = ByteLevelBPETokenizer()
        print("✅ Tokenizers ready.")

    # ------------------------------------------------------------
    def _encode(self, text_list, tokenizer, pad_id=0):
        """Encode texts and pad to uniform length."""
        encoded = [torch.tensor(tokenizer.encode(t).ids[:self.seq_len]) for t in text_list]
        lengths = torch.tensor([len(x) for x in encoded])
        padded = pad_sequence(encoded, batch_first=True, padding_value=pad_id)
        return padded, lengths

    # ------------------------------------------------------------
    def _collate_fn(self, batch):
        """Collate (src, tgt_in, tgt_out, src_len, tgt_len)."""
        src_texts = [ex[self.translation_field][self.src_lang] for ex in batch]
        tgt_texts = [ex[self.translation_field][self.tgt_lang] for ex in batch]

        src_enc, src_len = self._encode(src_texts, self.src_tok)
        tgt_enc, tgt_len = self._encode(tgt_texts, self.tgt_tok)

        # Shifted decoder input/output
        tgt_in = tgt_enc[:, :-1]
        tgt_out = tgt_enc[:, 1:]

        return src_enc, tgt_in, tgt_out, src_len, tgt_len

    # ------------------------------------------------------------
    def loaders(self):
        """Return train/validation DataLoaders."""
        dl_train = DataLoader(
            self.ds_split, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn
        )
        dl_valid = DataLoader(
            self.ds_split, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn
        )
        print(f"✅ Seq2Seq DataLoaders ready ({len(self.ds_split)} samples).")
        return dl_train, dl_valid

def collate_seq2seq(batch, pad_idx=0):
    """
    Pads src, tgt_input, and tgt_output sequences to the same batch length.
    Returns (src_padded, tgt_in_padded, tgt_out_padded, src_lengths, tgt_lengths)
    """
    srcs, tgts_in, tgts_out = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in srcs])
    tgt_lengths = torch.tensor([len(t) for t in tgts_in])

    src_max = src_lengths.max().item()
    tgt_max = tgt_lengths.max().item()

    src_padded = torch.stack([F.pad(s, (0, src_max - len(s)), value=pad_idx) for s in srcs])
    tgt_in_padded = torch.stack([F.pad(t, (0, tgt_max - len(t)), value=pad_idx) for t in tgts_in])
    tgt_out_padded = torch.stack([F.pad(t, (0, tgt_max - len(t)), value=pad_idx) for t in tgts_out])

    return src_padded, tgt_in_padded, tgt_out_padded, src_lengths, tgt_lengths

    
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
import random
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """
    Configuration dataclass for building datasets in the unified data pipeline.

    This structure controls *what data is loaded*, *how it is tokenized*,
    and *how it is structured* into training samples.

    It is used by `DataModule` and `build_dataset()` to initialize:
      - text corpora (local or Hugging Face datasets)
      - tokenization parameters
      - dataset slicing for LM, Typing, or Seq2Seq tasks
      - batching behavior for DataLoaders

    Each parameter is grouped and explained below.
    """

    # ============================================================
    # 🧱  Data Source Parameters
    # ============================================================

    files: Optional[List[str]] = None
    """
    List of local text file paths to load as the raw corpus.
    If specified, these files will be read line-by-line and joined.
    Example: ["data/wiki.txt", "data/books.txt"]

    NOTE: Ignored if `hf_name` is provided.
    """

    hf_name: Optional[str] = None
    """
    Hugging Face dataset repository name.
    Example: 'OpenAssistant/oasst1', 'wikitext', or 'Helsinki-NLP/opus-zh-en'

    When provided, the dataset will be automatically downloaded via
    `datasets.load_dataset(hf_name, hf_config)`.
    """

    hf_config: Optional[str] = None
    """
    Optional dataset configuration name for multi-config HF datasets.
    Example: 'wikitext-2-raw-v1', 'zh-en'
    """

    hf_split: str = "train"
    """
    Which dataset split to load ('train', 'validation', or 'test').
    Default: 'train'
    """

    # ============================================================
    # 🔤  Tokenizer Configuration
    # ============================================================

    tokenizer: str = "char"
    """
    Type of tokenizer to use.
    Options:
      - 'char': Character-level tokenizer (fast, small vocab)
      - 'word': Whitespace-based word tokenizer
      - 'bpe':  Custom Byte-Level BPE tokenizer (learned from text)
      - 'hf:<model>': Use pretrained Hugging Face tokenizer
                      Example: 'hf:Qwen/Qwen2.5-3B'
    """

    vocab_size: int = 8000
    """
    Target vocabulary size for trainable tokenizers (BPE/word-level).
    Ignored for HF tokenizers (their vocab is fixed).
    """

    lowercase: bool = False
    """
    If True, all text is lowercased before tokenization.
    Useful for small char/word-level models; unnecessary for HF tokenizers.
    """
    keep_emojis_math: bool = True
    keep_lang: str = "en_zh"

    # ============================================================
    # 🎯  Task Configuration
    # ============================================================

    task: str = "lm"
    """
    Defines how the dataset will be structured:
      - 'lm':      Standard causal language modeling (next-token prediction)
      - 'typing':  Predictive typing (prefix → next few tokens)
      - 'seq2seq': Encoder–decoder tasks (translation, summarization, etc.)
    """

    seq_len: int = 256
    """
    Maximum token sequence length per sample.
    Determines both model input length and memory footprint.
    Example: 128 (fast), 512 (standard), 1024+ (long context)
    """

    batch_size: int = 64
    """
    Number of samples per training batch.
    Can be reduced if GPU memory is limited or increased for faster convergence.
    """

    split_ratio: float = 0.9
    """
    Fraction of the dataset used for training (rest for validation).
    Example: 0.9 → 90% train / 10% val
    """

    mode: str = "teacher-forced"
    """
    Loss computation strategy for sequence models:
      - 'teacher-forced': Predict every next token (standard LM training)
      - 'final-token':    Predict only the final next token (prefix completion)
    """

    # ============================================================
    # ⌨️  Typing-Specific Parameters
    # ============================================================

    num_prefixes_per_sentence: int = 3
    """
    (Typing task only)
    Number of random prefix–next pairs generated from each sentence.
    Example: For "I love apples", we might sample prefixes:
        "I", "I love", "I love app"
    """

    next_token_window: int = 5
    """
    (Typing task only)
    Number of future tokens predicted after each prefix.
    Example: Given prefix "I love", model predicts the next 5 tokens.
    """

    # ============================================================
    # ⚙️  Optional Advanced Parameters (for LM)
    # ============================================================

    stride: Optional[int] = None
    """
    Step size between adjacent LM windows (for sliding-window datasets).
    Example:
      seq_len=512, stride=256 → 50% overlap between samples.
    If None, defaults to seq_len (non-overlapping).
    """

    max_tokens: Optional[int] = None
    """
    Cap on total tokens encoded from the corpus.
    Use this to limit dataset size for faster experimentation.
    Example: 2_000_000 → encode only first 2M tokens.
    """

    max_train_samples: Optional[int] = None
    """
    Limit on total number of training sequences created after tokenization.
    Useful for debugging or running small-scale experiments.
    """

    encode_batch_size: int = 1000
    """
    (For Hugging Face tokenizers)
    Number of text lines processed per batch when encoding text.
    Higher values = faster but uses more memory.
    """

    chunk_size: int = 50_000
    """
    (For non-HF tokenizers)
    Number of lines joined and encoded together as one chunk.
    Reduces tokenizer overhead for large corpora.
    """

class DataModule:
    """
    Universal DataModule for:
      - Causal language modeling (LM)
      - Predictive typing (prefix → next tokens)

    Unified logic flow:
      1️⃣ Setup tokenizer
      2️⃣ Load & prepare raw dataset text
      3️⃣ Encode efficiently (in chunks or via HF tokenizer)
      4️⃣ Build training and validation splits
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        # --- 1️⃣ Load raw text ---
        raw_text = self._load_text()
        if not isinstance(raw_text, str):
            raise ValueError("❌ _load_text() must return a single string.")

        # --- 2️⃣ Clean & normalize text ---
        print("🧹 Cleaning raw text ...")

        # Split into lines for cleaning
        lines = [l for l in raw_text.split("\n") if l.strip()]

        # Apply cleaning (you can tune keep_lang etc.)
        cleaned_lines = clean_texts(
            lines,
            keep_lang=getattr(cfg, "keep_lang", "en_zh"),
            keep_emojis_math=getattr(cfg, "keep_emojis_math", True),
            lowercase=getattr(cfg, "lowercase", True),
            keep_space=True,
            min_len=3,
        )

        # Re-join into one long text block for tokenization
        cleaned_text = "\n".join(cleaned_lines)
        print(f"✅ Cleaned text length: {len(cleaned_text):,} chars "
              f"({len(cleaned_lines):,} lines kept)")

        self.text = cleaned_text

        # --- 2️⃣ Setup tokenizer ---
        self._setup_tokenizer(cleaned_text)

        # --- 3️⃣ Build dataset ---
        if cfg.task == "typing":
            self.train_dataset, self.valid_dataset = self._build_typing_dataset()
        else:
            self._build_lm_dataset(cleaned_text)

    # ------------------------------------------------------------
    def _load_text(self) -> str:
        """Load text from local files or Hugging Face Hub datasets (auto-config)."""

        # --- 1️⃣ Local files ---
        if self.cfg.files:
            text = ""
            for p in self.cfg.files:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"❌ File not found: {p}")
                with open(p, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"
            print(f"📄 Loaded {len(self.cfg.files)} local text file(s).")
            return text

        # --- 2️⃣ Hugging Face datasets ---
        elif self.cfg.hf_name:
            repo_path = self.cfg.hf_name.strip()
            print(f"📚 Loading Hugging Face dataset: '{repo_path}'")

            # 🔍 Try to detect dataset configs
            try:
                configs = get_dataset_config_names(repo_path)
            except Exception as e:
                configs = []
                print(f"⚠️ Could not retrieve configs for '{repo_path}': {e}")

            cfg_name = None
            if self.cfg.hf_config and configs and self.cfg.hf_config in configs:
                cfg_name = self.cfg.hf_config
                print(f"✅ Using specified config: '{cfg_name}'")
            elif configs:
                cfg_name = configs[0]  # pick first available config
                print(f"🧩 Detected configs: {configs}")
                print(f"✅ Using first available config: '{cfg_name}'")
            else:
                raise ValueError(f"❌ Specified config '{self.cfg.hf_config}' not found and no other configs available: {configs}")
                

            # Try loading dataset (with config if available)
            try:
                ds = load_dataset(repo_path, cfg_name) if cfg_name else load_dataset(repo_path)
            except Exception as e:
                raise RuntimeError(
                    f"❌ Failed to load dataset '{repo_path}' (config={cfg_name}).\nReason: {e}"
                )

            # Select split
            split = self.cfg.hf_split if self.cfg.hf_split in ds else list(ds.keys())[0]
            print(f"📗 Using split: '{split}'")
            split_data = ds[split]

            # --- Prefer 'text' field ---
            if "text" in split_data.features:
                text_field = "text"
                print(f"🔤 Using text field: '{text_field}'")
            else:
                # fallback to auto-detect string field
                candidate_fields = [
                    k for k, feat in split_data.features.items()
                    if isinstance(feat, Value) and feat.dtype == "string"
                ]
                if not candidate_fields:
                    print(f"⚠️ No direct 'text' field found. Using first available field.")
                    candidate_fields = [list(split_data.features.keys())[0]]
                text_field = candidate_fields[0]
                print(f"🔍 Using detected text field: '{text_field}'")

            # --- Extract text samples ---
            raw_texts = split_data[text_field]
            if isinstance(raw_texts[0], dict):
                key = "en" if "en" in raw_texts[0] else list(raw_texts[0].keys())[0]
                raw_texts = [entry[key] for entry in raw_texts if isinstance(entry, dict) and key in entry]
                print(f"🌐 Flattened translation data using key: '{key}'")
            elif isinstance(raw_texts[0], list):
                raw_texts = [" ".join(turn for turn in dialog if isinstance(turn, str)) for dialog in raw_texts]
                print(f"💬 Flattened list-type dialogues into text strings.")

            text = "\n".join(raw_texts)
            print(f"✅ Loaded {len(raw_texts):,} samples from '{repo_path}' (config={cfg_name})")
            return text

        # --- 3️⃣ No source provided ---
        else:
            raise ValueError("❌ Provide either `files` or `hf_name` in DataConfig.")
    
    
    # ============================================================
    # TOKENIZER SETUP
    # ============================================================
    def _setup_tokenizer(self, raw_text: str):
        """Setup tokenizer based on cfg.tokenizer."""
        print(f"🔤 Setting up tokenizer: {self.cfg.tokenizer}")
        if self.cfg.tokenizer == "char":
            self.tok = CharTokenizer([raw_text])
            self.pad_id = 0

        elif self.cfg.tokenizer == "word":
            self.tok = WordTokenizer([raw_text])
            self.pad_id = 0

        elif self.cfg.tokenizer.startswith("hf:"):
            name = self.cfg.tokenizer.split("hf:")[1]
            self.tok = HFTokenizerWrapper(name)
            self.pad_id = self.tok.pad_id or 0

        elif self.cfg.tokenizer == "bpe":
            self._setup_bpe_tokenizer(raw_text)

        else:
            raise ValueError(f"❌ Unknown tokenizer type: {self.cfg.tokenizer}")

        self.vocab_size = self.tok.vocab_size
        print(f"✅ Tokenizer initialized | vocab_size={self.vocab_size} | pad_id={self.pad_id}")

    # ============================================================
    # BYTE-LEVEL BPE TOKENIZER
    # ============================================================
    def _setup_bpe_tokenizer(self, text):
        """Train or load byte-level BPE tokenizer (emoji + Unicode support)."""
        from tokenizers import ByteLevelBPETokenizer
        cache_dir = "tokenizer_cache"
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer_path = os.path.join(cache_dir, "byte_bpe_tokenizer.json")

        if os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer
            print(f"🔁 Loading existing Byte-BPE tokenizer from {tokenizer_path}")
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            print(f"🧩 Training new Byte-Level BPE tokenizer (vocab_size={self.cfg.vocab_size})...")
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator(
                [text],
                vocab_size=self.cfg.vocab_size,
                min_frequency=2,
                special_tokens=["<pad>", "<bos>", "<eos>"],
            )
            tokenizer.save(tokenizer_path)
            print(f"💾 Saved Byte-BPE tokenizer to {tokenizer_path}")

        class BPETokWrapper:
            def __init__(self, tok):
                self.tok = tok
                self.pad_id = tok.token_to_id("<pad>") or 0
                self.vocab_size = tok.get_vocab_size()
            def encode(self, s): return self.tok.encode(s).ids
            def decode(self, ids): return self.tok.decode(ids)

        self.tok = BPETokWrapper(tokenizer)
        self.pad_id = self.tok.pad_id

    # ============================================================
    # ENCODING UTILITIES
    # ============================================================
    def _encode_in_chunks(self, text: str, chunk_size: int = 50_000, max_tokens: int | None = None):
        """
        Encode very long raw text safely using a non-HF tokenizer (char/BPE).

        Args:
            text: full raw corpus
            chunk_size: number of *lines* to join per chunk
            max_tokens: optional hard limit on total tokens returned
        """
        from tqdm import tqdm
        print(f"🧩 Encoding text in chunks (chunk_size={chunk_size}) ...")

        ids: list[int] = []
        lines = [l for l in text.split("\n") if l.strip()]      # drop empties
        for i in tqdm(range(0, len(lines), chunk_size), desc="Encoding"):
            chunk = "\n".join(lines[i:i + chunk_size])
            try:
                chunk_ids = self.tok.encode(chunk)
            except Exception as e:
                print(f"⚠️  Encoding failed at chunk {i}: {e}")
                continue
            ids.extend(chunk_ids)
            if max_tokens and len(ids) >= max_tokens:
                ids = ids[:max_tokens]
                print(f"🧱 Reached token cap: {max_tokens:,}")
                break

        print(f"✅ Encoded {len(ids):,} tokens total")
        return ids


    def _encode_with_hf_tokenizer(self, text: str, batch_size: int = 1_000, max_tokens: int | None = None):
        """
        Encode raw text using a Hugging Face tokenizer in batched mode.

        ✅ This function is optimized for long corpora (millions of tokens).
        ✅ Automatically flattens nested token lists and validates output integrity.

        Expected behavior:
            Input:
                text: long string of text data (may contain newlines)
            Output:
                A single flattened list[int] of token IDs suitable for LM datasets.

        Example shapes:
            - Before encoding:
                len(text.split("\\n")) → N lines
            - After tokenizer:
                batch_enc: List[List[int]] of shape [batch_size, variable_length]
            - After flatten:
                all_ids: List[int] (flat sequence of token IDs)

        Args:
            text (str): Raw text corpus (already loaded or from HF dataset).
            batch_size (int): Number of lines to encode at once (speed vs. memory tradeoff).
            max_tokens (Optional[int]): Optional cap on total number of encoded tokens.

        Returns:
            all_ids (List[int]): Flattened list of encoded token IDs.
        """
        from tqdm import tqdm

        print("🧩 Encoding text with Hugging Face tokenizer (batched)...")

        # -----------------------------------------------------------
        # 1️⃣  Preprocess input text into lines
        # -----------------------------------------------------------
        lines = [l for l in text.split("\n") if l.strip()]
        total_lines = len(lines)
        if total_lines == 0:
            raise ValueError("❌ No valid non-empty lines found in text input.")

        tokenizer = self.tok.tokenizer
        all_ids: list[int] = []

        # -----------------------------------------------------------
        # 2️⃣  Encode lines in batches for efficiency
        # -----------------------------------------------------------
        for i in tqdm(range(0, total_lines, batch_size), desc="Tokenizing"):
            batch = lines[i:i + batch_size]

            # Expected: batch_enc = List[List[int]]
            try:
                batch_enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
            except Exception as e:
                print(f"⚠️  Tokenization failed at batch {i}: {e}")
                continue

            # Validate tokenizer output
            if not isinstance(batch_enc, list):
                raise TypeError(f"❌ Unexpected tokenizer output type: {type(batch_enc)} (expected list)")
            if len(batch_enc) == 0:
                print(f"⚠️ Empty batch returned by tokenizer (batch {i}); skipping.")
                continue

            # -------------------------------------------------------
            # 3️⃣  Flatten nested token lists safely
            # -------------------------------------------------------
            # HF tokenizers can sometimes output nested lists: [[ [int] ]]
            for seq_idx, seq in enumerate(batch_enc):
                if not isinstance(seq, list):
                    print(f"⚠️ Unexpected element type at batch {i}, seq {seq_idx}: {type(seq)}; skipping.")
                    continue

                # Handle nested structures: e.g., [[tokens], [tokens]]
                for token_or_list in seq:
                    if isinstance(token_or_list, list):
                        all_ids.extend(token_or_list)
                    elif isinstance(token_or_list, int):
                        all_ids.append(token_or_list)
                    else:
                        print(f"⚠️ Non-integer token detected: {type(token_or_list)} at batch {i}, seq {seq_idx}")

                # Optional early stop if token limit exceeded
                if max_tokens and len(all_ids) >= max_tokens:
                    all_ids = all_ids[:max_tokens]
                    print(f"🧱 Reached token cap: {max_tokens:,}")
                    print(f"✅ Encoded {len(all_ids):,} tokens total")
                    return all_ids

        # -----------------------------------------------------------
        # 4️⃣  Validation & summary
        # -----------------------------------------------------------
        if len(all_ids) == 0:
            raise ValueError("❌ No tokens were generated — check tokenizer or input data.")

        # Sanity checks
        if any(isinstance(x, list) for x in all_ids[:1000]):
            raise TypeError("❌ Nested token lists detected — flattening failed.")
        if not all(isinstance(x, int) for x in all_ids[:1000]):
            raise TypeError("❌ Non-integer token detected in token list.")

        print(f"✅ Encoded {len(all_ids):,} tokens total "
            f"(avg tokens/line ≈ {len(all_ids)//max(total_lines,1)})")
        return all_ids


    # ============================================================
    # LM DATASET
    # ============================================================
    def _build_lm_dataset(self, raw_text):
        """Build LM dataset: encode text, flatten tokens, and create fixed-length windows."""
        print("🧩 Building LM dataset ...")

        # 1️⃣ Encode text into token IDs
        if hasattr(self.tok, "tokenizer"):   # HF tokenizer
            ids = self._encode_with_hf_tokenizer(
                raw_text,
                batch_size=getattr(self.cfg, "encode_batch_size", 1000),
                max_tokens=getattr(self.cfg, "max_tokens", None),
            )
        else:
            ids = self._encode_in_chunks(
                raw_text,
                chunk_size=getattr(self.cfg, "chunk_size", 50_000),
                max_tokens=getattr(self.cfg, "max_tokens", None),
            )

        # 2️⃣ Validate and flatten
        if len(ids) == 0:
            raise ValueError("❌ No tokens produced during encoding.")
        if isinstance(ids[0], list):
            print(f"⚠️  Nested list detected (first element len={len(ids[0])}); flattening ...")
            ids = [t for sub in ids for t in sub]
            print(f"✅ Flattened token list length: {len(ids):,}")

        # 3️⃣ Determine stride safely
        stride = getattr(self.cfg, "stride", None)
        if stride is None or stride <= 0:
            stride = self.cfg.seq_len
        if stride < self.cfg.seq_len:
            print(f"⚙️  Using sliding window stride={stride}")
        else:
            print(f"⚙️  Using non-overlapping windows (stride={stride})")

        # 4️⃣ Build sequences of fixed length
        sequences = []
        for i in range(0, len(ids) - self.cfg.seq_len, stride):
            seq = ids[i:i + self.cfg.seq_len]
            if not all(isinstance(x, int) for x in seq):
                raise TypeError(f"❌ Non-integer token detected at sequence {i}–{i+self.cfg.seq_len}")
            sequences.append(seq)

        print(f"📏 Built {len(sequences):,} sequences of len={self.cfg.seq_len}")
        
        print(f"🧾 Sample types: type(sequences[0])={type(sequences[0])}")
        if isinstance(sequences[0], list):
            print(f"  Inner element type: {type(sequences[0][0])}")
            if isinstance(sequences[0][0], list):
                print("⚠️ Detected double nested lists!")

        # 5️⃣ Split train/val
        split_idx = int(len(sequences) * self.cfg.split_ratio)
        self.train_ids = sequences[:split_idx]
        self.valid_ids = sequences[split_idx:]
        print(f"✅ Dataset split: {len(self.train_ids):,} train | {len(self.valid_ids):,} val sequences")

        # 6️⃣ Validation check
        ex = self.train_ids[0]
        if isinstance(ex, list) and isinstance(ex[0], list):
            raise ValueError("❌ Double-nested list detected in train_ids — dataset will be 3D!")
        if not all(isinstance(x, int) for x in ex):
            raise ValueError("❌ Non-integer elements in sequence — tokenizer output corrupted.")
    
    # ============================================================
    # TYPING DATASET
    # ============================================================
    def _build_typing_dataset(self):
        """
        Construct prefix → next-token dataset for typing task.

        Automatically switches to DictionaryTypingDataset if
        a dictionary-style HF dataset is configured.
        """
        import random
        import torch
        from torch.utils.data import DataLoader

        # ---------------------------------------------------------
        # 1️⃣ Check if this is a Dictionary dataset (e.g., EnglishDictionary)
        # ---------------------------------------------------------
        if (
            self.cfg.hf_name
            and isinstance(self.cfg.hf_name, str)
            and "dictionary" in self.cfg.hf_name.lower()
        ):
            print("📘 Building DictionaryTypingDataset (word + definition)...")

            # Reuse the dedicated dataset builder
            dataset = DictionaryTypingDataset(
                hf_name=self.cfg.hf_name,
                seq_len=self.cfg.seq_len,
                next_window=getattr(self.cfg, "next_token_window", 6),
                max_prefix_len=getattr(self.cfg, "max_prefix_len", 12),
                lowercase=getattr(self.cfg, "lowercase", True),
            )

            # Split 90/10 for train/valid
            total = len(dataset)
            split = int(self.cfg.split_ratio * total)
            train_dataset = torch.utils.data.Subset(dataset, range(0, split))
            valid_dataset = torch.utils.data.Subset(dataset, range(split, total))

            print(
                f"✅ DictionaryTypingDataset ready | "
                f"train={len(train_dataset):,} | valid={len(valid_dataset):,} | "
                f"vocab={dataset.vocab_size}"
            )
            return train_dataset, valid_dataset

        # ---------------------------------------------------------
        # 2️⃣ Standard typing dataset from self.text
        # ---------------------------------------------------------
        print("⌨️ Building Typing dataset from raw text...")
        sentences = self.text.split("\n")
        samples = []
        for s in sentences:
            s = s.strip()
            if len(s) < 3:
                continue
            ids = self.tok.encode(s)
            L = len(ids)
            if L < 3:
                continue

            # Generate multiple prefix→target pairs per sentence
            for _ in range(self.cfg.num_prefixes_per_sentence):
                prefix_len = random.randint(1, max(2, L - 2))
                target_len = min(self.cfg.next_token_window, L - prefix_len)
                if target_len <= 0:
                    continue

                x = torch.tensor(ids[:prefix_len], dtype=torch.long)
                y = torch.tensor(ids[prefix_len:prefix_len + target_len], dtype=torch.long)
                samples.append((x, y))

        print(f"✅ Generated {len(samples):,} prefix→next-token pairs for typing.")
        split = int(self.cfg.split_ratio * len(samples))

        # Convert to Subset-like dataset lists (for collate)
        train_samples = samples[:split]
        valid_samples = samples[split:]

        return train_samples, valid_samples

    # ============================================================
    # DATALOADERS
    # ============================================================
    def loaders(self):
        if self.cfg.task == "typing":
            collate = lambda b: collate_teacher(b, pad_token_id=self.pad_id)
            dl_train = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                                shuffle=True, collate_fn=collate)
            dl_valid = DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size,
                                shuffle=False, collate_fn=collate)
            return dl_train, dl_valid

        # LM
        train_ds = SequenceDataset(self.train_ids, self.cfg.seq_len, self.cfg.mode)
        valid_ds = SequenceDataset(self.valid_ids, self.cfg.seq_len, self.cfg.mode)
        collate = collate_teacher if self.cfg.mode == "teacher-forced" else collate_final
        dl_train = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                            shuffle=True, drop_last=True, collate_fn=collate)
        dl_valid = DataLoader(valid_ds, batch_size=self.cfg.batch_size,
                            shuffle=False, drop_last=False, collate_fn=collate)
        return dl_train, dl_valid


def get_dataset_config_template():
    """
    Return a detailed configuration template for building datasets with explanations.
    Use this to construct or document the `args` Namespace for `build_dataset()`.

    Returns:
        dict: parameter names and their detailed descriptions.
    """
    return {
        "task": "Task type — one of ['lm', 'typing', 'seq2seq'].\n"
                "  - 'lm': standard language modeling\n"
                "  - 'typing': predictive typing (prefix → next)\n"
                "  - 'seq2seq': translation/summarization",

        # -------------------- Data source --------------------
        "hf_name": "Hugging Face dataset name, e.g. 'wikitext' or 'OpenAssistant/oasst1'.",
        "hf_config": "Optional dataset config (e.g., 'wikitext-2-raw-v1').",
        "hf_split": "Split name to use ('train', 'validation', 'test').",
        "files": "List of local text file paths (used if hf_name is None).",

        # -------------------- Tokenization --------------------
        "tokenizer": "Tokenizer type:\n"
                     "  'char' → character-level\n"
                     "  'word' → whitespace-separated words\n"
                     "  'bpe'  → train custom Byte-Level BPE\n"
                     "  'hf:<model>' → use Hugging Face tokenizer (e.g. 'hf:Qwen/Qwen2.5-3B')",
        "vocab_size": "Vocabulary size for BPE or word tokenizers (ignored for HF tokenizers).",
        "lowercase": "If True, lowercase text before tokenization.",

        # -------------------- Dataset structure --------------------
        "seq_len": "Maximum sequence length (tokens per training sample).",
        "batch_size": "Batch size per DataLoader.",
        "split_ratio": "Train/validation split ratio (e.g., 0.9 = 90% train).",
        "mode": "Loss computation mode: 'teacher-forced' or 'final-token'.",

        # -------------------- LM dataset (sliding window) --------------------
        "stride": "Stride (token step) between adjacent LM sequences.\n"
                  "  - = seq_len → non-overlapping\n"
                  "  - < seq_len → overlapping (sliding window)",
        "max_tokens": "Optional maximum number of tokens to encode from corpus (limits dataset size).",
        "max_train_samples": "Optional maximum number of training sequences to keep.",
        "encode_batch_size": "Batch size for Hugging Face tokenization (speed optimization).",
        "chunk_size": "Chunk size (lines) for non-HF tokenization.",

        # -------------------- Typing dataset --------------------
        "num_prefixes_per_sentence": "Number of prefixes generated per sentence (typing task).",
        "next_token_window": "Number of next tokens predicted after each prefix (typing task).",

        # -------------------- Seq2Seq dataset --------------------
        "src_lang": "Source language key (e.g., 'en') for translation datasets.",
        "tgt_lang": "Target language key (e.g., 'zh') for translation datasets.",

        # -------------------- Misc --------------------
        "lowercase": "Convert all text to lowercase (for char/word tokenizers).",
    }
    
# ============================================================
# Unified DATASET INITIALIZATION FUNCTION
# ============================================================
def build_dataset(task: str, args):
    """
    Build dataset dynamically based on the specified task.

    Tasks supported:
      - "lm":       standard causal Language Modeling dataset
      - "typing":   predictive typing dataset (prefix → next few tokens)
      - "seq2seq":  encoder–decoder dataset (translation, summarization, etc.)

    Args:
        task (str): Task type ("lm", "typing", "seq2seq").
        args: Namespace or configuration object with required attributes.

    Returns:
        data: A dataset or DataModule object with `.loaders()` for Trainer.
    """
    from datasets import load_dataset

    # --- 1️⃣  Seq2Seq tasks (encoder–decoder) ---
    if task == "seq2seq":
        if not args.hf_name:
            raise ValueError(
                "--hf_name (e.g., 'Helsinki-NLP/opus-zh-en' or 'wmt19') is required for seq2seq tasks."
            )

        print(f"📘 Building Seq2Seq dataset from Hugging Face repo: {args.hf_name}")
        data = Seq2SeqDataModuleHF(
            dataset_repo=args.hf_name,
            hf_split=getattr(args, "hf_split", "train"),
            hf_config=getattr(args, "hf_config", None),
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            src_lang=getattr(args, "src_lang", "en"),
            tgt_lang=getattr(args, "tgt_lang", "zh"),
        )
        return data

    # --- 2️⃣  Causal LM or Typing tasks ---
    elif task in ["lm", "typing"]:
        cfg = DataConfig(
            files=args.files,
            hf_name=args.hf_name,
            hf_config=getattr(args, "hf_config", None),
            hf_split=getattr(args, "hf_split", "train"),
            tokenizer=getattr(args, "tokenizer", "char"),     # 'char', 'word', 'bpe', or 'hf:<model>'
            vocab_size=getattr(args, "vocab_size", 8000),
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lowercase=getattr(args, "lowercase", False),
            keep_emojis_math=getattr(args, "keep_emojis_math", True),
            keep_lang=getattr(args, "keep_lang", "en_zh"),
            split_ratio=getattr(args, "split_ratio", 0.9),
            mode=getattr(args, "mode", "teacher-forced"),
            task=task,
            # Typing dataset options
            num_prefixes_per_sentence=getattr(args, "num_prefixes_per_sentence", 3),
            next_token_window=getattr(args, "next_token_window", 5),
            # LM dataset options
            stride=getattr(args, "stride", None),             # sliding-window stride
            max_tokens=getattr(args, "max_tokens", None),     # cap total tokens
            max_train_samples=getattr(args, "max_train_samples", None),  # optional subset
            encode_batch_size=getattr(args, "encode_batch_size", 1000),  # HF tokenizer batch size
            chunk_size=getattr(args, "chunk_size", 50_000),   # non-HF tokenizer chunk size
        )

        print(f"📗 Building DataModule for task='{task}' "
              f"(tokenizer={cfg.tokenizer}, seq_len={cfg.seq_len})")

        data = DataModule(cfg)
        return data

    # --- 3️⃣  Unknown task type ---
    else:
        raise ValueError(f"❌ Unknown task type '{task}'. Use 'lm', 'typing', or 'seq2seq'.")

# ============================================================
# DATASET TEST FUNCTION (supports typing, LM, seq2seq, HF)
# ============================================================

import torch
from typing import List, Optional
from collections import Counter
import numpy as np


def test_tokenizer(
    data,
    samples: Optional[List[str]] = None,
    num_samples: int = 3,
    show_tokens: int = 40,
):
    """
    Standalone tokenizer test utility.

    Performs a quick sanity check of the tokenizer used in a dataset.
    Fetches or accepts sample text, tokenizes it, prints token IDs, tokens,
    decoded text, and verifies round-trip consistency.

    Args:
        data: DataModule or object containing `.tok` (wrapper) or `.tokenizer`.
        samples (list[str], optional): custom text samples to test.
        num_samples (int): number of examples to use if pulling from `data.text`.
        show_tokens (int): how many tokens to display before truncating.

    Example:
        >>> data = build_dataset("lm", args)
        >>> test_tokenizer(data)
    """

    print("\n🔍 ===== TOKENIZER SANITY CHECK =====")

    # --- 1️⃣  Locate tokenizer object ---
    tok = getattr(data, "tok", None)
    if tok is None:
        raise ValueError("❌ No tokenizer found in data object (missing `.tok`).")

    # HFTokenizerWrapper exposes .tokenizer property; otherwise use itself
    tokenizer = getattr(tok, "tokenizer", tok)

    print(f"🧩 Tokenizer class: {type(tokenizer).__name__}")
    if hasattr(tokenizer, "name_or_path"):
        print(f"📘 Tokenizer name/path: {tokenizer.name_or_path}")

    # --- 2️⃣  Determine test samples ---
    if samples is None:
        if hasattr(data, "text") and isinstance(data.text, str):
            lines = [l.strip() for l in data.text.split("\n") if l.strip()]
            samples = lines[:num_samples]
        else:
            raise ValueError(
                "❌ No text samples provided and data.text not available."
            )
    else:
        samples = samples[:num_samples]

    print(f"📚 Testing {len(samples)} text samples.\n")

    # --- 3️⃣  Test each sample ---
    for i, s in enumerate(samples):
        print(f"🔹 Sample {i+1}")
        print(f"   Input text: {s[:200]}{'...' if len(s) > 200 else ''}")

        # Tokenize
        try:
            encoded = tokenizer(s, add_special_tokens=False)
            ids = encoded["input_ids"]
        except Exception as e:
            print(f"⚠️  Tokenization failed: {e}")
            continue

        tokens = (
            tokenizer.convert_ids_to_tokens(ids)
            if hasattr(tokenizer, "convert_ids_to_tokens")
            else None
        )

        print(f"   Token IDs  ({len(ids)}): {ids[:show_tokens]}{'...' if len(ids) > show_tokens else ''}")
        if tokens:
            print(f"   Tokens     ({len(tokens)}): {tokens[:show_tokens]}{'...' if len(tokens) > show_tokens else ''}")

        # Decode back
        try:
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"   🔁 Decoded : {decoded[:200]}{'...' if len(decoded) > 200 else ''}")
        except Exception as e:
            print(f"⚠️  Decode failed: {e}")

        print("")

    print("✅ Tokenizer test complete.\n")


import torch
import numpy as np
from itertools import islice
from collections import Counter


def inspect_dataset(data, task="lm", num_batches=1, num_samples=2, show_tokens=40):
    """
    Comprehensive dataset and dataloader inspection utility.

    Works for:
      - DataModule (LM, Typing)
      - Seq2SeqDataModuleHF
      - HFData (Hugging Face datasets)

    Args:
        data: Dataset or DataModule object returned by build_dataset().
        task (str): one of ['lm', 'typing', 'seq2seq', 'hf']
        num_batches (int): number of training batches to preview.
        num_samples (int): number of decoded samples per batch.
        show_tokens (int): number of token IDs to display before truncating.
    """

    print("\n🔍 ===== DATASET INSPECTION =====")
    print(f"🧩 Task type: {task}")

    # -----------------------------------------------------------
    # 1️⃣ Try to get DataLoaders
    # -----------------------------------------------------------
    try:
        loaders = data.loaders()
    except Exception as e:
        print(f"❌ Failed to call data.loaders(): {e}")
        return

    # Handle output formats
    if isinstance(loaders, tuple):
        if len(loaders) == 2:
            dl_train, dl_valid = loaders
            dl_test = None
        elif len(loaders) == 3:
            dl_train, dl_valid, dl_test = loaders
        else:
            print("⚠️ Unexpected number of loaders returned.")
            return
    else:
        print("⚠️ data.loaders() did not return a tuple.")
        return

    print(f"📦 Train batches: {len(dl_train)} | Validation batches: {len(dl_valid)}")

    vocab_size = getattr(data, "vocab_size", None)
    if vocab_size:
        print(f"🔡 Vocabulary size: {vocab_size:,}")

    pad_id = getattr(data, "pad_id", 0)

    # -----------------------------------------------------------
    # 2️⃣ Inspect sample batches
    # -----------------------------------------------------------
    for i, batch in enumerate(islice(dl_train, num_batches)):
        print(f"\n🧩 --- Inspecting train batch {i+1}/{num_batches} ---")

        # ---- Hugging Face dataset (dict format) ----
        if task == "hf" and isinstance(batch, dict):
            print(f"🔹 Batch type: dict")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  • {k}: shape={v.shape}, dtype={v.dtype}")
            if "input_ids" in batch:
                ids = batch["input_ids"][0][:show_tokens].tolist()
                print(f"  • Example input_ids[:{show_tokens}]: {ids}")
            continue

        # ---- General tuple-based datasets ----
        if not isinstance(batch, (list, tuple)):
            print(f"⚠️ Unexpected batch type: {type(batch)}")
            continue

        print(f"🔹 Batch type: tuple (len={len(batch)})")
        for j, t in enumerate(batch):
            if torch.is_tensor(t):
                print(f"  • tensor[{j}]: shape={tuple(t.shape)}, dtype={t.dtype}")

        # Assign common variables
        input_ids = batch[0]
        labels = batch[1] if len(batch) > 1 else None

        # ---- Compute padding / length statistics ----
        if torch.is_tensor(input_ids):
            pad_ratio = (input_ids == pad_id).float().mean().item()
            lengths = (input_ids != pad_id).sum(dim=1)
            print(f"  • Avg length: {lengths.float().mean():.1f} | Min: {lengths.min().item()} | Max: {lengths.max().item()}")
            print(f"  • Pad ratio: {pad_ratio*100:.2f}%  (pad_id={pad_id})")

        if torch.is_tensor(labels):
            unique_labels = torch.unique(labels)
            print(f"  • Label unique tokens: {len(unique_labels)}")
            if vocab_size:
                print(f"  • Label vocab coverage: {len(unique_labels)/vocab_size*100:.2f}% of vocab")

            y_np = labels.cpu().numpy().flatten()
            pad_count = np.sum(y_np == pad_id)
            pad_ratio_y = pad_count / len(y_np)
            print(f"  • Label pad ratio: {pad_ratio_y*100:.2f}%")
            most_common = Counter(y_np).most_common(5)
            print(f"  • Top 5 label IDs: {most_common}")

        # -----------------------------------------------------------
        # 3️⃣ Decode examples (if tokenizer is available)
        # -----------------------------------------------------------
        tok = getattr(data, "tok", None)
        if tok is not None:
            tokenizer = getattr(tok, "tokenizer", tok)
            if hasattr(tokenizer, "decode") and callable(tokenizer.decode):
                print("\n🗣️ Decoded examples:")
                for n in range(min(num_samples, input_ids.size(0))):
                    ids = input_ids[n].cpu().tolist()
                    ids = [x for x in ids if x != pad_id]
                    try:
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                    except TypeError:
                        text = tokenizer.decode(ids)
                    print(f"   {n+1}. {text[:200]}{'...' if len(text) > 200 else ''}")

                    if labels is not None:
                        y_ids = labels[n].cpu().tolist()
                        y_ids = [x for x in y_ids if x != pad_id]
                        try:
                            y_text = tokenizer.decode(y_ids, skip_special_tokens=True)
                        except TypeError:
                            y_text = tokenizer.decode(y_ids)
                        print(f"      → target: {y_text[:200]}{'...' if len(y_text) > 200 else ''}")
            else:
                print("ℹ️ Tokenizer found, but no valid decode() method.")
        else:
            print("ℹ️ No tokenizer attached; skipping decode.")

        # ---- Seq2Seq dataset details ----
        if task == "seq2seq" and len(batch) == 5:
            src, tgt_in, tgt_out, src_len, tgt_len = batch
            print(f"\n🌐 Seq2Seq details:")
            print(f"  • src shape: {src.shape} | tgt_in: {tgt_in.shape} | tgt_out: {tgt_out.shape}")
            print(f"  • src_len avg: {src_len.float().mean():.1f} | tgt_len avg: {tgt_len.float().mean():.1f}")

    # -----------------------------------------------------------
    # 4️⃣ Dataset summary
    # -----------------------------------------------------------
    print("\n📊 ===== DATASET SUMMARY =====")
    try:
        total_train = len(getattr(data, "train_dataset", getattr(dl_train.dataset, [])))
        print(f"  • Training samples: {total_train:,}")
    except Exception:
        print("  • Training samples: Unknown")

    try:
        val_count = len(getattr(data, "valid_dataset", getattr(dl_valid.dataset, [])))
        print(f"  • Validation samples: {val_count:,}")
    except Exception:
        print("  • Validation samples: Unknown")

    print("✅ Dataset inspection complete.\n")
                 
# ============================================================
# FULL DATASET PIPELINE TESTING FUNCTION 
# ============================================================
def run_all_dataset_tests():
    """
    🔬 Comprehensive dataset verification utility.

    Runs and validates all dataset types supported by `build_dataset()`:

      1️⃣ Predictive Typing dataset (Character & Byte-level BPE)
      2️⃣ Standard LM dataset (causal LM)
      3️⃣ Seq2Seq dataset (encoder–decoder translation)
      4️⃣ Hugging Face native dataset (AutoTokenizer)

    Each test:
      - Builds the dataset
      - Runs DataLoader inspection
      - Prints decoded samples & structure stats
    """

    print("\n🚀 ===== Running Complete Dataset Tests =====")

    # --------------------------------------------------------
    # Common Args template
    # --------------------------------------------------------
    class Args:
        def __init__(self):
            # Generic parameters
            self.files = None
            self.hf_name = None
            self.hf_config = None
            self.hf_model_name = None
            self.hf_split = "train"

            # Tokenizer settings
            self.tokenizer = "char"       # default (can override per test)
            self.vocab_size = 8000
            self.lowercase = False
            self.keep_lang = "en" #"ascii"#"en_zh"
            self.keep_emojis_math = False

            # Dataset core parameters
            self.seq_len = 128
            self.batch_size = 32
            self.split_ratio = 0.9
            self.mode = "teacher-forced"

            # Typing task parameters
            self.num_prefixes_per_sentence = 3
            self.next_token_window = 5

            # LM / Seq2Seq defaults
            self.task = "lm"
            self.stride = None
            self.max_tokens = None
            self.max_train_samples = None
            self.encode_batch_size = 1000
            self.chunk_size = 50_000

    # Reusable Args object
    args = Args()

    # --------------------------------------------------------
    # 1️⃣ Typing dataset (Character-level tokenizer)
    # --------------------------------------------------------
    print("\n⌨️ [1A] Testing Typing Dataset (Character-level)...")
    try:
        args.task = "typing"
        args.hf_name = "OpenAssistant/oasst1"
        args.hf_split = "train"
        args.tokenizer = "char"

        data = build_dataset(args.task, args)
        inspect_dataset(data, task="typing", num_batches=2, num_samples=2)
    except Exception as e:
        print(f"❌ Typing (char) dataset test failed: {e}")
    
    print("\n⌨️ [1B] Testing DictionaryTyping Dataset (Character-level)...")
    try:
        args.task = "typing"
        args.hf_name = "npvinHnivqn/EnglishDictionary"
        args.hf_split = "train"
        args.tokenizer = "char"
        args.seq_len = 64
        args.next_token_window = 6
        args.max_prefix_len = 12
        data = build_dataset(args.task, args)
        inspect_dataset(data, task="typing", num_batches=2, num_samples=2)
    except Exception as e:
        print(f"❌ Typing (char) dataset test failed: {e}")

    # --------------------------------------------------------
    # 1️⃣ Typing dataset (Byte-level BPE tokenizer)
    # --------------------------------------------------------
    print("\n⌨️ [1C] Testing Typing Dataset (Byte-level BPE)...")
    try:
        args.task = "typing"
        args.tokenizer = "bpe"
        args.hf_name = "OpenAssistant/oasst1"

        data = build_dataset(args.task, args)
        inspect_dataset(data, task="typing", num_batches=2, num_samples=2)
    except Exception as e:
        print(f"❌ Typing (BPE) dataset test failed: {e}")

    # --------------------------------------------------------
    # 2️⃣ Standard LM dataset (Causal LM)
    # --------------------------------------------------------
    print("\n📘 [2] Testing Standard LM Dataset (Causal LM)...")
    try:
        args.task = "lm"
        args.hf_name = "Salesforce/wikitext"
        args.hf_config = "wikitext-2-raw-v1"   # smaller version (~37k samples)
        args.hf_split = "train"
        args.tokenizer = "hf:gpt2"

        data = build_dataset(args.task, args)
        inspect_dataset(data, task="lm", num_batches=2, num_samples=2)
    except Exception as e:
        print(f"❌ LM dataset test failed: {e}")

    # --------------------------------------------------------
    # 3️⃣ Seq2Seq dataset (Encoder–Decoder)
    # --------------------------------------------------------
    print("\n📗 [3] Testing Seq2Seq Dataset (Encoder–Decoder)...")
    try:
        args.task = "seq2seq"
        args.hf_name = "wmt19"
        args.hf_config = "zh-en"
        args.tokenizer = "hf:gpt2"

        data = build_dataset(args.task, args)
        inspect_dataset(data, task="seq2seq", num_batches=1, num_samples=2)
    except Exception as e:
        print(f"❌ Seq2Seq dataset test failed: {e}")

    # --------------------------------------------------------
    # 4️⃣ Hugging Face native dataset (HF Tokenizer)
    # --------------------------------------------------------
    print("\n🤗 [4] Testing Hugging Face Dataset (AutoTokenizer)...")
    try:
        args.task = "lm"
        args.hf_name = "OpenAssistant/oasst1"
        args.tokenizer = "hf:Qwen/Qwen2.5-3B"

        data = build_dataset(args.task, args)
        inspect_dataset(data, task="hf", num_batches=1, num_samples=2)
    except Exception as e:
        print(f"❌ HF dataset test failed: {e}")

    # --------------------------------------------------------
    # ✅ Summary
    # --------------------------------------------------------
    print("\n✅ ===== All Dataset Tests Completed =====")
    print("  • Typing (char): checked")
    print("  • Typing (BPE):  checked")
    print("  • LM (GPT2 tokenizer): checked")
    print("  • Seq2Seq (WMT19 zh-en): checked")
    print("  • Hugging Face dataset (Qwen2.5-3B): checked\n")
    
if __name__ == "__main__":
    ds = SequenceDataset(list(range(1000)), seq_len=128)
    x,y = ds[0]
    print(x.shape, y.shape)   # torch.Size([128]) torch.Size([128])

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=32, collate_fn=collate_teacher)
    xb, yb, l = next(iter(dl))
    print(xb.shape, yb.shape) # torch.Size([32,128]) torch.Size([32,128])

    run_all_dataset_tests()