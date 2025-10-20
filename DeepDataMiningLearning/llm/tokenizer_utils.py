
from typing import Optional, List, Tuple, Dict

import os
import json
import numpy as np
import re
import unicodedata
from typing import List
from pathlib import Path


# Optional: English word validation set
try:
    import nltk #pip install nltk
    from nltk.corpus import words
    
    # Download once if needed
    # nltk.download('words')

    # Define the custom directory
    custom_dir = os.path.join('sampledata', 'nltk_data')
    # Prepend the new directory to NLTK's search path
    nltk.data.path.insert(0, custom_dir)
    nltk.download("words")

    EN_WORDS = set(w.lower() for w in words.words())
except:
    EN_WORDS = set()  # fallback if nltk words not available
    




def nltk_englishwords():
    custom_dir = os.path.join('sampledata', 'nltk_data')
    # Download the "words" corpus. NLTK will now use the new path.
    #nltk.download("words")
    # Define the full path for the output file
    file_path = custom_dir / "english_words.txt"
    # Open the file using the full path
    with file_path.open("w", encoding="utf-8") as f:
        for w in words.words():
            if w.isalpha() and len(w) > 2:
                f.write(w.lower() + "\n")

    print(f"File saved to: {file_path}")


import os
import json
import string
from typing import List


class CharTokenizer:
    """
    A simple character-level tokenizer.

    Features:
      ✅ Automatically builds vocabulary from input texts
      ✅ Option to include all printable ASCII characters (string.printable)
      ✅ Handles <unk>, <pad>, and <eos> tokens
      ✅ pad_id = -100 by default (matches PyTorch ignore_index)
      ✅ Supports save() / load() to persist vocab across sessions
    """

    def __init__(
        self,
        texts: List[str],
        add_eos: bool = True,
        add_pad: bool = True,
        add_unk: bool = True,
        pad_id: int = -100,
        lowercase: bool = False,
        use_all_printable: bool = False,  # ✅ new option
    ):
        """
        Args:
            texts (List[str]): training text corpus
            add_eos (bool): add <eos> token to vocab
            add_pad (bool): add <pad> token to vocab
            add_unk (bool): add <unk> token to vocab
            pad_id (int): padding token ID (default -100)
            lowercase (bool): convert input text to lowercase
            use_all_printable (bool): include full printable ASCII charset in vocab
        """

        # ------------------------------------------------------------
        # 1️⃣ Normalize input text
        # ------------------------------------------------------------
        if lowercase:
            texts = [t.lower() for t in texts]
        texts = [t.strip() for t in texts]

        # ------------------------------------------------------------
        # 2️⃣ Extract unique characters from corpus
        # ------------------------------------------------------------
        chars = sorted(set("".join(texts)))

        # ------------------------------------------------------------
        # 3️⃣ Optionally add all printable ASCII chars
        # ------------------------------------------------------------
        if use_all_printable:
            all_chars = list(string.printable)  # digits, letters, punctuation, whitespace, etc.
            chars = sorted(set(chars + all_chars))
            print("📚 Added all printable ASCII characters to vocabulary.")

        # ------------------------------------------------------------
        # 4️⃣ Add special tokens
        # ------------------------------------------------------------
        specials = []
        if add_pad:
            specials.append("<pad>")
        if add_eos:
            specials.append("<eos>")
        if add_unk:
            specials.append("<unk>")

        vocab = specials + chars

        # ------------------------------------------------------------
        # 5️⃣ Build mappings
        # ------------------------------------------------------------
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # ------------------------------------------------------------
        # 6️⃣ Store token indices
        # ------------------------------------------------------------
        self.pad_id = pad_id
        self.pad_idx = self.stoi.get("<pad>", pad_id)
        self.eos_idx = self.stoi.get("<eos>", None)
        self.unk_idx = self.stoi.get("<unk>", None)

        self.vocab_size = len(self.stoi)
        self.pad_token = "<pad>" if add_pad else None
        self.use_all_printable = use_all_printable

        print(
            f"✅ CharTokenizer initialized | vocab_size={self.vocab_size} "
            f"| pad_idx={self.pad_idx} | eos_idx={self.eos_idx} | unk_idx={self.unk_idx}"
        )
        if use_all_printable:
            print(f"✅ Printable ASCII characters included (total unique chars: {len(chars)}).")

    # ------------------------------------------------------------
    # 🔹 Encode
    # ------------------------------------------------------------
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """
        Convert text to a list of character token IDs.
        Unknown characters are mapped to <unk>.
        """
        ids = [self.stoi.get(ch, self.unk_idx) for ch in text]
        if add_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    # ------------------------------------------------------------
    # 🔹 Decode
    # ------------------------------------------------------------
    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Convert token IDs back to a string.
        """
        chars = []
        for i in ids:
            ch = self.itos.get(i, "")
            if ch == "<eos>":
                break
            if skip_specials and ch in ("<pad>", "<eos>"):
                continue
            chars.append(ch)
        return "".join(chars)

    # ------------------------------------------------------------
    # 🔹 Save
    # ------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab": self.stoi,
            "pad_id": self.pad_id,
            "pad_idx": self.pad_idx,
            "eos_idx": self.eos_idx,
            "unk_idx": self.unk_idx,
            "use_all_printable": self.use_all_printable,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved CharTokenizer → {path}")

    # ------------------------------------------------------------
    # 🔹 Load
    # ------------------------------------------------------------
    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = CharTokenizer([""], use_all_printable=data.get("use_all_printable", False))
        tok.stoi = data["vocab"]
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        tok.pad_id = data.get("pad_id", -100)
        tok.pad_idx = data.get("pad_idx", -100)
        tok.eos_idx = data.get("eos_idx", None)
        tok.unk_idx = data.get("unk_idx", None)
        tok.vocab_size = len(tok.stoi)
        tok.use_all_printable = data.get("use_all_printable", False)

        print(
            f"✅ Loaded CharTokenizer from {path} | vocab_size={tok.vocab_size} "
            f"| pad_idx={tok.pad_idx} | eos_idx={tok.eos_idx} | unk_idx={tok.unk_idx}"
        )
        if tok.use_all_printable:
            print("✅ Printable ASCII characters are enabled.")
        return tok


class WordTokenizer:
    """
    A robust multilingual word-level tokenizer with flexible options.

    Features:
      ✅ Handles both English and Chinese text
      ✅ Optional: keep or remove punctuation
      ✅ Optional: distinguish between Chinese and English punctuation
      ✅ Adds <pad>, <eos>, and <unk> special tokens
      ✅ Default pad_id = -100 (compatible with PyTorch ignore_index)
      ✅ Optional English vocabulary seeding (from nltk.corpus.words)
      ✅ Save/load support for persistence
    """

    def __init__(
        self,
        texts: List[str],
        lang: str = "en",
        add_eos: bool = True,
        add_pad: bool = True,
        add_unk: bool = True,
        pad_id: int = -100,
        lowercase: bool = True,
        use_english_vocab: bool = False,
        keep_punct: bool = False,           # ✅ whether to keep punctuation as separate tokens
        separate_zh_punct: bool = True,     # ✅ distinguish Chinese vs English punctuation
    ):
        self.lang = lang
        self.add_eos = add_eos
        self.add_pad = add_pad
        self.add_unk = add_unk
        self.pad_id = pad_id
        self.lowercase = lowercase
        self.use_english_vocab = use_english_vocab
        self.keep_punct = keep_punct
        self.separate_zh_punct = separate_zh_punct

        # ------------------------------------------------------------
        # 1️⃣ Normalize text: lowercase and strip whitespace
        # ------------------------------------------------------------
        if lowercase and lang != "zh":
            texts = [t.lower() for t in texts]
        texts = [t.strip() for t in texts]

        # ------------------------------------------------------------
        # 2️⃣ Define punctuation regex patterns
        # ------------------------------------------------------------
        # English punctuation (ASCII)
        EN_PUNCT = r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]"
        # Chinese punctuation (full-width)
        ZH_PUNCT = r"[，。！？、“”‘’《》（）【】……—～：；·]"

        # ------------------------------------------------------------
        # 3️⃣ Tokenize text into words (and optionally punctuation)
        # ------------------------------------------------------------
        tokens = []
        for t in texts:
            if lang == "zh":
                # For Chinese: treat each character as a token, or include punctuation if needed
                if keep_punct:
                    # Keep both Chinese and English punctuation
                    pattern = r"[\u4e00-\u9fff]+|" + EN_PUNCT + "|" + ZH_PUNCT
                    tokens.extend(re.findall(pattern, t))
                else:
                    # Only keep Chinese characters (ignore punctuation)
                    tokens.extend([ch for ch in t if "\u4e00" <= ch <= "\u9fff"])
            else:
                # English or mixed-language text
                if keep_punct:
                    if separate_zh_punct:
                        # Keep English words + both EN and ZH punctuation
                        pattern = r"\w+|" + EN_PUNCT + "|" + ZH_PUNCT
                    else:
                        # Keep words + any punctuation (mixed)
                        pattern = r"\w+|[^\w\s]"
                    tokens.extend(re.findall(pattern, t))
                else:
                    # Default: only keep words, ignore punctuation
                    tokens.extend(re.findall(r"\b\w+\b", t))

        # ------------------------------------------------------------
        # 4️⃣ English word dictionary filtering (if desired)
        # ------------------------------------------------------------
        if lang == "en" and EN_WORDS and not use_english_vocab:
            # Keep tokens that are in the English dictionary OR are punctuation/numbers
            tokens = [
                w for w in tokens
                if (w.isalpha() and w in EN_WORDS) or not w.isalpha()
            ]

        # ------------------------------------------------------------
        # 5️⃣ English dictionary seeding (optional vocab expansion)
        # ------------------------------------------------------------
        if lang == "en" and use_english_vocab and EN_WORDS:
            print("📚 Merging English dictionary words into vocabulary...")
            tokens.extend(list(EN_WORDS))

        # ------------------------------------------------------------
        # 6️⃣ Build vocabulary (unique tokens + special symbols)
        # ------------------------------------------------------------
        vocab = sorted(set(tokens))
        specials = []
        if add_pad:
            specials.append("<pad>")
        if add_eos:
            specials.append("<eos>")
        if add_unk:
            specials.append("<unk>")
        full_vocab = specials + vocab

        # ------------------------------------------------------------
        # 7️⃣ Build lookup tables
        # ------------------------------------------------------------
        self.stoi = {w: i for i, w in enumerate(full_vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

        self.pad_idx = self.stoi.get("<pad>", pad_id)
        self.eos_idx = self.stoi.get("<eos>", None)
        self.unk_idx = self.stoi.get("<unk>", None)
        self.vocab_size = len(self.stoi)

        print(
            f"✅ WordTokenizer initialized | lang={lang} | keep_punct={keep_punct} | "
            f"separate_zh_punct={separate_zh_punct} | vocab_size={self.vocab_size} "
            f"| pad_idx={self.pad_idx} | eos_idx={self.eos_idx} | unk_idx={self.unk_idx}"
        )
        if use_english_vocab:
            print("✅ English dictionary words added to vocabulary.")

    # ------------------------------------------------------------
    # 🔹 Encode: convert text → token IDs
    # ------------------------------------------------------------
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """
        Encode a string into a list of token IDs.
        Handles punctuation based on configuration.
        Unknown tokens are mapped to <unk>.
        """
        EN_PUNCT = r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]"
        ZH_PUNCT = r"[，。！？、“”‘’《》（）【】……—～：；·]"

        if self.lang == "zh":
            if self.keep_punct:
                pattern = r"[\u4e00-\u9fff]+|" + EN_PUNCT + "|" + ZH_PUNCT
                tokens = re.findall(pattern, text)
            else:
                tokens = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        else:
            text = text.lower() if self.lowercase else text
            if self.keep_punct:
                if self.separate_zh_punct:
                    pattern = r"\w+|" + EN_PUNCT + "|" + ZH_PUNCT
                else:
                    pattern = r"\w+|[^\w\s]"
                tokens = re.findall(pattern, text)
            else:
                tokens = re.findall(r"\b\w+\b", text)

        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens]
        if add_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    # ------------------------------------------------------------
    # 🔹 Decode: convert token IDs → string
    # ------------------------------------------------------------
    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Decode token IDs back into string text.
        For Chinese: returns concatenated string without spaces.
        For English: joins tokens with spaces.
        """
        words = []
        for i in ids:
            tok = self.itos.get(i, "")
            if tok == "<eos>":
                break
            if skip_specials and tok in ("<pad>", "<eos>"):
                continue
            words.append(tok)

        # Chinese: no spaces; English/mixed: use spaces
        if self.lang == "zh":
            return "".join(words)
        return " ".join(words)

    # ------------------------------------------------------------
    # 🔹 Save tokenizer configuration
    # ------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "lang": self.lang,
            "vocab": self.stoi,
            "add_eos": self.add_eos,
            "add_pad": self.add_pad,
            "add_unk": self.add_unk,
            "pad_id": self.pad_id,
            "keep_punct": self.keep_punct,
            "separate_zh_punct": self.separate_zh_punct,
            "lowercase": self.lowercase,
            "use_english_vocab": self.use_english_vocab,
            "pad_idx": self.pad_idx,
            "eos_idx": self.eos_idx,
            "unk_idx": self.unk_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved WordTokenizer → {path}")

    # ------------------------------------------------------------
    # 🔹 Load tokenizer from disk
    # ------------------------------------------------------------
    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = WordTokenizer(
            [""],
            lang=data.get("lang", "en"),
            add_eos=data.get("add_eos", True),
            add_pad=data.get("add_pad", True),
            add_unk=data.get("add_unk", True),
            pad_id=data.get("pad_id", -100),
            lowercase=data.get("lowercase", True),
            use_english_vocab=data.get("use_english_vocab", False),
            keep_punct=data.get("keep_punct", False),
            separate_zh_punct=data.get("separate_zh_punct", True),
        )
        tok.stoi = data["vocab"]
        tok.itos = {i: w for w, i in tok.stoi.items()}
        tok.vocab_size = len(tok.stoi)
        tok.pad_idx = data.get("pad_idx", -100)
        tok.eos_idx = data.get("eos_idx", None)
        tok.unk_idx = data.get("unk_idx", None)

        print(
            f"✅ Loaded WordTokenizer from {path} | lang={tok.lang} | keep_punct={tok.keep_punct} "
            f"| separate_zh_punct={tok.separate_zh_punct} | vocab_size={tok.vocab_size}"
        )
        return tok
    
class HFTokenizerWrapper:
    """
    Wrapper for any Hugging Face tokenizer (BPE / WordPiece).

    Features:
      ✅ Uses transformers.AutoTokenizer (fast version)
      ✅ Automatically sets <pad> token if missing
      ✅ Provides save() and load() methods for persistence
      ✅ Exposes simple encode/decode interface (like CharTokenizer)
      ✅ Access to underlying tokenizer via `.tokenizer`
    """

    def __init__(self, name_or_path: str, add_special_tokens: bool = False):
        """
        Initialize tokenizer from pretrained name or local path.

        Args:
            name_or_path (str): HF model name or tokenizer path.
            add_special_tokens (bool): Whether to include BOS/EOS tokens by default.
        """
        from transformers import AutoTokenizer

        # Load tokenizer (fast implementation preferred)
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

        # Ensure pad token exists
        if self.tok.pad_token_id is None:
            # Try to reuse existing eos or sep token
            self.tok.pad_token = self.tok.eos_token or self.tok.sep_token or "[PAD]"

        self.vocab_size = self.tok.vocab_size
        self.add_special_tokens = add_special_tokens

        print(
            f"✅ HFTokenizerWrapper initialized from '{name_or_path}' "
            f"| vocab_size={self.vocab_size} | pad_id={self.tok.pad_token_id}"
        )

    # ------------------------------------------------------------
    # 🔹 Expose underlying tokenizer
    # ------------------------------------------------------------
    @property
    def tokenizer(self):
        """Return the underlying Hugging Face tokenizer object."""
        return self.tok

    # ------------------------------------------------------------
    # 🔹 Encode text → list of IDs
    # ------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = None) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text (str): Input string.
            add_special_tokens (bool): Whether to include BOS/EOS tokens.
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens
        return self.tok(text, add_special_tokens=add_special_tokens)["input_ids"]

    # ------------------------------------------------------------
    # 🔹 Decode IDs → text
    # ------------------------------------------------------------
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode list of IDs back to string."""
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------
    # 🔹 Token ID accessors
    # ------------------------------------------------------------
    @property
    def pad_id(self): return self.tok.pad_token_id
    @property
    def bos_id(self): return self.tok.bos_token_id
    @property
    def eos_id(self): return self.tok.eos_token_id

    # ------------------------------------------------------------
    # 🔹 Save tokenizer to directory
    # ------------------------------------------------------------
    def save(self, path: str):
        """
        Save the tokenizer to a directory.

        Args:
            path (str): Directory path to save files.
        """
        os.makedirs(path, exist_ok=True)
        self.tok.save_pretrained(path)
        print(f"💾 Saved Hugging Face tokenizer → {path}")

    # ------------------------------------------------------------
    # 🔹 Load tokenizer from directory
    # ------------------------------------------------------------
    @staticmethod
    def load(path: str, add_special_tokens: bool = False):
        """
        Load tokenizer from a saved directory.

        Args:
            path (str): Directory path where tokenizer was saved.
            add_special_tokens (bool): Whether to include BOS/EOS tokens by default.

        Returns:
            HFTokenizerWrapper
        """
        print(f"🔍 Loading Hugging Face tokenizer from {path} ...")
        return HFTokenizerWrapper(path, add_special_tokens=add_special_tokens)


import os
from typing import List, Literal

class CustomTokenizer:
    """
    Custom reversible tokenizer (UTF-8 safe version).

    ✅ Full encode-decode symmetry for English, Chinese, emoji
    ✅ Unicode-safe: preserves all multi-byte characters
    ✅ Supports BPE, WordPiece, Unigram, WordLevel
    ✅ GPT-style reversible ByteBPE if desired
    """

    def __init__(
        self,
        texts: List[str] = None,
        tokenizer_type: Literal["bpe", "bytebpe", "wordpiece", "unigram", "wordlevel"] = "bytebpe",
        tokenizer_path: str = "outputs/custom_tokenizer.json",
        vocab_size: int = 8000,
        min_freq: int = 2,
        pad_id: int = -100,
        add_special_tokens: bool = True,
        lowercase: bool = True,
    ):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE, WordPiece, Unigram, WordLevel
        from tokenizers.trainers import (
            BpeTrainer, WordPieceTrainer, UnigramTrainer, WordLevelTrainer
        )
        from tokenizers.normalizers import NFKC, Lowercase, Sequence
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.decoders import BPEDecoder

        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.pad_id = pad_id
        self.lowercase = lowercase

        # ------------------------------------------------------------
        # 1️⃣ Load if exists
        # ------------------------------------------------------------
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"✅ Loaded existing tokenizer → {tokenizer_path}")
        else:
            if texts is None:
                raise ValueError("❌ Cannot train tokenizer without input texts.")

            print(f"🚀 Training new {tokenizer_type.upper()} tokenizer (vocab={vocab_size})")

            # --------------------------------------------------------
            # 2️⃣ Select model + trainer
            # --------------------------------------------------------
            if tokenizer_type in ("bytebpe", "bpe"):
                model = BPE(unk_token="<unk>")
                trainer = BpeTrainer(
                    vocab_size=vocab_size,
                    min_frequency=min_freq,
                    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
                    if add_special_tokens else [],
                )
                # ✅ Use Whitespace (Unicode-safe)
                pre_tokenizer = Whitespace()

            elif tokenizer_type == "wordpiece":
                model = WordPiece(unk_token="<unk>")
                trainer = WordPieceTrainer(
                    vocab_size=vocab_size,
                    min_frequency=min_freq,
                    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
                    if add_special_tokens else [],
                )
                pre_tokenizer = Whitespace()

            elif tokenizer_type == "unigram":
                model = Unigram()
                trainer = UnigramTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
                    if add_special_tokens else [],
                )
                pre_tokenizer = Whitespace()

            elif tokenizer_type == "wordlevel":
                model = WordLevel(unk_token="<unk>")
                trainer = WordLevelTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
                    if add_special_tokens else [],
                )
                pre_tokenizer = Whitespace()

            else:
                raise ValueError(f"❌ Unknown tokenizer_type: {tokenizer_type}")

            # --------------------------------------------------------
            # 3️⃣ Build tokenizer pipeline (UTF-8 safe)
            # --------------------------------------------------------
            tokenizer = Tokenizer(model)
            if lowercase:
                tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
            else:
                tokenizer.normalizer = NFKC()

            tokenizer.pre_tokenizer = pre_tokenizer
            tokenizer.decoder = BPEDecoder()  # ✅ Unicode-safe BPE decoder

            tokenizer.train_from_iterator(texts, trainer=trainer)

            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            tokenizer.save(tokenizer_path)
            print(f"💾 Saved UTF-8 safe custom tokenizer → {tokenizer_path}")

            self.tokenizer = tokenizer

        # ------------------------------------------------------------
        # 4️⃣ Load vocab
        # ------------------------------------------------------------
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab.get("<pad>", pad_id)
        self.eos_idx = self.vocab.get("<eos>", None)
        self.bos_idx = self.vocab.get("<bos>", None)
        self.unk_idx = self.vocab.get("<unk>", None)

        print(
            f"✅ CustomTokenizer initialized | type={tokenizer_type} | vocab_size={self.vocab_size} | "
            f"pad_idx={self.pad_idx} | eos_idx={self.eos_idx} | bos_idx={self.bos_idx} | unk_idx={self.unk_idx}"
        )
        print("🧩 UTF-8 safe normalization and decoding enabled (reversible).")

    # ------------------------------------------------------------
    # 🔹 Encode
    # ------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos and self.bos_idx is not None:
            ids = [self.bos_idx] + ids
        if add_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    # ------------------------------------------------------------
    # 🔹 Decode
    # ------------------------------------------------------------
    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Decode token IDs back into readable UTF-8 text.
        Handles byte-level Latin-1 → UTF-8 re-decoding automatically.
        """
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_specials)
        try:
            # 🔧 Re-interpret byte string correctly
            text = text.encode("latin1").decode("utf8")
        except Exception:
            pass
        # Optional: clean GPT-style 'Ġ' markers for readability
        text = text.replace("Ġ", " ")
        return text.strip()

    # ------------------------------------------------------------
    # 🔹 Save
    # ------------------------------------------------------------
    def save(self, path: str = None):
        path = path or self.tokenizer_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        print(f"💾 Saved CustomTokenizer → {path}")

    # ------------------------------------------------------------
    # 🔹 Load (static)
    # ------------------------------------------------------------
    @staticmethod
    def load(path: str):
        from tokenizers import Tokenizer
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Tokenizer not found: {path}")

        tok = Tokenizer.from_file(path)
        wrapper = CustomTokenizer.__new__(CustomTokenizer)
        wrapper.tokenizer = tok
        wrapper.tokenizer_path = path
        wrapper.vocab = tok.get_vocab()
        wrapper.vocab_size = len(wrapper.vocab)
        wrapper.pad_idx = wrapper.vocab.get("<pad>", -100)
        wrapper.eos_idx = wrapper.vocab.get("<eos>", None)
        wrapper.bos_idx = wrapper.vocab.get("<bos>", None)
        wrapper.unk_idx = wrapper.vocab.get("<unk>", None)
        print(f"✅ Loaded CustomTokenizer from {path} | vocab_size={wrapper.vocab_size}")
        return wrapper

class TokenizerFactory:
    """
    Unified Tokenizer Builder / Loader.

    Automatically builds or loads tokenizers based on user options:
      - "char"                   → CharTokenizer
      - "word"                   → WordTokenizer
      - "hf:xx"                  → HFTokenizerWrapper for pretrained Hugging Face tokenizer
      - "custom:tokenizer_type"  → CustomTokenizer (e.g. "custom:bytebpe")
      - "auto"                   → Auto-detect from provided path
    """

    @staticmethod
    def build(
        tokenizer: str,
        texts: List[str] = None,
        tokenizer_path: str = None,
        vocab_size: int = 8000,
        min_freq: int = 2,
        pad_id: int = -100,
        lowercase: bool = True,
    ):
        """
        Build or load tokenizer automatically based on user input.

        Args:
            tokenizer (str): Tokenizer specifier ("char", "word", "hf:gpt2", "custom:bytebpe", "auto").
            texts (List[str]): Training corpus (required for training custom tokenizers).
            tokenizer_path (str): Optional tokenizer path to load from.
            vocab_size (int): Vocabulary size for custom tokenizers.
            min_freq (int): Minimum token frequency for custom tokenizers.
            pad_id (int): Default pad_id, consistent with PyTorch ignore_index.
            lowercase (bool): Convert text to lowercase for char/word tokenizers.

        Returns:
            tokenizer_instance: A tokenizer object with unified encode()/decode()/save()/load() API.
        """

        # ✅ Case 1: Auto-detect from existing path
        if tokenizer == "auto":
            if tokenizer_path and os.path.exists(tokenizer_path):
                print(f"🔍 Auto-detecting tokenizer from {tokenizer_path} ...")
                # Try loading HF or custom first
                try:
                    return HFTokenizerWrapper.load(tokenizer_path)
                except Exception:
                    try:
                        return CustomTokenizer.load(tokenizer_path)
                    except Exception:
                        pass

                # Fallback to char/word tokenizers
                if "char" in tokenizer_path.lower():
                    return CharTokenizer.load(tokenizer_path)
                elif "word" in tokenizer_path.lower():
                    return WordTokenizer.load(tokenizer_path)
                raise ValueError(f"❌ Unable to detect tokenizer type from {tokenizer_path}.")
            else:
                raise ValueError("❌ 'auto' mode requires a valid tokenizer_path to load from.")

        # ✅ Case 2: Hugging Face tokenizer (e.g. hf:gpt2)
        elif tokenizer.startswith("hf:"):
            name_or_path = tokenizer.split("hf:")[1]
            print(f"🤗 Building Hugging Face tokenizer: {name_or_path}")
            return HFTokenizerWrapper(name_or_path)

        # ✅ Case 3: Custom tokenizer (e.g. custom:bytebpe)
        elif tokenizer.startswith("custom:"):
            algo = tokenizer.split("custom:")[1]
            print(f"🧠 Training/Loading Custom tokenizer ({algo}) ...")
            return CustomTokenizer(
                texts=texts,
                tokenizer_type=algo,
                tokenizer_path=tokenizer_path or f"outputs/custom_{algo}_tokenizer.json",
                vocab_size=vocab_size,
                min_freq=min_freq,
                pad_id=pad_id,
                lowercase=lowercase,
            )

        # ✅ Case 4: Character tokenizer
        elif tokenizer == "char":
            print("🔤 Building CharTokenizer ...")
            if tokenizer_path and os.path.exists(tokenizer_path):
                return CharTokenizer.load(tokenizer_path)
            return CharTokenizer(texts, add_eos=True, add_pad=True)

        # ✅ Case 5: Word tokenizer
        elif tokenizer == "word":
            print("🧩 Building WordTokenizer ...")
            if tokenizer_path and os.path.exists(tokenizer_path):
                return WordTokenizer.load(tokenizer_path)
            return WordTokenizer(texts, lang="en")

        else:
            raise ValueError(f"❌ Unknown tokenizer option: {tokenizer}")



def test_tokenizers(
    dataset_name: str = "ag_news",     # realistic English dataset
    num_samples: int = 1000,           # how many sentences to train custom tokenizers
    vocab_size: int = 5000,            # vocab for CustomTokenizer
    tokenizer_dir: str = "outputs/",
):
    """
    🔍 Comprehensive tokenizer test suite (realistic version).

    Trains and evaluates multiple tokenizer backends on a real dataset.
    Includes:
      - CharTokenizer
      - WordTokenizer
      - HFTokenizerWrapper (GPT-2)
      - CustomTokenizer (ByteBPE)
      - Auto-detected tokenizer reload

    Args:
        dataset_name (str): Hugging Face dataset to use ("ag_news", "wikitext", etc.).
        num_samples (int): Number of lines from dataset for training.
        vocab_size (int): Vocabulary size for custom tokenizers.
        tokenizer_dir (str): Directory to store tokenizer files.
    """
    from datasets import load_dataset
    #from DeepDataMiningLearning.llm.tokenizers.factory import TokenizerFactory

    print("\n🔍 ===== REALISTIC TOKENIZER TEST SUITE =====\n")

    # ------------------------------------------------------------
    # 1️⃣ Load a realistic text dataset
    # ------------------------------------------------------------
    print(f"📘 Loading dataset: {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")

    # Extract text field automatically
    text_field = "text" if "text" in ds.column_names else ds.column_names[0]
    texts = [t.strip() for t in ds[text_field] if isinstance(t, str) and len(t.strip()) > 0]
    texts = texts[:num_samples]
    print(f"✅ Loaded {len(texts)} text samples for tokenizer training.\n")

    # Prepare smaller sample set for testing readability
    test_samples = texts[:3]

    # ------------------------------------------------------------
    # 2️⃣ Helper: normalization for round-trip checks
    # ------------------------------------------------------------
    def normalize(text: str):
        """Normalize text for fair comparison (handles Ġ, case, and spacing)."""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("Ġ", " ")  # replace GPT-style space marker
        text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
        return text.strip().lower()

    # ------------------------------------------------------------
    # 3️⃣ Define test runner
    # ------------------------------------------------------------
    def _inspect_tokenizer(tok, samples: List[str]):
        print(f"🧩 Tokenizer class: {type(tok).__name__}")
        if hasattr(tok, "vocab_size"):
            print(f"🔡 Vocab size: {tok.vocab_size}")
        if hasattr(tok, "tokenizer_path"):
            print(f"💾 Path: {getattr(tok, 'tokenizer_path', None)}")

        # Special token indices
        specials = {name: getattr(tok, name) for name in ["pad_idx", "bos_idx", "eos_idx", "unk_idx"] if hasattr(tok, name)}
        if specials:
            print(f"🔖 Special token indices: {specials}")

        print(f"\n📚 Testing {len(samples)} samples:\n")
        for i, s in enumerate(samples, 1):
            print(f"🔹 Sample {i}: {s[:120]}{'...' if len(s) > 120 else ''}")
            try:
                ids = tok.encode(s)
                decoded = tok.decode(ids)
                print(f"   🔢 Token IDs ({len(ids)}): {ids[:40]}{'...' if len(ids) > 40 else ''}")
                print(f"   🔁 Decoded: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")

                # Normalized match check
                match = normalize(decoded) == normalize(s)
                if match:
                    print("   ✅ Round-trip match (normalized): True")
                else:
                    print("   ⚠️ Round-trip mismatch (normalized): False")
                    print(f"      expected: {normalize(s)}")
                    print(f"      got     : {normalize(decoded)}")

            except Exception as e:
                print(f"   ⚠️ Error during encode/decode: {e}")
            print("")
        print("")

    # ------------------------------------------------------------
    # 4️⃣ Run tokenizer tests
    # ------------------------------------------------------------
    print("\n==================== 🟦 CHAR TOKENIZER ====================")
    tok_char = TokenizerFactory.build("char", texts=texts)
    _inspect_tokenizer(tok_char, test_samples)

    print("\n==================== 🟩 WORD TOKENIZER ====================")
    lowercase_texts = [t.lower() for t in texts]
    tok_word = TokenizerFactory.build("word", texts=lowercase_texts)
    _inspect_tokenizer(tok_word, test_samples)

    print("\n==================== 🤗 HF TOKENIZER (gpt2) ====================")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # suppress warnings
    try:
        tok_hf = TokenizerFactory.build("hf:gpt2")
        _inspect_tokenizer(tok_hf, test_samples)
    except Exception as e:
        print(f"⚠️ Hugging Face tokenizer load failed: {e}")

    print("\n==================== 🧠 CUSTOM TOKENIZER (ByteBPE) ====================")
    tok_custom_path = os.path.join(tokenizer_dir, "custom_bytebpe_tokenizer.json")
    tok_custom = TokenizerFactory.build(
        "custom:bytebpe",
        texts=texts,
        tokenizer_path=tok_custom_path,
        vocab_size=vocab_size,
    )
    _inspect_tokenizer(tok_custom, test_samples)

    print("\n==================== 🔄 AUTO-DETECTED TOKENIZER ====================")
    tok_auto = TokenizerFactory.build("auto", tokenizer_path=tok_custom_path)
    _inspect_tokenizer(tok_auto, test_samples)

    print("\n✅ All realistic tokenizer tests completed successfully.\n")

def test_custom_tokenizer():
    texts = [
        "The cat sat on the mat.",
        "Byte-level BPE tokenizers handle punctuation very well.",
    ]

    # Train a new ByteLevelBPE tokenizer
    tok = CustomTokenizer(
        texts=texts,
        tokenizer_type="bytebpe",
        tokenizer_path="outputs/custom_bytebpe.json",
        vocab_size=5000,
    )

    # Encode & Decode
    ids = tok.encode("The cat sat.")
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
    #Decoded: Ġthe Ġ c at Ġ s at . Ġ (U+0120) is a special marker used in GPT-style Byte-Level BPEs to represent a leading space.
    # Save & Load
    tok.save("outputs/custom_bytebpe.json")
    tok2 = CustomTokenizer.load("outputs/custom_bytebpe.json")
    print("Decoded after load:", tok2.decode(ids))

def test_custom_tokenizer2():
    texts = [
        "Hello, world! 你好，世界！",
        "GPT-2 🤖🚀🔥 Tokenizer test. 字节级别可逆编码。",
    ]

    tok = CustomTokenizer(texts, tokenizer_type="bytebpe", vocab_size=5000)

    for s in texts:
        ids = tok.encode(s)
        dec = tok.decode(ids)
        print(f"\nOriginal: {s}")
        print(f"Decoded : {dec}")
        print("Match   :", s == dec)

def test_char_word_tokenizer():
    texts = [
        "The cat sat on the mat.",
        "Hello world! Python coding.",
        "机器学习是人工智能的一个分支。"
    ]

    # English tokenizer
    tok_en = WordTokenizer(texts, lang="en")
    ids_en = tok_en.encode("The dog sat on the mat")
    print(ids_en)
    print(tok_en.decode(ids_en))

    # Chinese tokenizer
    tok_zh = WordTokenizer(texts, lang="zh")
    ids_zh = tok_zh.encode("机器学习")
    print(ids_zh)
    print(tok_zh.decode(ids_zh))

    # Save & Load
    tok_en.save("outputs/word_tokenizer_en.json")
    tok_zh.save("outputs/word_tokenizer_zh.json")

    tok_en2 = WordTokenizer.load("outputs/word_tokenizer_en.json")
    tok_zh2 = WordTokenizer.load("outputs/word_tokenizer_zh.json")

def test_hftokenizer():
    #from DeepDataMiningLearning.llm.tokenizer_utils import HFTokenizerWrapper

    # 1️⃣ Load pretrained tokenizer
    tok = HFTokenizerWrapper("gpt2")

    # 2️⃣ Encode & decode text
    ids = tok.encode("Hello world!")
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))

    # 3️⃣ Save tokenizer
    tok.save("outputs/hf_tokenizer_gpt2")

    # 4️⃣ Load tokenizer
    tok2 = HFTokenizerWrapper.load("outputs/hf_tokenizer_gpt2")

    # 5️⃣ Verify it works
    print(tok2.encode("Hello world!"))
    print(tok2.decode(tok2.encode("Hello world!")))

if __name__ == "__main__":
    #huggingface-cli login
    test_char_word_tokenizer()
    test_hftokenizer()
    test_custom_tokenizer()
    test_custom_tokenizer2()
    texts = [
        "The cat sat on the mat.",
        "Machine learning is fun.",
        "机器学习是人工智能的一个分支。",
        "Let's test emojis 🤖🚀🔥!"
    ]

    test_tokenizers(dataset_name="ag_news", num_samples=1000, vocab_size=8000)
    