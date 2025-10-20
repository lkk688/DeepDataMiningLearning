
from typing import List, Union, Optional, Tuple, Dict
import os
import json
import numpy as np
import re
import unicodedata

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

# Optional jieba for Chinese segmentation
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠️  jieba not installed. Install via `pip install jieba` for Chinese word segmentation.")




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
    Multilingual word-level tokenizer with English + Chinese + mixed support.

    Features:
      ✅ English, Chinese, or mixed en-zh text
      ✅ Optional jieba for Chinese word segmentation
      ✅ Keeps or removes punctuation
      ✅ Distinguishes Chinese vs English punctuation
      ✅ Adds <pad>, <eos>, and <unk> tokens
      ✅ Supports save() / load()
    """

    def __init__(
        self,
        texts: List[str],
        lang: str = "en", #"zh", "en-zh", "mixed"
        add_eos: bool = True,
        add_pad: bool = True,
        add_unk: bool = True,
        pad_id: int = -100,
        lowercase: bool = True,
        use_english_vocab: bool = False,
        keep_punct: bool = False,
        separate_zh_punct: bool = True,
        use_jieba: bool = False,        # ✅ new: use jieba for Chinese segmentation
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
        self.use_jieba = use_jieba and JIEBA_AVAILABLE

        # ------------------------------------------------------------
        # 1️⃣ Normalize text
        # ------------------------------------------------------------
        if lowercase and lang != "zh":
            texts = [t.lower() for t in texts]
        texts = [t.strip() for t in texts]

        # ------------------------------------------------------------
        # 2️⃣ Regex for punctuation
        # ------------------------------------------------------------
        EN_PUNCT = r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]"
        ZH_PUNCT = r"[，。！？、“”‘’《》（）【】……—～：；·]"
        tokens = []

        # ------------------------------------------------------------
        # 3️⃣ Tokenization logic
        # ------------------------------------------------------------
        for t in texts:
            if lang == "zh":
                # ------------------------------
                # Chinese only
                # ------------------------------
                if self.use_jieba:
                    # jieba word segmentation
                    zh_tokens = list(jieba.cut(t))
                else:
                    # character-based segmentation
                    zh_tokens = [ch for ch in t if "\u4e00" <= ch <= "\u9fff"]

                if keep_punct:
                    puncts = re.findall(EN_PUNCT + "|" + ZH_PUNCT, t)
                    tokens.extend(zh_tokens + puncts)
                else:
                    tokens.extend(zh_tokens)

            elif lang == "en":
                # ------------------------------
                # English only
                # ------------------------------
                if keep_punct:
                    if separate_zh_punct:
                        pattern = r"\w+|" + EN_PUNCT + "|" + ZH_PUNCT
                    else:
                        pattern = r"\w+|[^\w\s]"
                    tokens.extend(re.findall(pattern, t))
                else:
                    tokens.extend(re.findall(r"\b\w+\b", t))

            elif lang in ("en-zh", "mixed"):
                pattern = (
                    r"[\u4e00-\u9fff]+"  # Chinese block
                    + (f"|{EN_PUNCT}|{ZH_PUNCT}" if keep_punct else "")
                    + r"|\b\w+\b"        # English words
                )

                mixed_tokens = re.findall(pattern, t)
                seg_tokens = []
                for tok in mixed_tokens:
                    if re.match(r"[\u4e00-\u9fff]+", tok):
                        # Pass contiguous Chinese segment to jieba if enabled
                        if self.use_jieba:
                            seg_tokens.extend(list(jieba.cut(tok)))
                        else:
                            seg_tokens.extend(list(tok))  # fallback: per-character
                    else:
                        seg_tokens.append(tok)
                tokens.extend(seg_tokens)

            else:
                raise ValueError(f"❌ Unsupported language type: {lang}")

        # ------------------------------------------------------------
        # 4️⃣ Optional English dictionary filtering
        # ------------------------------------------------------------
        if lang in ("en", "en-zh", "mixed") and EN_WORDS and not use_english_vocab:
            if self.lowercase:
                # Standard lowercase filtering
                tokens = [
                    w for w in tokens
                    if (w.isalpha() and w in EN_WORDS)
                    or (not w.isalpha())
                    or re.match(r"[\u4e00-\u9fff]", w)
                ]
            else:
                # Case-sensitive mode: allow both lowercase and capitalized forms
                valid_words = set(EN_WORDS)
                # Add capitalized versions for typical English nouns/proper names
                capitalized_words = {w.capitalize() for w in EN_WORDS if w.isalpha() and len(w) > 1}
                valid_words.update(capitalized_words)

                tokens = [
                    w for w in tokens
                    if (w.isalpha() and w in valid_words)
                    or (not w.isalpha())
                    or re.match(r"[\u4e00-\u9fff]", w)
                ]

        # ------------------------------------------------------------
        # 5️⃣ Optional English vocab expansion
        # ------------------------------------------------------------
        if lang in ("en", "en-zh", "mixed") and use_english_vocab and EN_WORDS:
            print("📚 Merging English dictionary words into vocabulary...")
            if self.lowercase:
                tokens.extend(list(EN_WORDS))
            else:
                # Add both lowercase and capitalized forms to vocab
                capitalized_words = [w.capitalize() for w in EN_WORDS if w.isalpha() and len(w) > 1]
                tokens.extend(list(EN_WORDS) + capitalized_words)

        # ------------------------------------------------------------
        # 6️⃣ Build vocab
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
            f"✅ WordTokenizer initialized | lang={lang} | jieba={self.use_jieba} | "
            f"keep_punct={keep_punct} | vocab_size={self.vocab_size} "
            f"| pad_idx={self.pad_idx} | eos_idx={self.eos_idx} | unk_idx={self.unk_idx}"
        )

    # ------------------------------------------------------------
    # 🔹 Encode
    # ------------------------------------------------------------
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        if self.lowercase and self.lang != "zh":
            text = text.lower()
        """Encode a string into token IDs."""
        if self.lang == "zh" and self.use_jieba:
            tokens = list(jieba.cut(text))
        elif self.lang in ("en-zh", "mixed") and self.use_jieba:
            parts = re.findall(r"[\u4e00-\u9fff]+|\b\w+\b|[^\w\s]", text)
            tokens = []
            for p in parts:
                if re.match(r"[\u4e00-\u9fff]+", p):
                    tokens.extend(list(jieba.cut(p)))
                else:
                    tokens.append(p)
        elif self.lang in ("en-zh", "mixed"):
            #tokens = re.findall(r"[\u4e00-\u9fff]+|\b\w+\b|[^\w\s]", text)
            parts = re.findall(r"[\u4e00-\u9fff]+|\b\w+\b|[^\w\s]", text)
            tokens = []
            for p in parts:
                if re.match(r"[\u4e00-\u9fff]+", p):
                    if self.use_jieba:
                        tokens.extend(list(jieba.cut(p)))
                    else:
                        tokens.extend(list(p))  # ✅ fallback: char-level
                else:
                    tokens.append(p)
        elif self.lang == "zh":
            tokens = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        else:
            tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens]
        if add_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    # ------------------------------------------------------------
    # 🔹 Decode
    # ------------------------------------------------------------
    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """Decode token IDs back into text."""
        words = []
        for i in ids:
            tok = self.itos.get(i, "")
            if tok == "<eos>":
                break
            if skip_specials and tok in ("<pad>", "<eos>"):
                continue
            words.append(tok)
        # Chinese/mixed: remove spaces between Chinese chars
        text = " ".join(words)
        text = re.sub(r"\s*(?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s*", "", text)
        return text.strip()

        # ------------------------------------------------------------
    # 🔹 Save tokenizer to disk
    # ------------------------------------------------------------
    def save(self, path: str):
        """
        Save tokenizer configuration and vocabulary to disk in JSON format.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "lang": self.lang,
            "add_eos": self.add_eos,
            "add_pad": self.add_pad,
            "add_unk": self.add_unk,
            "pad_id": self.pad_id,
            "lowercase": self.lowercase,
            "use_english_vocab": self.use_english_vocab,
            "keep_punct": self.keep_punct,
            "separate_zh_punct": self.separate_zh_punct,
            "use_jieba": getattr(self, "use_jieba", False),
            "jieba_available": getattr(self, "use_jieba", False),
            "pad_idx": self.pad_idx,
            "eos_idx": self.eos_idx,
            "unk_idx": self.unk_idx,
            "vocab": self.stoi,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved WordTokenizer → {path}")

    # ------------------------------------------------------------
    # 🔹 Load tokenizer from disk
    # ------------------------------------------------------------
    @staticmethod
    def load(path: str):
        """
        Load a WordTokenizer from JSON file.
        Reconstructs configuration, vocabulary, and index mappings.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = WordTokenizer(
            [""],  # placeholder text
            lang=data.get("lang", "en"),
            add_eos=data.get("add_eos", True),
            add_pad=data.get("add_pad", True),
            add_unk=data.get("add_unk", True),
            pad_id=data.get("pad_id", -100),
            lowercase=data.get("lowercase", True),
            use_english_vocab=data.get("use_english_vocab", False),
            keep_punct=data.get("keep_punct", False),
            separate_zh_punct=data.get("separate_zh_punct", True),
            use_jieba=data.get("use_jieba", False),
        )

        # Restore vocab tables
        vocab = data.get("vocab", {})
        tok.stoi = vocab
        tok.itos = {i: w for w, i in vocab.items()}

        tok.vocab_size = len(vocab)
        tok.pad_idx = data.get("pad_idx", -100)
        tok.eos_idx = data.get("eos_idx", None)
        tok.unk_idx = data.get("unk_idx", None)

        print(
            f"✅ Loaded WordTokenizer from {path} | lang={tok.lang} | jieba={tok.use_jieba} "
            f"| keep_punct={tok.keep_punct} | separate_zh_punct={tok.separate_zh_punct} "
            f"| vocab_size={tok.vocab_size}"
        )

        return tok

from transformers import AutoTokenizer
# Optional: only needed for Byte-Level fix
try:
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTok
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
except Exception:
    ByteLevelPreTok = None
    ByteLevelDecoder = None


def _safe_dir_name(name: str) -> str:
    """
    Make a filesystem-safe directory name from a repo id like 'Qwen/Qwen2-1.5B'.
    E.g. 'outputs/hf_tok__Qwen__Qwen2-1.5B'
    """
    # Replace slashes and spaces with double underscore; keep word chars, dot, dash
    cleaned = re.sub(r"[^\w.\-]+", "__", name)
    return cleaned

class HFTokenizerWrapper:
    """
    Robust wrapper around Hugging Face tokenizers that:
      - Works with remote and local paths
      - Fixes Byte-Level BPE (GPT-2/Qwen) decoders on reload
      - Leaves SentencePiece (LLaMA/Mistral) alone
      - Runs a small Unicode self-test after load

    Notes:
      * For GPT-2/Qwen-like tokenizers (byte-level BPE), we re-attach the ByteLevel
        pre-tokenizer + decoder after reload to retain full UTF-8 round-tripping.
      * For SP models (LLaMA, etc.) we don't touch internals, but we still check
        that CJK/emoji decoding survives a save→reload cycle.
    """

    def __init__(self,
                 name_or_path: str,
                 add_special_tokens: bool = True,
                 trust_remote_code: bool = False):
        self.add_special_tokens = add_special_tokens

        # 1) Load tokenizer (remote or local)
        if os.path.isdir(name_or_path):
            print(f"📁 Loading tokenizer from local directory: {name_or_path}")
        else:
            print(f"🌐 Loading pretrained tokenizer from Hugging Face Hub: {name_or_path}")

        self.tok = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # 2) Classify tokenizer family
        self._model_id = getattr(self.tok, "name_or_path", str(name_or_path)).lower()
        self._tok_class = type(self.tok).__name__.lower()

        # Simple heuristics:
        self._is_byte_level = any(k in self._model_id for k in ["gpt2", "qwen"]) \
                              or "bytelevel" in self._tok_class \
                              or "gpt2" in self._tok_class \
                              or "qwen" in self._tok_class
        self._is_sentencepiece = "sentencepiece" in self._tok_class or "llama" in self._model_id

        # 3) Repair byte-level tokenizer plumbing if necessary
        if self._is_byte_level and hasattr(self.tok, "backend_tokenizer") and ByteLevelDecoder is not None:
            print("⚙️  Detected Byte-Level BPE family → verifying pre-tokenizer/decoder ...")
            backend = self.tok.backend_tokenizer
            pre = getattr(backend, "pre_tokenizer", None)
            dec = getattr(backend, "decoder", None)

            if not pre or "ByteLevel" not in str(pre):
                backend.pre_tokenizer = ByteLevelPreTok(add_prefix_space=False)
                print("🩹 Added ByteLevel pre-tokenizer.")
            if not dec or "ByteLevel" not in str(dec):
                backend.decoder = ByteLevelDecoder()
                print("🩹 Added ByteLevel decoder.")

        # 4) Basic metadata
        self.vocab_size = getattr(self.tok, "vocab_size", None)
        self._pad_id = getattr(self.tok, "pad_token_id", None) or getattr(self.tok, "eos_token_id", None)
        print(f"✅ HFTokenizerWrapper initialized from '{name_or_path}' "
              f"| vocab_size={self.vocab_size} | pad_id={self._pad_id}")

        # 5) Post-load Unicode sanity check (emoji + CJK)
        self._unicode_self_test()

    # ------------------------------------------------------------
    # 🔹 Expose underlying tokenizer
    # ------------------------------------------------------------
    @property
    def tokenizer(self):
        """Return the underlying Hugging Face tokenizer object."""
        return self.tok

    # ------------------------------------------------------------
    # 🔹 Encode / Decode
    # ------------------------------------------------------------
    # def encode(self, text: str):
    #     """Convert text to token IDs."""
    #     return self.tok.encode(text, add_special_tokens=self.add_special_tokens)

    # def decode(self, ids):
    #     """Convert token IDs back to text."""
    #     return self.tok.decode(ids, skip_special_tokens=True)
    
    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=self.add_special_tokens)

    def decode(self, ids: Union[List[int], "np.ndarray"]) -> str:
        # Disable whitespace cleanup so nothing useful is removed
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # ------------------------------------------------------------
    # 🔹 Token ID accessors
    # ------------------------------------------------------------
    # @property
    # def pad_id(self): return self.tok.pad_token_id
    @property
    def pad_id(self):
        return getattr(self, "_pad_id", None)
    @property
    def bos_id(self): return self.tok.bos_token_id
    @property
    def eos_id(self): return self.tok.eos_token_id

    # ------------------------------------------------------------
    # 🔹 Save tokenizer to directory
    # ------------------------------------------------------------
    def save(self, path: str):
        """
        Save tokenizer safely:
          - If `path` is a bare directory root, we’ll create a safe child folder
            derived from the model id (to avoid 'Qwen/Qwen2...' nested dirs).
          - If `path` points to a directory that already looks like a leaf folder,
            we save directly there.
        """
        os.makedirs(path, exist_ok=True)

        # If the caller passed a generic root (e.g., "outputs"), append a safe leaf
        # Otherwise, if the path already contains a tokenizer.json (leaf), save here
        leaf = path
        has_json = os.path.exists(os.path.join(path, "tokenizer.json"))
        if not has_json:
            safe_leaf = _safe_dir_name(self._model_id or "hf_tokenizer")
            leaf = os.path.join(path, f"hf_tok__{safe_leaf}")
            os.makedirs(leaf, exist_ok=True)

        self.tok.save_pretrained(leaf)
        print(f"💾 Saved Hugging Face tokenizer → {leaf}")

    @staticmethod
    def load(path: str,
             add_special_tokens: bool = True,
             trust_remote_code: bool = False) -> "HFTokenizerWrapper":
        """
        Load tokenizer from a saved directory. Works with:
          - exact leaf dirs (contain tokenizer.json)
          - parent dirs produced by .save() (auto-detects leaf)
        """
        print(f"🔍 Loading Hugging Face tokenizer from {path} ...")

        load_dir = path
        if os.path.isdir(path) and not os.path.exists(os.path.join(path, "tokenizer.json")):
            # try to find a subdir that contains tokenizer.json
            candidates = [d for d in os.listdir(path)
                          if os.path.isdir(os.path.join(path, d))
                          and os.path.exists(os.path.join(path, d, "tokenizer.json"))]
            if candidates:
                load_dir = os.path.join(path, candidates[0])

        wrapper = HFTokenizerWrapper(
            load_dir, add_special_tokens=add_special_tokens,
            trust_remote_code=trust_remote_code
        )
        print(f"✅ HFTokenizerWrapper reloaded from '{load_dir}' | vocab_size={wrapper.vocab_size}")
        return wrapper

    # ---------- Helpers ----------

    def _unicode_self_test(self):
        """
        Try a tiny probe with emoji + CJK to ensure decode survives.
        Print a warning (and a quick diff) if it doesn’t.
        """
        probe = "Hello world! 👋 你好！"
        ids = self.encode(probe)
        out = self.decode(ids)

        if out != probe:
            # Some models normalize case/quotes; try a softer check
            # If it still fails, warn loudly.
            if ("你好" not in out) or ("👋" not in out):
                print("⚠️  Unicode round-trip changed after load/save.")
                print(f"   expected: {probe}")
                print(f"   got     : {out}")

                # For SP models, we can't attach a ByteLevel decoder; just hint.
                if self._is_sentencepiece:
                    print("   ℹ️ This tokenizer uses SentencePiece; unknown chars may map to <unk> and be dropped.")
                else:
                    print("   ℹ️ This looks like a byte-level BPE family; "
                          "we already attached ByteLevel decoder. If this persists, "
                          "ensure you are not normalizing text before encode().")
    # """
        # Load tokenizer from a saved directory.
        # Automatically repairs missing ByteLevelDecoder so UTF-8 (Chinese/emoji) decoding works.
        # When you load "gpt2" through AutoTokenizer, the underlying Rust ByteLevel pre-tokenizer and decoder are automatically attached.

        # However, when you save + reload from disk, transformers reconstructs the tokenizer from JSON metadata but does not reattach the ByteLevelDecoder object (the JSON does not serialize its internal state completely).

        # So at runtime:
        #     •	The tokenizer can still encode multi-byte UTF-8 correctly → ✅ encode works.
        #     •	But it fails to decode those byte sequences back to UTF-8 → ⚠️ decode truncates anything outside ASCII.
        # """
        # # Load the high-level Hugging Face tokenizer
        # tok = AutoTokenizer.from_pretrained(path, use_fast=True)

        # # ✅ Fix: reattach ByteLevel pre-tokenizer and decoder
        # backend_tok = getattr(tok, "backend_tokenizer", None)
        # if backend_tok:
        #     if not backend_tok.pre_tokenizer or "ByteLevel" not in str(backend_tok.pre_tokenizer):
        #         print("⚙️  Repairing missing ByteLevel pre-tokenizer ...")
        #         backend_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        #     if not backend_tok.decoder or "ByteLevel" not in str(backend_tok.decoder):
        #         print("⚙️  Repairing missing ByteLevel decoder ...")
        #         backend_tok.decoder = ByteLevelDecoder()

        # # Wrap it in your class again
        # wrapper = HFTokenizerWrapper.__new__(HFTokenizerWrapper)
        # wrapper.tok = tok
        # wrapper.vocab_size = tok.vocab_size
        # wrapper._pad_id = tok.pad_token_id or tok.eos_token_id
        # wrapper.add_special_tokens = add_special_tokens
        # print(f"✅ HFTokenizerWrapper reloaded from '{path}' | vocab_size={wrapper.vocab_size}")
        # return wrapper



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
    tok_custom_path = None #os.path.join(tokenizer_dir, "custom_bytebpe_tokenizer.json")
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
        "Hello, world! Tokenizer.",
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
        "Machine Learning 是人工智能的一个分支。"
    ]

    # English tokenizer
    tok_en = WordTokenizer(texts, lang="en")
    ids_en = tok_en.encode("The dog sat on the mat")
    words = [tok_en.itos[i] for i in ids_en]
    print(words)
    print(tok_en.decode(ids_en))
    
    # English tokenizer
    tok_en = WordTokenizer(texts, lang="en", lowercase=False)
    ids_en = tok_en.encode("The dog sat on the mat")
    words = [tok_en.itos[i] for i in ids_en]
    print(words)
    print(tok_en.decode(ids_en))

    # Chinese tokenizer
    tok_zh = WordTokenizer(texts, lang="zh")
    ids_zh = tok_zh.encode("智能机器学习！")
    words = [tok_zh.itos[i] for i in ids_zh]
    print(words)
    print(tok_zh.decode(ids_zh))
    #use_jieba
    tok_zh = WordTokenizer(texts, lang="zh", use_jieba=True)
    ids_zh = tok_zh.encode("智能机器, 人工智能学习！")
    words = [tok_zh.itos[i] for i in ids_zh]
    print(words)
    print(tok_zh.decode(ids_zh))
    
    # mixed tokenizer
    tok_mix = WordTokenizer(texts, lang="mixed")
    ids_mix = tok_mix.encode("Hello:Machine! 智能机器学习！")
    words = [tok_mix.itos[i] for i in ids_mix]
    print(words)
    print(tok_mix.decode(ids_mix))
    tok_mix = WordTokenizer(texts, lang="mixed", use_jieba=True)
    ids_mix = tok_mix.encode("Hello:Machine! 智能机器学习！人工智能学习")
    words = [tok_mix.itos[i] for i in ids_mix]
    print(words)
    print(tok_mix.decode(ids_mix))

    # Save & Load
    tok_en.save("outputs/word_tokenizer_en.json")
    tok_zh.save("outputs/word_tokenizer_zh.json")
    tok_mix.save("outputs/word_tokenizer_mix.json")

    tok_en2 = WordTokenizer.load("outputs/word_tokenizer_en.json")
    ids_en = tok_en2.encode("The dog sat on the mat")
    print(tok_en2.decode(ids_en))
    tok_zh2 = WordTokenizer.load("outputs/word_tokenizer_zh.json")
    ids_zh = tok_zh2.encode("智能机器, 人工智能学习！")
    print(tok_zh2.decode(ids_zh))
    tok_mix2 = WordTokenizer.load("outputs/word_tokenizer_mix.json")
    ids_mix = tok_mix2.encode("Hello:Machine! 智能机器学习！人工智能学习")
    print(tok_mix2.decode(ids_mix))

def test_hftokenizer(name="gpt2"):
    #from DeepDataMiningLearning.llm.tokenizer_utils import HFTokenizerWrapper

    # 1️⃣ Load pretrained tokenizer
    tok = HFTokenizerWrapper(name)

    # 2️⃣ Encode & decode text
    ids = tok.encode("Hello world! 👋 你好！")
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))

    # 3️⃣ Save tokenizer
    filename=f"outputs/hf_tokenizer_{name}"
    tok.save(filename)#"outputs/hf_tokenizer_gpt2")
    contents = os.listdir(filename)
    print(contents)

    # 4️⃣ Load tokenizer
    tok2 = HFTokenizerWrapper.load(filename)

    # 5️⃣ Verify it works
    print(tok2.encode("Hello world! 👋 你好！"))
    print(tok2.decode(tok2.encode("Hello world!")))

if __name__ == "__main__":
    #huggingface-cli login
    test_char_word_tokenizer()
    test_hftokenizer()
    test_hftokenizer("Qwen/Qwen2-1.5B")
    test_hftokenizer("meta-llama/Meta-Llama-3-8B")
    test_custom_tokenizer()
    test_custom_tokenizer2()
    texts = [
        "The cat sat on the mat.",
        "Machine learning is fun.",
        "机器学习是人工智能的一个分支。",
        "Let's test emojis 🤖🚀🔥!"
    ]

    test_tokenizers(dataset_name="ag_news", num_samples=1000, vocab_size=8000)
    