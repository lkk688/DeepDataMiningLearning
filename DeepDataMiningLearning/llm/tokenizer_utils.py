
from typing import List, Literal, Union, Optional, Tuple, Dict
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

# pip install sentencepiece tiktoken
import os
from typing import List, Literal, Optional, Dict, Any

class CustomTokenizer:
    """
    Unified tokenizer wrapper with two robust, production-grade backends:

    - "sp-unigram": SentencePiece Unigram + byte fallback (LLaMA3/Gemma/Mistral style)
    - "tiktoken-bpe": OpenAI cl100k_base (GPT-4-Turbo/Qwen 2.5 family)

    Design goals:
      ✅ True round-trip for multilingual (no mojibake) when normalization is off
      ✅ No training surprises: SP uses sentencepiece, BPE uses prebuilt tiktoken
      ✅ Keep encode()/decode()/show_tokens()/plot_vocab_distribution APIs
      ✅ Play nice with your existing Char/Word/HF tokenizer wrappers
    """

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        tokenizer_type: Literal["sp-unigram", "tiktoken-bpe"] = "sp-unigram",
        tokenizer_path: str = "outputs/custom_tokenizer",  # will use .model for SP, .json for metadata; tiktoken uses its built-in name
        vocab_size: int = 8000,
        min_freq: int = 2,             # kept for API symmetry; SP/tiktoken ignore this
        pad_id: int = -100,
        add_special_tokens: bool = True,
        lowercase: bool = False,       # 🚫 keep False for round-trip
        sp_normalization: str = "nmt_nfkc",  # SentencePiece normalization rule
        tiktoken_encoding: str = "cl100k_base",  # OpenAI/GPT-4/Qwen vocab
    ):
        self.kind = tokenizer_type
        self.pad_id = pad_id
        self.add_special_tokens = add_special_tokens
        self.lowercase = lowercase

        # shared attributes exposed for your training loop
        self.vocab_size: int = 0
        self.pad_idx: Optional[int] = None
        self.eos_idx: Optional[int] = None
        self.bos_idx: Optional[int] = None
        self.unk_idx: Optional[int] = None

        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        self.tokenizer_path = tokenizer_path

        if tokenizer_type == "sp-unigram":
            self._init_sentencepiece(
                texts=texts,
                model_prefix=tokenizer_path,     # will create tokenizer_path + ".model" & ".vocab"
                vocab_size=vocab_size,
                sp_norm=sp_normalization,
            )
        elif tokenizer_type == "tiktoken-bpe":
            self._init_tiktoken(encoding_name=tiktoken_encoding)
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        print("🧩 UTF-8 safe tokenizer initialized and reversible.")

    # ---------- SentencePiece backend ----------
    def _init_sentencepiece(self, texts: Optional[List[str]], model_prefix: str, vocab_size: int, sp_norm: str):
        import sentencepiece as spm

        model_file = f"{model_prefix}.model"
        if not os.path.exists(model_file):
            if not texts:
                raise ValueError("sp-unigram requires `texts` to train.")
            corpus = f"{model_prefix}.txt"
            with open(corpus, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write((t.lower() if self.lowercase else t) + "\n")

            print(f"🚀 Training SentencePiece Unigram (byte_fallback=True) → {model_file}")
            spm.SentencePieceTrainer.Train(
                input=corpus,
                model_prefix=model_prefix,
                model_type="unigram",
                vocab_size=vocab_size,
                byte_fallback=True,         # ✅关键：字节回退
                normalization_rule_name=sp_norm,
                character_coverage=1.0,
                input_sentence_size=2000000,
                shuffle_input_sentence=True,
                # add_dummy_prefix 默认 True（会产生空格标记），与 LLaMA 系一样
                # 特殊符号：sp 默认保留 <unk>=0，<s>=1，</s>=2，可与项目约定映射
            )
            print(f"💾 Saved SP model: {model_file}")

        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=model_file)
        self.sp = sp
        # map ids like LLaMA style (<unk>=0, <s>=1, </s>=2). We expose as eos/bos/unk.
        self.unk_idx = sp.unk_id()
        self.bos_idx = sp.bos_id() if sp.bos_id() >= 0 else None
        self.eos_idx = sp.eos_id() if sp.eos_id() >= 0 else None
        self.pad_idx = self.pad_id  # SentencePiece本身没有pad，训练时用 ignore_index/-100 即可
        self.vocab_size = sp.vocab_size()
        print(f"✅ SP loaded | vocab_size={self.vocab_size} | ids: unk={self.unk_idx}, bos={self.bos_idx}, eos={self.eos_idx}")

    # ---------- tiktoken backend ----------
    def _init_tiktoken(self, encoding_name: str):
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)  # e.g., "cl100k_base"
        self.tk = enc
        # tiktoken没有显式 pad/eos/bos；你可以在任务层面管理它们。
        self.pad_idx = self.pad_id
        self.eos_idx = None
        self.bos_idx = None
        self.unk_idx = None
        # 近似曝光 vocab size（tiktoken 暴露 n_vocab）
        self.vocab_size = getattr(enc, "n_vocab", 0)
        print(f"✅ tiktoken loaded | encoding={encoding_name} | vocab_size≈{self.vocab_size}")

    # ---------- Public API ----------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        t = text.lower() if self.lowercase else text
        if self.kind == "sp-unigram":
            ids = self.sp.encode(t, out_type=int)  # unicode safe
            if add_bos and self.bos_idx is not None:
                ids = [self.bos_idx] + ids
            if add_eos and self.eos_idx is not None:
                ids = ids + [self.eos_idx]
            return ids
        else:  # tiktoken
            import tiktoken
            ids = self.tk.encode(t, allowed_special=set())  # no special by default
            # 如果你想在任务层添加 eos，请在上层做（保持 encode/decode 可逆）
            return ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        if self.kind == "sp-unigram":
            # SentencePiece 会忽略 <s>,</s>；<unk> 会解为 <unk> 字符串（不乱码）
            return self.sp.decode(ids)
        else:
            # tiktoken 提供 decode
            return self.tk.decode(ids)

    def show_tokens(self, text: str, max_show: int = 50):
        ids = self.encode(text)
        print(f"\n🔍 Tokenization Debug — Input: {repr(text)}")
        print(f"Total tokens: {len(ids)} (showing up to {max_show})\n")
        for i, tid in enumerate(ids[:max_show]):
            if self.kind == "sp-unigram":
                piece = self.sp.id_to_piece(tid)
            else:
                # tiktoken: 单 token -> bytes，再转 utf-8
                piece = self.tk.decode_single_token_bytes(tid).decode("utf-8", errors="replace")
            print(f"[{i:3d}] {tid:<6} → {repr(piece)}")

    def save(self, path: Optional[str] = None):
        """
        For SP: model already saved as `.model`.
        For tiktoken: nothing to save; persist the chosen encoding name in a small json.
        """
        import json
        path = path or self.tokenizer_path
        meta_path = path + (".sp.json" if self.kind == "sp-unigram" else ".tk.json")
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "path": self.tokenizer_path,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "lowercase": self.lowercase,
        }
        if self.kind == "tiktoken-bpe":
            payload["tiktoken_encoding"] = getattr(self.tk, "name", "cl100k_base")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved tokenizer meta → {meta_path}")

    @staticmethod
    def load(path: str):
        """
        Auto-load by reading the side meta json.
        """
        import json
        # try SP meta first
        sp_meta = path + ".sp.json"
        tk_meta = path + ".tk.json"
        if os.path.exists(sp_meta):
            with open(sp_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            obj = CustomTokenizer.__new__(CustomTokenizer)
            obj.kind = "sp-unigram"
            obj.tokenizer_path = path
            obj.pad_id = meta.get("pad_id", -100)
            obj.lowercase = meta.get("lowercase", False)
            obj._init_sentencepiece(texts=None, model_prefix=path, vocab_size=0, sp_norm="nmt_nfkc")
            return obj
        elif os.path.exists(tk_meta):
            with open(tk_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            obj = CustomTokenizer.__new__(CustomTokenizer)
            obj.kind = "tiktoken-bpe"
            obj.tokenizer_path = path
            obj.pad_id = meta.get("pad_id", -100)
            obj.lowercase = meta.get("lowercase", False)
            enc_name = meta.get("tiktoken_encoding", "cl100k_base")
            obj._init_tiktoken(enc_name)
            return obj
        else:
            raise FileNotFoundError(f"No meta found for tokenizer at {path} (.sp.json/.tk.json)")

    # simple frequency plot on the given texts
    def plot_vocab_distribution(self, texts: List[str], top_k: int = 40, save_dir="outputs/tokenizer_analysis", show=False):
        import os, collections, matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        counter = collections.Counter()
        for t in texts:
            counter.update(self.encode(t))
        most = counter.most_common(top_k)
        xs = [tid for tid, _ in most]
        ys = [cnt for _, cnt in most]
        if self.kind == "sp-unigram":
            labels = [self.sp.id_to_piece(tid) for tid in xs]
        else:
            labels = [self.tk.decode_single_token_bytes(tid).decode("utf-8", "replace") for tid in xs]

        plt.figure(figsize=(max(8, top_k*0.3), 4))
        plt.bar(range(len(xs)), ys)
        plt.xticks(range(len(xs)), labels, rotation=90)
        plt.tight_layout()
        out_path = os.path.join(save_dir, "vocab_distribution_top{}.png".format(top_k))
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()
        print(f"💾 Saved vocabulary distribution figure → {out_path}")
    

import os
import tempfile

def estimate_vocab_size(
    corpus_path: str = None,
    texts: list[str] = None,
    max_vocab: int = 50000,
    safety_factor: float = 0.8,
) -> int:
    """
    Estimate a safe vocab_size for SentencePiece tokenizer training.

    Automatically adapts based on the diversity of the corpus.
    Works with either a file path or an in-memory list of texts.

    Args:
        corpus_path (str, optional): Path to corpus text file.
        texts (list[str], optional): In-memory corpus as list of strings.
        max_vocab (int): Upper limit for vocab size.
        safety_factor (float): Multiplier to stay below theoretical max.

    Returns:
        int: Recommended vocab_size for training.
    """
    if corpus_path is None and texts is None:
        raise ValueError("❌ Must provide either `corpus_path` or `texts`.")

    # ------------------------------------------------------------
    # 1️⃣ Load corpus text (from file or in-memory)
    # ------------------------------------------------------------
    if texts is not None:
        text_data = "\n".join(texts)
        # Optionally create a temporary file for SP training reuse
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="temp_corpus_")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text_data)
        corpus_path = tmp_path
    else:
        # Read directly from existing file
        with open(corpus_path, "r", encoding="utf-8") as f:
            text_data = f.read()

    # ------------------------------------------------------------
    # 2️⃣ Compute statistics
    # ------------------------------------------------------------
    total_chars = len(text_data)
    unique_chars = len(set(text_data))

    # Prevent division by zero
    if unique_chars == 0 or total_chars == 0:
        raise ValueError("❌ Empty corpus or no valid characters found.")

    # ------------------------------------------------------------
    # 3️⃣ Heuristic vocab estimation formula
    # ------------------------------------------------------------
    # Empirical heuristic: each unique char may expand into ~20–50 subwords.
    ratio = 20 + (unique_chars ** 0.5)  # adaptively increase with diversity
    est_vocab = int(unique_chars * ratio * safety_factor)

    # Bound vocab size to sensible range
    est_vocab = max(300, min(est_vocab, max_vocab))

    # ------------------------------------------------------------
    # 4️⃣ Display statistics
    # ------------------------------------------------------------
    num_lines = text_data.count("\n") + 1
    print(f"📊 Corpus statistics:")
    print(f"   • Lines          : {num_lines:,}")
    print(f"   • Total chars    : {total_chars:,}")
    print(f"   • Unique chars   : {unique_chars:,}")
    print(f"🧮 Estimated vocab_size ≈ {est_vocab}")

    # Cleanup temporary file (if created)
    if texts is not None and os.path.exists(corpus_path):
        os.remove(corpus_path)

    return est_vocab

class TokenizerFactory:
    """
    Unified Tokenizer Builder / Loader.

    Automatically builds or loads tokenizers based on user options:
      - "char"                   → CharTokenizer
      - "word"                   → WordTokenizer
      - "hf:xx"                  → HFTokenizerWrapper for pretrained Hugging Face tokenizers
      - "custom:tokenizer_type"  → CustomTokenizer (e.g. "custom:bytebpe", "custom:sp-unigram")
      - "auto"                   → Auto-detect and load from a provided path
    """

    @staticmethod
    def build(
        tokenizer: str,
        texts: List[str] = None,
        tokenizer_path: str = None,
        vocab_size: int = None,
        min_freq: int = 2,
        pad_id: int = -100,
        lowercase: bool = True,
    ):
        """
        Build or load tokenizer automatically based on user input.

        Args:
            tokenizer (str): Tokenizer specifier ("char", "word", "hf:gpt2", "custom:bytebpe", "custom:sp-unigram", etc.)
            texts (List[str]): Training corpus (required for training custom tokenizers).
            tokenizer_path (str): Optional tokenizer path to load from or save to.
            vocab_size (int): Vocabulary size for custom tokenizers. If None, will auto-detect.
            min_freq (int): Minimum token frequency for custom tokenizers.
            pad_id (int): Default pad_id, consistent with PyTorch ignore_index.
            lowercase (bool): Convert text to lowercase for char/word tokenizers.

        Returns:
            tokenizer_instance: A tokenizer object with unified encode()/decode()/save()/load() API.
        """

        # ✅ 1️⃣ Auto-detect and load existing tokenizer
        if tokenizer == "auto":
            if tokenizer_path and os.path.exists(tokenizer_path):
                print(f"🔍 Auto-detecting tokenizer from {tokenizer_path} ...")
                # Try known types in order of complexity
                for loader in (HFTokenizerWrapper, CustomTokenizer, CharTokenizer, WordTokenizer):
                    try:
                        tok = loader.load(tokenizer_path)
                        print(f"✅ Auto-loaded tokenizer type: {loader.__name__}")
                        return tok
                    except Exception:
                        continue
                raise ValueError(f"❌ Unable to auto-detect tokenizer type from {tokenizer_path}.")
            else:
                raise ValueError("❌ 'auto' mode requires a valid tokenizer_path to load from.")

        # ✅ 2️⃣ Hugging Face pretrained tokenizers
        elif tokenizer.startswith("hf:"):
            name_or_path = tokenizer.split("hf:")[1]
            print(f"🤗 Loading Hugging Face tokenizer: {name_or_path}")
            return HFTokenizerWrapper(name_or_path)

        # ✅ 3️⃣ Custom tokenizer family (ours)
        elif tokenizer.startswith("custom:"):
            algo = tokenizer.split("custom:")[1]
            save_path = tokenizer_path or f"outputs/custom_{algo}_tokenizer"

            # Auto-create save dir
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"🧠 Building CustomTokenizer ({algo}) → {save_path}")

            # Optional: auto-detect vocab size (only for SP-style tokenizers)
            if vocab_size is None and algo in ["sp-unigram", "bytebpe", "bpe"]:
                if texts is None or len(texts) == 0:
                    raise ValueError("❌ Need non-empty texts to estimate vocab_size automatically.")
                corpus_path = "outputs/temp_corpus.txt"
                with open(corpus_path, "w", encoding="utf-8") as f:
                    for t in texts:
                        f.write(t.strip() + "\n")
                vocab_size = estimate_vocab_size(corpus_path=corpus_path)
                print(f"📏 Auto-detected vocab_size={vocab_size}")

            # Instantiate CustomTokenizer with correct parameters
            return CustomTokenizer(
                texts=texts,
                tokenizer_type=algo,
                tokenizer_path=save_path,
                vocab_size=vocab_size or 8000,
                min_freq=min_freq,
                pad_id=pad_id,
                lowercase=lowercase,
            )

        # ✅ 4️⃣ Character tokenizer
        elif tokenizer == "char":
            print("🔤 Building CharTokenizer ...")
            if tokenizer_path and os.path.exists(tokenizer_path):
                return CharTokenizer.load(tokenizer_path)
            return CharTokenizer(texts, add_eos=True, add_pad=True)

        # ✅ 5️⃣ Word tokenizer
        elif tokenizer == "word":
            print("🧩 Building WordTokenizer ...")
            if tokenizer_path and os.path.exists(tokenizer_path):
                return WordTokenizer.load(tokenizer_path)
            return WordTokenizer(texts, lang="en")

        else:
            raise ValueError(f"❌ Unknown tokenizer option: {tokenizer}")



def test_tokenizers(
    dataset_name: str = "ag_news",
    num_samples: int = 1000,
    vocab_size: int = 5000,
    tokenizer_dir: str = "outputs/",
):
    """
    🔍 Comprehensive tokenizer test suite (realistic version).

    Tests:
      - CharTokenizer
      - WordTokenizer
      - HFTokenizerWrapper (GPT-2)
      - CustomTokenizer (ByteBPE)
      - CustomTokenizer (SentencePiece-Unigram)
      - CustomTokenizer (Tiktoken-BPE)
      - Auto-detected reload
    """
    import os, re, unicodedata
    from typing import List
    from datasets import load_dataset
    
    print("\n🔍 ===== REALISTIC TOKENIZER TEST SUITE =====\n")

    # ------------------------------------------------------------
    # 1️⃣  Load a realistic dataset
    # ------------------------------------------------------------
    print(f"📘 Loading dataset: {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")
    text_field = "text" if "text" in ds.column_names else ds.column_names[0]
    texts = [t.strip() for t in ds[text_field] if isinstance(t, str) and len(t.strip()) > 0]
    texts = texts[:num_samples]
    print(f"✅ Loaded {len(texts)} text samples for tokenizer training.\n")
    test_samples = texts[:3]

    # ------------------------------------------------------------
    # 2️⃣  Normalizer for fair comparisons
    # ------------------------------------------------------------
    def normalize(text: str):
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("Ġ", " ").replace("▁", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    # ------------------------------------------------------------
    # 3️⃣  Helper to inspect a tokenizer
    # ------------------------------------------------------------
    def _inspect_tokenizer(tok, samples: List[str]):
        print(f"🧩 Tokenizer class: {type(tok).__name__}")
        if hasattr(tok, "vocab_size"):
            print(f"🔡 Vocab size: {tok.vocab_size}")
        if hasattr(tok, "tokenizer_path"):
            print(f"💾 Path: {getattr(tok, 'tokenizer_path', None)}")

        specials = {n: getattr(tok, n)
                    for n in ["pad_idx", "bos_idx", "eos_idx", "unk_idx"]
                    if hasattr(tok, n)}
        if specials:
            print(f"🔖 Special token indices: {specials}")

        for i, s in enumerate(samples, 1):
            print(f"\n🔹 Sample {i}: {s[:100]}{'...' if len(s)>100 else ''}")
            try:
                ids = tok.encode(s)
                decoded = tok.decode(ids)
                print(f"   🔢 Token IDs ({len(ids)}): {ids[:40]}{'...' if len(ids) > 40 else ''}")
                print(f"   🔁 Decoded: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")
                ok = normalize(decoded) == normalize(s)
                print(f"   {'✅' if ok else '⚠️'} Round-trip match: {ok}")
            except Exception as e:
                print(f"   ⚠️ Encode/decode error: {e}")
        print("")

    # ------------------------------------------------------------
    # 4️⃣  Run the suite
    # ------------------------------------------------------------
    print("\n==================== 🟦 CHAR TOKENIZER ====================")
    tok_char = TokenizerFactory.build("char", texts=texts)
    _inspect_tokenizer(tok_char, test_samples)

    print("\n==================== 🟩 WORD TOKENIZER ====================")
    tok_word = TokenizerFactory.build("word", texts=[t.lower() for t in texts])
    _inspect_tokenizer(tok_word, test_samples)

    print("\n==================== 🤗 HF TOKENIZER (gpt2) ====================")
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        tok_hf = TokenizerFactory.build("hf:gpt2")
        _inspect_tokenizer(tok_hf, test_samples)
    except Exception as e:
        print(f"⚠️ HF tokenizer load failed: {e}")


    print("\n==================== 🧠 CUSTOM TOKENIZER (SentencePiece-Unigram) ====================")
    tok_sp = TokenizerFactory.build(
        "custom:sp-unigram",
        texts=None, #texts,
        tokenizer_path=os.path.join(tokenizer_dir, "custom_sp_unigram"),
        vocab_size=None,
    )
    _inspect_tokenizer(tok_sp, test_samples)

    print("\n==================== 🧠 CUSTOM TOKENIZER (Tiktoken-BPE) ====================")
    tok_tk = TokenizerFactory.build(
        "custom:tiktoken-bpe",
        texts=None,   # no training needed
        tokenizer_path=os.path.join(tokenizer_dir, "custom_tiktoken_bpe"),
    )
    _inspect_tokenizer(tok_tk, test_samples)

    print("\n✅ All tokenizer backend tests completed successfully.\n")

def test_custom_tokenizer():
    texts = [
        "The cat sat on the mat.",
        "Hello world! Python coding.",
        "Machine Learning 是人工智能的一个分支。",
        "Hello, world! 你好，世界！",
        "GPT tokenization test 🚀🔥",
    ]

    # 1) LLaMA/Gemma style (SP Unigram + byte-fallback)
    tok_sp = CustomTokenizer(texts, tokenizer_type="sp-unigram", tokenizer_path="outputs/sp_unigram", vocab_size=304)
    tok_sp.show_tokens("Hello, 世界！ LLaMA 3 模型测试。")
    print("SP round-trip:", tok_sp.decode(tok_sp.encode("Hello, 世界！ LLaMA 3 模型测试。")))

    # 2) GPT/Qwen style (tiktoken cl100k_base)
    tok_bpe = CustomTokenizer(tokenizer_type="tiktoken-bpe")  # 不需要 texts
    tok_bpe.show_tokens("GPT-4 Turbo tokenizer 测试。🚀🔥")
    print("BPE round-trip:", tok_bpe.decode(tok_bpe.encode("GPT-4 Turbo tokenizer 测试。🚀🔥")))

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
    #Train or load LLaMA3 / Gemma style (SentencePiece Unigram)
    texts = [
        "Hello world! 你好，世界！",
        "Machine learning 是人工智能的重要分支。",
        "LLaMA 3 uses Unigram tokenization.",
    ]
    tok = TokenizerFactory.build("custom:sp-unigram", texts=texts, vocab_size=None)
    tok.show_tokens("Hello, 世界！ LLaMA 3 模型测试。")

    #Load GPT / Qwen compatible tokenizer (Tiktoken)
    tok = TokenizerFactory.build("custom:tiktoken-bpe")
    tok.show_tokens("GPT-4 Turbo tokenizer 测试。🚀🔥")

    test_tokenizers(dataset_name="ag_news", num_samples=1000, vocab_size=8000)
    