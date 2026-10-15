"""Config system: YAML files + `key.sub=value` command-line overrides -> typed dataclasses.

Design goals (borrowed from how we ran parameter-golf sweeps):
  * an experiment is *a config file*, never an edited line of code;
  * configs can inherit (`defaults: [base.yaml]`) so an ablation is a 3-line diff;
  * every key is typed and validated -- a typo such as `model.n_layer=6` is an error,
    not a silently-ignored setting (the #1 source of "my ablation did nothing" bugs);
  * the fully-resolved config is written next to every run's results.

Usage:
    cfg = load_config("configs/03_gpt_fineweb.yaml", ["model.n_layers=6", "train.lr=1e-3"])
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


# ----------------------------------------------------------------------------- data
@dataclass
class DataConfig:
    # tinyshakespeare | text | hf | shards
    #   text   : local UTF-8 file(s) given in `files`
    #   hf     : HuggingFace dataset `hf_name` (column(s) `hf_text_fields`)
    #   shards : pre-tokenized parameter-golf style .bin shards (`shard_dir`)
    source: str = "tinyshakespeare"
    files: list = field(default_factory=list)
    hf_name: str = ""
    hf_config: str = ""
    hf_split: str = "train"
    hf_text_fields: list = field(default_factory=lambda: ["text"])
    hf_max_docs: int = 0                 # 0 = all
    # char | byte | gpt2 | sp:<path/to/model>   (sp = SentencePiece, e.g. golf's sp1024)
    tokenizer: str = "char"
    val_fraction: float = 0.1
    split_by: str = "tail"               # tail (contiguous last chunk) | doc (seeded doc shuffle)
    shard_dir: str = ""                  # for source=shards: dir with *_train_*.bin / *_val_*.bin
    train_pattern: str = "*_train_*.bin"
    val_pattern: str = "*_val_*.bin"
    max_train_shards: int = 0            # 0 = all (1 golf shard = 100M tokens)
    cache_dir: str = "data/nanolm"       # where tokenized .bin caches are written
    sampler: str = "random"              # random windows | sequential stream (golf style)


# ---------------------------------------------------------------------------- model
@dataclass
class ModelConfig:
    arch: str = "gpt"                    # rnn | gru | lstm | gpt
    vocab_size: int = 0                  # 0 = take from tokenizer
    dim: int = 256
    n_layers: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True
    # ---- recurrent models
    rnn_hidden: int = 0                  # 0 = same as dim
    # ---- transformer: the "textbook" knobs
    n_heads: int = 4
    n_kv_heads: int = 0                  # 0 = n_heads (MHA); < n_heads = GQA
    max_seq_len: int = 1024
    norm: str = "rmsnorm"                # layernorm | rmsnorm
    pos: str = "rope"                    # learned | rope | none
    rope_base: float = 10000.0
    mlp: str = "gelu"                    # gelu | swiglu | relu2 | leaky_relu2
    mlp_mult: float = 4.0
    bias: bool = False
    init_std: float = 0.02
    embed_init_std: float = 0.0          # 0 = init_std (golf tied embedding: 0.005)
    # ---- parameter-golf knobs (all default OFF => plain GPT)
    qk_norm: bool = False                # RMSNorm on q and k (golf baseline)
    qk_gain_init: float = 0.0            # >0: learnable per-head query gain (golf: 1.5 -> 5.0)
    logit_softcap: float = 0.0           # >0: cap*tanh(logits/cap)
    rope_dims: int = 0                   # partial RoPE: rotate only first k dims of each head (0 = all)
    zero_init_proj: bool = False         # zero-init output projections (block starts as identity)
    embed_norm: bool = False             # RMSNorm right after the token embedding
    layer_scales: bool = False           # learnable per-channel attn_scale / mlp_scale
    resid_mix: bool = False              # learnable mix of current stream with x0 (embedding)
    ln_scale: bool = False               # scale normed input of layer i by 1/sqrt(i+1)
    unet_skips: bool = False             # encoder-half outputs added to mirrored decoder layers
    recur_layers: list = field(default_factory=list)  # e.g. [3,4,5]: re-run these layers
    recur_times: int = 1                 # extra passes over recur_layers
    recur_start_frac: float = 0.0        # turn recurrence on after this fraction of training
    parallel_from: int = -1              # >=0: parallel attn/MLP residual from this layer on
    xsa_last_n: int = 0                  # exclusive self-attention in the last N layers
    smear_gate: bool = False             # gated mix of previous token's embedding
    bigram_buckets: int = 0              # >0: hashed (prev,cur) bigram embedding table
    bigram_dim: int = 128
    value_residual: bool = False         # mix layer-0 values into every later layer
    attn_gate: bool = False              # per-head sigmoid gate on the attention output


# ---------------------------------------------------------------------------- train
@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 32                 # sequences per micro-batch (per device)
    grad_accum: int = 1
    max_steps: int = 2000
    max_time_sec: float = 0.0            # >0: wall-clock budget (golf: 600 s), warmdown adapts
    optimizer: str = "adamw"             # adamw | muon (Muon on matrices + Adam elsewhere)
    lr: float = 3e-4                     # AdamW lr (adamw) / Adam lr for non-matrix params (muon)
    embed_lr: float = 0.0                # 0 = same as lr
    matrix_lr: float = 0.02              # Muon lr for 2-D hidden weights
    weight_decay: float = 0.1
    muon_wd: float = 0.0                 # decoupled WD on Muon params (golf: 0.04 - 0.09)
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    betas: list = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0               # 0 = off
    schedule: str = "cosine"             # cosine | wsd (warmup-stable-decay / golf warmdown) | constant
    warmup_steps: int = 100
    warmdown_frac: float = 0.2           # wsd: fraction of training spent decaying to min_lr
    min_lr_frac: float = 0.1
    ema_decay: float = 0.0               # >0: keep an EMA of weights, evaluate with it
    qat_bits: int = 0                    # >0: fake-quantize matrices (STE) during training
    qat_start_frac: float = 0.0
    compile: bool = False                # torch.compile (CUDA only; ignored elsewhere)
    loss_on: str = "all"                 # all = next-token loss at every position | last = final position only
    log_every: int = 50
    eval_every: int = 250
    eval_tokens: int = 262_144           # tokens used by the *periodic* (cheap) val estimate
    save_checkpoint: bool = True


# ----------------------------------------------------------------------------- eval
@dataclass
class EvalConfig:
    final_eval_tokens: int = 0           # 0 = full val split
    eval_seq_len: int = 0                # 0 = train.seq_len
    sliding_stride: int = 0              # >0: sliding-window eval (golf: 64)
    batch_size: int = 32
    sample_tokens: int = 300             # generate a text sample at the end (0 = off)
    sample_prompt: str = ""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 1.0                   # nucleus sampling (1.0 = off)


@dataclass
class ExportConfig:
    quant_bits: int = 0                  # 0 = skip; 8 / 6 / 4: per-row int quantization round-trip
    compressor: str = "lzma"             # zlib | lzma
    budget_mb: float = 16.0              # parameter-golf artifact limit (reported, not enforced)


@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 1337
    device: str = "auto"                 # auto | cuda | mps | cpu
    precision: str = "auto"              # auto | fp32 | bf16 | fp16
    out_dir: str = "outputs/nanolm"
    run_dir: str = ""                    # explicit run dir (set by sweep.py); "" = timestamped
    tags: list = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


# ======================================================= Part 2: pretrained-model configs
@dataclass
class FTModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B"      # any HuggingFace causal LM
    dtype: str = "auto"                  # auto (bf16 on CUDA, fp32 on MPS/CPU) | fp32 | bf16 | fp16
    lora: bool = True                    # False = full fine-tuning (needs ~4x more memory)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                        "gate_proj", "up_proj", "down_proj"])
    grad_checkpointing: bool = True      # trade compute for memory (laptops!)


@dataclass
class FTDataConfig:
    # alpaca       : instruction / input / output   (SFT)
    # ultrafeedback: chosen / rejected chat pairs    (DPO)
    # golf_val     : FineWeb validation documents    (bits-per-byte evaluation)
    format: str = "alpaca"
    dataset: str = "yahma/alpaca-cleaned"
    split: str = "train"
    val_split: str = ""                  # "" = carve n_val examples out of `split` (seeded)
    n_train: int = 2000
    n_val: int = 100
    max_len: int = 512                   # tokens (prompt + response); longer examples are dropped
    min_score_gap: float = 1.0           # DPO: keep pairs whose quality scores differ by >= this
    shard_dir: str = "data/golf/fineweb10B_sp1024"          # golf_val
    shard_tokenizer: str = "sp:data/golf/tokenizers/fineweb_1024_bpe.model"
    eval_docs: int = 200                 # golf_val: number of validation documents


@dataclass
class FTTrainConfig:
    method: str = "sft"                  # sft | dpo | eval_bpb
    steps: int = 300
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4                     # LoRA likes ~1e-4..3e-4; full fine-tuning ~1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 20
    dpo_beta: float = 0.1                # strength of the KL anchor to the reference model
    log_every: int = 10
    eval_every: int = 100
    max_time_sec: float = 0.0
    ctx_len: int = 1024                  # eval_bpb window


@dataclass
class FTRunConfig:
    name: str = "finetune"
    seed: int = 1337
    device: str = "auto"
    precision: str = "auto"
    out_dir: str = "outputs/nanolm"
    run_dir: str = ""
    tags: list = field(default_factory=list)
    model: FTModelConfig = field(default_factory=FTModelConfig)
    data: FTDataConfig = field(default_factory=FTDataConfig)
    train: FTTrainConfig = field(default_factory=FTTrainConfig)
    eval_prompts: list = field(default_factory=lambda: [
        "Give three tips for staying healthy.",
        "Explain what a language model is to a 10-year-old.",
        "Write a haiku about autumn.",
        "What is the capital of France, and why is it famous?",
        "Translate 'good morning' into Spanish and Chinese.",
    ])
    gen_tokens: int = 128


# ------------------------------------------------------------------ YAML machinery
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_defaults(path: str | Path) -> dict:
    """Load YAML; a top-level `defaults: [a.yaml, b.yaml]` is merged first (relative to file)."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = deep_merge(merged, load_yaml_with_defaults(path.parent / parent))
    return deep_merge(merged, raw)


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_overrides(overrides: list[str]) -> dict:
    """['model.dim=128', 'train.lr=1e-3', 'model.recur_layers=[3,4]'] -> nested dict."""
    out: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, val = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(val))
    return out


def _coerce(value: Any, typ: Any, where: str) -> Any:
    if typ is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if typ is float and isinstance(value, str):      # YAML reads "3e-4" as a string
        return float(value)
    if typ is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ in (int, float, str, bool) and not isinstance(value, typ):
        raise TypeError(f"{where}: expected {typ.__name__}, got {value!r}")
    return value


def from_dict(cls, d: dict, where: str = ""):
    """Build a (nested) dataclass from a dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} in section '{where or 'root'}'. "
                       f"Valid keys: {sorted(names)}")
    kwargs = {}
    for k, v in d.items():
        typ = hints[k]
        path = f"{where}.{k}" if where else k
        if dataclasses.is_dataclass(typ):
            kwargs[k] = from_dict(typ, v or {}, path)
        else:
            kwargs[k] = _coerce(v, typ, path)
    return cls(**kwargs)


def load_config(path: str | None = None, overrides: list[str] | None = None,
                extra: dict | None = None, cls=RunConfig):
    raw = load_yaml_with_defaults(path) if path else {}
    if extra:
        raw = deep_merge(raw, extra)
    raw = deep_merge(raw, parse_overrides(overrides or []))
    return from_dict(cls, raw)


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
