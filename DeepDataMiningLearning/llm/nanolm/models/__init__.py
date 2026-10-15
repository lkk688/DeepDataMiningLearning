"""Model registry: `build_model(cfg.model)` -> nn.Module with forward(idx[B,T]) -> logits[B,T,V]."""
from .gpt import GPT
from .rnn import RecurrentLM

RECURRENT = ("rnn", "gru", "lstm", "lstm_scratch")


def build_model(mcfg):
    if mcfg.vocab_size <= 0:
        raise ValueError("model.vocab_size must be set (train.py fills it from the tokenizer)")
    if mcfg.arch in RECURRENT:
        return RecurrentLM(mcfg)
    if mcfg.arch == "gpt":
        return GPT(mcfg)
    raise ValueError(f"unknown model.arch {mcfg.arch!r}; choose from {RECURRENT + ('gpt',)}")


def count_params(model, exclude_embeddings: bool = False) -> int:
    seen, n = set(), 0
    for name, p in model.named_parameters():
        if id(p) in seen:          # tied weights are counted once
            continue
        seen.add(id(p))
        if exclude_embeddings and ("emb" in name.split(".")[0] or name.startswith("embed")):
            continue
        n += p.numel()
    return n
