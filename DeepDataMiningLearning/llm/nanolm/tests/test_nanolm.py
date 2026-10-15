"""Correctness tests -- run before trusting any number.   python -m pytest nanolm/tests -q

The most important one is test_causality: a model that can peek at future tokens gets a
spectacular (and meaningless) bpb. Every input add-on (SmearGate, BigramHash, ...) is a chance
to leak, so each golf feature is checked individually.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nanolm.config import ModelConfig, TrainConfig, load_config
from nanolm.data import ByteTokenizer, CharTokenizer, byte_lut_tensors, count_bytes
from nanolm.device import get_device
from nanolm.evaluate import evaluate
from nanolm.models import build_model
from nanolm.optim import zeropower_via_newtonschulz5, lr_multiplier
from nanolm.quant import export_artifact

GOLF_FEATURES = {
    "plain": {},
    "golf_baseline": dict(qk_norm=True, qk_gain_init=1.5, logit_softcap=30.0, zero_init_proj=True,
                          embed_norm=True, layer_scales=True, resid_mix=True, unet_skips=True,
                          mlp="relu2", n_kv_heads=2),
    "smear": dict(smear_gate=True),
    "bigram": dict(bigram_buckets=257, bigram_dim=16),
    "xsa": dict(xsa_last_n=2),
    "value_residual": dict(value_residual=True),
    "partial_rope_lnscale": dict(rope_dims=4, ln_scale=True),
    "parallel": dict(parallel_from=1),
    "attn_gate": dict(attn_gate=True),
    "recurrence": dict(recur_layers=[1, 2], recur_times=2),
    "learned_pos_layernorm": dict(pos="learned", norm="layernorm", mlp="gelu", bias=True),
    "swiglu": dict(mlp="swiglu"),
}


def small_cfg(**kw):
    base = dict(arch="gpt", vocab_size=97, dim=32, n_layers=4, n_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("feature", list(GOLF_FEATURES))
def test_causality(feature):
    torch.manual_seed(0)
    model = build_model(small_cfg(**GOLF_FEATURES[feature])).eval()
    # zero-init projections make blocks identity at init -> perturb so the test is meaningful
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randint(0, 97, (2, 24))
    x2 = x.clone()
    x2[:, 15:] = torch.randint(0, 97, (2, 9))            # change only the future
    with torch.no_grad():
        a, b = model(x), model(x2)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5), f"{feature}: future tokens leak into the past"
    assert not torch.allclose(a[:, 15:], b[:, 15:]), f"{feature}: model ignores its input"


@pytest.mark.parametrize("arch", ["rnn", "gru", "lstm", "lstm_scratch"])
def test_recurrent_shapes_and_causality(arch):
    torch.manual_seed(0)
    m = build_model(ModelConfig(arch=arch, vocab_size=50, dim=16, n_layers=2)).eval()
    x = torch.randint(0, 50, (3, 12))
    x2 = x.clone()
    x2[:, 8:] = 0
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert a.shape == (3, 12, 50)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)


def test_gqa_matches_shapes_and_params():
    mha = build_model(small_cfg())
    gqa = build_model(small_cfg(n_kv_heads=1))
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(gqa) < n(mha)
    assert gqa(torch.randint(0, 97, (1, 8))).shape == (1, 8, 97)


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(64, 128) @ torch.diag(torch.logspace(-2, 1, 128))   # badly conditioned
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=5).float())
    # quintic NS is tuned for speed, not exactness: singular values land in roughly [0.7, 1.2]
    assert s.max() < 1.3 and s.min() > 0.5, s


def test_byte_luts_count_utf8_bytes():
    text = "héllo wörld — ok"
    for tok in (CharTokenizer.from_text(text), ByteTokenizer()):
        ids = torch.tensor([tok.encode(text)])
        base, lead, boundary = (torch.tensor(a) for a in tok.byte_luts())
        nbytes = count_bytes(ids[:, :-1], ids[:, 1:], (base.int(), lead, boundary)).sum().item()
        first = len(tok.decode(ids[0, :1].tolist()).encode())
        assert nbytes + first == len(text.encode("utf-8"))


def test_sentencepiece_luts_if_available():
    import os
    path = "data/golf/tokenizers/fineweb_1024_bpe.model"
    if not os.path.exists(path):
        pytest.skip("golf tokenizer not downloaded")
    from nanolm.data import SentencePieceTokenizer
    tok = SentencePieceTokenizer(path)
    text = "The quick brown fox jumps over the lazy dog. Ünïcödé too!"
    ids = torch.tensor([[1] + tok.encode(text)])               # BOS then text, as in the golf shards
    luts = tuple(torch.tensor(a) for a in tok.byte_luts())
    nbytes = count_bytes(ids[:, :-1], ids[:, 1:], (luts[0].int(), luts[1], luts[2])).sum().item()
    assert nbytes == len(text.encode("utf-8"))


@pytest.mark.parametrize("stride", [0, 8, 16])
def test_eval_scores_each_token_once(stride):
    torch.manual_seed(0)
    d = get_device("cpu")
    m = build_model(small_cfg())
    val = np.random.default_rng(0).integers(0, 97, size=400).astype(np.uint16)
    luts = (torch.ones(97, dtype=torch.int32), torch.zeros(97, dtype=torch.bool), torch.zeros(97, dtype=torch.bool))
    r = evaluate(m, val, 32, 4, luts, d, stride=stride)
    expected = (399 // 32) * 32 if stride == 0 else 399
    assert r["tokens"] == expected
    assert math.isclose(r["bpb"], r["loss"] / math.log(2), rel_tol=1e-6)    # 1 byte per token here


def test_quant_roundtrip_small_error():
    torch.manual_seed(0)
    m = build_model(small_cfg(dim=512, n_layers=1, n_heads=8))
    for bits, tol in ((8, 0.02), (6, 0.08)):
        blob, deq = export_artifact(m, bits, "lzma")
        sd = m.state_dict()
        w, wq = sd["blocks.0.mlp.fc.weight"], deq["blocks.0.mlp.fc.weight"]
        assert (w - wq).norm() / w.norm() < tol
        assert len(blob) < sum(t.numel() for t in sd.values()) * 4


def test_config_rejects_typos():
    with pytest.raises(KeyError):
        load_config(None, ["model.n_layer=6"])
    cfg = load_config(None, ["model.recur_layers=[3,4]", "train.lr=1e-3"])
    assert cfg.model.recur_layers == [3, 4] and cfg.train.lr == 1e-3


def test_wsd_schedule_shape():
    t = TrainConfig(schedule="wsd", warmup_steps=0, warmdown_frac=0.2, min_lr_frac=0.0)
    assert lr_multiplier(t, 0, 0.5) == 1.0
    assert abs(lr_multiplier(t, 0, 0.9) - 0.5) < 1e-9
    assert abs(lr_multiplier(t, 0, 1.0)) < 1e-9


# ------------------------------------------------------------------ tokenizers added for Part 1
SHAKESPEARE_LIKE = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
                    "All:\nSpeak, speak.  Naïve café — 1234 tokens!\n") * 20


def _exact_bytes(tok, text):
    ids = torch.tensor([tok.encode(text)])
    base, lead, boundary = (torch.tensor(a) for a in tok.byte_luts())
    n = count_bytes(ids[:, :-1], ids[:, 1:], (base.int(), lead, boundary)).sum().item()
    first = int(base[ids[0, 0]])
    return n + first


def test_word_tokenizer_lossless_with_byte_fallback():
    from nanolm.data import WordTokenizer
    tok = WordTokenizer.train(SHAKESPEARE_LIKE, vocab_size=280)
    unseen = "Zebra-quagga met Ünïcödé friends 😀 in 2026."
    for text in (SHAKESPEARE_LIKE, unseen):
        assert tok.decode(tok.encode(text)) == text
        assert _exact_bytes(tok, text) == len(text.encode("utf-8"))
    assert any(i < 256 for i in tok.encode(unseen))      # OOV words fell back to bytes


def test_trained_sentencepiece_is_lossless(tmp_path):
    from nanolm.data import SentencePieceTokenizer, train_sentencepiece
    model = train_sentencepiece(SHAKESPEARE_LIKE, "bpe", 300, str(tmp_path / "sp"))
    tok = SentencePieceTokenizer(model)
    text = "Before we proceed, hear me.\n\nAll:\n  Speak café 😀!"
    assert tok.decode(tok.encode(text)) == text
    # the very first piece carries SentencePiece's dummy-prefix space, which is not a real byte
    assert abs(_exact_bytes(tok, text) - len(text.encode("utf-8"))) <= 1


def test_uint32_shards_roundtrip(tmp_path):
    from nanolm.data import read_shard, write_shard
    ids = np.array([0, 5, 70000, 151000], dtype=np.int64)
    write_shard(tmp_path / "x.bin", ids, vocab_size=151_936)
    assert read_shard(tmp_path / "x.bin").tolist() == ids.tolist()
    write_shard(tmp_path / "y.bin", ids[:2], vocab_size=1024)
    assert read_shard(tmp_path / "y.bin").dtype.itemsize == 2


# ------------------------------------------------------------------ Part 2 (pretrained models)
def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_token_logprobs_matches_hf_loss():
    from nanolm.finetune import IGNORE, token_logprobs
    model = _tiny_llama()
    ids = torch.randint(0, 101, (3, 20))
    att = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :8] = IGNORE                        # prompt tokens are not scored
    with torch.no_grad():
        lp, n = token_logprobs(model, ids, att, labels, chunk=7)     # odd chunk size on purpose
        hf = model(input_ids=ids, attention_mask=att, labels=labels).loss
    assert torch.allclose(-(lp.sum() / n.sum()), hf, atol=1e-5)
    assert n.tolist() == [12.0, 12.0, 12.0]


def test_dpo_starts_at_zero_margin_with_lora():
    peft = pytest.importorskip("peft")
    from nanolm.finetune import IGNORE, dpo_terms
    model = peft.get_peft_model(_tiny_llama(), peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    model.eval()
    def batch():
        ids = torch.randint(0, 101, (2, 16))
        lab = ids.clone()
        lab[:, :6] = IGNORE
        return ids, torch.ones_like(ids), lab
    with torch.no_grad():
        loss, acc, margin = dpo_terms(model, (batch(), batch()), beta=0.1)
    assert abs(margin.item()) < 1e-6              # LoRA B starts at 0: policy == reference
    assert abs(loss.item() - math.log(2)) < 1e-5  # -log sigmoid(0)
    assert acc.item() == 0.5                      # ties count as half
