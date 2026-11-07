"""GRUScratch (models/rnn.py) must compute exactly what nn.GRU computes.   python -m pytest nanolm/tests/test_gru_scratch.py -v

All comparisons run in float64 so that any mismatch above rounding level (~1e-15) is a real bug.
The last test re-runs a trained character GRU through GRUScratch; it is skipped if that run
(python -m nanolm.sweep --sweep nanolm/configs/sweeps/01_char_models.yaml) does not exist.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanolm.config import ModelConfig
from nanolm.models import build_model
from nanolm.models.rnn import GRUScratch

B, T, D, H = 4, 50, 32, 64
LLM_ROOT = Path(__file__).resolve().parents[2]
TRAINED_GRU = LLM_ROOT / "outputs/nanolm/sweeps/char_models/gru_s1337"


@pytest.fixture
def ref_and_input():
    torch.manual_seed(0)
    ref = nn.GRU(D, H, batch_first=True).double()
    x = torch.randn(B, T, D, dtype=torch.float64, requires_grad=True)
    return ref, x


def test_forward_matches_nn_gru(ref_and_input):
    ref, x = ref_and_input
    mine = GRUScratch(D, H).double().load_from(ref)
    y_ref, h_ref = ref(x)
    y = mine(x)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(y[:, -1], h_ref[0], rtol=0, atol=1e-12)


def test_backward_matches_nn_gru(ref_and_input):
    ref, x = ref_and_input
    mine = GRUScratch(D, H).double().load_from(ref)
    w = torch.randn(B, T, H, dtype=torch.float64)
    g_ref = torch.autograd.grad((ref(x)[0] * w).sum(),
                                [x, ref.weight_ih_l0, ref.weight_hh_l0, ref.bias_ih_l0, ref.bias_hh_l0])
    g = torch.autograd.grad((mine(x) * w).sum(),
                            [x, mine.W_x.weight, mine.W_h.weight, mine.W_x.bias, mine.W_h.bias])
    for name, a, b in zip(["x", "W_ih", "W_hh", "b_ih", "b_hh"], g, g_ref):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10, msg=f"dL/d{name} differs")


def test_two_layer_stack_with_initial_state(ref_and_input):
    _, x = ref_and_input
    ref = nn.GRU(D, H, num_layers=2, batch_first=True).double()
    layers = [GRUScratch(D, H).double().load_from(ref, 0), GRUScratch(H, H).double().load_from(ref, 1)]
    h0 = torch.randn(2, B, H, dtype=torch.float64)
    y_ref, hT_ref = ref(x, h0)
    out, hT = x, []
    for layer, h in zip(layers, h0):
        out = layer(out, h)
        hT.append(out[:, -1])
    torch.testing.assert_close(out, y_ref, rtol=0, atol=1e-12)
    torch.testing.assert_close(torch.stack(hT), hT_ref, rtol=0, atol=1e-12)


def test_reset_before_variant_differs(ref_and_input):
    """Control: the Cho et al. (2014) cell with the same weights is a different function."""
    ref, x = ref_and_input
    cho = GRUScratch(D, H, reset_after=False).double().load_from(ref)
    assert (cho(x) - ref(x)[0]).abs().max().item() > 1e-3


def test_gru_scratch_arch_matches_gru_arch():
    """model.arch=gru_scratch with copied weights gives the same logits as model.arch=gru."""
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, dim=32, n_layers=2, dropout=0.0)
    ref = build_model(ModelConfig(arch="gru", **cfg)).double().eval()
    mine = build_model(ModelConfig(arch="gru_scratch", **cfg)).double().eval()
    state = {k: v for k, v in ref.state_dict().items() if not k.startswith("rnn.")}
    mine.load_state_dict(state, strict=False)
    for i, layer in enumerate(mine.rnn):
        layer.load_from(ref.rnn, i)
    idx = torch.randint(0, 50, (2, 40))
    with torch.no_grad():
        emb = ref.embed(idx)
        torch.testing.assert_close(mine.rnn(emb), ref.rnn(emb)[0], rtol=0, atol=1e-12)
        torch.testing.assert_close(mine(idx), ref(idx), rtol=0, atol=1e-6)


@pytest.mark.skipif(not (TRAINED_GRU / "model.pt").exists(), reason="trained char_models/gru run not found")
def test_trained_char_gru_unchanged():
    from nanolm.generate import load_run
    _, model, tok, _, _ = load_run(str(TRAINED_GRU), device="cpu")
    model.eval().double()
    text = (LLM_ROOT / "data/nanolm/raw/tinyshakespeare.txt").read_text()[-513:]    # tail = validation text
    idx = torch.tensor([tok.encode(text)])
    gru = model.rnn
    scratch = nn.Sequential(*[GRUScratch(gru.input_size if l == 0 else gru.hidden_size, gru.hidden_size)
                              .double().load_from(gru, l) for l in range(gru.num_layers)])
    with torch.no_grad():
        emb = model.embed(idx[:, :-1])
        torch.testing.assert_close(scratch(emb), gru(emb)[0], rtol=0, atol=1e-12)
        logits_ref = model(idx[:, :-1])
        model.rnn = scratch
        logits = model(idx[:, :-1])
    bpb = lambda lg: F.cross_entropy(lg[0].double(), idx[0, 1:]).item() / math.log(2)
    print(f"\nbpb on {idx.shape[1] - 1} val chars: nn.GRU {bpb(logits_ref):.6f}  GRUScratch {bpb(logits):.6f}")
    assert bpb(logits) == pytest.approx(bpb(logits_ref), abs=1e-9)
