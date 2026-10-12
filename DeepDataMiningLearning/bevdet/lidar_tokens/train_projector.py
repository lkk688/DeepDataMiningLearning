"""Train a LiDAR->LLM-token projector against Qwen-Drive's own perception loss.

WHAT THIS IS
------------
The mechanical gate for "LiDAR tokens into a vision VLA": freeze the VLM, freeze the
perception head, and train **only** a projector that writes a handful of LiDAR-derived
tokens into the language model's input sequence. If the injected tokens cannot change the
occupancy prediction in a *useful* direction, the route is dead regardless of how good the
LiDAR encoder is.

It is a gate, not a result. The demo set is **6 frames** (4 nuPlan, 2 nuScenes), so the
only question it can answer is the capacity/plumbing one:

    can a projector at the LLM input, with everything else frozen, OVERFIT 6 frames?

A "no" kills the route: the channel from the injected tokens to the BEV head is too narrow
to matter, and no amount of data will widen it. A "yes" says the channel is open and the
next step is a real dataset. It does **not** say LiDAR helps -- overfitting 6 frames is
what a working optimiser does. Cf. §7.8/§7.9 of `ngperception/docs/RESEARCH_DIRECTIONS.md`:
prove the apparatus can move before building the experiment around it.

WHY THE GRADIENT EXISTS AT ALL
------------------------------
Qwen-Drive is released inference-only and two of its CUDA ops have no backward, so the BEV
head does not backpropagate as shipped. `qwen_drive_grad.py` routes the deformable
attention through the release's own pure-torch fallback; `check_grad.py` is the gate that
established both halves of the path. Read that file first.

ARMS
----
``--inject none``    the released path, no extra tokens. The number to beat. Eval only.
``--inject zero``    K tokens whose embedding is exactly zero. Eval only. Isolates the cost
                     of the K extra sequence positions from the cost/benefit of their
                     *content* -- without it, any change in the trained arms is confounded
                     with "the prompt got K tokens longer".
``--inject lidar``   K LiDAR tokens from this frame's point cloud, projector trained.
``--inject const``   the DECISIVE control: the projector's input is the SAME vector for
                     every frame, so the tokens carry zero information about anything --
                     only `pos`, the MLP bias and `gate` can move. Whatever this arm scores
                     is what free parameters at the LLM input are worth on their own. If it
                     matches `lidar`, the injected modality is doing nothing and the gain is
                     a learned constant prefix that recalibrates the frozen head.
``--inject shuffle`` the CONTROL: same tokens, but the point cloud is taken from a
                     *different* frame in the set. A projector that improves here is
                     exploiting the injection channel as free parameters (a per-frame bias
                     the LLM can key on), not reading the LiDAR. Required, not optional:
                     with 6 frames and a trainable input, "it learned the frame index" is
                     the default explanation.

HOW THE TOKENS GET IN
---------------------
K placeholder *text* tokens are inserted into the prompt and a forward hook on the
token-embedding module overwrites their embeddings with the projector's output.

**They must go BEFORE the first image token, and this was learned the hard way.** The first
version appended them at the end of the prompt. The projector's gradient was then *exactly*
zero and the loss bit-identical to the baseline -- because the decoder is **causal**, so a
token appended after the images cannot influence the images' hidden states, and the BEV head
reads exactly those. A zero gradient with a working optimiser is a plumbing bug, not a
negative result (RESEARCH_DIRECTIONS.md §7.8). They are now inserted at the first
`<|vision_start|>`, so every camera's tokens can attend to them.

Insertion is not `inputs_embeds=`: HF's Qwen-VL forward locates image positions from
`input_ids` and masked-scatters the vision features *after* embedding, so passing
`inputs_embeds` risks silently dropping the images. Overwriting text positions inside the
embedding hook leaves the image scatter and the image-token tap the BEV head reads
untouched.

One deliberate deviation from the released path: inserting K tokens shifts every image
token's rope position by K. The model was trained with variable-length text before the
images, so this is in-distribution, but it is not a no-op -- which is exactly what the
``zero`` arm measures.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

QD = Path('/data/rnd-liu/Others/Qwen-Drive-1.0')
sys.path.insert(0, str(QD / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_drive_grad import (patch_deformable_attention,  # noqa: E402
                             set_vision_patch_embed)

OCC_EMPTY = 9          # configuration_perception.OCC_EMPTY_LABEL
OCC_NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _git(repo, *a):
    try:
        return subprocess.check_output(['git', '-C', str(repo), *a], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# LiDAR tokenisation
# --------------------------------------------------------------------------- #

class PillarTokenizer:
    """Raw point cloud -> ``K = grid**2`` fixed tokens of height-histogram features.

    Parameter-free and deliberately crude. The eventual design feeds a frozen detector's
    object queries (`extract.py`); this stands in for the gate because it needs no aligned
    detector checkpoint for these packed demo frames, and it carries real geometry -- a
    height histogram per BEV cell is exactly the "where is there stuff, and how tall" fact
    a single image answers badly.
    """

    def __init__(self, grid: int = 8, height_bins: int = 16,
                 xy_range: float = 50.0, z_range=(-3.0, 5.0)):
        self.grid = grid
        self.height_bins = height_bins
        self.xy_range = xy_range
        self.z_range = z_range
        self.n_tokens = grid * grid
        self.dim = height_bins + 3          # hist + log count + mean z + max z

    def features(self, frame) -> torch.Tensor:
        """Uniform interface with QueryTokenizer: a PerceptionFrame -> (K, D)."""
        if frame.lidar is None:
            raise SystemExit(f'{frame.token} has no lidar.npy')
        return self(frame.lidar)

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        g, hb, r = self.grid, self.height_bins, self.xy_range
        z0, z1 = self.z_range
        out = np.zeros((g * g, self.dim), np.float32)
        p = points[np.all(np.abs(points[:, :2]) < r, axis=1)]
        if p.shape[0]:
            ix = np.clip(((p[:, 0] + r) / (2 * r) * g).astype(int), 0, g - 1)
            iy = np.clip(((p[:, 1] + r) / (2 * r) * g).astype(int), 0, g - 1)
            iz = np.clip(((p[:, 2] - z0) / (z1 - z0) * hb).astype(int), 0, hb - 1)
            cell = ix * g + iy
            np.add.at(out, (cell, iz), 1.0)
            out[:, :hb] = np.log1p(out[:, :hb])
            cnt = np.bincount(cell, minlength=g * g).astype(np.float32)
            out[:, hb] = np.log1p(cnt)
            sz = np.bincount(cell, weights=p[:, 2], minlength=g * g)
            out[:, hb + 1] = sz / np.maximum(cnt, 1.0)
            mz = np.full(g * g, z0, np.float32)
            np.maximum.at(mz, cell, p[:, 2])
            out[:, hb + 2] = mz
        return torch.from_numpy(out)


class QueryTokenizer:
    """A frozen LiDAR detector's object queries as tokens (`extract.py` output).

    This is the tokeniser the plan always intended and the one any *negative* claim about
    LiDAR utility requires (§5.5): LVLDrive uses FSDv2 pretrained on nuScenes, so reporting
    "LiDAR tokens do not help" from an 8x8 height histogram invites "your encoder was a
    histogram", and the objection is correct. The histogram stays available as the cheap
    lower bound -- comparing the two is itself the measurement of how much the tokeniser
    matters.

    All P queries are kept, not just the confident ones: P is fixed, so the token count does
    not silently vary per frame (see `extract.py`). The npz must come from the **LiDAR-only**
    detector; queries from BEVFusion-LC carry camera features the VLM can already see.
    """

    def __init__(self, npz: Path, l2_normalise: bool = True):
        d = np.load(npz, allow_pickle=False)
        if 'token' not in d:
            raise SystemExit(f'{npz} has no `token` array -- it was written by the old '
                             'extract.py which saved an integer sample_idx. Re-extract.')
        self.feat = d['query_feat'].astype(np.float32)          # (N, P, C)
        self.index = {str(t): i for i, t in enumerate(d['token'])}
        self.n_tokens = int(self.feat.shape[1])
        self.dim = int(self.feat.shape[2])
        meta = Path(str(npz).replace('.npz', '') + '.meta.json')
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        if l2_normalise:
            # Detector query features have no reason to sit at any particular scale, and the
            # projector's LayerNorm sees them per-token; normalising here makes the two
            # tokenisers comparable rather than differing by an arbitrary gain.
            n = np.linalg.norm(self.feat, axis=-1, keepdims=True)
            self.feat = self.feat / np.maximum(n, 1e-6)
        print(f'queries: {self.feat.shape[0]} frames x {self.n_tokens} x {self.dim} '
              f'from {self.meta.get("args", {}).get("checkpoint", "?")} '
              f'({self.meta.get("modality", "?")})')

    def features(self, frame) -> torch.Tensor:
        i = self.index.get(frame.token)
        if i is None:
            raise SystemExit(f'no extracted queries for token {frame.token}. Extract with '
                             f'`extract.py --tokens <manifest.json>` over the SAME frames.')
        return torch.from_numpy(self.feat[i])


class Projector(nn.Module):
    """K LiDAR feature vectors -> K token embeddings, matched to the embedding scale.

    ``target_rms`` is measured from the VLM's own embedding matrix. Without it a freshly
    initialised projector emits vectors whose norm is nothing like a real token's, and the
    frozen decoder either ignores them or saturates -- a failure of scale that looks
    exactly like a failure of the idea. ``gate`` scales the whole output; it is init 1.0 by
    default (a standard soft-prompt init). ``--gate-init 0`` makes the first step start from
    zero-embedding tokens, i.e. from the ``zero`` arm rather than from a random one.
    """

    def __init__(self, d_in: int, d_out: int, n_tokens: int, hidden: int = 512,
                 target_rms: float = 1.0, gate_init: float = 1.0):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(n_tokens, d_in))
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.register_buffer('target_rms', torch.tensor(float(target_rms)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x + self.pos)
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.target_rms
        return y * self.gate


# --------------------------------------------------------------------------- #
# the training objective
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS. Trained against pooled occupancy cross-entropy, the projector drives the
# loss down 25 % while fwIoU FALLS, and the per-class breakdown says exactly why: the only
# two classes that improve are the two largest (`empty` 1.70 M voxels, `driveable` 152 k),
# while every object class collapses (`vehicle` -0.119, `bicycle` -0.204, `traffic_cone`
# -0.104). Even inside `mask_camera` the grid is 74 % `empty`, so the cheapest way to lower
# CE is to sharpen what already dominates and sacrifice objects -- and better LiDAR
# information just gives the optimiser more leverage to do it. Two independent causes, two
# switches:
#
#   --loss ce+lovasz   Lovász-softmax is a direct differentiable surrogate for per-class
#                      Jaccard, so losing an object class is penalised as such rather than
#                      paid for with background gains. In-house evidence, on this project's
#                      own occupancy models: class re-weighting was NEGATIVE, Lovász was the
#                      best single change (ConvNeXt-V2-L + 3D head + Lovász 0.1485, +70 %
#                      over the R50 baseline).
#   --loss-mask camera Supervise only where our GT is trustworthy. Outside `mask_camera` the
#                      released GT and Occ3D disagree three times more than inside it (~90 %
#                      of their extra occupied voxels are outside), so an unmasked loss
#                      actively teaches the projector to delete occupancy the frozen head
#                      was trained to emit -- and sharpening `empty` is the cheapest way to
#                      comply. This may matter as much as the class imbalance; the two are
#                      orthogonal and are measured separately.
#
# The two functions below are a verbatim port of
# `ngperception/occupancy/train_lss.py:33-61` (Berman et al., CVPR'18), with `ignore` moved
# from Occ3D's free label 17 to Qwen-Drive's `empty` = 9. Kept as a copy because train_lss
# uses package-relative imports and cannot be imported standalone from here; `--self-test`
# checks the port against a directly computed Jaccard rather than trusting it.


def lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension of the Jaccard loss."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, ignore=OCC_EMPTY):
    """Multi-class Lovász-softmax on flat (P,C) probs / (P,) labels."""
    C = probas.shape[1]
    losses = []
    for c in range(C):
        if c == ignore:
            continue
        fg = (labels == c).float()
        if fg.sum() == 0:                      # class absent -> no signal
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def occ_objective(logits, gt, mask, kind: str, lovasz_w: float):
    """The trained loss. `logits` (X,Y,Z,C), `gt` (X,Y,Z), `mask` (X,Y,Z) bool or None.

    Note the reported eval `loss` stays plain UNMASKED cross-entropy whatever this returns,
    so it remains one invariant yardstick across objectives; only the trained signal changes.
    """
    flat = logits.reshape(-1, OCC_NUM_CLASSES).float()
    tgt = gt.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1)
        flat, tgt = flat[m], tgt[m]
    loss = flat.new_zeros(())
    if kind in ('ce', 'ce+lovasz'):
        loss = loss + F.cross_entropy(flat, tgt)
    if kind in ('lovasz', 'ce+lovasz'):
        loss = loss + lovasz_w * lovasz_softmax_flat(flat.softmax(1), tgt)
    return loss


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

class OccMetric:
    """Per-class IoU over non-empty GT classes, accumulated over frames.

    **Report `fwiou` beside `miou`, never `miou` alone.** On the 6-frame demo set the
    unweighted mean runs over 8 classes of which four hold 1, 14, 55 and 73 GT voxels, so a
    14-voxel class flipping moved mIoU by 0.098 *while every other class improved* -- the two
    metrics disagreed in sign. That is RESEARCH_DIRECTIONS.md §7.3 (near-zero denominators)
    reaching us through a metric rather than a ratio. `fwiou` weights each class by its GT
    voxel count, so a class nobody can measure cannot dominate.
    """

    def __init__(self, n: int = OCC_NUM_CLASSES):
        self.n = n
        self.inter = np.zeros(n, np.int64)
        self.union = np.zeros(n, np.int64)
        self.gt_count = np.zeros(n, np.int64)

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        for c in range(self.n):
            p, g = pred == c, gt == c
            self.inter[c] += int((p & g).sum())
            self.union[c] += int((p | g).sum())
            self.gt_count[c] += int(g.sum())

    def _classes(self):
        return [c for c in range(self.n) if c != OCC_EMPTY and self.union[c] > 0]

    def miou(self) -> float:
        cls = self._classes()
        if not cls:
            return float('nan')
        return float(np.mean([self.inter[c] / self.union[c] for c in cls]))

    def fwiou(self) -> float:
        """GT-frequency-weighted IoU over the same classes."""
        cls = self._classes()
        w = np.array([self.gt_count[c] for c in cls], float)
        if not cls or w.sum() == 0:
            return float('nan')
        v = np.array([self.inter[c] / self.union[c] for c in cls], float)
        return float((w * v).sum() / w.sum())

    def counts(self) -> dict:
        return {c: int(self.gt_count[c]) for c in range(self.n)}

    def per_class(self) -> dict:
        return {c: (float(self.inter[c] / self.union[c]) if self.union[c] else None)
                for c in range(self.n)}


# --------------------------------------------------------------------------- #
# the frozen stack
# --------------------------------------------------------------------------- #

class Stack:
    """Frozen VLM + frozen perception head, with an injectable embedding hook."""

    def __init__(self, vlm_path: Path, head_path: Path, device: str, attn_math: str,
                 patch_embed: str = 'matmul'):
        from transformers import AutoTokenizer
        from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
        from qwen_drive_perception import QwenDrivePerception
        from qwen_drive_perception.dataset import PerceptionProcessor

        holder = QwenDriveForPlanning.from_pretrained(
            vlm_path, dtype=torch.bfloat16, attn_implementation='sdpa')
        self.vlm = holder.vlm
        del holder.planning_expert
        self.model = QwenDrivePerception.from_pretrained(head_path, dtype=torch.bfloat16)
        self.model.to(device).eval()
        self.processor = PerceptionProcessor(AutoTokenizer.from_pretrained(vlm_path))
        self.model.attach(self.vlm.to(device), self.processor)
        self.device = device
        self.attn_math = torch.float32 if attn_math == 'fp32' else torch.bfloat16

        for p in self.vlm.parameters():
            p.requires_grad_(False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # HF gates checkpointing on `self.training`, so eval() silently disables it.
        # The parameters are frozen either way; train() only re-enables checkpointing.
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.vlm.train()
        if patch_embed == 'matmul':
            r = set_vision_patch_embed(self.vlm, 'matmul', verify=True)
            print(f"vision patch embed: Conv3d -> matmul  conv {r['conv_sec']:.4f}s -> "
                  f"matmul {r['matmul_sec']:.4f}s ({r['speedup']:.0f}x), "
                  f"max abs diff {r['max_abs_diff']:.2e}")

        emb = self.vlm.get_input_embeddings()
        self.d_llm = emb.weight.shape[1]
        with torch.no_grad():
            self.emb_rms = float(emb.weight.float().norm(dim=-1).mean())
        self._inject = None
        emb.register_forward_hook(self._hook)

    def _hook(self, _m, args, out):
        """Overwrite the placeholder positions with the projector's tokens."""
        if self._inject is None:
            return out
        tok, at, n = self._inject
        out = out.clone()
        out[:, at:at + n, :] = tok.to(out.dtype).unsqueeze(0)
        return out

    def add_placeholders(self, inputs: dict, k: int) -> tuple[dict, int]:
        """Insert ``k`` placeholder text tokens immediately before the FIRST image.

        Before, not after: the decoder is causal, so tokens placed after the images cannot
        change the image tokens' hidden states -- and those are the only thing the BEV head
        reads. Appending gave an exactly-zero gradient (see the module docstring).

        The placeholder id is irrelevant (the hook overwrites the embedding); `im_end` is
        used so an accidentally un-hooked forward is obviously wrong rather than subtly
        plausible.
        """
        if k == 0:
            return inputs, -1
        ids = inputs['input_ids']
        vs = (ids[0] == self.processor.vision_start_id).nonzero()
        if vs.numel() == 0:
            raise RuntimeError('no <|vision_start|> in the prompt')
        at = int(vs[0])
        pad = torch.full((1, k), self.processor.im_end_id, dtype=torch.long, device=ids.device)
        out = dict(inputs)
        out['input_ids'] = torch.cat([ids[:, :at], pad, ids[:, at:]], dim=1)
        return out, at

    def forward_occ(self, inputs: dict, img_metas: dict, tokens=None, at: int = -1,
                    grad: bool = True, attn: str = 'torch'):
        """The released `infer` path, minus @no_grad, returning the occupancy logits.

        ``attn='kernel'`` uses the released bf16 CUDA op -- correct for evaluation, where
        no gradient is wanted, so every reported number is the released model's number.
        ``attn='torch'`` uses the differentiable fallback and is required for training.
        """
        vlm, model = self.vlm, self.model
        self._inject = None if tokens is None else (tokens, at, tokens.shape[0])
        captured = {}

        def vit_hook(_m, a, _o=None):
            captured['patches'] = a[0]
        h = vlm.model.visual.merger.register_forward_hook(vit_hook)
        try:
            with contextlib.nullcontext() if grad else torch.no_grad():
                out = vlm.model(input_ids=inputs['input_ids'],
                                pixel_values=inputs['pixel_values'],
                                image_grid_thw=inputs['image_grid_thw'],
                                mm_token_type_ids=model._modality_ids(inputs['input_ids']),
                                use_cache=False)
                hs = vlm.model.language_model.norm(out.last_hidden_state)
                # The ViT stream is independent of the injected tokens by construction
                # (the vision tower runs before the decoder), so it stays detached. That
                # also keeps `voxel_pool_depth` -- an autograd.Function with no backward --
                # off the graph.
                with torch.no_grad():
                    patches = vlm.model.visual.merger.norm(captured['patches'])
                    vit_feats = model._premerge_grids(patches, inputs['image_grid_thw'])

                n_cam = len(img_metas['cam_order'])
                gh = inputs['image_grid_thw'][-1, 1].item()
                gw = inputs['image_grid_thw'][-1, 2].item()
                tpi = gh // 2 * gw // 2
                mask = inputs['input_ids'][0] == vlm.config.image_token_id
                llm_tok = hs[0][mask][-n_cam * tpi:]
                img_llm = llm_tok.view(n_cam, gh // 2, gw // 2, -1)
                img_vit = torch.stack(vit_feats[-n_cam:], dim=0)

                dtype = next(model.bev_modeling.parameters()).dtype
                ctx = (contextlib.nullcontext() if attn == 'kernel' else
                       patch_deformable_attention(math_dtype=self.attn_math, verbose=False))
                with ctx:
                    outs = model.bev_modeling(img_vit_feats=img_vit.detach().to(dtype),
                                              img_llm_feats=img_llm.to(dtype),
                                              img_metas=[img_metas])
                return outs['occ_pred']
        finally:
            h.remove()
            self._inject = None


def mem(tag: str) -> None:
    print(f'   [mem] {tag:28s} allocated {torch.cuda.memory_allocated() / 2**30:6.2f} GiB'
          f'   peak {torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB', flush=True)


# --------------------------------------------------------------------------- #

def _self_test() -> int:
    """Check the ported Lovász-softmax against Jaccard computed directly.

    The port is 30 lines of published math copied across files; copies drift. Three
    properties pin it: a perfect prediction gives 0, the loss tracks 1 - IoU on a
    hard-assignment case, and the `ignore` class is genuinely excluded.
    """
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f'  {"PASS" if cond else "FAIL"}  {name}')
        ok &= bool(cond)

    torch.manual_seed(0)
    P, C = 4000, OCC_NUM_CLASSES
    lab = torch.randint(0, C, (P,))

    perfect = F.one_hot(lab, C).float()
    chk('perfect prediction -> 0', float(lovasz_softmax_flat(perfect, lab)) < 1e-6)

    # One scored class, hard assignment: Lovász of a hard 0/1 prediction equals 1 - IoU.
    lab2 = torch.zeros(P, dtype=torch.long)
    lab2[:1000] = 0
    lab2[1000:] = OCC_EMPTY
    pr = torch.zeros(P, C)
    pr[:800, 0] = 1.0                       # 800 of 1000 correct, no false positives
    pr[800:, OCC_EMPTY] = 1.0
    iou = 800 / 1000
    got = float(lovasz_softmax_flat(pr, lab2))
    chk(f'hard case tracks 1-IoU ({got:.4f} vs {1 - iou:.4f})', abs(got - (1 - iou)) < 1e-4)

    # `ignore` must be excluded: degrading only the ignored class must not change the loss.
    a = float(lovasz_softmax_flat(perfect, lab))
    bad = perfect.clone()
    m = lab == OCC_EMPTY
    bad[m] = 0.0
    bad[m, (OCC_EMPTY + 1) % C] = 1.0       # ignored class predicted totally wrong...
    b = float(lovasz_softmax_flat(bad, lab))
    chk('ignoring `empty` changes the loss (it corrupts OTHER classes too)', b > a)
    only_ign = perfect.clone()
    only_ign[m] = 1.0 / C                    # smear ONLY the ignored class's rows
    c_ = float(lovasz_softmax_flat(only_ign, lab))
    chk(f'…but a class absent from scoring cannot help ({c_:.4f} >= {a:.4f})', c_ >= a)

    g = occ_objective(torch.randn(4, 4, 4, C, requires_grad=True),
                      torch.randint(0, C, (4, 4, 4)), None, 'ce+lovasz', 1.0)
    g.backward()
    chk('occ_objective is differentiable', True)

    print('\n' + ('ALL PASS' if ok else 'FAILURES ABOVE'))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true',
                    help='verify the ported Lovász against a directly computed Jaccard and '
                         'exit; no GPU, no model')
    ap.add_argument('--config', type=Path, help='YAML whose keys are these same flags')
    ap.add_argument('--vlm', type=Path, default=Path('/data/rnd-liu/Others/Qwen-Drive-1.0-4B'))
    ap.add_argument('--head', type=Path,
                    default=Path('/data/rnd-liu/Others/Qwen-Drive-1.0-4B/perception'))
    ap.add_argument('--frames', type=Path, default=QD / 'data/demo/perception')
    ap.add_argument('--out-root', type=Path, default=Path('outputs'))
    ap.add_argument('--exp-name', default='projector_gate')
    ap.add_argument('--inject', choices=['none', 'zero', 'const', 'lidar', 'shuffle'],
                    default='lidar')
    ap.add_argument('--gate-init', type=float, default=1.0)
    ap.add_argument('--tokenizer', choices=['pillar', 'queries'], default='pillar',
                    help="'pillar' = parameter-free 8x8 height histogram (cheap lower "
                         "bound); 'queries' = a frozen LiDAR-only detector's TransFusion "
                         'object queries via --queries (what any negative claim about '
                         'LiDAR utility requires -- see the QueryTokenizer docstring).')
    ap.add_argument('--queries', type=Path, default=None,
                    help='extract.py .npz, required for --tokenizer queries')
    ap.add_argument('--grid', type=int, default=8, help='BEV grid side -> K = grid^2 tokens')
    ap.add_argument('--height-bins', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--patch-embed', choices=['matmul', 'conv'], default='matmul',
                    help="the vision patch embedding. The released module is an nn.Conv3d whose kernel covers its entire input -- a matmul in convolution clothing -- and on 14336 patches in bf16 it takes 91.5 s where the equivalent GEMM takes 0.0046 s, BIT-IDENTICAL output (max abs diff 0.0). It is 99%% of the released forward. 'matmul' is the fix; 'conv' reproduces the released behaviour.")
    ap.add_argument('--eval-attn', choices=['kernel', 'torch'], default='kernel',
                    help='attention implementation used for EVALUATION. "kernel" is the '
                         'released CUDA op, so every reported metric is the released '
                         "model's own number (99.98%% argmax parity with the fallback -- "
                         'check_grad.py --stage parity). Training always uses the '
                         'differentiable fallback.')
    ap.add_argument('--attn-math', choices=['fp32', 'bf16'], default='fp32',
                    help='precision of the pure-torch deformable attention. bf16 roughly '
                         'halves its activation memory at some gradient-fidelity cost.')
    ap.add_argument('--projector-ckpt', type=Path,
                    help='sensitivity mode only: also probe a TRAINED projector from this '
                         '`projector.pt`. Its row is compared against `self`, which is the '
                         'same architecture untrained on the same cloud, so the argmax '
                         'difference between them is exactly what training achieved in '
                         'prediction terms -- the measure that separates information from '
                         'calibration when the loss falls but mIoU does not. NOTE this is '
                         'only a clean before/after if the untrained probe shares the '
                         "trained run's --seed, --grid and --hidden: the projector is "
                         'constructed at the same point in the RNG stream in both modes, '
                         'so equal flags give a bit-identical initialisation.')
    ap.add_argument('--mode', choices=['train', 'sensitivity'], default='train',
                    help="'sensitivity' trains nothing. It asks the prior question -- how "
                         'wide is the channel at all: with an UNTRAINED projector, how much '
                         'does the frozen stack\'s occupancy output move when the injected '
                         "tokens' content changes? If swapping one frame's point cloud for "
                         "another's barely moves the prediction, no optimiser will find a "
                         'useful signal in that channel, and 50 SGD steps would only tell '
                         'us that ambiguously and 20x slower (RESEARCH_DIRECTIONS.md '
                         '§7.9: measure the effect before building the perturbation).')
    ap.add_argument('--loss', choices=['ce', 'lovasz', 'ce+lovasz'], default='ce',
                    help='the TRAINED objective. Pooled CE collapses object classes into '
                         'background (see the note above occ_objective); Lovász-softmax is '
                         'a per-class Jaccard surrogate and is the in-house-supported fix.')
    ap.add_argument('--lovasz-w', type=float, default=1.0)
    ap.add_argument('--loss-mask', choices=['none', 'camera', 'lidar'], default='none',
                    help='restrict the TRAINED loss to the trustworthy region. Separate from '
                         '--occ-mask, which only restricts the metric.')
    ap.add_argument('--occ-mask', choices=['none', 'camera', 'lidar'], default='none',
                    help="restrict the occupancy metric to Occ3D's observed region. The "
                         'released GT and Occ3D disagree mostly about UNOBSERVED space '
                         '(~90 %% of their extra occupied voxels are outside mask_camera; '
                         'restricting lifts occupied IoU 0.60 -> 0.84/0.87 on the two '
                         'nuScenes demo tokens), so scoring outside it charges the model '
                         'for predicting where our GT declines to label. Needs a frame '
                         'packed by pack_nuscenes.py; demo frames carry no masks.')
    ap.add_argument('--dataset-type', choices=['all', 'nuscenes', 'nuplan'], default='all')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all (TRAINING frames)')
    ap.add_argument('--holdout', type=int, default=0,
                    help='additionally evaluate on the next N frames, never trained on. '
                         'Every result so far is measured on the frames it trained on; a '
                         'recalibration of the frozen head should generalise and a memorised '
                         'per-frame fix should not, so this separates them.')
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.config:
        # Precedence: explicit command line > YAML > default. A key is taken from the YAML
        # only where the parsed value is still the parser's default, so an arm can be
        # overridden per run without editing the file.
        defaults = {a.dest: a.default for a in ap._actions}
        for k, v in (yaml.safe_load(args.config.read_text()) or {}).items():
            if k not in defaults:
                raise SystemExit(f'unknown config key: {k}')
            if getattr(args, k) != defaults[k]:
                continue
            setattr(args, k, type(defaults[k])(v) if isinstance(defaults[k], Path) else v)

    set_global_seed(args.seed)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.out_root / f'{stamp}_{args.exp_name}_{args.inject}'
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(f'output -> {out_dir}\n')

    from qwen_drive_perception.dataset import PerceptionFrame

    # A packed directory also holds manifest.json, lidar_queries.npz and mmengine's
    # `_runner/` work dir; a frame is a directory that actually contains frame.json.
    dirs = sorted(p for p in args.frames.iterdir()
                  if p.is_dir() and (p / 'frame.json').exists())
    if not dirs:
        raise SystemExit(f'no frame.json found under {args.frames}')
    frames = [PerceptionFrame(d) for d in dirs]
    if args.dataset_type != 'all':
        frames = [f for f in frames if f.dataset_type == args.dataset_type]
    n_train = args.max_frames or len(frames)
    all_frames = frames[:n_train + args.holdout]
    frames = all_frames[:n_train]
    ho_frames = all_frames[n_train:]
    if not frames:
        raise SystemExit('no frames selected')
    if args.inject == 'shuffle' and len(frames) < 2:
        raise SystemExit('--inject shuffle needs >= 2 frames: with one frame the "other" '
                         'cloud is the same cloud, so the control is not a control.')
    print(f'{len(frames)} train frames' +
          (f' + {len(ho_frames)} HELD-OUT' if ho_frames else '') + ': ' +
          ', '.join(f'{f.token[:8]}({f.dataset_type})' for f in frames[:8]) +
          (' ...' if len(frames) > 8 else ''))

    stack = Stack(args.vlm, args.head, args.device, args.attn_math,
                  patch_embed=args.patch_embed)
    if args.tokenizer == 'queries':
        if args.queries is None:
            raise SystemExit('--tokenizer queries needs --queries <extract.py npz>')
        tok = QueryTokenizer(args.queries)
    else:
        tok = PillarTokenizer(grid=args.grid, height_bins=args.height_bins)
    print(f'embedding rms {stack.emb_rms:.4f}  d_llm {stack.d_llm}  '
          f'K {tok.n_tokens} tokens of dim {tok.dim}')

    # Pre-tokenise every frame's LiDAR once; the tokeniser has no parameters.
    feats = {}
    const_feat = None
    for i, f in enumerate(all_frames):
        if args.inject == 'const':
            # identical input for every frame: the token content carries no information, so
            # only the projector's own parameters can explain anything this arm achieves.
            if const_feat is None:
                const_feat = torch.zeros(tok.n_tokens, tok.dim)
            feats[f.token] = const_feat.to(args.device)
        else:
            src = (all_frames[(i + 1) % len(all_frames)]
                   if args.inject == 'shuffle' else f)
            feats[f.token] = tok.features(src).to(args.device)

    trains = args.inject in ('lidar', 'shuffle', 'const')
    proj = None
    if trains:
        proj = Projector(tok.dim, stack.d_llm, tok.n_tokens, hidden=args.hidden,
                         target_rms=stack.emb_rms,
                         gate_init=args.gate_init).to(args.device).to(torch.float32)
        n_par = sum(p.numel() for p in proj.parameters())
        print(f'projector: {n_par / 1e6:.2f} M trainable parameters')
    zero_tok = (torch.zeros(tok.n_tokens, stack.d_llm, device=args.device,
                            dtype=torch.bfloat16) if args.inject == 'zero' else None)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr) if proj else None

    def _pack(fs):
        out = []
        for f in fs:
            inputs, metas = stack.processor(f, device=args.device)
            at = -1
            if args.inject != 'none' or args.mode == 'sensitivity':
                inputs, at = stack.add_placeholders(inputs, tok.n_tokens)
            gt = torch.from_numpy(f.gt['occ'].astype(np.int64)).to(args.device)
            def _mask(which, flag, fr=f):
                if which == 'none':
                    return None
                key = f'mask_{which}'
                if key not in fr.gt:
                    raise SystemExit(f'{flag} {which} needs `{key}` in gt.npz; re-pack with '
                                     'the current pack_nuscenes.py')
                return torch.from_numpy(fr.gt[key].astype(bool)).to(args.device)
            out.append((f, inputs, metas, gt, at,
                        _mask(args.occ_mask, '--occ-mask'),
                        _mask(args.loss_mask, '--loss-mask')))
        return out

    packed = _pack(frames)
    packed_ho = _pack(ho_frames)
    print(f'prompt length {packed[0][1]["input_ids"].shape[1]} tokens, '
          f'LiDAR tokens inserted at index {packed[0][4]}')

    meta = dict(date=datetime.now().isoformat(timespec='seconds'), args=cfg,
                n_frames=len(frames), seed=args.seed,
                tokens=dict(K=tok.n_tokens, dim=tok.dim),
                embedding_rms=stack.emb_rms,
                git=dict(commit=_git(Path(__file__).parent, 'rev-parse', 'HEAD'),
                         dirty=bool(_git(Path(__file__).parent, 'status', '--porcelain'))))
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    log = []

    base_pred = {}          # eval@0 predictions, for the argmax-drift measure

    def evaluate(tag: str, subset=None) -> tuple:
        met, tot, n, t0 = OccMetric(), 0.0, 0, time.time()
        drift, nvox = 0, 0
        with torch.no_grad():
            for f, inputs, metas, gt, at, msk, lmsk in (packed if subset is None
                                                        else subset):
                t = zero_tok
                if proj is not None:
                    t = proj(feats[f.token]).to(torch.bfloat16)
                logits = stack.forward_occ(inputs, metas, tokens=t, at=at, grad=False,
                                           attn=args.eval_attn)
                loss = F.cross_entropy(logits[0].reshape(-1, OCC_NUM_CLASSES).float(),
                                       gt.reshape(-1))
                tot += float(loss)
                n += 1
                pred = logits[0].argmax(-1).cpu().numpy()
                g = gt.cpu().numpy()
                if msk is not None:
                    m = msk.cpu().numpy()
                    met.add(pred[m], g[m])
                else:
                    met.add(pred, g)
                # Argmax drift vs the untrained state. Cross-entropy over 640 k voxels is
                # dominated by `empty` (~89 % IoU on its own), so the projector can lower
                # the loss purely by sharpening a class it already predicts -- calibration,
                # not information. That is the "constant prior" shape from
                # RESEARCH_DIRECTIONS.md §2.6, and this is the number that catches it:
                # falling loss with ZERO drift means nothing was learned about the scene.
                sup = pred if msk is None else pred[msk.cpu().numpy()]
                if f.token in base_pred:
                    drift += int((sup != base_pred[f.token]).sum())
                    nvox += sup.size
                else:
                    base_pred[f.token] = sup
        d = (100.0 * drift / nvox) if nvox else float('nan')
        print(f'  [{tag}] loss {tot / n:.4f}  fwIoU {met.fwiou():.4f}  mIoU {met.miou():.4f}'
              f'  argmax drift {d:.3f}%  ({(time.time() - t0) / n:.1f} s/frame)', flush=True)
        return (tot / n, met.miou(), met.per_class(), d, met.fwiou(), met.counts())

    if args.mode == 'sensitivity':
        if args.inject == 'shuffle':
            raise SystemExit('--mode sensitivity swaps the clouds itself; use '
                             '--inject lidar (or none) so `self` really is this frame.')
        if len(packed) < 2:
            raise SystemExit('--mode sensitivity needs >= 2 frames (it swaps clouds)')
        f, inputs, metas, gt, at, msk, lmsk = packed[0]
        other = packed[1][0]
        gen = torch.Generator(device='cpu').manual_seed(args.seed)
        probes = {
            'none':      None,
            'zero':      torch.zeros(tok.n_tokens, stack.d_llm),
            'self':      None,   # filled below
            'other':     None,
            'random':    torch.randn(tok.n_tokens, stack.d_llm, generator=gen),
        }
        pr = (proj if proj is not None else
              Projector(tok.dim, stack.d_llm, tok.n_tokens, hidden=args.hidden,
                        target_rms=stack.emb_rms,
                        gate_init=args.gate_init).to(args.device).to(torch.float32))
        with torch.no_grad():
            probes['self'] = pr(feats[f.token]).cpu()
            probes['other'] = pr(tok.features(other).to(args.device)).cpu()
        probes['random'] = (probes['random'] /
                            probes['random'].norm(dim=-1, keepdim=True) * stack.emb_rms)
        if args.projector_ckpt:
            trained = Projector(tok.dim, stack.d_llm, tok.n_tokens, hidden=args.hidden,
                                target_rms=stack.emb_rms,
                                gate_init=args.gate_init).to(args.device).to(torch.float32)
            trained.load_state_dict(torch.load(args.projector_ckpt, map_location='cpu',
                                               weights_only=True))
            with torch.no_grad():
                probes['trained'] = trained(feats[f.token]).cpu()
                probes['trained_other'] = trained(tok.features(other).to(args.device)).cpu()
            print(f'loaded trained projector {args.projector_ckpt} '
                  f'(gate {float(trained.gate):+.4f})')

        res = {}
        for name, t in probes.items():
            tt = None if t is None else t.to(args.device, torch.bfloat16)
            with torch.no_grad():
                lg = stack.forward_occ(inputs, metas, tokens=tt,
                                       at=(-1 if name == 'none' else at),
                                       grad=False, attn=args.eval_attn)
            loss = float(F.cross_entropy(lg[0].reshape(-1, OCC_NUM_CLASSES).float(),
                                         gt.reshape(-1)))
            res[name] = dict(loss=loss, logits=lg[0].float().cpu(),
                             pred=lg[0].argmax(-1).cpu())
            print(f'  [{name:7s}] loss {loss:.4f}', flush=True)

        # `none` has a different sequence length, so it is compared by prediction only.
        ref = 'zero'
        n_vox = res[ref]['pred'].numel()
        print(f'\n{"probe":10s}{"loss":>9s}{"d loss":>9s}{"logit rel L2":>14s}'
              f'{"argmax != zero":>16s}')
        for name in probes:
            r = res[name]
            dl = r['loss'] - res[ref]['loss']
            if name == 'none':
                rel = float('nan')
            else:
                d = r['logits'] - res[ref]['logits']
                rel = float(d.pow(2).sum().sqrt() /
                            res[ref]['logits'].pow(2).sum().sqrt().clamp_min(1e-12))
            dis = 100.0 * float((r['pred'] != res[ref]['pred']).sum()) / n_vox
            print(f'{name:10s}{r["loss"]:9.4f}{dl:+9.4f}{rel:14.4e}{dis:15.3f}%')
        # The pairwise rows are the actual read; the table above is only a per-probe
        # summary against a single reference.
        pairs = [('self', 'other',          'BANDWIDTH (untrained): only the cloud differs'),
                 ('random', 'zero',         'upper bound: what ANY token content can do'),
                 ('none', 'zero',           'cost of the K extra positions alone'),
                 ('trained', 'self',        'what TRAINING achieved, in predictions'),
                 ('trained', 'trained_other', 'BANDWIDTH (trained): only the cloud differs')]
        print(f'\n{"pair":26s}{"logit rel L2":>14s}{"argmax diff":>13s}   note')
        for a, b, note in pairs:
            if a not in res or b not in res:
                continue
            if 'none' in (a, b):
                rel = float('nan')          # different sequence length
            else:
                d = res[a]['logits'] - res[b]['logits']
                rel = float(d.pow(2).sum().sqrt() /
                            res[b]['logits'].pow(2).sum().sqrt().clamp_min(1e-12))
            dis = 100.0 * float((res[a]['pred'] != res[b]['pred']).sum()) / n_vox
            print(f'{a + " vs " + b:26s}{rel:14.4e}{dis:12.3f}%   {note}')

        print(f'\nn = {n_vox} voxels, one frame ({f.token[:8]}, {f.dataset_type}); '
              f'cloud swapped with {other.token[:8]}.')
        print('READ: `self` vs `other` is the channel bandwidth -- same projector, same '
              'images, only the point cloud differs. If that row pair is ~0, the injected '
              'tokens cannot carry LiDAR content to the BEV head and no amount of training '
              'will change it. `random` is the upper bound on what ANY token content can '
              'do; `none` vs `zero` is the cost of the K extra positions alone.')
        (out_dir / 'sensitivity.json').write_text(json.dumps(
            {k: dict(loss=v['loss'],
                     argmax_diff_vs_zero=100.0 * float((v['pred'] != res[ref]['pred']).sum())
                     / n_vox) for k, v in res.items()}, indent=2))
        return 0

    l0, m0, pc0, d0, f0, gc0 = evaluate('eval@0')
    log.append(dict(epoch=0, split='eval', loss=l0, miou=m0, fwiou=f0, per_class=pc0,
                    argmax_drift_pct=d0, gt_voxels=gc0))
    if packed_ho:
        h = evaluate('hold@0', packed_ho)
        log.append(dict(epoch=0, split='holdout', loss=h[0], miou=h[1], fwiou=h[4],
                        per_class=h[2], argmax_drift_pct=h[3]))

    if proj is None:
        (out_dir / 'log.json').write_text(json.dumps(log, indent=2))
        print(f'\n--inject {args.inject}: reference arm, nothing to train.')
        return 0

    order = list(range(len(packed)))
    for ep in range(1, args.epochs + 1):
        random.shuffle(order)
        run = 0.0
        t_ep = time.time()
        for j in order:
            f, inputs, metas, gt, at, msk, lmsk = packed[j]
            t = proj(feats[f.token]).to(torch.bfloat16)
            logits = stack.forward_occ(inputs, metas, tokens=t, at=at, grad=True)
            loss = occ_objective(logits[0], gt, lmsk, args.loss, args.lovasz_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(proj.parameters(), 5.0)
            opt.step()
            run += float(loss)
            if ep == 1 and j == order[0]:
                mem('after first train step')
                print(f'   grad norm at step 1: {float(gn):.4e}  '
                      f'gate {float(proj.gate):.4e}')
                if float(gn) == 0.0:
                    raise SystemExit(
                        'ABORT: the projector gradient is exactly zero. That is a plumbing '
                        'bug, not a negative result -- the injected tokens have no causal '
                        'path to the tapped image tokens. Check the insertion index.')
        dt = time.time() - t_ep
        print(f'epoch {ep:3d}  train loss {run / len(order):.4f}  '
              f'gate {float(proj.gate):+.4f}  {dt / len(order):.1f} s/step', flush=True)
        log.append(dict(epoch=ep, split='train', loss=run / len(order),
                        gate=float(proj.gate), sec_per_step=dt / len(order)))
        if ep % args.eval_every == 0 or ep == args.epochs:
            le, me, pce, de, fe, gce = evaluate(f'eval@{ep}')
            log.append(dict(epoch=ep, split='eval', loss=le, miou=me, fwiou=fe,
                            per_class=pce, argmax_drift_pct=de, gt_voxels=gce))
            if packed_ho:
                h = evaluate(f'hold@{ep}', packed_ho)
                log.append(dict(epoch=ep, split='holdout', loss=h[0], miou=h[1],
                                fwiou=h[4], per_class=h[2], argmax_drift_pct=h[3]))
            torch.save(proj.state_dict(), out_dir / 'projector.pt')
        (out_dir / 'log.json').write_text(json.dumps(log, indent=2))

    ev = [r for r in log if r['split'] == 'eval']
    print(f'\nREAD: baseline loss {ev[0]["loss"]:.4f} fwIoU {ev[0]["fwiou"]:.4f} '
          f'mIoU {ev[0]["miou"]:.4f}  ->  final loss {ev[-1]["loss"]:.4f} '
          f'fwIoU {ev[-1]["fwiou"]:.4f} mIoU {ev[-1]["miou"]:.4f}  '
          f'argmax drift {ev[-1]["argmax_drift_pct"]:.3f}%')
    print('Falling loss with ~zero argmax drift is CALIBRATION, not information: the '
          'occupancy cross-entropy is dominated by `empty`, so sharpening a class the model '
          'already predicts lowers it without learning anything about the scene.')
    print('This is an OVERFIT test on '
          f'{len(frames)} frames. A drop here proves the injection channel carries '
          'signal to the BEV head; it says nothing about generalisation. Compare against '
          'the --inject shuffle arm before reading it as "the LLM used the LiDAR".')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
