"""Optimizers, LR schedules and weight EMA.

AdamW  -- the textbook default.
Muon   -- "MomentUm Orthogonalized by Newton-Schulz" (Keller Jordan, modded-nanogpt), used by every
          parameter-golf record. For each 2-D weight matrix the momentum-averaged gradient G is
          replaced by its nearest semi-orthogonal matrix  U V^T  (from G = U S V^T): every singular
          direction gets the same step size, so rarely-updated directions are not starved.
          Embeddings, the output head and all 1-D/scalar params still use Adam.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


# ===================================================================================== Muon
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of G with a quintic Newton-Schulz iteration (no SVD needed).

    bf16 on CUDA (fast, and the iteration is robust to it); fp32 on MPS/CPU for portability.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)                 # spectral norm <= Frobenius norm <= 1
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters. Under DDP each rank orthogonalises a slice of the parameters and
    the updates are all-reduced (same scheme as the parameter-golf baseline)."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            flat = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=torch.float32)
            off = 0
            for i, p in enumerate(params):
                if i % world == rank and p.grad is not None:
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p.grad)
                    buf = st["buf"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g = g * max(1.0, g.size(0) / g.size(1)) ** 0.5   # keep RMS update size shape-independent
                    flat[off: off + p.numel()] = g.reshape(-1).float()
                off += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            off = 0
            for p in params:
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(flat[off: off + p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                off += p.numel()


# ============================================================================== param groups
def build_optimizers(model, tcfg, device_info) -> list[torch.optim.Optimizer]:
    """AdamW: decay on >=2-D weights, no decay on norms/biases/scalars.
    Muon:  Muon on hidden matrices; Adam (no decay) on embeddings, head, and scalars."""
    lr_embed = tcfg.embed_lr or tcfg.lr
    betas = tuple(tcfg.betas)
    fused = {"fused": True} if device_info.fused_adam else {}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    emb_ids = {id(p) for n, p in named if n.split(".")[0] in ("tok_emb", "pos_emb", "embed", "lm_head", "head")}

    if tcfg.optimizer == "adamw":
        decay = [p for n, p in named if p.ndim >= 2 and id(p) not in emb_ids]
        nodecay = [p for n, p in named if p.ndim < 2]
        emb = [p for n, p in named if id(p) in emb_ids]
        groups = [
            {"params": decay, "weight_decay": tcfg.weight_decay, "lr": tcfg.lr, "base_lr": tcfg.lr},
            {"params": nodecay, "weight_decay": 0.0, "lr": tcfg.lr, "base_lr": tcfg.lr},
            {"params": emb, "weight_decay": 0.0, "lr": lr_embed, "base_lr": lr_embed},
        ]
        groups = [g for g in groups if g["params"]]
        return [torch.optim.AdamW(groups, betas=betas, eps=1e-8, **fused)]

    if tcfg.optimizer == "muon":
        if not hasattr(model, "matrix_params"):
            raise ValueError("optimizer=muon needs a model with matrix_params() (arch=gpt)")
        mats = model.matrix_params()
        mat_ids = {id(p) for p in mats}
        emb = [p for n, p in named if id(p) in emb_ids]
        rest = [p for n, p in named if id(p) not in mat_ids and id(p) not in emb_ids]
        adam_groups = [g for g in (
            {"params": emb, "lr": lr_embed, "base_lr": lr_embed},
            {"params": rest, "lr": tcfg.lr, "base_lr": tcfg.lr},
        ) if g["params"]]
        adam = torch.optim.Adam(adam_groups, betas=betas, eps=1e-8, **fused)
        muon = Muon(mats, lr=tcfg.matrix_lr, momentum=tcfg.muon_momentum,
                    ns_steps=tcfg.muon_ns_steps, weight_decay=tcfg.muon_wd)
        for g in muon.param_groups:
            g["base_lr"] = tcfg.matrix_lr
        return [adam, muon]
    raise ValueError(f"unknown optimizer {tcfg.optimizer!r} (adamw | muon)")


# ================================================================================ schedules
def lr_multiplier(tcfg, step: int, progress: float) -> float:
    """progress in [0, 1] = max(step / max_steps, elapsed / max_time_sec).

    Using *progress* rather than step lets the same schedule work under a wall-clock budget
    (golf: "train for 600 s") -- the warmdown starts when 1 - warmdown_frac of the time is used.
    """
    if tcfg.warmup_steps > 0 and step < tcfg.warmup_steps:
        return (step + 1) / tcfg.warmup_steps
    lo = tcfg.min_lr_frac
    if tcfg.schedule == "constant":
        return 1.0
    if tcfg.schedule == "cosine":
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    if tcfg.schedule == "wsd":                       # warmup - stable - decay ("warmdown")
        start = 1.0 - tcfg.warmdown_frac
        if progress < start:
            return 1.0
        return lo + (1 - lo) * max(0.0, 1.0 - (progress - start) / max(tcfg.warmdown_frac, 1e-9))
    raise ValueError(f"unknown schedule {tcfg.schedule!r}")


def set_lr(optimizers, mult: float, step: int, tcfg) -> None:
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        if isinstance(opt, Muon):                    # golf: momentum warmup 0.85 -> 0.95
            frac = min(1.0, step / 300)
            for g in opt.param_groups:
                g["momentum"] = (1 - frac) * 0.85 + frac * tcfg.muon_momentum


# ===================================================================================== EMA
class EMA:
    """Exponential moving average of the weights; evaluated instead of the raw weights."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model, step: int):
        # warm-up: early on, average over ~step steps instead of ~1/(1-decay) steps, so a short
        # run is not evaluated on weights that are mostly the random initialisation
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.float(), 1 - d)

    def state_dict_for(self, model) -> dict:
        sd = model.state_dict()
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v) for k, v in sd.items()}
