"""Make Qwen-Drive's released perception head differentiable.

WHY THIS EXISTS
---------------
`Qwen-Drive-1.0` ships the perception stack **inference-only**: `QwenDrivePerception.infer`
is `@torch.no_grad()`, and -- less obviously -- two of its custom CUDA operators have no
backward:

* ``ops/__init__.py:ms_deform_attn_bf16_forward`` calls the JIT extension's ``.forward()``
  *directly*, with no ``torch.autograd.Function`` wrapper. Its output is a fresh tensor
  with ``grad_fn=None``, so **every deformable-attention call silently cuts the graph.**
* ``ops/__init__.py:_VoxelPoolDepthCuda`` *is* an ``autograd.Function`` but defines only
  ``forward``; a backward through it raises.

Consequence, measured: with grad enabled and ``img_llm_feats.requires_grad=True``, every
output of ``bev_modeling`` (``bev_embed``, ``all_cls_scores``, ``all_bbox_preds``,
``occ_pred``, ``seg_preds``) comes back ``requires_grad=False, grad_fn=None``. The
submodules on the way in (``adaptor``, ``vit_neck``) *do* keep the graph
(``grad_fn=AddBackward0``), which localises the break to the encoder's attention.

WHY IT IS CHEAP TO FIX
----------------------
The release also contains a **pure-torch fallback** for exactly this op --
``layers.py:multi_scale_deformable_attn_pytorch``, a ``F.grid_sample``-based
implementation that the dispatcher already uses when the tensors are not on CUDA
(``attention.py:52-53``). It is autograd-native. So the fix is not "write a backward
kernel": it is to route the CUDA path through the fallback the authors already shipped.

The second op needs no fix *for our use case*, and that is a property of the architecture
worth stating: ``voxel_pool_depth`` lives on the **ViT stream** (``view_transform.py``,
fed by ``img_vit_feats``). The ViT runs *before* the language model, so tokens injected at
the LLM's input cannot change ``img_vit_feats`` at all. Keep that stream detached and
``_VoxelPoolDepthCuda.apply`` receives no grad-requiring input, creates no autograd node,
and is never asked to go backward. The LLM stream (``adaptor`` -> BEV encoder spatial
cross-attention -> ``bev_embed`` -> det decoder / occ decoder / map decoder) is the one we
need, and it is deformable-attention all the way.

COST
----
The fallback is slower and holds more activation memory than the kernel (it materialises
``[bs*heads, C, n_query, n_level*n_point]`` per level, and the BEV encoder's n_query is
``bev_h*bev_w`` = 40000). Use ``patch(...)`` only for the training forward; leave the
kernel in place for evaluation. `math_dtype=torch.float32` additionally runs the
interpolation and the weighted sum in fp32 -- the kernel does this internally too, and
here it also keeps the *backward* out of bf16.

FIDELITY
--------
`check_grad.py --stage parity` compares the two paths on the real tensors. They are not
bit-identical (fp32 accumulation order differs, and the kernel uses ``--use_fast_math``);
what matters is that the fallback is the authors' own reference implementation of the same
math, so a projector trained through it is trained through the model's actual function.

USAGE
-----
    from qwen_drive_grad import patch_deformable_attention
    with patch_deformable_attention():
        outs = model.bev_modeling(img_vit_feats=vit.detach(), img_llm_feats=llm, ...)
        loss.backward()
"""

from __future__ import annotations

import contextlib

import torch

__all__ = ['patch_deformable_attention', 'differentiable_msda', 'audit_ops',
           'time_modules', 'set_vit_checkpointing', 'set_vision_patch_embed']


def differentiable_msda(value, value_spatial_shapes, value_level_start_index,
                        sampling_locations, attention_weights, im2col_step,
                        math_dtype=torch.float32):
    """Drop-in replacement for ``attention.multi_scale_deformable_attn_cuda``.

    Same signature (``level_start_index`` and ``im2col_step`` are kernel-tiling
    parameters the pure-torch path does not need), autograd-native, and returns the
    input dtype so the caller sees no change.
    """
    from qwen_drive_perception.layers import multi_scale_deformable_attn_pytorch
    out_dtype = value.dtype
    out = multi_scale_deformable_attn_pytorch(
        value.to(math_dtype), value_spatial_shapes,
        sampling_locations.to(math_dtype), attention_weights.to(math_dtype))
    return out.to(out_dtype)


@contextlib.contextmanager
def patch_deformable_attention(math_dtype=torch.float32, verbose: bool = True):
    """Route Qwen-Drive's deformable attention through its differentiable fallback.

    All three call sites (``attention.py`` lines 139/206/341) resolve
    ``multi_scale_deformable_attn_cuda`` as a module global, so patching the one module
    attribute covers the whole head. Restored on exit.
    """
    from qwen_drive_perception import attention as _attn
    orig = getattr(_attn, 'multi_scale_deformable_attn_cuda', None)
    if orig is None:                       # fail loud rather than silently no-op
        raise RuntimeError(
            'qwen_drive_perception.attention.multi_scale_deformable_attn_cuda is gone -- '
            'the upstream release changed; re-derive the patch point before trusting any '
            'gradient measured here.')

    def _shim(*a, **kw):
        kw.pop('math_dtype', None)
        return differentiable_msda(*a, **kw, math_dtype=math_dtype)

    _attn.multi_scale_deformable_attn_cuda = _shim
    if verbose:
        print(f'[qwen_drive_grad] deformable attention -> pure-torch fallback '
              f'(math in {math_dtype}); CUDA kernel bypassed')
    try:
        yield
    finally:
        _attn.multi_scale_deformable_attn_cuda = orig


def audit_ops() -> dict:
    """Report which released ops lack a backward. Evidence, not a guess."""
    import inspect

    from qwen_drive_perception import ops
    rep = {}
    src = inspect.getsource(ops.ms_deform_attn_bf16_forward)
    rep['ms_deform_attn'] = dict(
        is_autograd_function=False,
        raw_extension_call='.forward(' in src,
        verdict='cuts the graph (no autograd.Function)')
    vp = ops._VoxelPoolDepthCuda
    rep['voxel_pool_depth'] = dict(
        is_autograd_function=issubclass(vp, torch.autograd.Function),
        has_backward=('backward' in vp.__dict__),
        verdict='autograd.Function with no backward -> raises if reached')
    return rep


@contextlib.contextmanager
def time_modules(named: dict, out: dict):
    """Time the ``forward`` of each ``{label: module}``, cumulatively.

    ``out[label] = {'sec': float, 'calls': int}``. Wraps the bound method rather than using
    hooks, so a module called N times per forward accumulates all N intervals into one
    entry. **Nested labels are inclusive** -- timing ``head`` and ``head.encoder`` together
    means the first contains the second; read the tree, not the sum.
    ``torch.cuda.synchronize()`` on both sides: without it every number is just the kernel
    launch cost and the whole profile is meaningless.

    Why this exists: one forward of the released stack costs ~100 s/frame here, which makes
    any real dataset impossible, and the three timings we have (kernel / fallback-fp32 /
    fallback-bf16, all ~100 s) prove it is neither the gradient nor the deformable
    attention. That narrows it to "somewhere else", which is not an answer.
    """
    import time
    originals = {}

    def wrap(label, mod):
        orig = mod.forward

        def timed(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                return orig(*a, **kw)
            finally:
                torch.cuda.synchronize()
                e = out.setdefault(label, {'sec': 0.0, 'calls': 0})
                e['sec'] += time.time() - t0
                e['calls'] += 1
        return orig, timed

    try:
        for label, mod in named.items():
            if mod is None:
                continue
            orig, timed = wrap(label, mod)
            originals[label] = (mod, orig)
            mod.forward = timed
        yield out
    finally:
        for mod, orig in originals.values():
            mod.forward = orig


def set_vit_checkpointing(vlm, enabled: bool) -> int:
    """Turn gradient checkpointing on/off for the **vision tower only**. Returns #modules hit.

    Why: `vlm.train()` is required for HF to apply checkpointing at all (it gates on
    `self.training`), but `gradient_checkpointing_enable()` is global, so it also wraps the
    24 `Qwen3_5VisionBlock`s -- which are `GradientCheckpointingLayer`s. Under `no_grad`,
    non-reentrant checkpoint still executes the block inside a TorchDispatchMode, so every
    aten op pays Python dispatch. Measured on one 8-camera frame, `vlm.visual` was **91 s of
    a 92 s forward (98.9 %)** while the 4 B language model took 0.63 s and the whole BEV
    head 0.37 s.

    For this project the vision tower never needs a backward: tokens injected at the LLM's
    input cannot affect it (the ViT runs first), so its stream is detached. Checkpointing it
    buys nothing and costs everything.
    """
    n = 0
    visual = getattr(getattr(vlm, 'model', vlm), 'visual', None)
    if visual is None:
        return 0
    for m in visual.modules():
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = enabled
            n += 1
    return n


def set_vision_patch_embed(vlm, mode: str = 'matmul', verify: bool = True) -> dict:
    """Replace the vision patch embedding's Conv3d with the equivalent matmul.

    **This is the single biggest cost in the released forward.** Profiled on one 8-camera
    frame (`check_grad.py --stage profile`), steady state:

        vlm.visual (ViT)   91.85 s   99.0 %  of a 92.76 s infer()
          patch_embed      91.79 s   98.9 %
          blocks[0..23]     0.00 s
        vlm.language_model  0.57 s
        bev_head            0.33 s

    `Qwen3_5VisionPatchEmbed` is an `nn.Conv3d(3, 1024, kernel_size=[2,16,16],
    stride=[2,16,16])` applied to `(N, 3, 2, 16, 16)`, i.e. the kernel covers the *entire*
    input and the output is 1x1x1 per patch. That is a matmul expressed as a convolution,
    and with N = 14336 in bf16 it lands on a cudnn path ~10^4 times slower than the same
    arithmetic as a GEMM. Not gradient checkpointing: measured on and off, 91.85 vs 92.86 s.

    The contraction runs over ``(in_ch, kt, kh, kw)`` in C order for both operands, so
    ``x.reshape(N, -1) @ w.reshape(out, -1).T + b`` is the *same* computation -- and the
    incoming tensor is already ``(N, 1536)``, so even the reshape is free. bf16 accumulation
    order differs, hence ``verify``.

    Returns a dict with the verification numbers (and timings when verify=True).
    """
    import time
    visual = getattr(getattr(vlm, 'model', vlm), 'visual', None)
    pe = getattr(visual, 'patch_embed', None)
    if pe is None:
        raise RuntimeError('no visual.patch_embed -- upstream layout changed')
    if mode == 'conv':
        if hasattr(pe, '_orig_forward'):
            pe.forward = pe._orig_forward
            del pe._orig_forward
        return {'mode': 'conv'}

    conv = pe.proj
    if not isinstance(conv, torch.nn.Conv3d):
        raise RuntimeError(f'expected Conv3d, found {type(conv).__name__}; re-derive this')
    out_dim = conv.weight.shape[0]
    in_dim = int(conv.weight[0].numel())

    def matmul_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        w = conv.weight.reshape(out_dim, in_dim)
        x = hidden_states.reshape(-1, in_dim).to(w.dtype)
        out = x @ w.t()
        if conv.bias is not None:
            out = out + conv.bias
        return out

    rep = {'mode': 'matmul', 'in_dim': in_dim, 'out_dim': out_dim}
    if verify:
        dev, dt = conv.weight.device, conv.weight.dtype
        x = torch.randn(4096, in_dim, device=dev, dtype=dt)
        with torch.no_grad():
            a = pe.forward(x)
            torch.cuda.synchronize()
            t0 = time.time()
            a = pe.forward(x)
            torch.cuda.synchronize()
            rep['conv_sec'] = time.time() - t0
            b = matmul_forward(x)
            torch.cuda.synchronize()
            t0 = time.time()
            b = matmul_forward(x)
            torch.cuda.synchronize()
            rep['matmul_sec'] = time.time() - t0
        d = (a.float() - b.float()).abs()
        rep['max_abs_diff'] = float(d.max())
        rep['rel_l2'] = float((a.float() - b.float()).pow(2).sum().sqrt()
                              / a.float().pow(2).sum().sqrt().clamp_min(1e-12))
        rep['speedup'] = rep['conv_sec'] / max(rep['matmul_sec'], 1e-9)

    if not hasattr(pe, '_orig_forward'):
        pe._orig_forward = pe.forward
    pe.forward = matmul_forward
    return rep
