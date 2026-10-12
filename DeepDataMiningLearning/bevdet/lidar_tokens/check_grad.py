"""Can a projector at the LLM's input be trained by a loss at Qwen-Drive's BEV head?

This is step (a)-second-half of the LiDAR-tokens-into-a-VLA plan, and it is a *feasibility
gate*, not an experiment. The plan is: frozen VLM + frozen LiDAR detector + train only a
projector that maps detector queries into the language model's token space. That only works
if gradient flows from the BEV head's output, back through the 4 B language model, to the
input embeddings. Qwen-Drive is released inference-only (`QwenDrivePerception.infer` is
decorated `@torch.no_grad()`), so this is not obvious and must be checked before any
projector is written -- cf. §7.8/§7.9 of `ngperception/docs/RESEARCH_DIRECTIONS.md`: prove
the apparatus can move before building it.

WHY THE MEASUREMENT IS CLEAN. `infer` taps the LLM at *image-token positions only*
(`image_mask = input_ids[0] == image_token_id`), reshapes them per camera, and hands that to
the BEV head. Injected LiDAR tokens would therefore never reach the BEV head directly --
they can only act by changing the image tokens' representations through attention. So there
is no plumbing shortcut: any downstream change is genuine propagation through the LLM.

WHAT THIS CHECKS
  1. the tap + BEV head run at all outside `@torch.no_grad()`
  2. a scalar loss on the occupancy logits produces a *non-zero* gradient at the output of
     the token embedding layer -- the exact tensor a projector would write into
  3. the gradient is finite (bf16 through 36 layers is a plausible place to underflow)

A zero or NaN gradient here kills the projector route as released, and the fallback would be
to tap the VLM at a shallower layer or to fine-tune with the head unfrozen.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

QD = Path('/data/rnd-liu/Others/Qwen-Drive-1.0')
sys.path.insert(0, str(QD / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_drive_grad import (audit_ops, patch_deformable_attention,  # noqa: E402
                             set_vision_patch_embed, set_vit_checkpointing,
                             time_modules)


class _Captured(Exception):
    """Raised to abort a forward once the stream shapes are known."""


def _pick(o):
    """First tensor whose key mentions occ, else the largest tensor in the output."""
    import torch
    if torch.is_tensor(o):
        return o
    if isinstance(o, dict):
        for k, v in o.items():
            if 'occ' in k.lower() and torch.is_tensor(v):
                return v
        c = [v for v in o.values() if torch.is_tensor(v)]
        if c:
            return max(c, key=lambda t: t.numel())
    if isinstance(o, (list, tuple)):
        for v in o:
            r = _pick(v)
            if r is not None:
                return r
    return None


def mem(tag: str) -> None:
    import torch
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.max_memory_allocated() / 2**30
    print(f'   [mem] {tag:34s} allocated {a:6.2f} GiB   peak {r:6.2f} GiB', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vlm', type=Path, default=Path('/data/rnd-liu/Others/Qwen-Drive-1.0-4B'))
    ap.add_argument('--model', type=Path,
                    default=Path('/data/rnd-liu/Others/Qwen-Drive-1.0-4B/perception'))
    ap.add_argument('--frames', type=Path, default=QD / 'data/demo/perception')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--stage', choices=['ops', 'parity', 'profile', 'llm', 'head', 'full'], default='llm',
                    help="which half of the gradient path to verify. The full path OOMed at "
                         "57-60 GiB for one frame even with checkpointing, so verify it in "
                         "halves: 'llm' = embedding -> LLM output at image positions (a "
                         "synthetic loss, no BEV head); 'head' = img_llm_feats -> occupancy "
                         "output (BEV head only). Both passing implies the composition. "
                         "'ops' only audits which released CUDA ops have a backward and "
                         "touches no GPU. 'parity' runs the BEV head twice on identical "
                         "inputs -- once through the CUDA kernel, once through the "
                         "pure-torch fallback -- and reports how far apart they are; a "
                         "projector trained through the fallback is only training against "
                         "the released model if the two agree.")
    ap.add_argument('--patch-embed', choices=['matmul', 'conv'], default='conv',
                    help="the vision patch embedding. The released module is an nn.Conv3d whose kernel covers its entire input -- a matmul in convolution clothing -- and on 14336 patches in bf16 it takes 91.5 s where the equivalent GEMM takes 0.0046 s, BIT-IDENTICAL output (max abs diff 0.0). It is 99%% of the released forward. 'matmul' is the fix; 'conv' reproduces the released behaviour.")
    ap.add_argument('--vit-checkpointing', choices=['on', 'off'], default='on',
                    help='gradient checkpointing on the VISION TOWER. `gradient_'
                         'checkpointing_enable()` is global and wraps the 24 vision blocks '
                         'too; under no_grad each still runs inside a TorchDispatchMode, so '
                         'every aten op pays Python dispatch. The vision tower never needs '
                         'a backward here (injected tokens cannot affect it), so `off` '
                         'costs nothing. Default `on` reproduces the measured 91 s.')
    ap.add_argument('--attn', choices=['kernel', 'torch'], default='kernel',
                    help="which deformable-attention implementation the BEV head uses. "
                         "'kernel' is the released bf16 CUDA op, which has NO autograd "
                         "wrapper and therefore cuts the graph -- keep it to reproduce the "
                         "FAIL. 'torch' routes through the release's own pure-torch "
                         "fallback (layers.multi_scale_deformable_attn_pytorch), which is "
                         "autograd-native. See qwen_drive_grad.py.")
    args = ap.parse_args()

    if args.stage == 'ops':
        import json
        print(json.dumps(audit_ops(), indent=2))
        return 0

    from transformers import AutoTokenizer
    from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation='sdpa')
    vlm = holder.vlm
    del holder.planning_expert
    model = QwenDrivePerception.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    model.attach(vlm.to(args.device), processor)

    # Everything stays frozen: we are training a projector, not the model.
    for p in vlm.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    # MEMORY. A first attempt OOMed at 60.5 GiB for ONE frame. Two causes, both fixed here
    # and both decision-relevant for the projector's training cost:
    #   * `output_hidden_states=True` (what the released `infer` uses) materialises all 36
    #     layers' hidden states; with a grad graph attached they are all retained, and we
    #     only ever read the last one.
    #   * no activation checkpointing on a 4 B decoder.
    # HF only applies checkpointing when `self.gradient_checkpointing and self.training`,
    # so `eval()` silently disables it -- the first two attempts reported "enabled" and
    # saved 3.5 of 60 GiB for exactly this reason. Params stay frozen either way; train()
    # here only re-enables checkpointing (Qwen3.5 has no dropout by default).
    try:
        vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        vlm.train()
        print('gradient checkpointing: enabled (vlm.train() so HF actually applies it)')
        if args.patch_embed == 'matmul':
            print('vision patch embed: Conv3d -> matmul  '
                  + repr(set_vision_patch_embed(vlm, 'matmul', verify=True)))
        if args.vit_checkpointing == 'off':
            n = set_vit_checkpointing(vlm, False)
            print(f'gradient checkpointing: DISABLED on the vision tower '
                  f'({n} modules) -- it needs no backward here')
    except Exception as e:
        print(f'gradient checkpointing: unavailable ({type(e).__name__})')

    if args.stage == 'llm':
        # "Does gradient cross the frozen 4B decoder to its input embeddings?" does not
        # need a driving frame -- the driving frame is what made this OOM (8.8 GiB of
        # pixels plus ~48 GiB of activation graph over ~6k image tokens). A short
        # text-only sequence answers the same question for ~1/50 of the memory.
        emb0 = vlm.get_input_embeddings()
        got = {}

        def keep0(_m, _a, out):
            out.requires_grad_(True)
            out.retain_grad()
            got['e'] = out
            return out
        h = emb0.register_forward_hook(keep0)
        try:
            ids = torch.randint(0, 1000, (1, 256), device=args.device)
            o = vlm.model.language_model(input_ids=ids, use_cache=False)
            last = getattr(o, 'last_hidden_state', o[0] if isinstance(o, tuple) else o)
            mem('after LLM forward (text-only, 256 tok)')
            last.float().pow(2).mean().backward()
            mem('after backward')
        finally:
            h.remove()
        e = got.get('e')
        g = None if e is None else e.grad
        ok = g is not None and float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
        if g is not None:
            g = g.float()
            print(f'\nembedding-output grad: shape {tuple(g.shape)}  '
                  f'|g|_1 {float(g.abs().sum()):.4e}  max {float(g.abs().max()):.4e}  '
                  f'finite {bool(torch.isfinite(g).all())}  nonzero_rows '
                  f'{int((g.abs().sum(-1) > 0).sum())}/{g.shape[-2]}')
        print(f'\nVERDICT[llm]: {"PASS -- gradient crosses the frozen decoder to the input embeddings" if ok else "FAIL"}')
        return 0 if ok else 1

    frame_dir = sorted(p for p in args.frames.iterdir() if p.is_dir())[0]
    frame = PerceptionFrame(frame_dir)
    inputs, img_metas = processor(frame, device=args.device)
    mem('after processor')

    if args.stage in ('parity', 'head'):
        # The two streams have DIFFERENT widths (the ViT-stream FPN wants 1024, the
        # LLM-stream 2560), so both shapes are captured from a real no-grad forward
        # rather than guessed -- an earlier version used `randn_like(llm)` for the ViT
        # stream and died on a channel mismatch.
        shapes = {}
        real = model.bev_modeling.forward

        def spy(img_vit_feats, img_llm_feats, img_metas, **kw):
            shapes['vit'] = tuple(img_vit_feats.shape)
            shapes['llm'] = tuple(img_llm_feats.shape)
            shapes['dtype'] = img_llm_feats.dtype
            shapes['vit_t'] = img_vit_feats.detach().clone()
            shapes['llm_t'] = img_llm_feats.detach().clone()
            raise _Captured
        model.bev_modeling.forward = spy
        try:
            with torch.no_grad():
                model.infer(inputs, img_metas)
        except _Captured:
            pass
        finally:
            model.bev_modeling.forward = real
        torch.cuda.empty_cache()
        print(f'captured stream shapes: vit {shapes["vit"]}  llm {shapes["llm"]}')

    if args.stage == 'profile':
        # Where do the ~100 s go? Labels are nested and INCLUSIVE (`bev_head` contains
        # `bev_head.encoder`), so read the indentation, not the column sum. Two passes: the
        # first pays every one-off cost (kernel JIT load, cudnn autotune, lazy buffers such
        # as the 56x32x118 frustum grid), the second is the steady state -- reporting only
        # a first pass would attribute those to whichever module happened to trigger them.
        bm = model.bev_modeling
        tr = getattr(bm.head, 'transformer', None)
        vis = vlm.model.visual
        blocks = getattr(vis, 'blocks', [])
        named = {
            'vlm.visual (ViT)':        vis,
            '  patch_embed':           getattr(vis, 'patch_embed', None),
            '  pos_embed':             getattr(vis, 'pos_embed', None),
            '  blocks[0] of %d' % len(blocks): blocks[0] if len(blocks) else None,
            '  blocks[-1]':            blocks[-1] if len(blocks) else None,
            '  merger':                getattr(vis, 'merger', None),
            'vlm.language_model':      vlm.model.language_model,
            'bev_head':                bm,
            '  adaptor (LLM stream)':  bm.adaptor,
            '  vit_neck':              bm.vit_neck,
            '  depth_net':             bm.depth_net,
            '  view_trans (LSS)':      bm.view_trans,
            '  head':                  bm.head,
            '    transformer':         tr,
            '      encoder':           getattr(tr, 'encoder', None),
            '      decoder':           getattr(tr, 'decoder', None),
            '      occ_decoder':       getattr(tr, 'occ_decoder', None),
        }
        for tag in ('pass 1 (cold)', 'pass 2 (steady)'):
            acc = {}
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), time_modules(named, acc):
                model.infer(inputs, img_metas)
            torch.cuda.synchronize()
            total = time.time() - t0
            print(f'\n{tag}: infer() total {total:6.2f} s')
            print(f'{"module":28s}{"sec":>9s}{"calls":>7s}{"% of total":>12s}')
            for k in named:
                e = acc.get(k)
                if e is None:
                    continue
                print(f'{k:28s}{e["sec"]:9.2f}{e["calls"]:7d}'
                      f'{100 * e["sec"] / max(total, 1e-9):11.1f}%')
            unacc = total - sum(acc[k]['sec'] for k in ('vlm.visual (ViT)',
                                                        'vlm.language_model', 'bev_head')
                                if k in acc)
            print(f'{"unaccounted (glue/decode)":28s}{unacc:9.2f}'
                  f'{"":7s}{100 * unacc / max(total, 1e-9):11.1f}%')
        return 0

    if args.stage == 'parity':
        # Fidelity control for the fallback swap. Uses the REAL tapped features, not
        # noise: the kernel and the fallback differ in accumulation order and the kernel
        # is built with --use_fast_math, so exact equality is not the bar. The bar is
        # that the decision the head makes is the same one.
        vit, llm = shapes['vit_t'], shapes['llm_t']
        with torch.no_grad():
            a = model.bev_modeling(img_vit_feats=vit, img_llm_feats=llm,
                                   img_metas=[img_metas])
            mem('after kernel forward')
            with patch_deformable_attention():
                b = model.bev_modeling(img_vit_feats=vit, img_llm_feats=llm,
                                       img_metas=[img_metas])
            mem('after fallback forward')
        keys = [k for k in a if torch.is_tensor(a[k])] if isinstance(a, dict) else []
        print(f'\n{"tensor":16s}{"shape":>26s}{"max|d|":>12s}{"rel L2":>12s}')
        for k in keys:
            x, y = a[k].float(), b[k].float()
            d = (x - y).abs()
            rel = float(d.pow(2).sum().sqrt() / x.pow(2).sum().sqrt().clamp_min(1e-12))
            print(f'{k:16s}{str(tuple(x.shape)):>26s}{float(d.max()):>12.4e}{rel:>12.4e}')
        if 'occ_pred' in keys:
            am = (a['occ_pred'].argmax(-1) == b['occ_pred'].argmax(-1)).float().mean()
            print(f'\noccupancy argmax agreement: {float(am) * 100:.3f}% of '
                  f'{a["occ_pred"].shape[1] * a["occ_pred"].shape[2] * a["occ_pred"].shape[3]} voxels')
        if 'seg_preds' in keys:
            am = (a['seg_preds'].argmax(1) == b['seg_preds'].argmax(1)).float().mean()
            print(f'map argmax agreement:       {float(am) * 100:.3f}%')
        return 0

    if args.stage == 'head':
        # Does gradient reach `img_llm_feats`, the tensor the BEV head reads?
        llm = torch.randn(*shapes['llm'], device=args.device,
                          dtype=shapes['dtype'], requires_grad=True)
        vit = torch.randn(*shapes['vit'], device=args.device, dtype=shapes['dtype'])
        # The ViT stream is deliberately NOT grad-requiring: the vision tower runs before
        # the language model, so tokens injected at the LLM input cannot change it. That
        # also keeps `voxel_pool_depth` (autograd.Function with no backward) off the graph.
        ctx = (patch_deformable_attention() if args.attn == 'torch'
               else contextlib.nullcontext())
        with ctx:
            outs = model.bev_modeling(img_vit_feats=vit, img_llm_feats=llm,
                                      img_metas=[img_metas])
            mem('after BEV head forward')
            tgt = _pick(outs)
            if tgt is None:
                print('FAIL: no tensor in BEV head output')
                return 1
            print(f'loss tensor: shape {tuple(tgt.shape)} dtype {tgt.dtype} '
                  f'grad_fn {type(tgt.grad_fn).__name__ if tgt.grad_fn else None}')
            if tgt.grad_fn is None:
                print('\nFAIL: the BEV head output carries no grad_fn -- the graph was cut '
                      'inside the head.\n      Run `--stage ops` for which operator, and '
                      '`--attn torch` for the fix.')
                return 1
            tgt.float().pow(2).mean().backward()
            mem('after backward')
        g = llm.grad
        ok = g is not None and float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
        if g is not None:
            print(f'\nimg_llm_feats grad: |g|_1 {float(g.abs().sum()):.4e}  '
                  f'max {float(g.abs().max()):.4e}  finite {bool(torch.isfinite(g).all())}')
        print(f'\nVERDICT[head]: {"PASS -- the BEV head backpropagates into its image features" if ok else "FAIL"}')
        return 0 if ok else 1

    # Retain the gradient at the token-embedding output -- the tensor a projector writes to.
    emb = vlm.get_input_embeddings()
    print(f'input embedding module: {type(emb).__name__}')
    grabbed = {}

    def keep(_m, _a, out):
        out.requires_grad_(True)
        out.retain_grad()
        grabbed['e'] = out
        return out
    h_emb = emb.register_forward_hook(keep)

    # Replicate `infer`'s VLM tap, without the no_grad decorator.
    captured = {}

    def vit_hook(_m, a, _o=None):
        captured['patches'] = a[0]
    visual = vlm.model.visual
    h_vit = visual.merger.register_forward_hook(vit_hook)

    try:
        # `vlm.model(...)` returns the base output, whose `last_hidden_state` is the same
        # tensor as `hidden_states[-1]` without materialising the other 35.
        out = vlm.model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                        image_grid_thw=inputs['image_grid_thw'],
                        mm_token_type_ids=model._modality_ids(inputs['input_ids']),
                        use_cache=False)
        last = getattr(out, 'last_hidden_state', None)
        if last is None:                      # fall back to the released path
            out = vlm(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                      image_grid_thw=inputs['image_grid_thw'],
                      mm_token_type_ids=model._modality_ids(inputs['input_ids']),
                      use_cache=False, output_hidden_states=True)
            last = out.hidden_states[-1]
        hs = vlm.model.language_model.norm(last)
        with torch.no_grad():
            patches = visual.merger.norm(captured['patches'])
        vit_feats = model._premerge_grids(patches, inputs['image_grid_thw'])

        n_cam = len(img_metas['cam_order'])
        gh, gw = inputs['image_grid_thw'][-1, 1].item(), inputs['image_grid_thw'][-1, 2].item()
        tpi = gh // 2 * gw // 2
        mask = inputs['input_ids'][0] == vlm.config.image_token_id
        llm_tok = hs[0][mask][-n_cam * tpi:]
        img_llm = llm_tok.view(n_cam, gh // 2, gw // 2, -1)
        img_vit = torch.stack(vit_feats[-n_cam:], dim=0)

        mem('after LLM forward + tap')
        if args.stage == 'llm':
            # Synthetic loss straight on the tapped image tokens: this isolates
            # "does gradient cross the 4B decoder to the embeddings" from the BEV head.
            img_llm.float().pow(2).mean().backward()
            mem('after backward (llm only)')
            e = grabbed.get('e')
            g = None if e is None else e.grad
            ok = g is not None and float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
            if g is not None:
                g = g.float()
                print(f'\nembedding-output grad: shape {tuple(g.shape)}  '
                      f'|g|_1 {float(g.abs().sum()):.4e}  max {float(g.abs().max()):.4e}  '
                      f'finite {bool(torch.isfinite(g).all())}  nonzero_rows '
                      f'{int((g.abs().sum(-1) > 0).sum())}/{g.shape[-2]}')
            print(f'\nVERDICT[llm]: {"PASS -- gradient crosses the decoder to the input embeddings" if ok else "FAIL"}')
            h_vit.remove(); h_emb.remove()
            return 0 if ok else 1

        dtype = next(model.bev_modeling.parameters()).dtype
        ctx = (patch_deformable_attention() if args.attn == 'torch'
               else contextlib.nullcontext())
        with ctx:
            outs = model.bev_modeling(img_vit_feats=img_vit.detach().to(dtype),
                                      img_llm_feats=img_llm.to(dtype),
                                      img_metas=[img_metas])
            mem('after BEV head forward')
    finally:
        h_vit.remove()
        h_emb.remove()

    def pick(o):
        if torch.is_tensor(o):
            return o
        if isinstance(o, dict):
            for k, v in o.items():
                if 'occ' in k.lower() and torch.is_tensor(v):
                    return v
            cands = [v for v in o.values() if torch.is_tensor(v)]
            if cands:
                return max(cands, key=lambda t: t.numel())
        if isinstance(o, (list, tuple)):
            for v in o:
                r = pick(v)
                if r is not None:
                    return r
        return None

    tgt = _pick(outs)
    if tgt is None:
        print('FAIL: could not find a tensor in the BEV head output')
        print('       output type:', type(outs),
              list(outs.keys()) if isinstance(outs, dict) else '')
        return 1
    print(f'loss tensor: shape {tuple(tgt.shape)} dtype {tgt.dtype}')
    loss = tgt.float().pow(2).mean()
    loss.backward()

    e = grabbed.get('e')
    if e is None or e.grad is None:
        print('\nFAIL: no gradient reached the input embeddings.')
        print('      -> the projector-at-the-input route does not work as released;')
        print('         fall back to a shallower tap or an unfrozen head.')
        return 1
    g = e.grad.float()
    nz = float(g.abs().sum())
    print(f'\nembedding-output grad: shape {tuple(g.shape)}  '
          f'|g|_1 {nz:.4e}  max {float(g.abs().max()):.4e}  '
          f'finite {bool(torch.isfinite(g).all())}  nonzero_rows '
          f'{int((g.abs().sum(-1) > 0).sum())}/{g.shape[-2]}')
    ok = nz > 0 and bool(torch.isfinite(g).all())
    print(f'\nVERDICT: {"PASS -- a projector at the LLM input is trainable from the BEV head loss" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
