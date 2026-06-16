"""
CPU forward-pass + loss test for QueryOccHead.

Verifies:
  1. Module instantiates and parameter count matches expectations.
  2. Inference path with explicit query_xyz returns [B, N, K] correctly.
  3. Inference path with query_xyz=None (full grid) returns [B, K, Z, H, W].
  4. Training loss with subsampling returns finite scalar + grad flow OK.
  5. CPU memory rough comparison vs hypothetical dense voxel head.

Run:
    cd /data/rnd-liu/MyRepo/mmdetection3d
    conda run -n py310 python projects/bevdet/test_query_occ.py
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F

# Bootstrap import of train module (it lives in DeepDataMiningLearning, also
# mirrored at mmdetection3d/projects/bevdet/train via the existing symlink/sync).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _PROJECTS not in sys.path:
    sys.path.insert(0, _PROJECTS)
from projects.bevdet.train.query_occ_head import (   # noqa: E402
    QueryOccHead,
    _SmallBEVUNet,
    _sin_pos_encode,
    _flat_bilinear_sample,
)


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _heading(s):
    print("\n" + "=" * 78)
    print(f" {s}")
    print("=" * 78)


def _scenario(label, **kw):
    _heading(f"SCENARIO: {label}")
    head = QueryOccHead(**kw)
    head.eval()
    n = _count_params(head)
    print(f"  num_classes        = {head.num_classes}")
    print(f"  world_channels     = {head.world_channels}")
    print(f"  grid_size  (Z,H,W) = ({head.grid_z}, {head.grid_h}, {head.grid_w})")
    print(f"  pos_freqs          = {head.pos_freqs}")
    print(f"  total params       = {n:,}  ({n/1e6:.3f} M)")
    print(f"  UNet params        = {_count_params(head.unet):,}")
    print(f"  MLP  params        = {_count_params(head.mlp):,}")
    return head


def test_full_grid_inference(head, B=2, C=256, H=180, W=180):
    _heading("Test 1 — full-grid inference (query_xyz=None)")
    fused_bev = torch.randn(B, C, H, W) * 0.1
    print(f"  Input fused_bev: {tuple(fused_bev.shape)}")
    with torch.no_grad():
        out = head(fused_bev, query_xyz=None)
    print(f"  Output:          {tuple(out.shape)}  (expected [B, K, Z, H, W])")
    expected = (B, head.num_classes, head.grid_z, head.grid_h, head.grid_w)
    assert tuple(out.shape) == expected, f"Shape mismatch: {tuple(out.shape)} vs {expected}"
    assert torch.isfinite(out).all().item()
    print(f"  Output stats: min={out.min():+.4f}  max={out.max():+.4f}  "
          f"mean={out.mean():+.4f}  std={out.std():+.4f}  finite=True")
    print("  ✓ Full-grid inference path OK")


def test_explicit_query(head, B=2, C=256, H=180, W=180, N=1234):
    _heading(f"Test 2 — explicit-query inference (query_xyz given, N={N})")
    fused_bev = torch.randn(B, C, H, W) * 0.1
    # Random queries in normalized [-1, 1]^3
    query_xyz = (torch.rand(B, N, 3) * 2 - 1)
    with torch.no_grad():
        out = head(fused_bev, query_xyz=query_xyz)
    expected = (B, N, head.num_classes)
    print(f"  Output:          {tuple(out.shape)}  (expected {expected})")
    assert tuple(out.shape) == expected
    assert torch.isfinite(out).all().item()
    print(f"  Output stats: min={out.min():+.4f}  max={out.max():+.4f}  "
          f"mean={out.mean():+.4f}  std={out.std():+.4f}")
    print("  ✓ Explicit-query inference path OK")


def test_loss_with_subsample(head, B=2, C=256, H=180, W=180, ignore_frac=0.4):
    _heading("Test 3 — loss with sub-sampling + grad flow")
    head.train()
    # IMPORTANT: create the leaf tensor first, scale, THEN set requires_grad,
    # otherwise the multiplication produces a non-leaf tensor with no .grad.
    fused_bev = (torch.randn(B, C, H, W) * 0.1).requires_grad_(True)
    Z, Hg, Wg = head.grid_z, head.grid_h, head.grid_w

    # Build a random occ_gt with both valid labels and -1 ignored cells
    occ_gt = torch.randint(0, head.num_classes, (B, Z, Hg, Wg), dtype=torch.long)
    ignore_mask = torch.rand(B, Z, Hg, Wg) < ignore_frac
    occ_gt[ignore_mask] = -1

    out = head.loss(fused_bev, occ_gt)
    loss = out["loss_occ_query"]
    n_q  = out["occ_query_count"]
    print(f"  loss_occ_query     = {loss.item():.4f}")
    print(f"  queries used       = {int(n_q.item()):,}  "
          f"(cap = {head.train_max_queries:,} per sample × B={B})")
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"

    # Verify backward works
    loss.backward()
    g_norm = sum(p.grad.norm().item() ** 2 for p in head.parameters() if p.grad is not None) ** 0.5
    g_in   = fused_bev.grad.norm().item() if fused_bev.grad is not None else float("nan")
    print(f"  ∥grad(head params)∥ = {g_norm:.4f}")
    print(f"  ∥grad(fused_bev)∥   = {g_in:.4f}")
    assert g_norm > 0.0, "no gradient flowed to head params"
    assert g_in   > 0.0, "no gradient flowed back to fused_bev"
    print("  ✓ Loss + backward OK")


def test_pos_encode():
    _heading("Test 4 — sinusoidal positional encoding")
    xyz = torch.randn(5, 7, 3)
    for F_freqs in [0, 1, 4, 6]:
        pe = _sin_pos_encode(xyz, F_freqs)
        if F_freqs == 0:
            expected_d = 3
        else:
            expected_d = 3 * 2 * F_freqs
        print(f"  F={F_freqs:>2}  -> shape {tuple(pe.shape)}  (expected last-dim {expected_d})")
        assert pe.shape[-1] == expected_d
        if F_freqs > 0:
            assert torch.isfinite(pe).all().item()
    print("  ✓ pos_encode OK")


def test_flat_bilinear():
    _heading("Test 5 — flat batched bilinear sampling")
    B, C, H, W = 3, 4, 8, 8
    x = torch.arange(B * C * H * W, dtype=torch.float).reshape(B, C, H, W)
    # Sample at center → should equal feature value at (H/2, W/2) approximately
    n = 7
    xy = torch.zeros(n, 2)
    bidx = torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.long)
    out = _flat_bilinear_sample(x, xy, bidx)
    print(f"  feature_map: {tuple(x.shape)}, queries: {tuple(xy.shape)}, batch_idx: {bidx.tolist()}")
    print(f"  output: {tuple(out.shape)}  (expected ({n}, {C}))")
    assert tuple(out.shape) == (n, C)
    assert torch.isfinite(out).all().item()
    print("  ✓ flat_bilinear_sample OK")


def memory_estimate(head, B=16):
    _heading("Memory estimate vs hypothetical dense voxel head (B=16, BF16)")
    Z, H, W = head.grid_z, head.grid_h, head.grid_w
    K = head.num_classes
    Cw = head.world_channels
    train_q = head.train_max_queries

    # Dense voxel head (similar to current BEVOccHead)
    feat_vol = B * 128 * Z * H * W * 2          # 3D conv feature, BF16
    logits_vol = B * K * Z * H * W * 4           # CE logits, fp32
    dense_total = feat_vol + logits_vol
    print(f"  Hypothetical DENSE voxel head:")
    print(f"    3D conv feature volume @ hidden=128: {feat_vol / 1024**3:>6.2f} GB")
    print(f"    Dense logits [B,K,Z,H,W] (fp32 CE)  : {logits_vol / 1024**3:>6.2f} GB")
    print(f"    TOTAL (rough)                      : {dense_total / 1024**3:>6.2f} GB")

    # Q-Occ head with subsampling
    world_t = B * Cw * H * W * 2                 # world tensor, BF16
    q_total = B * train_q
    feat_t  = q_total * (Cw + 3 * 2 * head.pos_freqs) * 2
    mlp_t   = q_total * 128 * 2 * 2              # rough: 2 hidden layers active
    logits_t= q_total * K * 4
    qocc_total = world_t + feat_t + mlp_t + logits_t
    print(f"  Q-Occ head (subsampled, max {train_q:,} / sample):")
    print(f"    World tensor [B,Cw,H,W]           : {world_t  / 1024**2:>6.1f} MB")
    print(f"    Sampled feat [N,Cw+pe]            : {feat_t   / 1024**2:>6.1f} MB")
    print(f"    MLP intermediate (rough)          : {mlp_t    / 1024**2:>6.1f} MB")
    print(f"    Logits [N,K] (fp32 CE)            : {logits_t / 1024**2:>6.1f} MB")
    print(f"    TOTAL (rough)                     : {qocc_total / 1024**2:>6.1f} MB")
    print(f"  Reduction vs dense: {dense_total / qocc_total:.1f}×")


def main():
    print("\n>>> P2.11 QueryOccHead — CPU forward + loss test <<<")

    # Test the small utility functions first
    test_pos_encode()
    test_flat_bilinear()

    # ---- Scenario A: binary (K=2), our current pseudo-occ setup ----
    head_bin = _scenario(
        "Binary pseudo-occ (K=2)",
        in_channels=256, world_channels=64, mlp_hidden=128,
        num_classes=2, pos_freqs=6,
        grid_size=(16, 180, 180),
        train_max_queries=50_000,
    )
    test_full_grid_inference(head_bin)
    test_explicit_query(head_bin)
    test_loss_with_subsample(head_bin)
    memory_estimate(head_bin)

    # ---- Scenario B: Occ3D-nuScenes 17 classes ----
    head_o3d = _scenario(
        "Occ3D-nuScenes (K=17)",
        in_channels=256, world_channels=64, mlp_hidden=128,
        num_classes=17, pos_freqs=6,
        grid_size=(16, 180, 180),
        train_max_queries=50_000,
    )
    test_full_grid_inference(head_o3d)
    test_loss_with_subsample(head_o3d)
    memory_estimate(head_o3d)

    print("\nALL TESTS PASSED ✓\n")


if __name__ == "__main__":
    main()
