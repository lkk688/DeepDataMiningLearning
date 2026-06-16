"""
CPU forward-pass test for the GQA-augmented CrossAttnLSSTransform.

Verifies:
  1. Single-head path (num_heads=1) — backward-compatible, no regression.
  2. Multi-head MHA (num_heads=8, num_kv_groups=1).
  3. GQA (num_heads=8, num_kv_groups=4) — H_kv = 2.
  4. MQA (num_heads=8, num_kv_groups=8) — all heads share one K/V head.

For each path, prints:
  • Param counts
  • Output shape and dtype
  • Output stats (min / max / mean / nan check)
  • Approximate per-chunk K/V activation memory savings vs MHA

Run:
    cd /data/rnd-liu/MyRepo/mmdetection3d
    conda run -n py310 python projects/bevdet/test_gqa_calss.py
"""
from __future__ import annotations

import os
import sys
import torch

# Bootstrap mmengine + mmdet3d registry
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _PROJECTS not in sys.path:
    sys.path.insert(0, _PROJECTS)
import projects.bevdet.cross_attn_lss2 as ca_mod  # noqa: F401
from projects.bevdet.cross_attn_lss2 import CrossAttnLSSTransform


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _make_dummy_inputs(B=2, Nc=6, C=128, Hf=32, Wf=88, device="cpu", seed=42):
    """Build dummy FPN feature + projection matrices for one CPU forward pass."""
    torch.manual_seed(seed)
    feats = torch.randn(B, Nc, C, Hf, Wf, device=device) * 0.1

    # Realistic-ish lidar2img: project a unit cube to [0, W_img] x [0, H_img].
    # We just synthesize an invertible 4x4 with a sensible scale for testing.
    lidar2img = torch.zeros(B, Nc, 4, 4, device=device)
    for b in range(B):
        for c in range(Nc):
            # Rotate around Z by 60° per camera, scale up so points project inside frame
            theta = torch.tensor(60.0 * c * 3.14159 / 180.0)
            cos, sin = torch.cos(theta), torch.sin(theta)
            R = torch.tensor([
                [ cos, -sin, 0],
                [ sin,  cos, 0],
                [   0,    0, 1],
            ])
            # focal length ~700 px, principal point = image center
            f = 700.0
            cx, cy = 352.0, 128.0
            T = torch.eye(4)
            T[:3, :3] = torch.tensor([
                [f * cos, -f * sin, cx],
                [f * sin,  f * cos, cy],
                [   0   ,     0  ,  1],
            ])[:3, :3]
            T[2, 3] = 1.0  # depth offset to keep z>0
            lidar2img[b, c] = T

    # Identity image-aug
    img_aug = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, Nc, -1, -1).contiguous()
    return feats, lidar2img, img_aug


def _build_layer(num_heads: int, num_kv_groups: int, in_channels: int = 128,
                 out_channels: int = 128) -> CrossAttnLSSTransform:
    return CrossAttnLSSTransform(
        in_channels=in_channels,
        out_channels=out_channels,
        image_size=(256, 704),
        feature_size=(32, 88),
        xbound=(-54.0, 54.0, 0.3),
        ybound=(-54.0, 54.0, 0.3),
        zbound=(-5.0, 5.0, 10.0),
        dbound=(1.0, 60.0, 0.5),
        downsample=2,
        num_z=2,
        use_cam_embed=True,
        attn_chunk=4096,
        num_heads=num_heads,
        num_kv_groups=num_kv_groups,
    )


def _kv_memory_bytes(num_heads: int, num_kv_groups: int,
                     B=16, q=16384, NcK=36, C=128, dtype_bytes=2) -> int:
    """Approximate per-chunk K+V activation memory (bytes), assuming bf16."""
    if num_heads == 1:
        # Original: K=V=raw `sampled` tensor; one tensor of shape [B,q,Nc,K,C]
        return B * q * NcK * C * dtype_bytes
    H_kv = num_heads // num_kv_groups
    d_h = C // num_heads
    # K and V are SEPARATE projected tensors; each shape [B,q,NcK,H_kv,d_h]
    return 2 * B * q * NcK * H_kv * d_h * dtype_bytes


def _scenario(label: str, num_heads: int, num_kv_groups: int):
    print("=" * 78)
    print(f"SCENARIO: {label}   (num_heads={num_heads}, num_kv_groups={num_kv_groups})")
    print("=" * 78)
    layer = _build_layer(num_heads=num_heads, num_kv_groups=num_kv_groups)
    layer.eval()
    n_params = _count_params(layer)
    feats, lidar2img, img_aug = _make_dummy_inputs()
    B, Nc, C, Hf, Wf = feats.shape

    print(f"  Layer:")
    print(f"    in_channels        = {layer.in_channels}")
    print(f"    out_channels       = {layer.out_channels}")
    print(f"    use_multihead      = {layer.use_multihead}")
    if layer.use_multihead:
        print(f"    num_heads (H_q)    = {layer.num_heads}")
        print(f"    num_kv_groups (G)  = {layer.num_kv_groups}")
        print(f"    num_kv_heads (H_kv)= {layer.num_kv_heads}")
        print(f"    head_dim (d_h)     = {layer.head_dim}")
        print(f"    Wq:  {tuple(layer.wq.weight.shape)}")
        print(f"    Wk:  {tuple(layer.wk.weight.shape)}")
        print(f"    Wv:  {tuple(layer.wv.weight.shape)}")
        print(f"    Wo:  {tuple(layer.wo.weight.shape)}")
    print(f"    total params       = {n_params:,}")

    # Estimate per-chunk K/V memory for a "real" workload (bs=16, q=16384, NcK=36, C=128)
    kv_bytes = _kv_memory_bytes(num_heads, num_kv_groups)
    print(f"  Estimated K/V activation per chunk @ bs=16, q=16k, BF16:")
    print(f"    {kv_bytes / (1024**2):.1f} MB")

    # Forward pass
    with torch.no_grad():
        out = layer(feats, lidar2img=lidar2img, img_aug_matrix=img_aug)
    print(f"  Output:")
    print(f"    shape  = {tuple(out.shape)}")
    print(f"    dtype  = {out.dtype}")
    print(f"    finite = {torch.isfinite(out).all().item()}")
    print(f"    min    = {out.min().item():+.6f}")
    print(f"    max    = {out.max().item():+.6f}")
    print(f"    mean   = {out.mean().item():+.6f}")
    print(f"    std    = {out.std().item():+.6f}")

    # Expected output shape: [B, out_channels, Hy, Hx]
    Hy = int((54 - (-54)) / (0.3 * 2))   # 180
    Hx = Hy
    expected = (B, layer.out_channels, Hy, Hx)
    assert tuple(out.shape) == expected, f"Expected {expected}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all().item(), "Output contains NaN/Inf"
    print(f"  ✓ shape ok, output finite")
    print()
    return out, n_params, kv_bytes


def main():
    print("\n>>> P2.3 Multi-Head + GQA Cross-Attention test <<<\n")

    # 1) single-head: original baseline
    out_sh, p_sh, kv_sh = _scenario("Single-head (original baseline)",
                                    num_heads=1, num_kv_groups=1)
    # 2) MHA: 8 query heads, 8 K/V heads (no GQA)
    out_mha, p_mha, kv_mha = _scenario("MHA (8 heads, no GQA)",
                                       num_heads=8, num_kv_groups=1)
    # 3) GQA: 8 query heads, 2 K/V heads (G=4)
    out_gqa, p_gqa, kv_gqa = _scenario("GQA (8 query heads, 2 K/V heads, G=4)",
                                       num_heads=8, num_kv_groups=4)
    # 4) MQA: 8 query heads, 1 K/V head (G=8)
    out_mqa, p_mqa, kv_mqa = _scenario("MQA (8 query heads, 1 K/V head, G=8)",
                                       num_heads=8, num_kv_groups=8)

    # Sanity: outputs should differ across configurations (they have different params/structure)
    print("=" * 78)
    print("CROSS-CONFIGURATION SANITY CHECKS")
    print("=" * 78)
    diffs = {
        "MHA  vs single-head": (out_mha  - out_sh ).abs().mean().item(),
        "GQA  vs single-head": (out_gqa  - out_sh ).abs().mean().item(),
        "MQA  vs single-head": (out_mqa  - out_sh ).abs().mean().item(),
        "GQA  vs MHA        ": (out_gqa  - out_mha).abs().mean().item(),
        "MQA  vs GQA        ": (out_mqa  - out_gqa).abs().mean().item(),
    }
    for label, d in diffs.items():
        assert d > 1e-7, f"{label}: outputs are suspiciously identical (Δ={d})"
        print(f"  {label}: mean|Δ| = {d:.4e}")
    print()

    # Param-count summary
    print("=" * 78)
    print("PARAM COUNT SUMMARY")
    print("=" * 78)
    print(f"  Single-head        : {p_sh:>8,} params  (baseline)")
    print(f"  MHA  (H=8, G=1)    : {p_mha:>8,} params  (+{p_mha - p_sh:,})")
    print(f"  GQA  (H=8, G=4)    : {p_gqa:>8,} params  (+{p_gqa - p_sh:,})")
    print(f"  MQA  (H=8, G=8)    : {p_mqa:>8,} params  (+{p_mqa - p_sh:,})")
    print()

    # K/V memory comparison @ realistic workload (bs=16, q=16k, BF16)
    print("=" * 78)
    print("ESTIMATED PER-CHUNK K/V ACTIVATION MEMORY  (bs=16, q=16k, NcK=36, C=128, BF16)")
    print("=" * 78)
    print(f"  Single-head (raw)  : {kv_sh  / 1024**2:>7.1f} MB  (1 tensor, K=V=sampled)")
    print(f"  MHA  (H=8, G=1)    : {kv_mha / 1024**2:>7.1f} MB  ({kv_mha / kv_sh:.2f}× single-head)")
    print(f"  GQA  (H=8, G=4)    : {kv_gqa / 1024**2:>7.1f} MB  ({kv_gqa / kv_sh:.2f}× single-head, "
          f"{kv_gqa / kv_mha:.2f}× MHA)")
    print(f"  MQA  (H=8, G=8)    : {kv_mqa / 1024**2:>7.1f} MB  ({kv_mqa / kv_sh:.2f}× single-head, "
          f"{kv_mqa / kv_mha:.2f}× MHA)")
    print()

    print("ALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
