"""
flashocc.smoke_model
====================
Build FlashOccBEVDet on CUDA, feed random imgs + identity-ish calib, and verify the
output occ-logit shape (must be 200x200x16x18 in some order) and that values are finite.

Run:
  cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning && \
  PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning \
  CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=/data/rnd-liu/cuda_home2/bin:$PATH \
  LD_LIBRARY_PATH=/data/rnd-liu/cuda_home2/lib64:$LD_LIBRARY_PATH \
  TORCH_CUDA_ARCH_LIST=9.0 conda run -n py310 \
  python -m DeepDataMiningLearning.ngperception.flashocc.smoke_model
"""
import torch

from DeepDataMiningLearning.ngperception.flashocc.model import FlashOccBEVDet


def main():
    dev = "cuda"
    B, N = 2, 6
    torch.manual_seed(0)

    model = FlashOccBEVDet().to(dev)
    model.train()

    imgs = torch.randn(B, N, 3, 256, 704, device=dev)

    # identity-ish calibration: cams evenly rotated around ego-z, ~1.5m up, focal ~500.
    rots = torch.eye(3, device=dev).view(1, 1, 3, 3).repeat(B, N, 1, 1).clone()
    for i in range(N):
        a = torch.tensor(i * 2 * 3.14159265 / N, device=dev)
        c, s = torch.cos(a), torch.sin(a)
        rots[:, i] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=dev)
    trans = torch.zeros(B, N, 3, device=dev)
    trans[..., 2] = 1.5
    K = torch.tensor([[500.0, 0.0, 352.0],
                      [0.0, 500.0, 128.0],
                      [0.0, 0.0, 1.0]], device=dev)
    intrins = K.view(1, 1, 3, 3).repeat(B, N, 1, 1).clone()

    occ = model(imgs, rots, trans, intrins)

    print("output shape:", tuple(occ.shape))
    print("output dtype:", occ.dtype)
    print("finite:", bool(torch.isfinite(occ).all().item()))
    print("min/max/mean: %.4f / %.4f / %.4f" %
          (occ.min().item(), occ.max().item(), occ.float().mean().item()))

    assert tuple(occ.shape) == (B, 18, 200, 200, 16), \
        f"expected (B,18,200,200,16), got {tuple(occ.shape)}"
    assert torch.isfinite(occ).all(), "non-finite values in output!"

    # quick backward sanity (bev_pool_v2 grad path).
    loss = occ.float().pow(2).mean()
    loss.backward()
    g = model.img_view_transformer.depth_net.weight.grad
    print("backward ok; depth_net grad finite:", bool(torch.isfinite(g).all().item()))
    print("SMOKE TEST PASSED: layout (B, num_classes=18, Dx=200, Dy=200, Dz=16)")


if __name__ == "__main__":
    main()
