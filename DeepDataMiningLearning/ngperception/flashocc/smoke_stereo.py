"""
Smoke test for the standalone FlashOcc BEVStereo4DOCC port.

Builds FlashOccBEVStereo4D, loads the official checkpoint with strict=True (prints
missing/unexpected key counts -> must be 0/0), then runs a forward on dummy but
correctly-shaped inputs (B=1, num_frame=3 -> N=18 cam-views) and prints the output
shape (must be (1,18,200,200,16)) and finiteness.

Run:
  cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning && \
  PYTHONPATH=/fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning \
  CUDA_HOME=/data/rnd-liu/cuda_home2 PATH=/data/rnd-liu/cuda_home2/bin:$PATH \
  LD_LIBRARY_PATH=/data/rnd-liu/cuda_home2/lib64:$LD_LIBRARY_PATH \
  TORCH_CUDA_ARCH_LIST=9.0 conda run -n py310 python -m \
  DeepDataMiningLearning.ngperception.flashocc.smoke_stereo
"""
import torch

from DeepDataMiningLearning.ngperception.flashocc.model_stereo import (
    FlashOccBEVStereo4D, _CKPT_PATH)


def make_dummy_inputs(B=1, num_frame=3, num_cams=6, H=256, W=704, device='cuda'):
    N = num_cams * num_frame  # 18
    imgs = torch.randn(B, N, 3, H, W, device=device)

    # identity extrinsics (cam == ego == global)
    eye4 = torch.eye(4, device=device)
    sensor2egos = eye4.view(1, 1, 4, 4).repeat(B, N, 1, 1).clone()
    ego2globals = eye4.view(1, 1, 4, 4).repeat(B, N, 1, 1).clone()

    # a sane pinhole intrinsic at 256x704
    K = torch.tensor([[500.0, 0.0, W / 2.0],
                      [0.0, 500.0, H / 2.0],
                      [0.0, 0.0, 1.0]], device=device)
    intrins = K.view(1, 1, 3, 3).repeat(B, N, 1, 1).clone()

    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, N, 1, 1).clone()
    post_trans = torch.zeros(B, N, 3, device=device)
    bda = torch.eye(3, device=device).view(1, 3, 3).repeat(B, 1, 1).clone()

    return [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]


def main():
    assert torch.cuda.is_available(), "CUDA required (bev_pool_v2 is a CUDA op)."
    device = 'cuda'

    print(f"[build] FlashOccBEVStereo4D ...")
    model = FlashOccBEVStereo4D(pretrained_img=False)

    print(f"[load]  {_CKPT_PATH}")
    sd = torch.load(_CKPT_PATH, map_location='cpu', weights_only=False)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[strict-load] missing = {len(missing)}   unexpected = {len(unexpected)}")
    if missing:
        print("  missing keys:")
        for k in missing:
            print("    ", k)
    if unexpected:
        print("  unexpected keys:")
        for k in unexpected:
            print("    ", k)
    assert len(missing) == 0 and len(unexpected) == 0, \
        "strict-load FAILED: architecture does not match checkpoint 1:1"
    # Prove strict=True also passes.
    model.load_state_dict(sd, strict=True)
    print("[strict-load] strict=True load OK (0 missing / 0 unexpected)")

    model = model.to(device).eval()

    inputs = make_dummy_inputs(device=device)
    print("[forward] img_inputs shapes:")
    names = ['imgs', 'sensor2egos', 'ego2globals', 'intrins',
             'post_rots', 'post_trans', 'bda']
    for n, t in zip(names, inputs):
        print(f"    {n:12s} {tuple(t.shape)}")

    with torch.no_grad():
        out = model(inputs)
    print(f"[forward] output shape = {tuple(out.shape)}   dtype = {out.dtype}")
    print(f"[forward] finite = {bool(torch.isfinite(out).all())}   "
          f"min={out.min().item():.3f}  max={out.max().item():.3f}")

    assert tuple(out.shape) == (1, 18, 200, 200, 16), \
        f"unexpected output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "output has non-finite values"
    print("\nSMOKE PASS: strict-load 0/0, forward -> (1,18,200,200,16) finite.")


if __name__ == '__main__':
    main()
