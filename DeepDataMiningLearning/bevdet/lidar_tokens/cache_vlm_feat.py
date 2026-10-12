"""Cache a frozen driving-VLM's image tap as `<token>.npy`, for use as an LSS backbone.

WHY, AND WHY THIS IS NOT A RERUN OF A KNOWN NULL
------------------------------------------------
The LiDAR-into-the-token-stream route is closed (`QWEN_DRIVE_LIDAR_TOKENS.md`): the entire
gain there is an information-free constant prefix. What is *not* closed is the other
topology -- **fuse LiDAR in the 3D head, where fusion demonstrably works, and let the VLM
supply the camera prior.** In this project's own numbers LiDAR at the BEV level is one of
the largest effects measured (occupancy mIoU 0.302 camera-only -> **0.558** fused, geo-IoU
0.669 -> 0.838), so the question is not whether LiDAR helps but what the *camera* branch
should be.

Two prior results look like they close this, and neither does:

* **§2.1's frozen-backbone ranking was killed by a seed null (σ ≈ 0.014 occ mIoU).** But that
  was measured on *overall* mIoU, which is dominated by observed space -- where direct
  observation already saturates and the backbone cannot matter much. A null there is fully
  compatible with a real effect in the **occluded subset**, a different population that was
  never stratified.
* **§4.6's occluded-region plateau (pred_mIoU 0.070) was attributed to "a property of the
  available information".** That conclusion came from varying the **loss** -- two weightings
  (occluded region at 84 % vs 50 % of the loss) landing on 0.069 and 0.070. Neither arm
  varied the **feature prior**. A claim about what information is available was never tested
  by changing what the features encode.

So the open question is sharp: *is the occluded-space ceiling a property of single-frame
information, or a property of vision-only features?* A language-grounded backbone is the one
intervention that separates them, and it has never been run.

PRE-REGISTERED PREDICTION (written before any of this is trained)
-----------------------------------------------------------------
`pred_mIoU` (semantic) will **not** move materially. Occ3D's occluded labels are
observations from *other timestamps*; no single-frame prior, however semantic, knows whether
*this* car is behind *that* truck right now.

`pred_geo` (class-agnostic occupancy) **may** move, because a language-grounded prior
encodes typical scene structure -- road continues under the truck, buildings are solid --
and that is exactly what produced §4.6's one unpredicted result, the 2.5× jump in pred_geo
(0.028 -> 0.070) when the occluded region entered the loss at all.

If neither moves beyond the seed null, the information-ceiling conclusion is confirmed on a
second, independent axis and §4.6's semantic form is closed for good. That is a useful
outcome, not a wasted one -- but it is a *negative*, and it should be called one.

⚠ **σ on `pred_geo`/`pred_mIoU` has never been measured** (§4.6 ran one seed). Rule 7.1: get
≥2 seeds of the DINOv2-L baseline arm before interpreting any number from a new backbone.

GEOMETRY -- WHY THIS DROPS IN
-----------------------------
Qwen-Drive resizes every camera to 896×512 and its merged image-token grid is 28×16, so
`image_hw=(512, 896)` with `downsample=32` gives exactly the cached grid (16, 28). The LSS
dataset already rescales intrinsics to `image_hw` (it does this for DINOv2's 252×700), so
the lift geometry is consistent by construction rather than by a correction factor.

The tap is the same one the released BEV head reads -- decoder output at image-token
positions, after `language_model.norm` -- so these features are literally what Qwen-Drive
itself uses for 3D. Shape per frame: `(n_cam, 2560, 16, 28)`, fp16, ~13.8 MB.

USAGE
-----
    python cache_vlm_feat.py --frames <packed dir> --out <cache dir>
    # then, in ngperception/occupancy:
    python train_lss.py --backbone qwendrive --vlm-feat-cache <cache dir> --lidar-fusion ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

QD = Path('/data/rnd-liu/Others/Qwen-Drive-1.0')
sys.path.insert(0, str(QD / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_drive_grad import set_vision_patch_embed  # noqa: E402

#: The occupancy trainer's camera order (`ngperception/occupancy/geom.py:CAMS`).
#: Qwen-Drive's own order is CLOCKWISE -- [FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT,
#: FRONT_LEFT] -- so positions 3 and 6 are SWAPPED relative to this one. Caching in the VLM's
#: order would lift BACK_RIGHT features through FRONT_LEFT's extrinsics: no crash, no error,
#: just garbage that reads as "the VLM tap is a bad backbone". The permutation is derived from
#: each frame's own `cam_order` rather than hardcoded, and asserted.
OCC_CAM_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vlm', type=Path, default=Path('/data/rnd-liu/Others/Qwen-Drive-1.0-4B'))
    ap.add_argument('--frames', type=Path, required=True, help='pack_nuscenes.py output dir')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation='sdpa')
    vlm = holder.vlm.to(args.device).eval()
    del holder.planning_expert
    for p in vlm.parameters():
        p.requires_grad_(False)
    # 99 % of the released forward is one Conv3d that is a matmul in disguise; the swap is
    # bit-identical and ~91x faster end to end. Without it this cache is a week, not an hour.
    rep = set_vision_patch_embed(vlm, 'matmul', verify=True)
    print(f"patch embed -> matmul ({rep['speedup']:.0f}x, max abs diff "
          f"{rep['max_abs_diff']:.1e})")
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    image_token_id = vlm.config.image_token_id

    dirs = sorted(p for p in args.frames.iterdir()
                  if p.is_dir() and (p / 'frame.json').exists())
    if args.limit:
        dirs = dirs[:args.limit]
    print(f'{len(dirs)} frames -> {args.out}')

    np_dtype = np.float16 if args.dtype == 'float16' else np.float32
    done = 0
    for i, d in enumerate(dirs):
        dst = args.out / f'{d.name}.npy'
        if dst.exists():
            done += 1
            continue
        frame = PerceptionFrame(d)
        if sorted(frame.cam_order) != sorted(OCC_CAM_ORDER):
            raise SystemExit(f'{d.name}: cameras {frame.cam_order} != the occupancy set')
        perm = [frame.cam_order.index(c) for c in OCC_CAM_ORDER]
        inputs, img_metas = processor(frame, device=args.device)
        with torch.no_grad():
            out = vlm.model(input_ids=inputs['input_ids'],
                            pixel_values=inputs['pixel_values'],
                            image_grid_thw=inputs['image_grid_thw'],
                            mm_token_type_ids=(inputs['input_ids'] == image_token_id).long(),
                            use_cache=False)
            hs = vlm.model.language_model.norm(out.last_hidden_state)
            n_cam = len(img_metas['cam_order'])
            gh = inputs['image_grid_thw'][-1, 1].item()
            gw = inputs['image_grid_thw'][-1, 2].item()
            tpi = gh // 2 * gw // 2
            mask = inputs['input_ids'][0] == image_token_id
            tok = hs[0][mask][-n_cam * tpi:].view(n_cam, gh // 2, gw // 2, -1)
            # (N, h, w, C) -> (N, C, h, w), the layout `encoder.forward_feat` expects,
            # and reorder the cameras into the occupancy trainer's order (see OCC_CAM_ORDER).
            feat = tok.permute(0, 3, 1, 2)[perm].float().cpu().numpy().astype(np_dtype)
        np.save(dst, feat)
        done += 1
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(dirs)}  shape {feat.shape}  '
                  f'{feat.nbytes / 2**20:.1f} MiB/frame', flush=True)

    meta = dict(date=datetime.now().isoformat(timespec='seconds'), args={
        k: str(v) for k, v in vars(args).items()}, n_frames=done,
        tap='language_model.norm(last_hidden_state) at image-token positions '
            '(the same tensor the released BEV head reads)',
        layout='(n_cam, C, h, w)', dtype=args.dtype,
        cam_order=OCC_CAM_ORDER,
        cam_order_note='reordered from Qwen-Drive clockwise order into '
                       'ngperception/occupancy/geom.py:CAMS -- positions 3 and 6 differ',
        note='image_hw=(512,896) downsample=32 -> grid (16,28); train_lss rescales '
             'intrinsics to image_hw, so the lift geometry matches by construction.')
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\ncached {done} frames -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
