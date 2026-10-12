"""STEP 1 — native in-domain sanity for GaussianOcc.

Builds the repo's own Runer (loads encoder/depth/pose weights from ckpts/nusc-sem-gs),
runs the FIRST --n nuScenes val frames through the exact inference path used by
`teacherkit/teachers/gaussianocc_dump.py` (encoder -> depth decoder), and scores them
with the repo's own occ_metrics.Metric_mIoU against Occ3D GT. Proves weights + CUDA
rasteriser + geometry produce a non-degenerate semantic occ (~11 mIoU reproduction)
BEFORE we touch AV2. Must run under py311 + torchrun with the _stubs PYTHONPATH:

  cd /data/rnd-liu/Others/GaussianOcc
  CUDA_HOME=/home/010796032/miniconda3/envs/py311 \\
  PYTHONPATH=/data/rnd-liu/Others/GaussianOcc/_stubs:/data/rnd-liu/Others/GaussianOcc \\
  /home/010796032/miniconda3/envs/py311/bin/torchrun --nproc_per_node=1 --master_port=29533 \\
      <this_file> --config configs/nusc-sem-gs.txt \\
      --load_weights_folder ckpts/nusc-sem-gs --eval_only --n 10
"""
import os, sys, argparse
import numpy as np

GAUSSIANOCC = '/data/rnd-liu/Others/GaussianOcc'


def main():
    os.chdir(GAUSSIANOCC)
    for p in (GAUSSIANOCC, os.path.join(GAUSSIANOCC, '_stubs')):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from options import MonodepthOptions
    from runner import Runer
    from utils import occ_metrics

    n = int(os.environ.get('GSOCC_N', '10'))

    opts = MonodepthOptions().parse()
    runner = Runer(opts)
    runner.set_eval()

    from torch.utils.data import DataLoader
    ds = runner.val_loader.dataset
    stride = int(os.environ.get('GSOCC_STRIDE', '1'))
    ds.filenames = ds.filenames[::stride][:n]
    loader = DataLoader(ds, batch_size=runner.val_loader.batch_size,
                        collate_fn=runner.my_collate, num_workers=0,
                        pin_memory=False, drop_last=False, shuffle=False)

    metric = occ_metrics.Metric_mIoU(num_classes=18, use_lidar_mask=False, use_image_mask=True)
    occ_frac_list, fg_frac_list = [], []
    OBJ = [2, 3, 4, 5, 6, 7, 9, 10]

    with torch.no_grad():
        for k, data in enumerate(loader):
            colour = data[("color", 0, 0)][:opts.cam_N].cuda()
            feats = runner.models["encoder"](colour)
            output = runner.models["depth"](feats, data, epoch=0, is_train=False)
            pred = output['pred_occ_logits'][0].argmax(0).detach().cpu().numpy()  # (200,200,16)
            gt = data['semantics_3d'].detach().cpu().numpy()
            mcam = data['mask_camera_3d'].detach().cpu().numpy().astype(bool)
            metric.add_batch(semantics_pred=pred, semantics_gt=gt, mask_camera=mcam, mask_lidar=None)
            occ_frac_list.append(float((pred != 17).mean()))
            fg_frac_list.append(float(np.isin(pred, OBJ).mean()))
            uniq, cnts = np.unique(pred, return_counts=True)
            top = sorted(zip(cnts, uniq), reverse=True)[:5]
            print(f"[{k}] occ_frac {occ_frac_list[-1]:.3f}  fg_frac {fg_frac_list[-1]:.4f}  "
                  f"top classes {[(int(c), int(n_)) for n_, c in top]}", flush=True)

    names, mIoU, cnt = metric.count_miou()
    idx_no_empty = [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    print(f"\n=== NATIVE IN-DOMAIN ({cnt} frames) ===")
    print(f"mIoU (all 17):        {round(float(np.nanmean(mIoU[:17])) * 100, 2)}")
    print(f"mIoU (without empty): {round(float(np.nanmean(mIoU[idx_no_empty])) * 100, 2)}")
    print(f"mean occ_frac {np.mean(occ_frac_list):.3f}  mean fg_frac {np.mean(fg_frac_list):.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
