"""In-domain reference for GaussianOcc: run it via its NATIVE NuscDataset (correct geometry) on
nuScenes Occ3D val, computing the SAME FG_REC/obj_IoU/occ_frac metrics used on AV2 (object GT =
Occ3D semantics in OBJ_CLASSES, restricted to the camera mask). This gives the retention denominator,
so any AV2 change can be attributed to the domain gap rather than the adapter. Mirrors
opus_indomain_ref.py exactly. Writes gaussianocc_INDOMAIN.json with the AV2 schema.

Run under py311 + torchrun with the _stubs PYTHONPATH (same invocation as the smoke/AV2 scripts):
  cd /data/rnd-liu/Others/GaussianOcc
  CUDA_HOME=/home/010796032/miniconda3/envs/py311 GSOCC_N=40 GSOCC_STRIDE=150 \
  GSOCC_OUT=/tmp/shield/xdomain_results \
  PYTHONPATH=/data/rnd-liu/Others/GaussianOcc/_stubs:/data/rnd-liu/Others/GaussianOcc \
  /home/010796032/miniconda3/envs/py311/bin/torchrun --nproc_per_node=1 --master_port=29539 \
      <this_file> --config configs/nusc-sem-gs.txt --load_weights_folder ckpts/nusc-sem-gs --eval_only
"""
import os, sys
import numpy as np

GAUSSIANOCC = '/data/rnd-liu/Others/GaussianOcc'
XDOMAIN = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(GAUSSIANOCC)
    for p in (GAUSSIANOCC, os.path.join(GAUSSIANOCC, "_stubs")):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.path.insert(0, XDOMAIN)

    import torch
    from options import MonodepthOptions
    from runner import Runer
    import av2_common as C
    import metrics as MET

    n = int(os.environ.get("GSOCC_N", "40"))
    stride = int(os.environ.get("GSOCC_STRIDE", "150"))
    out_dir = os.environ.get("GSOCC_OUT", "/tmp/shield/xdomain_results")
    os.makedirs(out_dir, exist_ok=True)

    opts = MonodepthOptions().parse()
    runner = Runer(opts)
    runner.set_eval()

    from torch.utils.data import DataLoader
    ds = runner.val_loader.dataset
    ds.filenames = ds.filenames[::stride][:n]
    loader = DataLoader(ds, batch_size=runner.val_loader.batch_size,
                        collate_fn=runner.my_collate, num_workers=0,
                        pin_memory=False, drop_last=False, shuffle=False)

    nusc = ds.nusc
    pf = []
    with torch.no_grad():
        for data in loader:
            tok = data["token"][0]
            colour = data[("color", 0, 0)][:opts.cam_N].cuda()
            feats = runner.models["encoder"](colour)
            output = runner.models["depth"](feats, data, epoch=0, is_train=False)
            pred = output["pred_occ_logits"][0].argmax(0).detach().cpu().numpy()  # (200,200,16)

            # object GT = Occ3D semantics in OBJ_CLASSES, within the camera mask (mirrors OPUS ref)
            scene = nusc.get("scene", nusc.get("sample", tok)["scene_token"])["name"]
            g = np.load(os.path.join(ds.gts_path, scene, tok, "labels.npz"))
            sem = g["semantics"]; mcam = g["mask_camera"].astype(bool)
            obj_gt = np.isin(sem, MET.OBJ_CLASSES)

            r_occ, _ = MET.object_recall_precision(pred, obj_gt, mcam)
            fg_rec, fg_frac = MET.object_class_recall(pred, obj_gt, mcam)
            pf.append(dict(
                occ_frac=float(((pred != 17) & mcam).sum() / mcam.sum()),
                fg_frac=fg_frac, occ_recall=r_occ, fg_recall=fg_rec,
                obj_iou=MET.object_occ_iou(pred, obj_gt, mcam), geo_iou=0.0))

    m = C.save_results(os.path.join(out_dir, "gaussianocc_INDOMAIN.json"),
                       "GaussianOcc (self-sup, cam)", pf)
    print(f"GaussianOcc IN-DOMAIN (nuScenes, {len(pf)} frames): "
          f"FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  obj_IoU {m['obj_iou']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
