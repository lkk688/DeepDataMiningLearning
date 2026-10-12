"""Cross-domain eval: run GaussianOcc (nuScenes-trained, self-supervised, camera-only, single-frame)
on AV2 surround renders. The adapter REUSES GaussianOcc's own dataset preprocessing: it subclasses
the repo's NuscDataset and overrides only the raw-frame loader (get_info) to serve AV2 frame-0 data
(6 cameras = av2_common.AV2_6, each with its own K_ori + pose_spatial cam->ego). Everything geometry-
relevant downstream — the anisotropic resize to 640x384, the K scaling to the encoder frame
(('K',0,0)) and render frame (('K_render',0,0)), all_cam_center — is formed by the repo's own
MonoDataset.__getitem__, so nothing is hand-rolled. The occupancy output is produced by the exact
inference path in teacherkit/teachers/gaussianocc_dump.py:
    feats = encoder(color);  output = depth(feats, data, is_train=False)
    pred  = output['pred_occ_logits'][0].argmax(0)   # (200,200,16), Occ3D-nuScenes, free=17

Must run under py311 + torchrun with the _stubs PYTHONPATH:
  cd /data/rnd-liu/Others/GaussianOcc
  CUDA_HOME=/home/010796032/miniconda3/envs/py311 \
  GSOCC_SCENES=/tmp/shield/xdomain_scenes GSOCC_OUT=/tmp/shield/xdomain_results \
  PYTHONPATH=/data/rnd-liu/Others/GaussianOcc/_stubs:/data/rnd-liu/Others/GaussianOcc \
  /home/010796032/miniconda3/envs/py311/bin/torchrun --nproc_per_node=1 --master_port=29535 \
      <this_file> --config configs/nusc-sem-gs.txt --load_weights_folder ckpts/nusc-sem-gs --eval_only
"""
import os, sys, glob
import numpy as np

GAUSSIANOCC = '/data/rnd-liu/Others/GaussianOcc'
XDOMAIN = os.path.dirname(os.path.abspath(__file__))


def build_av2_dataset_cls():
    """Defined after chdir/sys.path so the repo modules import."""
    import PIL.Image as pil
    from datasets.nusc_dataset import NuscDataset
    from datasets.mono_dataset import MonoDataset, pil_loader

    sys.path.insert(0, XDOMAIN)
    import av2_common as C

    class AV2Dataset(NuscDataset):
        """Serves ONE AV2 frame (frame-0) through GaussianOcc's own preprocessing pipeline."""

        def __init__(self, opt, npz_path, frames_dir):
            # bypass NuscDataset.__init__ (no NuScenes devkit); init the MonoDataset base only
            MonoDataset.__init__(self, opt, opt.height, opt.width, [0], num_scales=1,
                                 is_train=False, volume_depth=opt.volume_depth)
            self.opt = opt
            self.npz_path = npz_path
            self.frames_dir = frames_dir
            self.filenames = [npz_path]
            self.loader = pil_loader

        def __len__(self):
            return 1

        def get_info(self, inputs, index_temporal, do_flip):
            d = np.load(self.npz_path, allow_pickle=True)
            cams = list(d["cams"])
            idx6 = C.select6(cams)                 # AV2_6 order
            base = os.path.basename(self.npz_path)[:-4]
            scene = os.path.basename(self.frames_dir.rstrip("/"))

            inputs[("color", 0, -1)] = []
            inputs["pose_spatial"] = []
            inputs[("K_ori", 0)] = []
            inputs["id"] = []
            inputs["width_ori"], inputs["height_ori"] = [], []
            inputs["token"] = [f"{scene}/{base}"]

            # AV2 mixes portrait (front_center 1550x2048) and landscape (others 2048x1550), but
            # GaussianOcc's preprocess stacks the native-resolution tensors, assuming a UNIFORM input
            # size (nuScenes is all 1600x900). GaussianOcc ultimately resizes every image to
            # (width=640, height=384) in a single LANCZOS step anyway, so we do exactly that single
            # step here (same interpolation) and scale each K by the same anisotropic factors. Feeding
            # 640x384 as the "native" frame makes the repo's later Resize a no-op and its K scaling
            # identity -> geometry is identical to letting the repo resize the native image directly.
            Wt, Ht = self.opt.width, self.opt.height   # 640, 384
            for ci in idx6:
                cam = cams[ci]
                img = self.loader(os.path.join(self.frames_dir, f"{base}_t0_{cam}.jpg"))
                Wn, Hn = img.size                       # native (W, H)
                img = img.resize((Wt, Ht), pil.LANCZOS)
                inputs[("color", 0, -1)].append(img)
                inputs["width_ori"].append(Wt)
                inputs["height_ori"].append(Ht)
                inputs["id"].append(cam)

                K = np.eye(4, dtype=np.float32)
                K[:3, :3] = d["K"][0, ci]                       # native-resolution intrinsic
                K[0, :] *= Wt / float(Wn)                       # scale to the 640x384 frame
                K[1, :] *= Ht / float(Hn)
                inputs[("K_ori", 0)].append(K)

                s2e = d["s2e"][0, ci].astype(np.float32)         # cam -> ego (pose_spatial)
                inputs["pose_spatial"].append(s2e)

            inputs[("K_ori", 0)] = np.stack(inputs[("K_ori", 0)], axis=0)
            inputs["pose_spatial"] = np.stack(inputs["pose_spatial"], axis=0)
            for k in ("width_ori", "height_ori"):
                inputs[k] = np.stack(inputs[k], axis=0)
            return

    return AV2Dataset, C


def geom_validate(sample, npz_path, W=640, H=384):
    """STEP 3: project forward actor-box centers (ego frame) through the SAME ('K',0,0) + pose_spatial
    the model consumes; a forward center must land in-frame for a front-facing camera. Returns
    (n_forward, n_hit, details)."""
    import torch
    K = sample[("K", 0, 0)].numpy()            # (6,4,4) intrinsic in 640x384 frame
    pose = sample["pose_spatial"].numpy()      # (6,4,4) cam->ego
    d = np.load(npz_path, allow_pickle=True)
    centers = d["dyn"][:, :3].astype(np.float64)          # (N,3) ego frame
    # forward = ahead of ego and within ~40 m
    fwd = centers[(centers[:, 0] > 2) & (centers[:, 0] < 40) & (np.abs(centers[:, 1]) < 20)]
    n_hit = 0
    for c in fwd:
        hit = False
        for ci in range(6):
            ego2cam = np.linalg.inv(pose[ci])
            pc = ego2cam @ np.array([c[0], c[1], c[2], 1.0])
            if pc[2] <= 0.1:
                continue
            uv = K[ci, :3, :3] @ (pc[:3] / pc[2])
            if 0 <= uv[0] < W and 0 <= uv[1] < H:
                hit = True
                break
        n_hit += int(hit)
    return len(fwd), n_hit


def main():
    os.chdir(GAUSSIANOCC)
    for p in (GAUSSIANOCC, os.path.join(GAUSSIANOCC, "_stubs")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from options import MonodepthOptions
    from runner import Runer

    scenes_root = os.environ.get("GSOCC_SCENES", "/tmp/shield/xdomain_scenes")
    out_dir = os.environ.get("GSOCC_OUT", "/tmp/shield/xdomain_results")
    os.makedirs(out_dir, exist_ok=True)

    opts = MonodepthOptions().parse()
    runner = Runer(opts)
    runner.set_eval()

    AV2Dataset, C = build_av2_dataset_cls()

    scene_dirs = sorted(glob.glob(os.path.join(scenes_root, "*/")))
    all_pf = []
    geom_tot, geom_hit = 0, 0
    limit = int(os.environ.get("GSOCC_LIMIT", "0"))  # 0 = all
    seen = 0

    with torch.no_grad():
        for sd in scene_dirs:
            scene = os.path.basename(sd.rstrip("/"))
            pf = []
            frames = C.scene_frames(sd)
            if limit:
                frames = frames[:limit]
            for npz in frames:
                seen += 1
                ds = AV2Dataset(opts, npz, sd)
                sample = ds[0]
                data = runner.my_collate([sample])

                # geometric validation (once-per-frame, cheap)
                gt_, gh_ = geom_validate(sample, npz)
                geom_tot += gt_; geom_hit += gh_

                colour = data[("color", 0, 0)][:opts.cam_N].cuda()
                feats = runner.models["encoder"](colour)
                output = runner.models["depth"](feats, data, epoch=0, is_train=False)
                pred = output["pred_occ_logits"][0].argmax(0).detach().cpu().numpy()  # (200,200,16)

                dyn, allg = C.gt_masks(npz)
                pf.append(C.eval_pred(pred, dyn, allg))
                all_pf.append(pf[-1])
            m = C.save_results(os.path.join(out_dir, f"gaussianocc_{scene}.json"),
                               "GaussianOcc (self-sup, cam)", pf)
            print(f"  {scene}: FG_REC {m['fg_recall']:.3f}  occ_frac {m['occ_frac']:.3f}  "
                  f"obj_IoU {m['obj_iou']:.3f}  (n={len(pf)})", flush=True)

    mm = C.save_results(os.path.join(out_dir, "gaussianocc_ALL.json"),
                        "GaussianOcc (self-sup, cam)", all_pf)
    print(f"\n[GEOM VALIDATION] forward actor centers landing in-frame: "
          f"{geom_hit}/{geom_tot} = {geom_hit / max(geom_tot,1):.1%}")
    print(f"GaussianOcc on AV2 (ALL {len(all_pf)} frames): FG_REC {mm['fg_recall']:.3f}  "
          f"occ_frac {mm['occ_frac']:.3f}  obj_IoU {mm['obj_iou']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
