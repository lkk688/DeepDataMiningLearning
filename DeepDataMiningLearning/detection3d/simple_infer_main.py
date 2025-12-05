"""
simple_infer_main.py

Thin main script for BEVFusion / mmdetection3d inference on NuScenes (and KITTI).

DESIGN GOAL
-----------
This file should stay SHORT and EASY to modify (or hand to an AI assistant).
All heavy logic lives in `bevfusion_infer_utils.py`.

If you later need to change behavior, you should:
  - Prefer to tweak HOW we call the utility functions here
  - Avoid regenerating or rewriting the utility module

UTILITY FUNCTIONS OVERVIEW
--------------------------
Imported from `bevfusion_infer_utils`:

1) setup_env(init_scope: bool = True)
   - Registers all mmdet3d modules and sets default scope.
   - Usually called once at startup.

2) configure_torch_for_inference()
   - Sets global PyTorch flags for inference:
       * torch.backends.cudnn.benchmark = True
       * torch.set_grad_enabled(False)

3) get_system_info() -> dict
   - Returns basic system info:
       * mmdet3d version
       * GPU name (if available)
       * total memory
       * timestamp

4) load_model_from_cfg(config_path, checkpoint_path, device="cuda",
                       dataroot=None, ann_file="", work_dir=None)
   - Loads an mmdet3d model using MMEngine Config and a checkpoint.
   - IMPORTANT CONFIG OPTIONS:
       * config_path: path to your mmdet3d config .py
       * checkpoint_path: path to .pth checkpoint
       * device: "cuda", "cuda:0", "cpu", etc.
       * dataroot: if not None, patches cfg.test_dataloader data_root & ann_file
       * ann_file: optional override for test_dataloader dataset.ann_file
       * work_dir: where Runner / logs will write (if used)

5) build_loader_pack(data_source, cfg, dataroot, nus_version="v1.0-trainval",
                     ann_file="", max_samples=-1, crop_policy="center",
                     workers=4)
   - Builds a dict:
       { "loader": DataLoader, "iter_fn": iterator, "nusc": NuScenes object }
   - MODES:
       * data_source == "cfg":
           - Uses cfg.test_dataloader.dataset (e.g., nuscenes info pkl)
       * data_source == "custom":
           - Uses raw NuScenesLoader (reads lidar sweeps & raw images)
   - IMPORTANT OPTIONS:
       * nus_version: e.g., "v1.0-trainval", "v1.0-mini"
       * max_samples: limit sample count for quick tests (-1 = all)
       * crop_policy: currently only "center" is implemented
       * workers: DataLoader num_workers

6) run_manual_benchmark(model, pack, class_names, out_dir, device="cuda",
                         eval_set="val", detection_cfg_name="detection_cvpr_2019",
                         score_thresh=0.05, max_samples=-1, sys_info=None)
   - Custom manual evaluation pipeline:
       * Iterates through pack["loader"] & pack["iter_fn"]
       * Runs model.test_step
       * Converts predictions into NuScenes JSON
       * Calls `nuscenes.eval` (NuScenesEval) directly
       * Writes:
           - nuscenes_results.json
           - eval metrics
           - benchmark_perf.json (latency & peak memory)
   - KEY CONFIG KNOBS:
       * eval_set: which split ("val", etc.)
       * detection_cfg_name: NuScenes config_factory key
       * score_thresh: filter low-confidence predictions
       * max_samples: used to filter GT/preds when sub-sampling

7) inference_loop(model, pack, out_dir, device="cuda",
                  score_thresh=0.25, metrics=None,
                  save_images=True, save_ply_if_headless=True,
                  show_open3d=True)
   - Visualization-oriented loop:
       * Runs inference
       * Saves multi-view 2D projections (JPG) if save_images=True
       * Saves PLY or shows Open3D viz depending on DISPLAY / flags
       * Tracks per-sample latency and peak memory
   - KEY CONFIG KNOBS:
       * score_thresh: visualization threshold
       * save_images: toggle 2D multi-view images
       * save_ply_if_headless: write PLY when running headless (no DISPLAY)
       * show_open3d: show interactive Open3D viewer when DISPLAY is present

8) run_benchmark_evaluation(args, sys_info)
   - Uses MMEngine Runner's `test()` method (the "official" pipeline).
   - This is the recommended way to obtain CORRECT NDS / mAP as defined by
     the mmdet3d config.
   - Behavior:
       * Loads Config.fromfile(args.config)
       * Sets cfg.work_dir, cfg.load_from
       * Overrides cfg.test_dataloader.dataset.data_root with args.dataroot
       * Optionally overrides ann_file based on args.dataset (nuscenes/kitti)
       * Registers a PerfHook (measures latency & memory)
       * Calls runner.test()
       * Flattens metric dict (if dict) and merges perf stats
       * Writes benchmark_results.json with:
           - system_info
           - accuracy_metrics
           - performance_metrics
   - KEY CONFIG KNOBS (controlled via CLI args in this main file):
       * args.dataset: "nuscenes" or "kitti" (controls ann_file override)
       * args.device: device used by PerfHook
       * args.config / args.checkpoint / args.dataroot / args.out_dir

When modifying behavior, prefer to:
  - Adjust the arguments passed to these utilities in `main()`
  - Add new flags to `parse_args()` for additional options
"""

import argparse
import os
import os.path as osp
import json

import os

# # --- FIX FOR NUMBA CUDA ERRORS ---
# # Force Numba to use the CUDA 12.6 libraries found on your cluster
# os.environ['NUMBAPRO_LIBDEVICE'] = "/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/libdevice/libdevice.10.bc"
# os.environ['NUMBAPRO_NVVM'] = "/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/lib64/libnvvm.so"
# # ---------------------------------

# All heavy logic is in this module:
from simple_infer_utils import (
    setup_env,
    configure_torch_for_inference,
    get_system_info,
    load_model_from_cfg,
    build_loader_pack,
    run_manual_benchmark,
    inference_loop,
    run_runner_benchmark,      # optional: keep the simple runner path
    run_benchmark_evaluation,  # NEW: full runner-based eval with PerfHook
)


def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    NOTE FOR FUTURE MODIFICATIONS (AI / HUMAN):
    ------------------------------------------
    If you want to expose more configuration knobs from the utility
    functions, add new arguments here, and then pass them down in main().

    Example:
        - To change score threshold for manual eval, you could add:
            ap.add_argument("--manual-score-thresh", type=float, default=0.05)
          and then pass it to run_manual_benchmark(..., score_thresh=args.manual_score_thresh)
    """
    ap = argparse.ArgumentParser(
        description="Thin main for BEVFusion / mmdetection3d inference and evaluation"
    )

    # -------------------------------------------------------------------------
    # CORE PATHS
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--config", required=True,
        help="Path to mmdet3d config (.py)"
    )
    ap.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint (.pth)"
    )
    ap.add_argument(
        "--dataroot", required=True,
        help="Dataset root directory (e.g., NuScenes dataroot)"
    )
    ap.add_argument(
        "--out-dir", default="results",
        help="Output directory for logs, metrics, and visualizations"
    )

    # -------------------------------------------------------------------------
    # MODE SELECTION
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--benchmark-type", choices=['runner', 'manual'],
        default='manual',
        help=(
            "Historical switch; current logic mainly uses --eval-backend. "
            "You can still use this for your own branching logic if desired."
        )
    )
    ap.add_argument(
        "--data-source", choices=['cfg', 'custom'],
        default='custom',
        help=(
            "Data loading backend:\n"
            "  - 'cfg': use cfg.test_dataloader (info pkl, etc.)\n"
            "  - 'custom': use raw NuScenesLoader (direct from sensor files)"
        )
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu')"
    )

    # -------------------------------------------------------------------------
    # DATASET / DATALOADER OPTIONS
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--nus-version", default="v1.0-trainval",
        help="NuScenes version (e.g., 'v1.0-mini', 'v1.0-trainval')"
    )
    ap.add_argument(
        "--ann-file", default="",
        help=(
            "Path to dataset info pkl (for cfg data_source). "
            "If empty, the config's default ann_file is used."
        )
    )
    ap.add_argument(
        "--max-samples", type=int, default=20,
        help=(
            "Limit number of samples (for custom data_source). "
            "Use -1 to process ALL samples."
        )
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Number of DataLoader workers"
    )

    # -------------------------------------------------------------------------
    # EVAL / VISUALIZATION OPTIONS
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--eval", action="store_true",
        help="Enable evaluation mode (either manual or runner backend)"
    )
    ap.add_argument(
        "--crop-policy", default="center",
        help="Image crop policy (currently only 'center' is implemented in utils)"
    )
    ap.add_argument(
        "--no-save-images", action="store_true",
        help="Disable saving 2D multiview visualization images"
    )
    ap.add_argument(
        "--no-save-ply", action="store_true",
        help="Disable saving PLY point clouds when running headless"
    )
    ap.add_argument(
        "--no-open3d", action="store_true",
        help="Disable interactive Open3D visualization even if DISPLAY is set"
    )

    # -------------------------------------------------------------------------
    # DATASET TYPE (for Runner-based evaluation)
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--dataset", choices=["nuscenes", "kitti"],
        default="nuscenes",
        help=(
            "Dataset type, used by run_benchmark_evaluation() to choose "
            "which ann_file to override for Runner-based evaluation."
        )
    )

    # -------------------------------------------------------------------------
    # EVALUATION BACKEND
    # -------------------------------------------------------------------------
    ap.add_argument(
        "--eval-backend", choices=["manual", "runner"],
        default="manual",
        help=(
            "Backend used when --eval is set:\n"
            "  - 'manual': custom JSON + NuScenesEval via run_manual_benchmark()\n"
            "  - 'runner': use mmdetection3d Runner.test() via run_benchmark_evaluation()\n"
            "For official NDS/mAP as defined by the config, use 'runner'."
        )
    )

    return ap.parse_args()


def main() -> None:
    """
    Main entrypoint.

    HIGH-LEVEL LOGIC:
    -----------------
    1. Parse arguments.
    2. Initialize environment and Torch inference settings.
    3. Get system info (for logging).
    4. If `--eval --eval-backend runner`, use run_benchmark_evaluation() and exit.
    5. Otherwise, load the model & config via load_model_from_cfg().
    6. Build loader pack via build_loader_pack().
    7. If `--eval` with manual backend, run_manual_benchmark().
    8. Else, run visualization-only inference_loop() and save metrics.json.

    MODIFYING LOGIC (AI / HUMAN):
    -----------------------------
    - To change thresholds, toggles, or evaluation details, edit the calls to:
        * run_benchmark_evaluation(...)
        * run_manual_benchmark(...)
        * inference_loop(...)
    - To expose new options, add CLI flags in parse_args() and pass them here.
    - Avoid editing bevfusion_infer_utils.py unless you need new core behaviors.
    """
    args = parse_args()

    # 1) Environment setup (MMDet3D registry + Torch inference configuration)
    setup_env(init_scope=True)
    configure_torch_for_inference()

    # Gather system info once; reused for metrics / JSON output
    sys_info = get_system_info()
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # PATH 1: Runner-based evaluation backend (recommended for official NDS)
    # -------------------------------------------------------------------------
    if args.eval and args.eval_backend == "runner":
        # This will:
        #   - Load the config
        #   - Override data_root and ann_file for the test dataset
        #   - Build a Runner and run test()
        #   - Use PerfHook to collect latency & memory
        #   - Write benchmark_results.json (metrics + perf + sys_info)
        run_benchmark_evaluation(args, sys_info)
        return

    # -------------------------------------------------------------------------
    # PATH 2: Manual backend (custom JSON + NuScenesEval + our own loaders)
    # -------------------------------------------------------------------------
    # If doing manual NuScenes evaluation, ensure we don't sub-sample
    if args.eval and args.eval_backend == "manual":
        # We want full dataset coverage, so disable max_samples limit.
        args.max_samples = -1

    # 3) Load model & config (this also applies path patches when dataroot is set)
    print("Loading Config & Model...")
    model, cfg = load_model_from_cfg(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        # For data_source == 'cfg', utils will patch cfg.test_dataloader paths
        dataroot=args.dataroot if args.data_source == 'cfg' else None,
        ann_file=args.ann_file,
        work_dir=args.out_dir,
    )

    # 4) Build loader pack (loader + iterator + NuScenes instance)
    #    NOTE: You can change:
    #       - data_source: "cfg" vs "custom"
    #       - max_samples: for debugging with fewer samples
    #       - workers: DataLoader parallelism
    pack = build_loader_pack(
        data_source=args.data_source,
        cfg=cfg,
        dataroot=args.dataroot,
        nus_version=args.nus_version,
        ann_file=args.ann_file,
        max_samples=args.max_samples,
        crop_policy=args.crop_policy,
        workers=args.workers,
        dataset=args.dataset
    )

    # 5) Either run evaluation (manual backend) or visual inference
    if args.eval and args.eval_backend == "manual":
        # Manual benchmark = export JSON + run NuScenesEval
        # If you want different thresholds or config name, edit here.
        run_manual_benchmark(
            model=model,
            pack=pack,
            class_names=cfg.class_names,
            out_dir=args.out_dir,
            device=args.device,
            eval_set="val",                    # You can change this if needed
            detection_cfg_name="detection_cvpr_2019",
            score_thresh=0.05,                 # Manual eval threshold
            max_samples=args.max_samples,      # Usually -1 for full eval
            sys_info=sys_info,
            dataset=args.dataset
        )
    else:
        # Visualization mode only:
        #   - multi-view 2D images (unless --no-save-images)
        #   - PLY / Open3D 3D viz (controlled by flags)
        #   - metrics.json with per-sample latency & max_conf
        metrics = {"system_info": sys_info, "samples": []}

        metrics = inference_loop(
            model=model,
            pack=pack,
            out_dir=args.out_dir,
            device=args.device,
            score_thresh=0.25,                 # Visualization threshold
            metrics=metrics,
            save_images=not args.no_save_images,
            save_ply_if_headless=not args.no_save_ply,
            show_open3d=not args.no_open3d,
            max_samples=args.max_samples,      # <-- now honored by the loop
        )

        with open(osp.join(args.out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()

"""
#Runner-based evaluation (official NDS via Runner backend)
python simple_infer_main.py \
    --config /data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth \
    --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes \
    --out-dir ./bevfusion_infer_results_v4 \
    --data-source cfg \
    --ann-file /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl \
    --dataset nuscenes \
    --eval \
    --eval-backend runner

#Manual backend evaluation (custom JSON + NuScenesEval)
python simple_infer_main.py \
    --config /data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth \
    --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes \
    --out-dir ./bevfusion_infer_results_v4 \
    --data-source custom \
    --ann-file /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl \
    --dataset nuscenes \
    --eval \
    --eval-backend manual
#NDS: 0.5743 | mAP: 0.5580

#Visualization-only (no evaluation, just multiview images + metrics)
python simple_infer_main.py \
    --config /data/rnd-liu/MyRepo/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth \
    --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes \
    --out-dir ./bevfusion_infer_results_v4 \
    --data-source custom \
    --ann-file /data/rnd-liu/MyRepo/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl
"""

#KITTI:
"""
#Numba issues
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/lib64/
ln -sf /opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/lib64/libnvvm.so /home/010796032/miniconda3/envs/py310/lib/libnvvm.so
ln -sf /opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/libdevice/libdevice.10.bc /home/010796032/miniconda3/envs/py310/lib/libdevice.10.bc

python simple_infer_main.py \
  --config /data/rnd-liu/MyRepo/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
  --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/kitti \
  --ann-file /data/rnd-liu/MyRepo/mmdetection3d/data/kitti/kitti_infos_val.pkl \
  --out-dir ./results_kitti \
  --dataset kitti \
  --data-source cfg \
  --eval \
  --eval-backend runner

python simple_infer_main.py \
  --config /data/rnd-liu/MyRepo/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --checkpoint /data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
  --dataroot /data/rnd-liu/MyRepo/mmdetection3d/data/kitti \
  --ann-file /data/rnd-liu/MyRepo/mmdetection3d/data/kitti/kitti_infos_val.pkl \
  --out-dir ./results_kitti \
  --dataset kitti \
  --data-source cfg \
  --eval \
  --eval-backend manual
"""