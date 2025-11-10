# simple_nuscenes_evaluator.py
# -*- coding: utf-8 -*-
"""
A clean, minimal wrapper around mmdetection3d's NuScenesMetric.

Why this exists:
- NuScenesMetric.process(data_batch, data_samples) expects the *second* arg to be
  a sequence of dicts, each with:
    {
      "pred_instances_3d": InstanceData (with bboxes_3d/scores_3d/labels_3d),
      "pred_instances":    InstanceData (optional; can be empty),
      "sample_idx":        int
    }
- Many custom training scripts return predictions as Det3DDataSample objects that
  are NOT subscriptable (i.e., p["pred_instances_3d"] fails). This adapter wraps
  them into dicts the metric can consume directly.

Usage (pseudo-code):
    evaluator = SimpleNuScenesEvaluator(
        data_root=..., ann_file=..., dataset_meta={"classes": CLASSES, "version": "v1.0-trainval"},
        jsonfile_prefix="work_dirs/nusc_eval/preds"
    )
    evaluator.reset()

    model.eval()
    with torch.no_grad():
        for vb in val_loader:
            vb = move_batch_to_device(vb, device)       # your move helper
            preds = model_wrapper.predict_step(vb)       # your predict_step
            evaluator.process_batch(vb, preds, device)   # <-- one line

    results = evaluator.compute()  # dict of NDS/mAP/etc.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

# Import the official NuScenesMetric. If you copied the class locally, import from your path instead.
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmengine.evaluator import BaseMetric


class SimpleNuScenesEvaluator:
    """
    A tiny façade over NuScenesMetric that:
      1) Extracts ground-truth samples from your validation batch.
      2) Normalizes predictions into list[dict] with required keys.
      3) Feeds them into NuScenesMetric.process(...) correctly.
      4) Calls NuScenesMetric.evaluate(...) to get final metrics.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        dataset_meta: Dict[str, Any],
        metric: Union[str, List[str]] = "bbox",
        modality: Optional[Dict[str, bool]] = None,
        jsonfile_prefix: Optional[str] = None,
        eval_version: str = "detection_cvpr_2019",
        collect_device: str = "cpu",
        format_only: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        """
        Args:
            data_root: NuScenes data root (e.g., ".../data/nuscenes").
            ann_file:  Annotation file path used by the dataset (the same as in cfg).
            dataset_meta: Must contain at least:
                {
                    "classes": [...],  # list of class names matching your dataset
                    "version": "v1.0-trainval" or "v1.0-mini"
                }
            metric: Which metric to compute ("bbox" by default).
            modality: Dict specifying modalities, e.g., {"use_camera": False, "use_lidar": True}
            jsonfile_prefix: Where to dump result jsons for official nuScenes eval.
            eval_version: NuScenes detection config version.
            collect_device: "cpu" or "gpu" for result collection (single-node fine with "cpu").
            format_only: If True, only formats json without computing metrics.
            backend_args: Optional storage backend args.
        """
        if modality is None:
            modality = dict(use_camera=False, use_lidar=True)

        # Create the official metric
        self._metric: BaseMetric = NuScenesMetric(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            modality=modality,
            prefix=None,
            format_only=format_only,
            jsonfile_prefix=jsonfile_prefix,
            eval_version=eval_version,
            collect_device=collect_device,
            backend_args=backend_args,
        )

        # Attach dataset meta (NuScenesMetric requires this to compute)
        # Expected keys: "classes" and "version"
        self._metric.dataset_meta = dataset_meta

        # Reset internal buffers
        self.reset()

    # ---------- Public API ----------

    def reset(self) -> None:
        """Clear metric internal buffers (results per-batch)."""
        self._metric.results.clear()

    def process_batch(self, val_batch: Dict[str, Any], preds: Any, device: torch.device) -> None:
        """
        Normalize and feed one validation batch to the metric.

        Args:
            val_batch: The raw batch dict from your val_loader (must contain "data_samples").
            preds:     Model predictions for this batch. Accepts:
                        - list[Det3DDataSample] (attr-only, not subscriptable)
                        - list[dict] with "pred_instances_3d"
                        - Det3DDataSample or dict (single item)
                        - tuple where first element is the above (e.g., (preds, aux))
            device:    Torch device used for empty InstanceData creation if needed.
        """
        gt_samples = self._extract_gt_samples_strict(val_batch)
        pred_records = self._build_nuscenes_pred_records(preds, gt_samples, device)

        # Call the official metric's process:
        #   data_batch: {"data_samples": <GT list>}
        #   data_samples: <PRED list[dict]> with required keys
        self._metric.process({"data_samples": gt_samples}, pred_records)

    def compute(self) -> Dict[str, float]:
        """
        Compute final nuScenes metrics over all accumulated batches.

        Returns:
            Dict[str, float]: NDS, mAP, class APs, and error metrics (mATE/mASE/...).
        """
        # n_samples is used by mmengine aggregator; give total number of seen samples
        # (each GT data_sample corresponds to one sample)
        n_samples = len(self._metric.results)
        return self._metric.evaluate(n_samples)

    # ---------- Helpers (private) ----------

    @staticmethod
    def _extract_gt_samples_strict(batch: Any) -> List[Any]:
        """
        Extract the ground-truth Det3DDataSample list from a validation batch.

        The val batch produced by mmdet3d datasets/dataloaders typically looks like:
            {"inputs": {...}, "data_samples": [Det3DDataSample, ...]}

        This method never returns a string and will raise with clear errors if
        the structure is unexpected.
        """
        if isinstance(batch, dict):
            if "data_samples" not in batch:
                raise RuntimeError("val batch dict missing key 'data_samples'")
            ds = batch["data_samples"]
            return ds if isinstance(ds, list) else [ds]

        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
            out = []
            for i, item in enumerate(batch):
                if "data_samples" not in item:
                    raise RuntimeError(f"val batch item[{i}] missing key 'data_samples'")
                ds = item["data_samples"]
                out.extend(ds if isinstance(ds, list) else [ds])
            return out

        raise RuntimeError(f"Unsupported val batch type for GT extraction: {type(batch)}")

    @staticmethod
    def _empty_instancedata(device: torch.device):
        """
        Create an empty mmengine.structures.InstanceData with the minimal fields
        nuScenes expects on the 3D side. bboxes_3d can be absent when empty.
        """
        from mmengine.structures import InstanceData
        idata = InstanceData()
        idata.scores_3d = torch.empty(0, device=device)
        idata.labels_3d = torch.empty(0, dtype=torch.long, device=device)
        return idata

    def _build_nuscenes_pred_records(
        self, preds: Any, gt_samples: List[Any], device: torch.device
    ) -> List[Dict[str, Any]]:
        """
        Normalize model outputs into list[dict] items required by NuScenesMetric.process.

        Each element contains:
          - 'pred_instances_3d': InstanceData with bboxes_3d/scores_3d/labels_3d (or empty)
          - 'pred_instances':    InstanceData for 2D (optional; we provide empty if absent)
          - 'sample_idx':        int (we fetch from prediction metainfo or fallback to GT)
        """
        # 1) Normalize container shape: single -> list; (preds, aux) -> preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds if isinstance(preds, (list, tuple)) else [preds]
        preds = list(preds)
        if len(preds) == 1 and isinstance(preds[0], (list, tuple)):
            preds = list(preds[0])

        from mmengine.structures import InstanceData

        records: List[Dict[str, Any]] = []
        for i, p in enumerate(preds):
            # 2) Fetch 3D predictions:
            #    Works for Det3DDataSample (attr-only) and dict-based preds.
            pred3d = getattr(p, "pred_instances_3d", None)
            if pred3d is None and isinstance(p, dict):
                pred3d = p.get("pred_instances_3d", None)
            if pred3d is None:
                pred3d = self._empty_instancedata(device)

            # 3) Fetch optional 2D predictions, or create an empty container.
            pred2d = getattr(p, "pred_instances", None)
            if pred2d is None and isinstance(p, dict):
                pred2d = p.get("pred_instances", None)
            if pred2d is None:
                pred2d = InstanceData()  # empty is fine

            # 4) Resolve sample_idx:
            sample_idx = None
            meta = getattr(p, "metainfo", None)
            if isinstance(meta, dict):
                sample_idx = meta.get("sample_idx", None)
            if sample_idx is None:
                # Fall back to ground-truth sample at aligned index
                gi = gt_samples[min(i, len(gt_samples) - 1)]
                if hasattr(gi, "metainfo") and isinstance(gi.metainfo, dict):
                    sample_idx = gi.metainfo.get("sample_idx", None)
                if sample_idx is None:
                    sample_idx = getattr(gi, "sample_idx", None)
            if sample_idx is None:
                raise RuntimeError(f"Cannot resolve sample_idx for pred #{i}")

            records.append(
                {
                    "pred_instances_3d": pred3d,
                    "pred_instances": pred2d,
                    "sample_idx": int(sample_idx),
                }
            )

        # Final sanity: ensure we return list[dict] (never string)
        if isinstance(records, (str, bytes)):
            raise TypeError("BUG: pred_records is a string; expected list[dict].")
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            raise TypeError(f"BUG: pred_records must be list[dict], got {type(records)}")

        return records