"""Q-Mix v0 — reliability-weighted multi-source training for cross-dataset
3D detection.

Three components:
  1. ``QMixLoadAnnotations3D`` — extends ``LoadAnnotations3D`` to pull
     the per-instance ``pseudo_weight`` from ``info['instances'][i]``
     into ``results['gt_pseudo_weights']`` (np.ndarray, shape (N,)).
     For GT-only datasets (nuScenes, Waymo GT) the field defaults to
     1.0 — no behavioural change.

  2. ``QMixPack3DDetInputs`` — extends ``Pack3DDetInputs`` to recognise
     ``gt_pseudo_weights`` as an instance-level field and route it into
     ``data_sample.gt_instances_3d.pseudo_weights``.

  3. ``QMixTransFusionHead`` — subclass of the project's TransFusion
     head that, in ``get_targets_single``, captures the assigner's
     ``pos_assigned_gt_inds`` and scales the per-positive
     ``label_weights`` / ``bbox_weights`` by the reliability of the
     assigned GT instance. Without ``pseudo_weights`` it falls back
     to the standard behaviour (weight = 1.0 everywhere).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from mmdet.models.task_modules import AssignResult
from mmengine.structures import InstanceData

from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.transforms_3d import (
    ObjectRangeFilter, ObjectNameFilter,
)
from mmdet3d.registry import MODELS, TRANSFORMS

# Use the bevdet-namespace TransFusionHead (the same one the rest of
# this project uses). Importing from projects.BEVFusion.* would trigger
# a duplicate-registration KeyError because both namespaces register
# BEVFusion under the same name in the mmdet3d registry.
from projects.bevdet.bevfusion.transfusion_head import (
    TransFusionHead, draw_heatmap_gaussian, gaussian_radius,
)


# ---------------------------------------------------------------------------
# 1) LoadAnnotations: also surface pseudo_weight per instance
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module()
class QMixLoadAnnotations3D(LoadAnnotations3D):
    """Drop-in for :class:`LoadAnnotations3D` that additionally loads a
    per-instance ``pseudo_weight`` (default 1.0) from the info dict's
    ``instances`` list."""

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        # The parent's transform filters instances (out-of-range, invalid
        # class, low-point-count, etc.) so the raw `instances` length
        # does not match the kept `gt_labels_3d`. To stay aligned we
        # only emit `gt_pseudo_weights` when the lengths match exactly;
        # otherwise we fall back to all-1.0 of the correct length.
        # This is correct for GT streams (their pseudo_weight is 1.0
        # by default) and for the pseudo stream (which is built from
        # the already-filtered PL JSONL and survives filtering).
        if 'gt_labels_3d' not in results:
            return results
        n_kept = len(results['gt_labels_3d'])
        instances = (results.get('ann_info', {}) or {}).get('instances') \
            or results.get('instances') or []
        if len(instances) == n_kept:
            weights = [float(inst.get('pseudo_weight', 1.0))
                       for inst in instances]
        else:
            weights = [1.0] * n_kept
        results['gt_pseudo_weights'] = np.asarray(weights, dtype=np.float32)
        return results


# ---------------------------------------------------------------------------
# 1b) Range / name filters that keep gt_pseudo_weights aligned
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module()
class QMixObjectRangeFilter(ObjectRangeFilter):
    """ObjectRangeFilter that also filters ``gt_pseudo_weights`` by the
    same BEV-range mask. Without this, the filter would drop GT boxes
    but leave the weight array at the original length, causing an
    InstanceData length-mismatch in Pack3DDetInputs."""

    def transform(self, input_dict: dict) -> dict:
        # We re-derive the mask before delegating so we can capture it
        # for the weights too.
        from mmdet3d.structures import (LiDARInstance3DBoxes,
                                         DepthInstance3DBoxes,
                                         CameraInstance3DBoxes)
        if 'gt_pseudo_weights' in input_dict \
                and 'gt_bboxes_3d' in input_dict:
            if isinstance(input_dict['gt_bboxes_3d'],
                          (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
                bev_range = self.pcd_range[[0, 1, 3, 4]]
            elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
                bev_range = self.pcd_range[[0, 2, 3, 5]]
            else:
                bev_range = self.pcd_range[[0, 1, 3, 4]]
            mask = input_dict['gt_bboxes_3d'].in_range_bev(
                bev_range).numpy().astype(bool)
            input_dict['gt_pseudo_weights'] = \
                input_dict['gt_pseudo_weights'][mask]
        return super().transform(input_dict)


@TRANSFORMS.register_module()
class QMixObjectNameFilter(ObjectNameFilter):
    """ObjectNameFilter that also filters ``gt_pseudo_weights`` by the
    same class mask."""

    def transform(self, input_dict: dict) -> dict:
        if 'gt_pseudo_weights' in input_dict \
                and 'gt_labels_3d' in input_dict:
            mask = np.isin(input_dict['gt_labels_3d'], self.labels)
            input_dict['gt_pseudo_weights'] = \
                input_dict['gt_pseudo_weights'][mask]
        return super().transform(input_dict)


# ---------------------------------------------------------------------------
# 2) Pack3DDetInputs: route gt_pseudo_weights into gt_instances_3d
# ---------------------------------------------------------------------------
@TRANSFORMS.register_module()
class QMixPack3DDetInputs(Pack3DDetInputs):
    """Drop-in for :class:`Pack3DDetInputs` that registers
    ``gt_pseudo_weights`` as a per-instance 3D field."""

    INSTANCEDATA_3D_KEYS = list(Pack3DDetInputs.INSTANCEDATA_3D_KEYS) + [
        'gt_pseudo_weights',
    ]


# ---------------------------------------------------------------------------
# 3) TransFusionHead: reliability-weighted positive supervision
# ---------------------------------------------------------------------------
@MODELS.register_module()
class QMixTransFusionHead(TransFusionHead):
    """Reliability-weighted TransFusion head.

    Overrides :meth:`get_targets_single` (copying the base body so we
    can capture ``sampling_result.pos_assigned_gt_inds`` instead of
    re-deriving it). When ``gt_instances_3d.pseudo_weights`` is present,
    scales per-positive ``label_weights`` and ``bbox_weights`` by
    the reliability of each assigned GT.
    """

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        pseudo_weights = getattr(gt_instances_3d, 'pseudo_weights', None)
        num_proposals = preds_dict['center'].shape[-1]

        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        vel = (copy.deepcopy(preds_dict['vel'].detach())
               if 'vel' in preds_dict else None)

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center,
                                              height, vel)
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        num_layer = self.num_decoder_layers if self.auxiliary else 1
        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:
                                      self.num_proposals * (idx_layer + 1)]
            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d,
                    score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, gt_bboxes_tensor, None,
                    gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        assign_result_ensemble = AssignResult(
            num_gts=sum(r.num_gts for r in assign_result_list),
            gt_inds=torch.cat([r.gt_inds for r in assign_result_list]),
            max_overlaps=torch.cat([r.max_overlaps
                                     for r in assign_result_list]),
            labels=torch.cat([r.labels for r in assign_result_list]),
        )

        gt_instances, pred_instances = (
            InstanceData(bboxes=gt_bboxes_tensor),
            InstanceData(priors=bboxes_tensor))
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, pred_instances, gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        bbox_targets = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros(
            [num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = torch.clamp(assign_result_ensemble.max_overlaps,
                            min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals,
                                                  dtype=torch.long)

        if gt_labels_3d is not None:
            labels += self.num_classes

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            # ----- Q-Mix v0 hook: scale per-positive weights -----
            if pseudo_weights is not None:
                pw = torch.as_tensor(pseudo_weights, device=bbox_weights.device, dtype=torch.float32)
                # one weight per positive, indexed by pos_assigned_gt_inds
                per_pos_w = pw[pos_assigned_gt_inds]
                bbox_weights[pos_inds, :] = per_pos_w.unsqueeze(-1).expand(
                    -1, self.bbox_coder.code_size)
            else:
                bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                base_pos_w = bboxes_tensor.new_ones(pos_inds.shape[0])
            else:
                base_pos_w = bboxes_tensor.new_full(
                    (pos_inds.shape[0],), float(self.train_cfg.pos_weight))
            if pseudo_weights is not None:
                pw = torch.as_tensor(pseudo_weights, device=bbox_weights.device, dtype=torch.float32)
                base_pos_w = base_pos_w * pw[pos_assigned_gt_inds]
            label_weights = label_weights.float()
            label_weights[pos_inds] = base_pos_w

        if len(neg_inds) > 0:
            label_weights = label_weights.float()
            label_weights[neg_inds] = 1.0

        # ---- Heatmap (dense; left unchanged by Q-Mix) ----
        device = labels.device
        gt_bboxes_3d_cat = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']
        heatmap = gt_bboxes_3d_cat.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d_cat)):
            width = gt_bboxes_3d_cat[idx][3] / voxel_size[0] / \
                    self.train_cfg['out_size_factor']
            length = gt_bboxes_3d_cat[idx][4] / voxel_size[1] / \
                     self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d_cat[idx][0], gt_bboxes_3d_cat[idx][1]
                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])
                centre = torch.tensor([coor_x, coor_y], dtype=torch.float32,
                                       device=device)
                centre_int = centre.to(torch.int32)
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                       centre_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None],
                bbox_weights[None], ious[None], int(pos_inds.shape[0]),
                float(mean_iou), heatmap[None])


# ---------------------------------------------------------------------------
# 6) Q-Mix v2 sampler: informativeness-densified, unbiased resampling
# ---------------------------------------------------------------------------
# Dense-learning insight (Feng & Liu, Nat. Commun. 2026): overcome the
# Curse of Rarity by RESAMPLING proportional to each sample's informativeness
# (gradient contribution x exposure) and EXCLUDING non-informative samples,
# WITHOUT biasing the gradient. We translate this to supervised 3D detection:
#   * resample frames (never scale the loss -- that was v0's biasing mistake)
#   * weight a frame by sum of per-instance (class-rarity x hardness),
#     where hardness uses cheap label-free proxies (LiDAR-point sparsity,
#     range). True model-loss hardness is the natural v3 upgrade.
#   * use pseudo-label reliability as a HARD GATE (drop noisy PLs), not a
#     loss weight.
#   * CAP the per-frame weight so rare-class frames cannot dominate
#     (fixes v1b's x5 overshoot that collapsed Pedestrian).
# Weights are computed from the actual (post-filter) dataset at init, so they
# stay aligned with the ConcatDataset index order regardless of frame drops.
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class QMixWeightedSampler(Sampler):
    """Weighted-with-replacement sampler over a (Concat)Dataset.

    Args mirror the informativeness recipe; defaults give car-only/dense/near
    frames weight ~1 and boost rare/sparse/far frames up to ``w_max``.
    """

    # rarity bonus by nuScenes-10 class index
    # (car=0 .. motorcycle=6, bicycle=7, pedestrian=8 ..)
    DEFAULT_RARITY = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.3,
                      6: 2.0, 7: 3.0, 8: 1.5, 9: 0.3}

    def __init__(self, dataset, rarity_bonus=None, geom_n_target=50.0,
                 sparsity_floor=0.3, range_norm=50.0, reliability_tau=0.0,
                 w_max=4.0, seed=None, round_up=True):
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.rarity = dict(self.DEFAULT_RARITY)
        if rarity_bonus:
            self.rarity.update({int(k): float(v)
                                for k, v in rarity_bonus.items()})
        self.geom_n_target = geom_n_target
        self.sparsity_floor = sparsity_floor
        self.range_norm = range_norm
        self.reliability_tau = reliability_tau
        self.w_max = w_max
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        self.weights = self._compute_weights()
        n = len(self.weights)
        if self.round_up:
            self.num_samples = int(np.ceil(n / world_size))
            self.total_size = self.num_samples * world_size
        else:
            self.num_samples = len(range(rank, n, world_size))
            self.total_size = n

    def _instances_of(self, idx):
        info = self.dataset.get_data_info(idx)
        if isinstance(info, dict) and info.get('instances'):
            return info['instances']
        ann = info.get('ann_info') if isinstance(info, dict) else None
        if not ann:
            return []
        labels = ann.get('gt_labels_3d', [])
        npts = ann.get('num_lidar_pts', [50] * len(labels))
        boxes = ann.get('gt_bboxes_3d', None)
        centers = boxes.tensor[:, :2].tolist() if boxes is not None \
            else [[0.0, 0.0]] * len(labels)
        out = []
        for j, lab in enumerate(labels):
            out.append({'bbox_label_3d': int(lab),
                        'num_lidar_pts': int(npts[j]) if j < len(npts) else 50,
                        'bbox_3d': [centers[j][0], centers[j][1], 0, 0, 0, 0, 0]})
        return out

    def _frame_weight(self, instances):
        # MAX over instances (presence of the single most-informative object),
        # NOT sum -- summing would oversample dense scenes by object count.
        best = 0.0
        for inst in instances:
            pw = inst.get('pseudo_weight', None)
            if pw is not None and pw < self.reliability_tau:
                continue  # reliability gate (noise exclusion)
            cls = int(inst.get('bbox_label_3d', inst.get('bbox_label', 0)))
            rb = self.rarity.get(cls, 0.5)
            if rb <= 0:
                continue
            n = float(inst.get('num_lidar_pts', self.geom_n_target))
            sparsity = min(1.0, max(self.sparsity_floor,
                                    1.0 - n / self.geom_n_target))
            box = inst.get('bbox_3d', [0, 0])
            rng = (float(box[0]) ** 2 + float(box[1]) ** 2) ** 0.5
            rangef = min(1.5, max(0.5, rng / self.range_norm))
            best = max(best, rb * sparsity * rangef)
        return float(min(self.w_max, 1.0 + best))

    def _source_bounds(self):
        # ConcatDataset segment boundaries so we can normalize per source.
        cs = getattr(self.dataset, 'cumulative_sizes', None)
        if cs:
            return [0] + list(cs)
        return [0, len(self.dataset)]

    def _compute_weights(self):
        n = len(self.dataset)
        w = np.ones(n, dtype=np.float64)
        for i in range(n):
            try:
                w[i] = self._frame_weight(self._instances_of(i))
            except Exception:
                w[i] = 1.0
        # Per-source normalization to mean 1.0: keeps each source's TOTAL
        # sampling mass = its size (mix ratio unchanged -> no source skew /
        # anti-seesaw), while densifying informative frames WITHIN a source.
        bounds = self._source_bounds()
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = w[a:b]
            m = seg.mean()
            if m > 0:
                w[a:b] = seg / m
        if self.rank == 0:
            print(f'[QMixWeightedSampler] {n} frames, {len(bounds)-1} '
                  f'sources, per-source normalized; global max={w.max():.2f} '
                  f'frac>1.5={np.mean(w > 1.5):.3f}')
        return torch.as_tensor(w, dtype=torch.double)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # weighted sampling WITH replacement = unbiased densification
        idx = torch.multinomial(self.weights, self.total_size,
                                replacement=True, generator=g).tolist()
        idx = idx[self.rank:self.total_size:self.world_size]
        return iter(idx)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
