"""ngperception.detection — pure-PyTorch 3D object detection (PointPillars), harvested from
2D3DFusion/mydetector3d (OpenPCDet lineage), no spconv/mmcv/compiled ops. See README.md."""
from .pointpillars import PointPillars, pillarize
from .box_utils import ResidualCoder, boxes_bev_iou_aligned, rotated_iou_bev, nms_aligned
