#https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/visualization/local_visualizer.py
#In configuration file: https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet3d.structures import (DepthInstance3DBoxes, Det3DDataSample)
from mmdet3d.visualization import Det3DLocalVisualizer

det3d_local_visualizer = Det3DLocalVisualizer()

image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8') #HWC
points = np.random.rand(1000, 3)
gt_instances_3d = InstanceData()
gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(torch.rand((5, 7))) #[5, 7]
gt_instances_3d.labels_3d = torch.randint(0, 2, (5,)) #(5,)
gt_det3d_data_sample = Det3DDataSample()
gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
data_input = dict(img=image, points=points)
det3d_local_visualizer.add_datasample('3D Scene', data_input, gt_det3d_data_sample)
det3d_local_visualizer.show()

#input_meta['depth2img']