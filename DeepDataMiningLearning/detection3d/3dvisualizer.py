#https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/visualization/local_visualizer.py
#In configuration file: https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet3d.structures import (DepthInstance3DBoxes, Det3DDataSample)
#from mmdet3d.visualization import Det3DLocalVisualizer
from local_visualizer import Det3DLocalVisualizer
import mmcv
import mmengine
import os

det3d_local_visualizer = Det3DLocalVisualizer()

# image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8') #HWC
# points = np.random.rand(1000, 3)
# gt_instances_3d = InstanceData()
# gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(torch.rand((5, 7))) #[5, 7]
# gt_instances_3d.labels_3d = torch.randint(0, 2, (5,)) #(5,)
# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
# data_input = dict(img=image, points=points)
# det3d_local_visualizer.add_datasample('3D Scene', data_input, gt_det3d_data_sample)
# det3d_local_visualizer.show()

#input_meta['depth2img']

#load npy file
lidarresults = np.load('data/testmulti_vis.npy', allow_pickle=True).item()
points = lidarresults['points']
boxes_3d = lidarresults['bboxes_3d']
scores_3d = np.asarray(lidarresults['scores_3d'])
labels_3d = np.asarray(lidarresults['labels_3d'])

img_path = os.path.join(r'D:\Developer', 'mmdetection3d/demo/data/kitti/000008.png')
if isinstance(img_path, str):
    img_bytes = mmengine.fileio.get(img_path)
    img = mmcv.imfrombytes(img_bytes) #(375, 1242, 3)
    img = img[:, :, ::-1] #(375, 1242, 3) bgr to rgb
data_input = dict(points=points, img=img)

pred = InstanceData()
pred.bboxes_3d = DepthInstance3DBoxes(boxes_3d)
pred.scores_3d = scores_3d
pred.labels_3d = labels_3d
det3d_data_sample = Det3DDataSample(
        metainfo=dict(box_type_3d=DepthInstance3DBoxes))
#det3d_data_sample = Det3DDataSample()

det3d_data_sample.pred_instances_3d = pred

meta_info = dict()
meta_info['depth2img'] = np.array(
    [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
        [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
        [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])
meta_info['lidar2img'] = np.array(
    [[5.23289349e+02, 3.68831943e+02, 6.10469439e+01],
        [1.09560138e+02, 1.97404735e+02, -5.47377738e+02],
        [1.25930002e-02, 9.92229998e-01, -1.23769999e-01]])
det3d_data_sample.set_metainfo(meta_info)

det3d_local_visualizer.add_datasample(
        "lidar",
        data_input,
        det3d_data_sample, #Det3DDataSample
        show=True,
        draw_gt=False,
        # wait_time=wait_time,
        # draw_pred=draw_pred,
        # pred_score_thr=pred_score_thr,
        # o3d_save_path=o3d_save_path,
        # out_file=out_file,
        vis_task='multi-modality_det',
    )
det3d_local_visualizer.show()