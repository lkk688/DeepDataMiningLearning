#mmdet3d/apis/inferencers/lidar_det3d_inferencer.py
#https://github.com/open-mmlab/mmdetection3d/blob/main/demo/pcd_demo.py
#https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py

# from argparse import ArgumentParser

# from mmdet3d.apis import inference_detector, init_detector, show_result_meshlab
# from os import path as osp

from argparse import ArgumentParser
import mmdet3d
print(mmdet3d.__version__)
from mmdet3d.structures import Box3DMode, Det3DDataSample
from mmdet3d.apis import LidarDet3DInferencer #https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inferencers/lidar_det3d_inferencer.py

from mmdet3d.apis import inference_detector, init_model, inference_multi_modality_detector
#from mmdet3d.apis import show_result_meshlab
#from os import path as osp
import os
import numpy as np

#'/data/cmpe249-fa22/WaymoKitti/4c_train5678/training/velodyne/008118.bin'
#demo/data/kitti/000008.bin
#demo/data/kitti/000008.png
#demo/data/kitti/000008.pkl #anno

#pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
#hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

#pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
#hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

#hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
#pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py

#demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
#hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth
#pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py

#multimodal
#demo/data/kitti/000008.bin
#demo/data/kitti/000008.png
#demo/data/kitti/000008.pkl #anno
#configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py
#mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth
def main():
    parser = ArgumentParser()
    parser.add_argument('--expname',  type=str, default='bevfusion')
    parser.add_argument('--mode',  type=str, default='multi')
    parser.add_argument('--pcd',  type=str, default='/data/rnd-liu/MyRepo/mmdetection3d/demo/data/kitti/000008.bin', help='Point cloud file')#
    parser.add_argument('--config', type=str, default='/data/rnd-liu/MyRepo/mmdetection3d/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth', help='Checkpoint file')
    parser.add_argument('--img', type=str, default='/data/rnd-liu/MyRepo/mmdetection3d/demo/data/kitti/000008.png')
    parser.add_argument('--ann', type=str, default='/data/rnd-liu/MyRepo/mmdetection3d/demo/data/kitti/000008.pkl')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM2', #'CAM_FRONT',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    parser.add_argument(
        '--out_dir', type=str, default='output', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # build the model from a config file and a checkpoint file
    #model = init_detector(args.config, args.checkpoint, device=args.device)
    model = init_model(args.config, args.checkpoint, device=args.device) ##https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py

    # test a single image
    #https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py
    if args.mode == "multi":
        data_sample, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                     args.ann, args.cam_type)
    else:
        data_sample, data = inference_detector(model, args.pcd)
    #result is Det3DDataSample https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/det3d_data_sample.py
    print(data_sample.gt_instances_3d)
    print(data_sample.gt_instances)
    print(data_sample.pred_instances_3d) #InstanceData 
    print(data_sample.pred_instances)
    # pts_pred_instances_3d # 3D instances of model predictions based on point cloud.
    #``img_pred_instances_3d`` (InstanceData): 3D instances of model predictions based on image.
    #https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inferencers/base_3d_inferencer.py#L30
    result = {}
    if 'pred_instances_3d' in data_sample:
        pred_instances_3d = data_sample.pred_instances_3d.numpy() #InstanceData
        #three keys: 'boxes_3d', 'scores_3d', 'labels_3d'
        result = {
            'labels_3d': pred_instances_3d.labels_3d.tolist(), #11
            'scores_3d': pred_instances_3d.scores_3d.tolist(), #11
            'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist() #11 len list, each 7 points
        }

    if 'pred_pts_seg' in data_sample:
        pred_pts_seg = data_sample.pred_pts_seg.numpy()
        result['pts_semantic_mask'] = \
            pred_pts_seg.pts_semantic_mask.tolist()

    if data_sample.box_mode_3d == Box3DMode.LIDAR:
        result['box_type_3d'] = 'LiDAR'
    elif data_sample.box_mode_3d == Box3DMode.CAM:
        result['box_type_3d'] = 'Camera'
    elif data_sample.box_mode_3d == Box3DMode.DEPTH:
        result['box_type_3d'] = 'Depth'

    print(data.keys())# ['data_samples', 'inputs'] ['inputs']['points']:[59187, 4] 
    points = data['inputs']['points'].cpu().numpy() #[59187, 4]

    
    #boxes_3d, scores_3d, labels_3d
    pred_bboxes = result['bboxes_3d']
    print(pred_bboxes)# 11 list, Each row is (x, y, z, x_size, y_size, z_size, yaw) 
    print(type(pred_bboxes))#<class 'list'>

    lidarresults = {}
    lidarresults['points'] = points
    lidarresults['boxes_3d'] = pred_bboxes
    lidarresults['scores_3d'] = result['scores_3d']
    lidarresults['labels_3d'] = result['labels_3d']
    #np.savez_compressed(os.path.join(args.out_dir, 'lidarresults.npz'), lidarresults)
    np.save(os.path.join(args.out_dir, args.expname+'_lidarresult.npy'), lidarresults)


if __name__ == '__main__':
    main()