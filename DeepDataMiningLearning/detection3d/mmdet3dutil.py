#mmdet3d/apis/inferencers/lidar_det3d_inferencer.py
#https://github.com/open-mmlab/mmdetection3d/blob/main/demo/pcd_demo.py
#https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py

# from argparse import ArgumentParser

# from mmdet3d.apis import inference_detector, init_detector, show_result_meshlab
# from os import path as osp

from argparse import ArgumentParser
from mmdet3d.apis import LidarDet3DInferencer

from mmdet3d.apis import inference_detector, init_model
#from mmdet3d.apis import show_result_meshlab
#from os import path as osp
import os
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('--pcd',  type=str, default='/data/cmpe249-fa22/WaymoKitti/4c_train5678/training/velodyne/008118.bin', help='Point cloud file')#
    parser.add_argument('--config', type=str, default='./3DDetection/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='../modelzoo_mmdetection3d/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test a single image
    result, data = inference_detector(model, args.pcd)
    print(len(result))# three keys: 'boxes_3d', 'scores_3d', 'labels_3d'
    
    points = data['points'][0][0].cpu().numpy()# points number *4
    pts_filename = data['img_metas'][0][0]['pts_filename']
    print("pts_filename:", pts_filename)
    file_name = os.path.split(pts_filename)[-1].split('.')[0] #006767
    print(data['img_metas'])
    print(data['img_metas'][0][0]['box_mode_3d']) #Box3DMode.LIDAR

    print("results len:", len(result[0]))# len=3, 
    for res in result[0].keys():
        print(res)
    #boxes_3d, scores_3d, labels_3d
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    print(pred_bboxes)# [11,7], Each row is (x, y, z, x_size, y_size, z_size, yaw) in Box3DMode.LIDAR
    print(type(pred_bboxes))#numpy.ndarray

    print("box_mode_3d:", data['img_metas'][0][0]['box_mode_3d'])#Box3DMode.LIDAR

    lidarresults = {}
    lidarresults['points'] = points
    lidarresults['boxes_3d'] = pred_bboxes
    lidarresults['scores_3d'] = result[0]['scores_3d'].numpy()
    lidarresults['labels_3d'] = result[0]['labels_3d'].numpy()
    #np.savez_compressed(os.path.join(args.out_dir, 'lidarresults.npz'), lidarresults)
    np.save(os.path.join(args.out_dir, 'lidarresultsnp.npy'), lidarresults)


    # show the results (points save to xxx_points.obj file, pred3d box save to xxx_pred.ply, these two files can opend by meshlab)
    #show_result_meshlab(data, result, args.out_dir)
    # show the results
    # show_result_meshlab(
    #     data,
    #     result,
    #     args.out_dir,
    #     args.score_thr,
    #     show=args.show,
    #     snapshot=args.snapshot,
    #     task='det')


if __name__ == '__main__':
    main()