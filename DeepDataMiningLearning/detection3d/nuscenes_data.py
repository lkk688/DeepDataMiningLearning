#https://www.nuscenes.org/nuscenes#download
#https://github.com/nutonomy/nuscenes-devkit
#python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
import argparse
from os import path as osp
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', type=str, default='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuScenes/', #'/mnt/f/Dataset/nuScenes/v1.0-mini',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0-mini',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/nuScenes/', #'/mnt/f/Dataset/nuScenes/v1.0-mini',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes') #kitti
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
args = parser.parse_args()

def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, max_sweeps=max_sweeps) #version=version, 

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')

#https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html
#(base) lkk@Alienware-LKKi7G8:~/Developer/DeepDataMiningLearning$ ln -s /mnt/f/Dataset/nuScenes/v1.0-mini ./data/nuScenes/
    #unlink ./data/nuScenes/v1.0-mini
    #$ ln -s /mnt/f/Dataset/nuScenes/ ./data/nuScenes/
if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            train_version = f'{args.version}'
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=train_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)