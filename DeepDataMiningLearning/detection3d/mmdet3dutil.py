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

def test_inference(args):
    # build the model from a config file and a checkpoint file
    #model = init_detector(args.config, args.checkpoint, device=args.device)
    model = init_model(args.config, args.checkpoint, device=args.device) ##https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py

    # test a single image
    #https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py
    if args.mode == "multi":
        data_sample, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                     args.infos, args.cam_type)
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

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.apis import convert_SyncBN
from mmdet3d.registry import DATASETS, MODELS, VISUALIZERS
from mmengine.runner import load_checkpoint
import mmcv
import mmengine
#https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/visualization/local_visualizer.py
#from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer
from local_visualizer import Det3DLocalVisualizer
from mmengine.dataset import Compose
from mmdet3d.utils import ConfigType
#from typing import Dict, List, Optional, Sequence, Union
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)
from mmengine.infer.infer import ModelType
from mmengine.dataset import pseudo_collate
from mmengine.model.utils import revert_sync_batchnorm
from rich.progress import track
import os.path as osp
from mmengine import dump
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)

from mmengine.structures import InstanceData
InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]

#ref from multi_modality_det3d_inferencer.py
class myInference():
    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 mode: str = 'lidar', 
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.mode = mode #newly added to differentiate different mode

        self.num_visualized_frames = 0
        self.scope = scope
        self.palette = palette
        self.model = self.initmodel(configfile=model, checkpoint=weights, device=device)
        if self.mode == "lidar":
            self.pipeline = self.lidar_init_pipeline(self.model.cfg)
        else:
            self.pipeline = self.multi_init_pipeline(self.model.cfg)
        self.collate_fn = pseudo_collate
            #visualizer=VISUALIZERS.build(config.visualizer) #Det3DLocalVisualizer
        self.visualizer= Det3DLocalVisualizer()
        self.visualizer.dataset_meta =self.model.dataset_meta
        self.model = revert_sync_batchnorm(self.model)

        #from mmengine.registry import FUNCTIONS
        # with FUNCTIONS.switch_scope_and_registry(self.scope) as registry: #self.scope=mmdet3d
        #collate_fn = registry.get(cfg.test_dataloader.collate_fn)
        #collate_fn = pseudo_collate

        #self.visualizer = self._init_visualizer(cfg)
        #cfg.visualizer.name = name
            #return VISUALIZERS.build(cfg.visualizer)
        #visualizer.dataset_meta = self.model.dataset_meta #classes, categories, palette
        #self.model = revert_sync_batchnorm(self.model)
        
    def initmodel(self, configfile, checkpoint, device):
        if isinstance(configfile, (str, Path)):
            config = Config.fromfile(configfile)
        elif not isinstance(config, Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        
        convert_SyncBN(config.model)
        config.model.train_cfg = None
        init_default_scope(config.get('default_scope', 'mmdet3d'))

        #build the model
        model = MODELS.build(config.model)

        has_dataset_meta = True
        if checkpoint is not None:
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint.get('meta', {}):
                # mmdet3d 1.x
                model.dataset_meta = checkpoint['meta']['dataset_meta'] #contain 'classes'
            elif 'CLASSES' in checkpoint.get('meta', {}):
                # < mmdet3d 1.x
                classes = checkpoint['meta']['CLASSES']
                model.dataset_meta = {'classes': classes}
                if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                    model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
            else:
                # < mmdet3d 1.x
                model.dataset_meta = {'classes': config.class_names}

                if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                    model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
    
            test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
            # lazy init. We only need the metainfo.
            test_dataset_cfg['lazy_init'] = True
            metainfo = DATASETS.build(test_dataset_cfg).metainfo
            cfg_palette = metainfo.get('palette', None) ##contain 'classes'
            if cfg_palette is not None and has_dataset_meta:
                model.dataset_meta['palette'] = cfg_palette
        
        model.cfg = config  # save the config in the model for convenience
        if device != 'cpu':
            torch.cuda.set_device(device)

        model.to(device)
        model.eval()
        return model
        
    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1

    def lidar_init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile') #0
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim'] #LIDAR, 4
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim'] #[0,1,2,3]

        pipeline_cfg[load_point_idx]['type'] = 'LidarDet3DInferencerLoader'
        return Compose(pipeline_cfg)
    
    def multi_init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                    'LoadPointsFromFile')
        load_mv_img_idx = self._get_transform_idx(
            pipeline_cfg, 'LoadMultiViewImageFromFiles')
        if load_mv_img_idx != -1:
            print(
                'LoadMultiViewImageFromFiles is not supported yet in the '
                'multi-modality inferencer. Please remove it')
        # Now, we only support ``LoadImageFromFile`` as the image loader in the
        # original piepline. `LoadMultiViewImageFromFiles` is not supported
        # yet.
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                                'LoadImageFromFile')

        if load_point_idx == -1 or load_img_idx == -1:
            raise ValueError(
                'Both LoadPointsFromFile and LoadImageFromFile must '
                'be specified the pipeline, but LoadPointsFromFile is '
                f'{load_point_idx == -1} and LoadImageFromFile is '
                f'{load_img_idx}')

        load_cfg = pipeline_cfg[load_point_idx]#1, {'type': 'LoadPointsFromFile'
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim'] #LIDAR, load_dim = 4
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim'] #[0,1,2,3]

        load_point_args = pipeline_cfg[load_point_idx]
        load_point_args.pop('type')
        load_img_args = pipeline_cfg[load_img_idx] #{'type': 'LoadImage
        load_img_args.pop('type') #become empty

        load_idx = min(load_point_idx, load_img_idx) #0
        pipeline_cfg.pop(max(load_point_idx, load_img_idx)) #remove 1 (LoadImage)

        pipeline_cfg[load_idx] = dict(
            type='MultiModalityDet3DInferencerLoader',
            load_point_args=load_point_args,
            load_img_args=load_img_args) #load_idx=0, replace the 0 as MultiModalityDet3DInferencerLoader

        return Compose(pipeline_cfg) #self.pipeline
    
    """
    preprocess_kwargs: set = {'cam_type'}
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'pred_score_thr',
        'img_out_dir', 'no_save_vis', 'cam_type_dir'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_dir', 'return_datasample', 'no_save_pred'
    }
    """
    def __call__(self,
                inputs: InputsType,
                batch_size: int = 1,
                return_datasamples: bool = False,
                show: bool = True, #NEW ADD
                return_vis: bool = True,
                no_save_vis: bool = False,
                img_out_dir: str = 'output/',
                cam_type_dir: str = 'CAM2',#used in visualization
                wait_time: int =-1, #block program, enable interaction of the figure
                no_save_pred: bool = False,
                pred_out_dir: str = 'output/',
                **kwargs) -> Optional[dict]:
        #cam_type='CAM2'
        visualize_kwargs = {'show': show, 'no_save_vis': no_save_vis, 'img_out_dir': img_out_dir, 'cam_type_dir': cam_type_dir, 'wait_time': wait_time, 'return_vis': return_vis}
        postprocess_kwargs = {'no_save_pred': no_save_pred, 'pred_out_dir': pred_out_dir}

        if self.mode == "lidar":
            ori_inputs = self.lidar_inputs_to_list(inputs)
        else:
            ori_inputs = self.multi_inputs_to_list(inputs, cam_type=cam_type_dir) #cam_type='CAM2'
            #list of item dict, each dict has 'points' 'img' 'cam2img' 'lidar2cam' 'lidar2img'
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size)
        
        preds = []
        results_dict = {'predictions': [], 'visualization': []}
        for data in inputs:
        #'data_samples': [<Det3DDataSample] each contains 'cam2img' 'lidar2cam' 'lidar2img'
        #'inputs'->'points'(list of n,4) 'img' (lists of [CHW])
            with torch.no_grad():
                result = self.model.test_step(data)
            #preds.append(result)
            preds.extend(result) #[Det3DDataSample] each contains 'pred_instances_3d' 'cam2img' 'lidar2cam' 'lidar2img'
            visualization = self.visualize(ori_inputs, preds,
                                           **visualize_kwargs)
            results = self.postprocess(preds, visualization,
                                       return_datasamples,
                                       **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions']) # add to list
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict
    
    def lidar_inputs_to_list(self, inputs: Union[dict, list], **kwargs) -> list:
        """Preprocess the inputs to a list.
        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list:
              List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, dict) and isinstance(inputs['points'], str):
            pcd = inputs['points']
            backend = get_file_backend(pcd)
            if hasattr(backend, 'isdir') and isdir(pcd):
                # Backends like HttpsBackend do not implement `isdir`, so
                # only those backends that implement `isdir` could accept
                # the inputs as a directory
                filename_list = list_dir_or_file(pcd, list_dir=False)
                inputs = [{
                    'points': join_path(pcd, filename)
                } for filename in filename_list]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def multi_inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type: str = 'CAM2',
                        **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, dict):
            assert 'infos' in inputs
            infos = inputs.pop('infos')

            if isinstance(inputs['img'], str):
                img, pcd = inputs['img'], inputs['points']
                backend = get_file_backend(img)
                if hasattr(backend, 'isdir') and isdir(img) and isdir(pcd):
                    # Backends like HttpsBackend do not implement `isdir`, so
                    # only those backends that implement `isdir` could accept
                    # the inputs as a directory
                    img_filename_list = list_dir_or_file(
                        img, list_dir=False, suffix=['.png', '.jpg'])
                    pcd_filename_list = list_dir_or_file(
                        pcd, list_dir=False, suffix='.bin')
                    assert len(img_filename_list) == len(pcd_filename_list)

                    inputs = [{
                        'img': join_path(img, img_filename),
                        'points': join_path(pcd, pcd_filename)
                    } for pcd_filename, img_filename in zip(
                        pcd_filename_list, img_filename_list)]

            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # get cam2img, lidar2cam and lidar2img from infos
            info_list = mmengine.load(infos)['data_list']
            assert len(info_list) == len(inputs)
            for index, input in enumerate(inputs):
                data_info = info_list[index]
                img_path = data_info['images'][cam_type]['img_path']
                if isinstance(input['img'], str) and \
                        osp.basename(img_path) != osp.basename(input['img']):
                    raise ValueError(
                        f'the info file of {img_path} is not provided.')
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img
        elif isinstance(inputs, (list, tuple)):
            # get cam2img, lidar2cam and lidar2img from infos
            for input in inputs:
                assert 'infos' in input
                infos = input.pop('infos')
                info_list = mmengine.load(infos)['data_list']
                assert len(info_list) == 1, 'Only support single sample' \
                    'info in `.pkl`, when input is a list.'
                data_info = info_list[0]
                img_path = data_info['images'][cam_type]['img_path']
                if isinstance(input['img'], str) and \
                        osp.basename(img_path) != osp.basename(input['img']):
                    raise ValueError(
                        f'the info file of {img_path} is not provided.')
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'],
                    dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'],
                        dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                input['cam2img'] = cam2img
                input['lidar2cam'] = lidar2cam
                input['lidar2img'] = lidar2img

        return list(inputs)
    
    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(
            map(self.pipeline, inputs), batch_size)
        yield from map(self.collate_fn, chunked_data)
    
    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from dataset.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    processed_data = next(inputs_iter)
                    chunk_data.append(processed_data)
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = '',
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (np.ndarray, optional): Visualized predictions.
                Defaults to None.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
                Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_dir (str): Directory to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
              :meth:`forward` and processed in :meth:`postprocess`.
              If ``return_datasample=False``, it usually should be a
              json-serializable dict containing only basic data elements such
              as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ''

        result_dict = {}
        results = preds
        if not return_datasample:
            results = []
            for pred in preds:
                result = self.pred2dict(pred, pred_out_dir)#<Det3DDataSample->dict['bboxes_3d']
                results.append(result)
        elif pred_out_dir != '':
            print("not support")
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization
        return result_dict
    
    def pred2dict(self,
                  data_sample: Det3DDataSample,
                  pred_out_dir: str = '') -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if 'pred_instances_3d' in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                'labels_3d': pred_instances_3d.labels_3d.tolist(),
                'scores_3d': pred_instances_3d.scores_3d.tolist(),
                'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().tolist()
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

        if pred_out_dir != '':
            if 'lidar_path' in data_sample:
                lidar_path = osp.basename(data_sample.lidar_path)
                lidar_path = osp.splitext(lidar_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         lidar_path + '.json')
            elif 'img_path' in data_sample:
                img_path = osp.basename(data_sample.img_path)
                img_path = osp.splitext(img_path)[0]
                out_json_path = osp.join(pred_out_dir, 'preds',
                                         img_path + '.json')
            else:
                out_json_path = osp.join(
                    pred_out_dir, 'preds',
                    f'{str(self.num_visualized_imgs).zfill(8)}.json')
            dump(result, out_json_path)

        return result
    
    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = -1, #0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            no_save_vis (bool): Whether to save visualization results.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            #pred:
            points_input = single_input['points'] #.bin file
            if isinstance(points_input, str):
                pts_bytes = mmengine.fileio.get(points_input)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim] #(n,4)
                pc_name = osp.basename(points_input).split('.bin')[0]
                pc_name = f'{pc_name}.png'
            elif isinstance(points_input, np.ndarray):
                points = points_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f'{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(points_input)}')

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

            if self.mode == "multi": #adding image
                img_input = single_input['img']
                if isinstance(single_input['img'], str):
                    img_bytes = mmengine.fileio.get(img_input)
                    img = mmcv.imfrombytes(img_bytes)
                    img = img[:, :, ::-1]
                    img_name = osp.basename(img_input)
                elif isinstance(img_input, np.ndarray):
                    img = img_input.copy()
                    img_num = str(self.num_visualized_frames).zfill(8)
                    img_name = f'{img_num}.jpg'
                else:
                    raise ValueError('Unsupported input type: '
                                    f'{type(img_input)}')

                out_file = osp.join(img_out_dir, 'vis_camera', cam_type_dir,
                                    img_name) if img_out_dir != '' else None

                data_input = dict(points=points, img=img)
                vis_task = "multi-modality_det"
            else:
                data_input = dict(points=points)
                out_file = o3d_save_path
                vis_task = "lidar_det"

            ##vis_task='mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg', 'multi-modality_det'
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                out_file=out_file,
                vis_task=vis_task,
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results

#from mmdet3d.apis import MultiModalityDet3DInferencer
#from mmdet3d.apis import LidarDet3DInferencer
def test_MultiModalityDet3DInferencer(args):
    call_args = vars(args)
    mode = call_args.pop('mode')
    init_kws = ['model', 'weights', 'device']
    init_args = {}
    init_args['model'] = call_args.pop('config')
    init_args['weights'] = call_args.pop('checkpoint')
    init_args['device'] = call_args.pop('device')
    init_args['scope'] = 'mmdet3d'
    init_args['mode'] = mode

    # for init_kw in init_kws:
    #     init_args[init_kw] = in_args.pop(init_kw)
    #inferencer = MultiModalityDet3DInferencer(model=args.config, weights=args.checkpoint, device='cuda:0')
    print(init_args)
    #{'model': 'D:\\Developer\\mmdetection3d\\configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py', 'weights': 'D:\\Developer\\mmdetection3d\\modelzoo_mmdetection3d/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth', 'device': 'cuda:0'}
    #inferencer = MultiModalityDet3DInferencer(**init_args)
    inferencer = myInference(**init_args)
    #inferencer = LidarDet3DInferencer(**init_args)

    # call_args['inputs'] = dict(
    #     points=call_args.pop('pcd'),
    #     img=call_args.pop('img'),
    #     infos=call_args.pop('infos'))
    # call_args['no_save_vis'] = False
    # call_args['no_save_pred']= False
    # call_args['out_dir'] = 'output/'
    # call_args['show'] = True
    # call_args.pop('expname')
    # call_args.pop('mode')
    # call_args.pop('basefolder')
    # call_args.pop('snapshot')
    # call_args.pop('score_thr')
    

    input_args={}
    if mode=="multi":
        input_args['inputs']=dict(
            points=call_args.pop('pcd'),
            img=call_args.pop('img'),
            infos=call_args.pop('infos'))
        input_args['cam_type'] = 'CAM2'
    elif mode=='lidar':
        input_args['inputs'] = dict(points=call_args.pop('pcd'))
    input_args['pred_score_thr'] = 0.3
    input_args['no_save_vis'] = False
    input_args['no_save_pred']= False
    input_args['out_dir'] = 'output/'
    input_args['show'] = True
    input_args['wait_time'] = -1
    print(input_args)
    #{'cam_type': 'CAM2', 'out_dir': 'output/', 'show': True, 'inputs': {'points': 'D:\\Developer\\mmdetection3d\\demo/data/kitti/000008.bin', 'img': 'D:\\Developer\\mmdetection3d\\demo/data/kitti/000008.png', 'infos': 'D:\\Developer\\mmdetection3d\\demo/data/kitti/000008.pkl'}, 'no_save_vis': False, 'no_save_pred': False}
    result_dict = inferencer(**input_args)#__call__ in Base3DInferencer(BaseInferencer)
    return result_dict


def test_inference2(args, device='cuda:0'):
    # build the model from a config file and a checkpoint file
    #model = init_detector(args.config, args.checkpoint, device=args.device)
    #model = init_model(args.config, args.checkpoint, device=args.device) 
    ##https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py
    if isinstance(args.config, (str, Path)):
        config = Config.fromfile(args.config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet3d'))

    #build the model
    model = MODELS.build(config.model)

    checkpoint = args.checkpoint
    has_dataset_meta = True
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmdet3d 1.x
            model.dataset_meta = checkpoint['meta']['dataset_meta'] #contain 'classes'
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmdet3d 1.x
            classes = checkpoint['meta']['CLASSES']
            model.dataset_meta = {'classes': classes}
            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
        else:
            # < mmdet3d 1.x
            model.dataset_meta = {'classes': config.class_names}

            if 'PALETTE' in checkpoint.get('meta', {}):  # 3D Segmentor
                model.dataset_meta['palette'] = checkpoint['meta']['PALETTE']
   
        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None) ##contain 'classes'
        if cfg_palette is not None and has_dataset_meta:
            model.dataset_meta['palette'] = cfg_palette
    
    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)

    model.to(device)
    model.eval()

    # _init_pipeline(cfg) 
    #pipeline_cfg = cfg.test_dataloader.dataset.pipeline

    #https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/apis/inference.py
    if args.mode == "multi":
        result, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                     args.infos, args.cam_type)
    else:
        result, data = inference_detector(model, args.pcd)

    
    #visualizer=VISUALIZERS.build(config.visualizer) #Det3DLocalVisualizer
    visualizer= Det3DLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    points = data['inputs']['points'].cpu().numpy() #[59187, 4]
    if isinstance(result.img_path, list):
        img = []
        for img_path in result.img_path:#6 images
            single_img = mmcv.imread(img_path)
            single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
            img.append(single_img)
    else:
        img = mmcv.imread(result.img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
    # if isinstance(args.img, str):
    #     img_bytes = mmengine.fileio.get(args.img)
    #     img = mmcv.imfrombytes(img_bytes) #(375, 1242, 3)
    #     img = img[:, :, ::-1] #(375, 1242, 3) bgr to rgb

    data_input = dict(points=points, img=img)

    #vis_task='mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg', 'multi-modality_det'
    if args.mode == "multi":
        vis_task = 'multi-modality_det'
    else:
        vis_task = 'lidar_det'
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        show=True,
        draw_gt=False,
        wait_time=-1,
        # draw_pred=draw_pred,
        pred_score_thr=args.score_thr,
        # o3d_save_path=o3d_save_path,
        out_file=args.out_dir,
        vis_task=vis_task,
    )
    #visualizer.show()
    # set bev image in visualizer

    # _inputs_to_list
    result_dict = {}
    if 'pred_instances_3d' in result:
        pred_instances_3d = result.pred_instances_3d.numpy() #InstanceData
        #three keys: 'boxes_3d', 'scores_3d', 'labels_3d'
        result_dict = {
            'labels_3d': pred_instances_3d.labels_3d.tolist(), #11
            'scores_3d': pred_instances_3d.scores_3d.tolist(), #11
            'bboxes_3d': pred_instances_3d.bboxes_3d.tensor.cpu().numpy() #tolist() #11 len list, each 7 points
        }
    #pred = result['bboxes_3d']
    result_dict['points'] = points
    #np.save(os.path.join(args.out_dir, args.expname+'_vis.npy'), result_dict)

    visualizer.set_bev_image(bev_shape=100)
    # draw bev bboxes
    #gt_bboxes_3d = CameraInstance3DBoxes(result_dict['bboxes_3d'])
    gt_bboxes_3d = BaseInstance3DBoxes(result_dict['bboxes_3d'])
    visualizer.draw_bev_bboxes(gt_bboxes_3d, scale=1, edge_colors='orange')
    visualizer.show()

    print("end of test")

#'/data/cmpe249-fa22/WaymoKitti/4c_train5678/training/velodyne/008118.bin'
#demo/data/kitti/000008.bin
#demo/data/kitti/000008.png
#demo/data/kitti/000008.pkl #anno

#configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
#hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

#configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
#hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth

#configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
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
    
#scp -r 010796032@coe-hpc2.sjsu.edu:/data/rnd-liu/MyRepo/mmdetection3d/modelzoo_mmdetection3d/ .
        
#https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html
#https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars
#python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show

#python demo/multi_modality_demo.py demo/data/kitti/000008.bin demo/data/kitti/000008.png demo/data/kitti/000008.pkl configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py modelzoo_mmdetection3d/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth --cam-type CAM2 --show

#https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion
#python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py modelzoo_mmdetection3d/bevfusion_converted.pth --cam-type all --score-thr 0.2 --show
    
#https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
#https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

#python projects/BEVFusion/demo/multi_modality_demo.py \
#    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin 
#    demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
#    modelzoo_mmdetection3d/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth --cam-type all --score-thr 0.2 --show

import numpy as np
from mmengine import load

#from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes, BaseInstance3DBoxes
def test_BEV(args):
    info_file = os.path.join(args.basefolder, 'mmdetection3d', args.infos)
    info_file = load(info_file) #('demo/data/kitti/000008.pkl')
    bboxes_3d = []
    for instance in info_file['data_list'][0]['instances']:
        bboxes_3d.append(instance['bbox_3d'])
    gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32) #list to np array (10,7)
    #gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)
    gt_bboxes_3d = BaseInstance3DBoxes(gt_bboxes_3d)
    

    visualizer = Det3DLocalVisualizer()
    # set bev image in visualizer
    visualizer.set_bev_image(bev_shape=100)
    # draw bev bboxes
    visualizer.draw_bev_bboxes(gt_bboxes_3d, scale=1, edge_colors='orange')
    visualizer.show()

import pickle
def check_annofile(args):
    with open(args.infos, 'rb') as f:
        data = pickle.load(f)
        print(data[0].keys())

def getindividualpkl_frominfospkl(infos_pklfile, save_path, use_num=50):
    with open(infos_pklfile, 'rb') as f:
        data = pickle.load(f)
    print("Data len:", len(data))
    print(data[0]['image']['image_path'].split('/')[-1].split('.')[0])#get filename
    for i in range(0, use_num, 1):
        list=[]
        list.append(data[i])
        cur_data = list
        file_name = data[i]['image']['image_path'].split('/')[-1].split('.')[0]
        with open(save_path+file_name+'_infos.pkl', 'wb') as f:
            pickle.dump(cur_data, f)

from mmdet3d.datasets import build_dataset
from mmdet3d.core.visualizer import show_multi_modality_result
def test_dataset(args, model, pkl_dir, out_dir):
    if isinstance(args.config, (str, Path)):
        config = Config.fromfile(args.config)
    
    test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
    dataset =DATASETS.build(test_dataset_cfg)

    dataset2 = [build_dataset(config.data.train)]
    print(dataset2[0][0].keys())

    num = 50
    for i in range(0, num, 1):
        cur_data = dataset[0][i]
        img_metas = cur_data.get('img_metas').data
        pts_file = img_metas.get('pts_filename')
        img = cur_data.get('img').data
        img_file_path = img_metas.get('filename')
        name = img_file_path.split('/')[-1] .split('.')[0]
        ann_file = pkl_dir + name + '_infos.pkl'
        project_mat =img_metas.get('lidar2img')#get projection matrix
        result, data = inference_multi_modality_detector(model, pts_file, img_file_path, ann_file)
        bboxes_data = result[0]['pts_bbox']['boxes_3d']
        show_multi_modality_result(img=img, box_mode='lidar', gt_bboxes=None, img_metas=img_metas, \
                                   pred_bboxes=bboxes_data, proj_mat=project_mat, out_dir=out_dir, filename=name, show=True)



def main():
    parser = ArgumentParser()
    parser.add_argument('--expname',  type=str, default='test')
    parser.add_argument('--mode',  type=str, default='multi') #multi, lidar
    parser.add_argument('--basefolder', type=str, default='/home/lkk/Developer/') # r'D:\Developer'  '/data/rnd-liu/MyRepo/mmdetection3d/'
    parser.add_argument('--pcd',  type=str, default='demo/data/kitti/000008.bin', help='Point cloud file')#
    parser.add_argument('--config', type=str, default='configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='modelzoo_mmdetection3d/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth', help='Checkpoint file')
    parser.add_argument('--img', type=str, default='demo/data/kitti/000008.png')
    parser.add_argument('--infos', type=str, default='demo/data/kitti/000008.pkl')
    # parser.add_argument('--pcd',  type=str, default='demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin', help='Point cloud file')#
    # parser.add_argument('--config', type=str, default='projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py', help='Config file')
    # parser.add_argument('--checkpoint', type=str, default='modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth', help='Checkpoint file')
    # parser.add_argument('--img', type=str, default='demo/data/nuscenes/')
    # parser.add_argument('--infos', type=str, default='demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM2', #'all' 'CAM2', #'CAM_FRONT',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
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

    args.pcd = os.path.join(args.basefolder, 'mmdetection3d', args.pcd)
    args.config = os.path.join(args.basefolder, 'mmdetection3d', args.config)
    args.checkpoint = os.path.join(args.basefolder, 'mmdetection3d', args.checkpoint)
    args.img = os.path.join(args.basefolder, 'mmdetection3d', args.img)
    args.infos = os.path.join(args.basefolder, 'mmdetection3d', args.infos)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    test_BEV(args)

    #test_MultiModalityDet3DInferencer(args) #BEVFusion has pipeline problem

    test_inference2(args, device=args.device) #BEVFusion works

    test_inference(args)

if __name__ == '__main__':
    main()