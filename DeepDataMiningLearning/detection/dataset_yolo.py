import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
from glob import glob
import os
import math
import itertools
import torch.utils.data as data
from copy import deepcopy
from pathlib import Path
from typing import Optional
from multiprocessing.pool import ThreadPool
import glob
import math
import cv2
import numpy as np
import psutil
from itertools import repeat
import contextlib
from tqdm import tqdm
from DeepDataMiningLearning.detection.modules.yolotransform import LetterBox
# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
DATASET_CACHE_VERSION = '1.0.3'

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def load_dataset_cache_file(path):
    """Load an yolo *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache

def save_dataset_cache_file(path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if os.access(str(path.parent), os.W_OK): #is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        #path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        print(f'New cache created: {path}')
    else:
        print(f'WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')

def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == 'JPEG':  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s

def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y

def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    #ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    #msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
                    print('ignore corrupt JPEG files')

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb[:, 1:] <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]

cococlassdict={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
INSTANCE_CATEGORY_NAMES = ['__background__','Vehicles', 'Pedestrians', 'Cyclists', 'Signs']
cococlasses=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class YOLODataset(torch.utils.data.Dataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.
    In yolo, a bounding box is represented by four values [x_center, y_center, width, height]. x_center and y_center are the normalized coordinates of the center of the bounding box. 

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """
    def __init__(self, root, annotation, train=True, transform=None, data=None, imgsz=640, format='yolo', use_segments=False, use_keypoints=False, **kwargs):
        #super().__init__()
        self.root = root #/data/cmpe249-fa23/coco/
        if train:
            foldername='train2017'
        else:
            foldername='val2017'
        self.cache_path = os.path.join(root, foldername+'.cache.npy')
        self.img_path = annotation #os.path.join(root, foldername, 'images') #contains test2017(all.jpg fiels) train2017 val2017(all.jpg)
        #self.annotation = annotation
        self.imgsz = imgsz
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.classes=data['names']
        #print(data['names'])
        classlist = list(data['names'].values())
        #print(classlist)
        self.numclass = len(classlist)
        self.transform = transform #not used
        self.letterbox = LetterBox((640, 640), auto=False, stride=32) #only support cv2
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        #super().__init__(*args, **kwargs)
        self.fraction = 1 # a fraction of all image files
        self.prefix = 'yolo'  # Set prefix before calling get_labels
        self.im_files = self.get_img_files(self.img_path) #all image files list
        self.labels = self.get_labels(Path(self.cache_path)) #list of dicts
        # if classes:
        #     self.single_cls = False
        #     self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = False
        self.augment = False
        self.stride =32
        self.fp16 = False
        #did not use image cache, these code can be reduced
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        self.format=format #yolo uses normalized xywh format, coco uses [xmin, ymin, width height], 
                            #torchvision format is [xmin, ymin, xmax, ymax]
    
    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'No images found in {img_path}'
        except Exception as e:
            raise FileNotFoundError(f'Error loading data from {img_path}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files
    
    def get_labels(self, cache_path): #new add cache_path
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files) #replace image path to labels path, point to txt label file
        #cache_path = Path(self.label_files[0]).parent.with_suffix('.cache') #'/data/cmpe249-fa23/coco/labels/train2017.cache'
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            #assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            #assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            print('cache not found, create a new cache labels')
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            #TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                print('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('version', 'msgs')]  # remove items
        labels = cache['labels'] #list of dicts, 
        if not labels:
            print(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly.')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            print(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            print(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly.')
        return labels
    
    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'Scanning {path.parent / path.stem}...'
        total = len(self.im_files) #118287
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(total=total, desc=desc)#TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in results: #pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.update(1)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            print('\n'.join(msgs))
        if nf == 0:
            print(f'{self.prefix}WARNING ⚠️ No labels found in {path}.')
        #x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files) #nf =117266 imges, nm = 1021 backgrounds
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(path, x)
        return x

    # def update_labels(self, include_class: Optional[list]):
    #     """include_class, filter labels to include only these classes (optional)."""
    #     include_class_array = np.array(include_class).reshape(1, -1)
    #     for i in range(len(self.labels)):
    #         if include_class is not None:
    #             cls = self.labels[i]['cls']
    #             bboxes = self.labels[i]['bboxes']
    #             segments = self.labels[i]['segments']
    #             keypoints = self.labels[i]['keypoints']
    #             j = (cls == include_class_array).any(1)
    #             self.labels[i]['cls'] = cls[j]
    #             self.labels[i]['bboxes'] = bboxes[j]
    #             if segments:
    #                 self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
    #             if keypoints is not None:
    #                 self.labels[i]['keypoints'] = keypoints[j]
    #         if self.single_cls:
    #             self.labels[i]['cls'][:, 0] = 0


    #ultralytics\data\base.py
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        imageandlabel = self.get_image_and_label(index) #image in label['img']
        img=imageandlabel['img'] #(480, 640, 3)
        #imgh, imgw = img.shape[:2]
        originalimgshape = img.shape

        imageandlabel = self.letterbox(labels=imageandlabel, image=img) #do letterbox here
        #change box from normalized xcycwh, to unnormalized xmin, ymin, xmax, ymax
        img=imageandlabel.pop('img')
        imgh_letterbox, imgw_leterbox = img.shape[:2]

        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1]) #BGR to RGB, HWC to CHW, (3, h, w)
        img = torch.from_numpy(img) #torch.Size([3, 480, 640])
        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        
        # if self.transform:
        #     transferred=self.transforms(imageandlabel)
        cls = imageandlabel.pop('cls') #(8, 1) array
        nl = len(cls)
        cls = np.squeeze(cls, axis=1) #(8,1) to array(8,)
        bbox = imageandlabel.pop('bboxes') #(8,4) array
        #bbox_format = imageandlabel.pop('bbox_format') #xywh
        #normalized = imageandlabel.pop('normalized') 
        
        
        # target['labels']=torch.from_numpy(cls) if nl else torch.zeros(nl)
        # target['boxes']=torch.from_numpy(bbox) if nl else torch.zeros((nl, 4))
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []
        for i in range(nl):
            ##change box from normalized xcycwh, to unnormalized xmin, ymin, xmax, ymax
            xmin,ymin,xmax,ymax = bbox[i] #0-1 normalized value
            w = xmax - xmin
            h = ymax - ymin
            xc=xmin+w/2
            yc=ymin+h/2
            if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0:
                if self.format=='yolo': #normalized xcenter, ycenter, width, height
                    target_bbox.append([xc/imgw_leterbox, yc/imgh_letterbox, w/imgw_leterbox, h/imgh_letterbox])
                    #target_bbox.append([xc, yc, w, h])
                    target_labels.append(int(cls[i]))
                else:
                    target_bbox.append([xmin, ymin, xmax, ymax]) #torchvison format is xmin, ymin, xmax, ymax
                    target_labels.append(int(cls[i])+1) #0 means background
                target_crowds.append(0)
                area=w*h #w*imgw*h*imgh
                target_areas.append(area)
        nl=len(target_bbox)
        target = {}
        #self.format = 'coco'
        if self.format=='yolo':
            target['img']=img #CHW
            target['bboxes'] = torch.as_tensor(target_bbox, dtype=torch.float32) if nl else torch.zeros((nl, 4))
            target['cls'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64) if nl else torch.zeros(nl)
            target['batch_idx'] = torch.zeros(nl) #new added in yolo
            target['orig_shape'] = originalimgshape #torch.as_tensor(originalimgshape, dtype=torch.int64)
            target['image_id'] = int(index)
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32) if nl else torch.zeros(nl)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64) if nl else torch.zeros(nl)
            return target #dict
        else:
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32) if nl else torch.zeros((nl, 4))
            # Labels int value for class
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64) if nl else torch.zeros(nl)
            target['image_id'] = int(index)
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32) if nl else torch.zeros(nl)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64) if nl else torch.zeros(nl)
            target['batch_idx'] = torch.zeros(nl) #new added in yolo
            target['orig_shape'] = originalimgshape # torch.as_tensor(originalimgshape, dtype=torch.int64)
            return img, target
    
    #ultralytics\data\base.py
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        # if self.rect:
        #     label['rect_shape'] = self.batch_shapes[self.batch[index]]
        #return self.update_labels_info(label)
        return label
    
    # def update_labels_info(self, label):
    #     """custom your label format here."""
    #     # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
    #     # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
    #     bboxes = label.pop('bboxes')
    #     segments = label.pop('segments')
    #     keypoints = label.pop('keypoints', None)
    #     bbox_format = label.pop('bbox_format')
    #     normalized = label.pop('normalized')
    #     #label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    #     #in ultralytics\utils\instance.py
    #     return label
    
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
    
    def __len__(self):
        return len(self.labels)

import yaml
if __name__ == "__main__":
    root='/data/cmpe249-fa23/coco/'
    annotation='/data/cmpe249-fa23/coco/train2017.txt'
    dataset_cfgfile = './DeepDataMiningLearning/detection/dataset.yaml'
    with open(dataset_cfgfile, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        data['yaml_file'] = str(dataset_cfgfile)
        data['kpt_shape'] = [17, 3] #for keypoint
    yolodataset = YOLODataset(root=root, annotation=annotation, train=True, transform=None, data=data,classes=None,use_segments=False,use_keypoints=False)
    target=yolodataset[1]
    print(target['img'].shape) #torch.Size([3, 480, 640])
    print(target.keys()) #dict_keys(['boxes', 'labels', 'image_id', 'area', 'iscrowd', 'batch_idx'])
    #image, targets = iter(next(yolodataset))
