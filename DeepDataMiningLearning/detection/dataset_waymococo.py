#ref: https://github.com/lkk688/WaymoObjectDetection/blob/master/MyDetector/torchvision_waymococo_train.py
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
from pycocotools.coco import COCO

class WaymoCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))#

        #
        dataset=self.coco.dataset
        imgToAnns=self.coco.imgToAnns
        catToImgs =self.coco.catToImgs
        cats=self.coco.cats

    
    def _get_target(self, id):
        'Get annotations for sample'

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=id)
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        imginfo=self.coco.imgs[img_id]
        path = imginfo['file_name']
        #print(f'index: {index}, img_id:{img_id}, info: {imginfo}')

        # path for input image
        #loadedimglist=coco.loadImgs(img_id)
        # print(loadedimglist)
        #path = coco.loadImgs(img_id)[0]['file_name']
        #print("image path:", path)
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        #img = Image.open(os.path.join(self.root, path)).convert('RGB')


        # List: get annotation id from coco
        #ann_ids = coco.getAnnIds(imgIds=img_id)
        annolist=[self.coco.imgToAnns[img_id]]
        anns = list(itertools.chain.from_iterable(annolist))
        ann_ids = [ann['id'] for ann in anns]
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        targets  = coco.loadAnns(ann_ids)
        #targets=self.anns[ann_ids]
        #print("targets:", targets)
        
        #image_id = targets["image_id"].item()

        # number of objects in the image
        num_objs = len(targets)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []
        for i in range(num_objs):
            xmin = targets[i]['bbox'][0]
            ymin = targets[i]['bbox'][1]
            width=targets[i]['bbox'][2]
            xmax = xmin + width
            height = targets[i]['bbox'][3]
            ymax = ymin + height
            if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and width>1 and height>1:
                target_bbox.append([xmin, ymin, xmax, ymax])
                target_labels.append(targets[i]['category_id'])
                target_crowds.append(targets[i]['iscrowd'])
                target_areas.append(targets[i]['area'])
        num_objs=len(target_bbox)
        #print("target_bbox len:", num_objs)
        if num_objs>0:
            #print("target_labels:", target_labels)
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
            # Labels int value for class
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
            target['image_id'] = torch.tensor([int(img_id)])
            #torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#torch.zeros((len(target['boxes'])), dtype=torch.int64)
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)#empty
            target['image_id'] = torch.tensor([int(img_id)])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty

        if self.transforms is not None:
            img = self.transforms(img)
        #print("target:", target)
        return img, target

    def __len__(self):
        return len(self.ids)