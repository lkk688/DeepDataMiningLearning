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
    def __init__(self, root, annotation, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.coco = COCO(annotation)
        self.is_train = train
        self.ids = list(sorted(self.coco.imgs.keys()))#id string list

        #
        dataset=self.coco.dataset #'images': image filename (images/xxx.jpg) with image_id (0000001)
        imgToAnns=self.coco.imgToAnns #image_id to list of annotations
        catToImgs =self.coco.catToImgs #three classes, 1,2,4
        cats=self.coco.cats
        self.numclass = 5 #len(catToImgs) + 1 #three classes + background
        #num_classes=5 # ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
        #previous_num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
        #Real data only has 
        self.INSTANCE_CATEGORY_NAMES = ['__background__','Vehicles', 'Pedestrians', 'Cyclists', 'Signs']
        #self.INSTANCE2id = {'__background__':0, 'Vehicles': 1, 'Pedestrians': 2, 'Cyclists': 4} #background is 0
        #self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        #In annotation, class is 1,2,4

    
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
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            #torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#torch.zeros((len(target['boxes'])), dtype=torch.int64)
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)#empty
            #target['image_id'] = torch.tensor([int(img_id)])
            #target['image_id'] = torch.tensor(int(img_id))
            target['image_id'] = int(img_id)
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty

        # if self.transforms is not None:
        #     img = self.transforms(img)
        if self.transform:
            img, target = self.transform(img, target)
        #print("target:", target)
        return img, target

    def __len__(self):
        return len(self.ids)

import DeepDataMiningLearning.detection.transforms as T
def get_transformsimple(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from pathlib import Path
if __name__ == "__main__":
    # path to your own data and coco file
    data_root = '/data/cmpe249-fa23/WaymoCOCO/'

    #jsonfile = 'annotations_train684step8allobject.json' #4 classes, 0,1,2,4 85008
    jsonfile = 'annotations_valallnew.json' #4 classes, 0,1,2,4 199935
    #jsonfile = 'annotations_trainallnew.json' #4 classes, 0,1,2,4 677530
    #jsonfile = '3classsub_annotations_trainall.json' #4 classes, 0,1,2,4 20820
    ann_file = os.path.join(data_root, jsonfile)#'annotations_train20new.json'
    # create own Dataset
    mywaymodataset = WaymoCOCODataset(root=data_root,  
                          annotation=ann_file,
                          transforms=get_transformsimple(False)
                          )
    length = len(mywaymodataset)
    print("Dataset",len(mywaymodataset))#85008
    img, target = mywaymodataset[0]
    print(target.keys())