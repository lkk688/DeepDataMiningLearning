import random
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple
import os
import numpy as np
import json
from PIL import Image
from pathlib import Path

# Nuscenes dataset class
# This class is used to load nuScenes dataset.
# Ref: https://www.nuscenes.org/
#
# Mini dataset for development
# metadata and image -
# https://motional-nuscenes.s3-ap-northeast-1.amazonaws.com/public/nuimages-v1.0/nuimages-v1.0-mini.tgz
# Full dataset for training
# metdata - https://motional-nuscenes.s3-ap-northeast-1.amazonaws.com/public/nuimages-v1.0/nuimages-v1.0-all-metadata.tgz
# images - https://motional-nuscenes.s3-ap-northeast-1.amazonaws.com/public/nuimages-v1.0/nuimages-v1.0-all-samples.tgz
# Extract all files in <data_dir>
class NuscenesDataset(Dataset):
    ## Nuscenes dataset class initialization
    # Args:
    # data_dir: str: Path to the data directory
    # train: bool: True if training dataset, False otherwise
    # sample_data_file: str: Name of the sample_data.json file
    # transform: Optional[Callable]: Transform to apply to the image
    def __init__(self, 
                 data_dir: str,
                 metadata_subdir: str = 'v1.0-mini',
                 train: bool = True,
                 transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.metadata_dir = os.path.join(data_dir, metadata_subdir)

        # Read categories
        self.read_categories()
        self.numclass = len(self.INSTANCE_CATEGORY_NAMES)

        # Read annotations
        self.read_annotations()

        # Read sample_data.json
        self.sample_data_list = []
        sample_data_path = os.path.join(self.metadata_dir, 'sample_data.json')
        with open(sample_data_path, 'r') as f:
            samples_list = json.load(f)
            # only keep files for which annotations are available
            i = -1
            for sample in samples_list:
                token = sample['token']
                if token in self.annotations:
                    i = i + 1
                    self.sample_data_list.append({
                        'image_id': i,
                        'filename': sample['filename'],
                        'token': sample['token']
                    })
        

    def __len__(self):
        return len(self.sample_data_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            idx (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        if idx>len(self.sample_data_list):
            print("Index out-of-range")
            image = None
        else:
            sample_data = self.sample_data_list[idx]
            image_file = sample_data['filename']
            image_path = Path(os.path.join(self.data_dir, image_file))
            if not image_path.exists():
                raise FileNotFoundError(f"Image file {image_path} does not exist")
            image = Image.open(image_path)
            if self.train:
                target = self.get_label(idx) #list of dicts

        if self.transform:
            image, target = self.transform(image, target)
        return image, target
    
    # read annotation files
    # In nuScene dataset, each image has a unique token and corresponding annotations
    # are stored in object_ann.json file. 
    # Token in sample_data.json maps to sample_data_token in object_ann.json
    def read_annotations(self):
        anns = {}
        with open(os.path.join(self.metadata_dir, 'object_ann.json'), 'r') as f:
            ann_list = json.load(f)
            for ann in ann_list:
                image_key = ann['sample_data_token']
                image_ann_list = []
                # check if anns has the key 
                if image_key not in anns:
                    anns[image_key] = image_ann_list
                else:
                    image_ann_list = anns[image_key]

                image_ann_list.append({
                    'bbox': ann['bbox'],
                    'category_token': ann['category_token']
                })
        self.annotations = anns
        return

    # Read category.jso and create a list of category names
    # nuScene dataset has token (unique id) for each category, create a map of token
    # to index in the list of category names
    def read_categories(self):
        categories = []
        category_token_idx_map = {}
        category_names = []

        with open(os.path.join(self.metadata_dir, 'category.json'), 'r') as f:
            categories = json.load(f)
        # populate category_token_id_map and category_names
        for i, category in enumerate(categories):
            category_token_idx_map[category['token']] = i
            category_names.append(category['name'])

        self.INSTANCE_CATEGORY_NAMES = category_names
        self.INSTANCE_CATEGORY_TOKEN2IDX = category_token_idx_map
        return
    
    # Get label for image at a given index
    # Returns a dictionary with keys: boxes, labels, image_id, area, iscrowd
    # Values are tensors for each key
    def get_label(self, idx: int) -> dict:
        annotations = self.annotations.get(self.sample_data_list[idx]['token'], [])
        boxes = []
        labels = []
        for ann in annotations:
            category_idx = self.INSTANCE_CATEGORY_TOKEN2IDX[ann['category_token']]
            boxes.append(ann['bbox'])
            labels.append(category_idx)

        num_objs = len(labels) #update num_objs
        newtarget = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        if num_objs >0:
            newtarget["boxes"] = boxes
            newtarget["labels"] = labels
            #newtarget["masks"] = masks
            newtarget["image_id"] = idx
            newtarget["area"] = area
            newtarget["iscrowd"] = iscrowd
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32) #not empty
        return newtarget
    

# Test the dataset
# It loads the dataset and modify a random image with bounding boxes and corresponding
# labels. Modified image is saved to disk (<data_dir>/out.png) for verification.
#
# This demonstrates that the dataset is loaded properly and can be used for training
if __name__ == "__main__":
    data_dir = '/home/nadeem/sjsu/data/nuScenes/'

    # sample data file
    # train
    metadata_subdir='v1.0-train'
    # validation
    metadata_subdir='v1.0-val'
    # mini
    metadata_subdir='v1.0-mini'

    import DeepDataMiningLearning.detection.transforms as T
    def get_transformsimple():
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ToDtype(torch.float, scale=True))
        return T.Compose(transforms)

    ds = NuscenesDataset(data_dir=data_dir, metadata_subdir=metadata_subdir, train=True, transform=get_transformsimple())
    print("Size of dataset", len(ds))

    test_idx = random.randint(0, len(ds) - 1)
    img, target = ds[test_idx]

    print(img.size) 
    print(target.keys()) #['boxes', 'labels', 'image_id', 'area', 'iscrowd']
    boxes = target['boxes']
    labels = target['labels']
    print(boxes.shape)  #torch.Size([3, 4]) n,4
    print(labels)
    labels = [ds.INSTANCE_CATEGORY_NAMES[int(label)] for label in labels]
    print(labels)

    # Draw bounding box
    import torchvision
    img = torchvision.utils.draw_bounding_boxes(img, boxes, labels, width=2, colors="yellow",  fill=False) 
    
    img = torchvision.transforms.ToPILImage()(img) 

    # save output 
    img.save(os.path.join(data_dir, "out.png")) 
