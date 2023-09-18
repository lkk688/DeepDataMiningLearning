import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from DeepDataMiningLearning.detection.coco_utils import get_coco
from DeepDataMiningLearning.detection import presets
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import csv
from DeepDataMiningLearning.detection.dataset_kitti import KittiDataset

WrapNewDict = False

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))


def get_dataset(datasetname, is_train, is_val, args):
    if datasetname.lower() == 'coco':
        ds, num_classes = get_cocodataset(is_train, is_val, args)
    elif datasetname.lower() == 'kitti':
        ds, num_classes = get_kittidataset(is_train, is_val, args)
    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)
    
def get_cocodataset(is_train, is_val, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes

class MyKittiDetection(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2"):
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        self._location = "training" if self.train else "testing"
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir
        # load all image files, sorting them to
        # ensure that they are aligned
        image_dir = os.path.join(self.root, "raw", self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self.root, "raw", self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
        #self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.INSTANCE_CATEGORY_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.INSTANCE2id = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7, 'Misc':8, 'DontCare':9} #background is 0
        self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        self.numclass = 9 #including background, excluding the 'DontCare'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
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
        #img, target = super().__getitem__(idx)
        if index>len(self.images):
            print("Index out-of-range")
            image = None
        else:
            image = Image.open(self.images[index])
            target, image_id = self._parse_target(index) if self.train else None

        if WrapNewDict:
            target = dict(image_id=image_id, annotations=target) #new changes
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def _parse_target(self, index: int) -> List:
        target = []
        labelfile = self.targets[index]
        full_name = os.path.basename(labelfile)
        file_name = os.path.splitext(full_name)
        imageidx=int(file_name[0]) #filename index 000001
        #img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #assert img_file.exists()

        with open(labelfile) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": imageidx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
            #Convert to the required format by Torch
            num_objs = len(target)
            boxes = []
            labels = []
            for i in range(num_objs):
                bbox = target[i]['bbox']
                xmin = bbox[0]
                xmax = bbox[2]
                ymin = bbox[1]
                ymax = bbox[3]
                objecttype=target[i]['type']
                #if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            num_objs = len(labels) #update num_objs
            newtarget = {}
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            if num_objs >0:
                newtarget["boxes"] = boxes
                newtarget["labels"] = labels
                #newtarget["masks"] = masks
                newtarget["image_id"] = image_id
                newtarget["area"] = area
                newtarget["iscrowd"] = iscrowd
            else:
                #negative example, ref: https://github.com/pytorch/vision/issues/2144
                newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
                target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
                target['image_id'] =image_id
                target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
                target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget, imageidx

    def __len__(self) -> int:
        return len(self.images)
    
def get_kittidataset(is_train, is_val, args):
    rootPath=args.data_path
    #dataset = MyKittiDetection(rootPath, train=True, transform=get_transform(is_train, args))
    if is_val == True:
        transformfunc=get_transform(False, args)
        dataset = KittiDataset(rootPath, train=True, split='val', transform=transformfunc)
    else:
        transformfunc=get_transform(True, args) #add augumentation
        dataset = KittiDataset(rootPath, train=is_train, split='train', transform=transformfunc)
    
    num_classes = dataset.numclass
    return dataset, num_classes
    #mykitti = datasets.Kitti(root=rootPath, train= True, transform = get_transform(is_train, args), target_transform = None, download = False)

def countobjects(alltypes):
    counter = {}
    for type in alltypes:
        if type not in counter:
            counter[type] = 0
        counter[type] += 1
    return counter

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    import DeepDataMiningLearning.detection.transforms as T
    def get_transformsimple(train):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ToDtype(torch.float, scale=True))
        # if train:
        #     transforms.append(RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
    
    class args:
        data_augmentation = 'hflip'
        backend = 'PIL'
        use_v2 = False
        weights = ''
        test_only = False
    
    rootPath = '/data/cmpe249-fa23/torchvisiondata/Kitti'
    is_train = True
    kittidataset = MyKittiDetection(rootPath, train=True, transform=get_transform(is_train, args))
    print(kittidataset.INSTANCE_CATEGORY_NAMES)
    print("Dataset len:", len(kittidataset))
    img, target = kittidataset[0]
    print(img.shape) #CHW format
    print(target.keys()) #['boxes', 'labels', 'image_id', 'area', 'iscrowd']
    print(target['boxes'].shape)  #torch.Size([3, 4]) n,4
    imgdata=img.permute(1, 2, 0) #CHW -> HWC
    #plt.imshow(imgdata)

    # labels in training set
    if WrapNewDict:
        alltargets = [target['annotations'] for _, target in kittidataset]
        print(target['annotations'][0].keys()) #['type', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y']
    else:
        alltargets = [target['labels'] for _, target in kittidataset]
    # typelist=[]
    # for target in alltargets: #each target is [N,4]
    #     for object in target: #list
    #         typelist.append(object['type'])
    # #alltypes = [target['type'] for target in alltargets]
    # counter=countobjects(typelist)
    # print(counter)
    #{'Cyclist': 1627, 'Pedestrian': 4487, 'Car': 28742, 'DontCare': 11295, 'Van': 2914, 'Misc': 973, 'Truck': 1094, 'Tram': 511, 'Person_sitting': 222}
