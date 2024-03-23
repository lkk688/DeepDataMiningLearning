#ref: https://github.com/lkk688/myyolov8/blob/main/ultralytics/data/augment.py
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import nn, Tensor
from DeepDataMiningLearning.detection.modules.utils import LOGGER, make_divisible, yolov8_non_max_suppression, non_max_suppression, scale_boxes, xyxy2xywh, xywh2xyxy

# same_shapes = all(x.shape == im[0].shape for x in im)
# [self.letterbox(image=x) for x in im]
class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        #if auto is True, minimum rectangle without fill

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width] 426, 640
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #(640, 426)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding dh=214
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides, 0
            dh /= 2 #107

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1)) #107, 107
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1)) #0, 0
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))  # for evaluation (0,107)

        if len(labels):
            #yolo uses normalized xywh format
            #labels = self._update_labels(labels, ratio, dw, dh)
            #normalized xcyc to xmin, ymin, xmax, ymax
            labels['bboxes']=xywh2xyxy(labels['bboxes'])
            #denormalize bboxes
            origh=shape[0]
            origw=shape[1]
            labels['bboxes']=self.scale_box(labels['bboxes'], (origw, origh, origw, origh))
            labels['normalized']=False
            rw, rh = ratio  # width, height ratios
            labels['bboxes']=self.scale_box(labels['bboxes'], (rw, rh, rw, rh))
            labels['bboxes'][:, 0] += dw #xmin+xoffset(dw)
            labels['bboxes'][:, 1] += dh #ymin+yoffset
            labels['bboxes'][:, 2] += dw #xmax+xoffset
            labels['bboxes'][:, 3] += dh #ymax+yoffset
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img
    
    def scale_box(self, bboxes, scale):#box is normalized xmin, ymin, w, h
        assert len(scale) == 4
        bboxes[:, 0] *= scale[0] #w
        bboxes[:, 1] *= scale[1] #h
        bboxes[:, 2] *= scale[2] #h
        bboxes[:, 3] *= scale[3] #w
        return bboxes

    #new added from ultralytics\utils\instance.py    
    # def _update_labels(self, labels, ratio, padw, padh):
    #     """Update labels."""
    #     labels['instances'].convert_bbox(format='xyxy')
    #     labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
    #     labels['instances'].scale(*ratio)
    #     labels['instances'].add_padding(padw, padh)
    #     return labels

class YoloTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a model.

    The transformations it performs are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        device: str = 'cuda:0',
        fp16: bool = False,
        cfgs: dict = None,
        size_divisible: int = 32,
    ):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.letterbox = LetterBox((640, 640), auto=True, stride=size_divisible) #only support cv2
        #if auto is True, the size of the image is not square, if auto=False, the size of the image is square.

        self.device = device
        self.fp16 = fp16
        #self.original_image_sizes: List[Tuple[int, int]] = []
        #self.newimage_size = [640, 640]
                #set parameters
        self.imgsz = 640
        #self.detcttransform = detcttransform
        self.nms_conf = cfgs['conf'] if cfgs['conf'] else 0.25
        self.nms_iou = cfgs['iou'] if cfgs['iou'] else 0.45
        self.nms_agnostic = cfgs['agnostic_nms'] if cfgs['agnostic_nms'] else False
        self.nms_max_det = cfgs['max_det'] if cfgs['max_det'] else 300
        self.classes = cfgs['classes']
    
    def forward(self, images, targets=None):
        #cv2 image list [(1080, 810, 3)]
        if isinstance(images, list): #inference mode
            # for img in images:
            #     val = img.shape[0:2]#img.shape[-2:]
            #     torch._assert(
            #         len(val) == 2,
            #         f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            #     )
            #     self.original_image_sizes.append((val[0], val[1]))
            images=[self.letterbox(image=x) for x in images] #list of (640, 480, 3)
            #if self.detcttransform:
                #imageslist, targets = self.detcttransform(images, targets)
                #images = imageslist.tensors
                #images=[self.detcttransform(image=x) for x in images] #letterbox
            #self.newimage_size = images[0].shape[0:2] #(640, 480, 3)
            images = self.pre_processing(images)
            return images #tensor output
        elif isinstance(images, torch.Tensor):
            images = self.pre_processing(images)
            return images
        else: #single image input, numpy array
            val = images.shape[-2:] #HWC to CHW format
            #self.original_image_sizes.append((val[0], val[1]))
            images=[self.letterbox(image=images)]
            images = self.pre_processing(images)
            return images #tensor output BCHW
    
    def pre_processing(self, im):
        """Prepares input image before inference.
        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        #ref: https://github.com/lkk688/myyolov8/blob/main/ultralytics/engine/predictor.py
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:#im is image list
            im = np.stack(im) #image list
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    #ref: https://github.com/lkk688/myyolov8/blob/main/ultralytics/models/yolo/detect/predict.py
    def postprocess(self, preds, newimagesize, origimageshapes):
        """Post-processes predictions and returns a list of Results objects."""

        if isinstance(newimagesize, torch.Tensor):
            newimagesize = newimagesize.detach().cpu().numpy()
        if isinstance(origimageshapes, torch.Tensor):
            origimageshapes = origimageshapes.detach().cpu().numpy()
        
        #y,x output, training mode, direct output x (three items), inference mode: y,x output
        #non_max_suppression only use the detection results (y)
        preds = yolov8_non_max_suppression(preds,
                                        self.nms_conf,
                                        self.nms_iou,
                                        agnostic=self.nms_agnostic,
                                        classes=self.classes) #max_det=self.nms_max_det, max_det = 300
        #output: Returns:
            # (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            #     shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            #     (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).

        # if self.detcttransform:
        #     #map image backto original_image_sizes
        #     detections = self.detcttransform.postprocess(preds, images.image_sizes, self.original_image_sizes) 
        # else:
        detections = []
        #result: List[Dict[str, Tensor]] in fasterrcnn
        for i, pred in enumerate(preds):
            #orig_img = origimages[i]
            origimageshape = origimageshapes[i]#orig_img.shape
            pred[:, :4] = scale_boxes(newimagesize, pred[:, :4], origimageshape)
            #img_path = self.batch[0][i]
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            resdict={}
            resdict["boxes"] = pred[:, :4].detach().cpu()#.numpy()
            resdict["scores"] = pred[:, 4].detach().cpu()#.numpy()
            resdict["labels"] = pred[:, 5].detach().cpu()#.numpy() #class
            detections.append(resdict)
        return detections