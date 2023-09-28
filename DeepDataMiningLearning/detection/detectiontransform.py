#refer: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
#from torchvision.io.image import read_image
from PIL import Image
import DeepDataMiningLearning.detection.transforms as T

def max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]#[[3, 800, 1295]] -> [3, 800, 1295]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def _resize_image_and_masks(
    image: Tensor,
    self_min_size: int,
    self_max_size: int,
    target: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    # if torchvision._is_tracing():
    #     im_shape = _get_shape_onnx(image)
    # elif torch.jit.is_scripting():
    #     im_shape = torch.tensor(image.shape[-2:])
    # else:
    #     im_shape = image.shape[-2:]
    im_shape = image.shape[-2:]

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None

    min_size = min(im_shape)
    max_size = max(im_shape)
    scale_factor = min(self_min_size / min_size, self_max_size / max_size)
    recompute_scale_factor = True

    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target

    return image, target

def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
    

class DetectionTransform(nn.Module):
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
        image_mean: List[float],
        image_std: List[float],
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
        

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets
    
    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            # if self._skip_resize:
            #     return image, target
            size = self.min_size[-1] #self.torch_choice(self.min_size)
        else:
            size = self.min_size[-1]
        image, target = _resize_image_and_masks(image, size, self.max_size, target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target
    
    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        # if torchvision._is_tracing():
        #     # batch_images() does not export well to ONNX
        #     # call _onnx_batch_images() instead
        #     return self._onnx_batch_images(images, size_divisible)

        max_size = max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs
    
    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            # if "masks" in pred:
            #     masks = pred["masks"]
            #     masks = paste_masks_in_image(masks, boxes, o_im_s)
            #     result[i]["masks"] = masks
            # if "keypoints" in pred:
            #     keypoints = pred["keypoints"]
            #     keypoints = resize_keypoints(keypoints, im_s, o_im_s)
            #     result[i]["keypoints"] = keypoints
        return result


def get_transformsimple(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def test_imagetransform(images, target, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):

    #https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py
    #GeneralizedRCNNTransform: List[Tensor] ->
    #image Tensor list input
    for i in range(len(images)):
        image = images[i]
        #normalize the image
        mean = torch.as_tensor(image_mean, dtype=image.dtype)
        std = torch.as_tensor(image_std, dtype=image.dtype)
        #mean[:, None, None]: torch.Size([3]) => torch.Size([3, 1, 1])
        image=(image - mean[:, None, None]) / std[:, None, None] #torch.Size([3, 1142, 1850])

        #resize the image
        im_shape = image.shape[-2:]#torch.Size([1142, 1850])
        min_size = min(im_shape)#1142
        max_size = max(im_shape)#1850
        set_min_size=800
        set_max_size=1333
        scale_factor = min(set_min_size / min_size, set_max_size / max_size) #0.7 follow min_size
        image_input=image[None] #torch.Size([1, 3, 1142, 1850])
        image = torch.nn.functional.interpolate(
            image_input,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )[0]
        print(image.shape) #torch.Size([3, 800, 1295]) #H size=min_size, W size<max_size
        images[i] = image

    image_sizes = [img.shape[-2:] for img in images] #[torch.Size([800, 1295])]
    #batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor
    max_size = max_by_axis([list(img.shape) for img in images]) #[3, 800, 1295]
    size_divisible=32
    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride) #[3, 800, 1312]

    batch_shape = [len(images)] + max_size #[2, 3, 800, 1312]
    batched_imgs = images[0].new_full(batch_shape, 0) #[2, 3, 800, 1312]
    for i in range(batched_imgs.shape[0]):
        img = images[i] #[3, 800, 1295]
        batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img) #copy the elements from img to self
    #return batched_imgs [2, 3, 800, 1312]
    image_sizes_list: List[Tuple[int, int]] = []
    for image_size in image_sizes:#original image size
        torch._assert(
            len(image_size) == 2,
            f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
        )
        image_sizes_list.append((image_size[0], image_size[1]))

    image_list = ImageList(batched_imgs, image_sizes_list)#images: list of [3, 800, 1295], image_sizes_list: [(800, 1295)]

    return image_list, target


#ref: https://github.com/lkk688/myyolov8/blob/main/ultralytics/data/augment.py
import cv2
import numpy as np
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

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


if __name__ == "__main__":
    #imgpath = "../../sampledata/sjsupeople.jpg"
    imgpath = "/home/010796032/MyRepo/DeepDataMiningLearning/sampledata/sjsupeople.jpg"#"../../sampledata/sjsupeople.jpg"
    image = Image.open(imgpath) #PIL image

    image_transfunc=get_transformsimple(train=True)
    image, target = image_transfunc(image, target=None) #image tensor [3, 1142, 1850]
    #image = torch.rand(3, 300, 400) #tensor CHW
    print(image.is_floating_point()) #Expected input images to be of floating type (in range [0, 1])
    print(image.dtype) #torch.float32
    
    images = [image]#[img for img in images]
    #add another random image
    images.append(torch.rand(3, 400, 600))
    #image list input
    image, target = test_imagetransform(image, target) #torch.Size([3, 800, 1295])
    #x=image.tensors
    #imagelist object, tensors section is list of tensors