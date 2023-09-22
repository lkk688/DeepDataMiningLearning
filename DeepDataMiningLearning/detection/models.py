import torchvision
import torch
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo

    # model = torchvision.models.get_model(args.model)
    #     args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    # )
def get_torchvision_detection_models(modelname, box_score_thresh=0.9):
    weights_enum = get_model_weights(modelname) #<enum 'FasterRCNN_MobileNet_V3_Large_320_FPN_Weights'>
    weights = weights_enum.DEFAULT #get the default weights
    preprocess = weights.transforms()
    classes = weights.meta["categories"]
    pretrained_model=get_model(modelname, box_score_thresh=0.9, weights="DEFAULT")
    
    return pretrained_model, preprocess, weights, classes

def modify_fasterrcnnheader(model, num_classes, freeze=True):
    if freeze == True:
        for p in model.parameters():
            p.requires_grad = False
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    #num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_trained_model(modelname, num_classes, checkpointpath):
    model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
    model = modify_fasterrcnnheader(model, num_classes, freeze=False)

    if checkpointpath:
        model.load_state_dict(torch.load(checkpointpath))#'./saved_models2/model_9.pth'))

    return model, preprocess

def modify_backbone(model, num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=2,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    return model
    # model = torchvision.models.get_model(
    #     args.model, weights=args.weights, weights_backbone=args.weights_backbone, num_classes=num_classes, **kwargs
    # )
import os
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from collections import OrderedDict
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x
    
class CustomRCNN(nn.Module):
    """
    Implements Custom Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

        During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    """
    def __init__(
        self,
        backbone_modulename,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        ):

        self.body, self.fpn = self.create_fpnbackbone(backbone_modulename)
        if not hasattr(self.fpn, "out_channels"):
            print("error")
        #self.backbone = backbone
        self.out_channels = self.fpn.out_channels

        #transform: 
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        # The transformations it performs are:
        # - input normalization (mean subtraction and std division)
        # - input / target resizing to match min_size / max_size
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        #It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets

        #RPN part
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(self.out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        #RCNN part
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(self.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        #roi_heads (nn.Module): takes the features + the proposals from the RPN and computes detections / masks from it.
        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
    
    def create_fpnbackbone(self, backbone_modulename):
        backbone = get_model(backbone_modulename, weights="DEFAULT")
        trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        extra_blocks = LastLevelMaxPool()
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #the number of channels in the FPN
        out_channels = 256
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=None,
        )
        return body, fpn

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            for target in targets:
                boxes = target["boxes"] #boxes only used for format validation
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        images, targets = self.transform(images, targets)
        cnnfeatures = self.body(images.tensors, targets)
        features = self.fpn(cnnfeatures)
        #features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections

#from torchvision.io.image import read_image
from PIL import Image
import DeepDataMiningLearning.detection.transforms as T
def get_transformsimple(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
def create_testdata():
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4] #xywh->xyxy
    labels = torch.randint(1, 91, (4, 11)) #[4, 11] size
    images = list(image for image in images) #list of tensor (torch.Size([3, 600, 1200]))
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    return images, targets

def test_backbone(body,fpn,x= torch.rand(1,3,64,64)):
    # compute the output
    x = body(x)
    output = fpn(x)
    #output = backbone(x)
    print([(k, v.shape) for k, v in output.items()])
        #     >>> # returns
        # >>>   [('0', torch.Size([1, 256, 16, 16])),
        # >>>    ('1', torch.Size([1, 256, 8, 8])),
        # >>>    ('2', torch.Size([1, 256, 4, 4])),
        # >>>    ('3', torch.Size([1, 256, 2, 2])),
        # >>>    ('pool', torch.Size([1, 256, 1, 1]))]

def test_imagetransform(imgpath, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):
    #imgpath = "../../sampledata/sjsupeople.jpg"
    image = Image.open(imgpath) #PIL image

    image_transfunc=get_transformsimple(train=True)
    image, target = image_transfunc(image, target=None) #image tensor [3, 1142, 1850]
    #image = torch.rand(3, 300, 400) #tensor CHW
    print(image.is_floating_point()) #Expected input images to be of floating type (in range [0, 1])
    print(image.dtype) #torch.float32

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
    return image

if __name__ == "__main__":
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH='/data/cmpe249-fa23/torchvisiondata/'

    model_names=list_models(module=torchvision.models)
    print("Torchvision buildin models:", model_names)
    detectionmodel_names=list_models(module=torchvision.models.detection)
    print("Torchvision detection models:", detectionmodel_names)

    imgpath = "/home/010796032/MyRepo/DeepDataMiningLearning/sampledata/sjsupeople.jpg"#"../../sampledata/sjsupeople.jpg"
    image = test_imagetransform(imgpath)
    #


    modelname = 'fasterrcnn_resnet50_fpn_v2'
    model, preprocess, weights, classes = get_torchvision_detection_models(modelname, box_score_thresh=0.9)
    INSTANCE_CATEGORY_NAMES = weights.meta["categories"]
    print(model.backbone.out_channels) #256
    modulelist=list(model.named_children())
    for m in modulelist:#'transform', 'backbone', 'rpn', 'roi_heads'
        print(m[0])
        print(len(m))

    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    #for inference
    predictions = model(x)
    print(predictions)#list of dicts
    #for training
    images, targets = create_testdata()
    output = model(images, targets)

    #export the model to ONNX:
    torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

    #returns the post-processed predictions as a List[Dict[Tensor]], one for each input image
    # The fields of the Dict are as
    # follows:
    #     - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
    #       ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
    #     - labels (Int64Tensor[N]): the predicted labels for each image
    #     - scores (Tensor[N]): the scores or each prediction

    summary(model=model, 
        input_size=(1, 3, 300, 400), #(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 