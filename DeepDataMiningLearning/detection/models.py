import torchvision
import torch
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from DeepDataMiningLearning.detection.modules.yolomodels import create_yolomodel, freeze_yolomodel
#from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN
import os
from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN

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
    pretrained_model=get_model(modelname, box_score_thresh=box_score_thresh, weights="DEFAULT")
    
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

#Pass
import torchinfo
def test_defaultmodels():
    model_names=list_models(module=torchvision.models)
    print("Torchvision buildin models:", model_names)
    detectionmodel_names=list_models(module=torchvision.models.detection)
    print("Torchvision detection models:", detectionmodel_names)

    #image = test_imagetransform(imgpath) #torch.Size([3, 800, 1295])
    #x=image.tensors
    #imagelist object, tensors section is list of tensors

    modelname = 'fasterrcnn_resnet50_fpn_v2'
    model, preprocess, weights, classes = get_torchvision_detection_models(modelname, box_score_thresh=0.2)
    try:
        torchinfo.summary(model=model,
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )
    except:
        print(model)
    
    INSTANCE_CATEGORY_NAMES = weights.meta["categories"]
    print(model.backbone.out_channels) #256
    #torch.save(model.state_dict(), "/data/cmpe249-fa23/modelzoo/fasterrcnn_resnet50_fpn_v2.pt")

    x=torch.rand(1,3,800,800)
    output = model.backbone(x) 
    print([(k, v.shape) for k, v in output.items()])
    
    x=torch.rand(1,3,64,64) #image.tensors #[2, 3, 800, 1312] list of tensors x= torch.rand(1,3,64,64)
    output = model.backbone(x) 
    print([(k, v.shape) for k, v in output.items()])
    #[('0', torch.Size([1, 256, 16, 16])), #/4
    # ('1', torch.Size([1, 256, 8, 8])), #/8
    # ('2', torch.Size([1, 256, 4, 4])),   #/16
    # ('3', torch.Size([1, 256, 2, 2])),   #/32
    # ('pool', torch.Size([1, 256, 1, 1]))]

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
    torch.onnx.export(model, x, "/data/cmpe249-fa23/trainoutput/faster_rcnn.onnx", opset_version = 11)

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

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def load_checkpoint(model, ckpt_file, fp16=False):
    ckpt=torch.load(ckpt_file, map_location='cpu')
    #print(ckpt.keys()) #'0.conv.weight', '0.bn.weight', '0.bn.bias'
    currentmodel_statedict = model.state_dict()
    csd = intersect_dicts(ckpt, currentmodel_statedict)  # intersect
    model.load_state_dict(ckpt, strict=False)
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights')
    #524/566 items
    #names = ckpt.module.names if hasattr(ckpt, 'module') else ckpt.names  # get class names
    model.half() if fp16 else model.float()
    return model

def create_detectionmodel(modelname, num_classes=None, trainable_layers=0, ckpt_file = None, fp16=False, device= 'cuda:0', scale='n'):
    model = None
    preprocess = None
    classes = None
    if trainable_layers==0:
        freezemodel = True
    if modelname == 'fasterrcnn_resnet50_fpn_v2':
        model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
        if num_classes is not None and len(classes) != num_classes:
            model = modify_fasterrcnnheader(model, num_classes, freeze=freezemodel)
        if ckpt_file:
            model = load_checkpoint(model, ckpt_file, fp16)
    elif modelname.startswith('customrcnn'):
        x = modelname.split("_")
        if x[0]== 'customrcnn' and x[1].startswith('resnet'):
            backbonename = x[1]
            model=CustomRCNN(backbone_modulename=backbonename,trainable_layers=trainable_layers,num_classes=num_classes,out_channels=256,min_size=800,max_size=1333)
            if ckpt_file:
                model = load_checkpoint(model, ckpt_file, fp16)
        else:
            print("Model name not supported")
    elif modelname.startswith('yolo') or 'yolo' in modelname.lower():
        model, preprocess, classes=create_yolomodel(modelname, num_classes, ckpt_file, fp16, device, scale)
        model= freeze_yolomodel(model, freeze=[])
        #ckpt file is already loaded in create_yolomodel
    else:
        print('Model name not supported')

    if model:
        summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        ) 
        if device:
            currentdevice=next(model.parameters()).device #simply getting the device name for the first parameter of the nn module
            model=model.to(device)
    return model, preprocess, classes
    

if __name__ == "__main__":
    test_defaultmodels()

    #os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    #DATAPATH='/data/cmpe249-fa23/torchvisiondata/'


    