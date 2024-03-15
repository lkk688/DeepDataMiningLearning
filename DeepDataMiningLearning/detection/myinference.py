import torch
#https://debuggercafe.com/anchor-free-object-detection-inference-using-fcos-fully-connected-one-stage-object-detection/

import cv2
import torch
import argparse
import time
import torchvision
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from DeepDataMiningLearning.detection.models import create_detectionmodel, get_torchvision_detection_models, load_trained_model
import numpy as np

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo

def myread_image(preprocess, imgpath, usecv2=True, uspil=False):
    if usecv2==True:
        im0 = cv2.imread(imgpath) #(1080, 810, 3) HWC, BGR format
        imaglist = [im0]
        imgtensors = preprocess(imaglist) #return #[1, 3, 640, 480]
        return imgtensors, imaglist #for yolo
    else:
        img = read_image(imgpath)
        batch = [preprocess(img)]
        return batch, [img] #for faster rcnn

def savepred_toimage(im0, onedetection, classes=None, usecv2=True, boxformat='xyxy', resultfile="results.jpg"):
    #labels = [names[i] for i in detections["labels"]] #classes[i]
    #img=im0.copy() #HWC (1080, 810, 3)
    if usecv2:
        im0=im0[..., ::-1].transpose((2,0,1))  # BGR to RGB, HWC to CHW
    imgtensor = torch.from_numpy(im0.copy()) #[3, 1080, 810]
    if boxformat =='xyxy':
        pred_bbox_tensor=onedetection["boxes"] #torch.from_numpy(onedetection["boxes"])
    else:
        #pred_bbox_tensor=torchvision.ops.box_convert(torch.from_numpy(onedetection["boxes"]), 'xywh', 'xyxy')
        pred_bbox_tensor=torchvision.ops.box_convert(onedetection["boxes"], 'xywh', 'xyxy')
    
    #print(pred_bbox_tensor)
    pred_labels = onedetection["labels"].numpy().astype(int).tolist()
    if classes:
        labels = [classes[i] for i in pred_labels]
    else:
        labels = [str(i) for i in pred_labels]
    #img: Tensor of shape (C x H x W) and dtype uint8.
    #box: Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
    #labels: Optional[List[str]]
    box = draw_bounding_boxes(imgtensor, boxes=pred_bbox_tensor,
                            labels=labels,
                            colors="red",
                            width=4, font_size=50)
    im = to_pil_image(box.detach())
    # save a image using extension
    im = im.save(resultfile)
    return im

def multimodel_inference(modelname, imgpath, ckpt_file, device='cuda:0', scale='n'):

    model, imgtransform, classes = create_detectionmodel(modelname=modelname, num_classes=80, trainable_layers=0, ckpt_file = ckpt_file, fp16=False, device= device, scale='n')

    if modelname.startswith("yolo"):
        imgtensors, imaglist = myread_image(imgtransform, imgpath, usecv2=True)
    else:
        imgtensors, imaglist= myread_image(imgtransform, imgpath, usecv2=False)
    #inference
    preds = model(imgtensors)
    
    newimgsize = imgtensors.shape[2:] #640, 480
    origimageshapes=[img.shape for img in imaglist]
    detections = imgtransform.postprocess(preds, newimgsize, origimageshapes)

    idx=0
    onedetection = detections[idx]
    im0=imaglist[idx]
    savepred_toimage(im0, onedetection, classes=classes, usecv2=True, boxformat='xyxy', resultfile="results.jpg")

def test_inference(modelname, imgpath):
    img = read_image(imgpath)
    pretrained_model, preprocess, weights, classes = get_torchvision_detection_models(modelname)
    pretrained_model.eval()
    #Apply inference preprocessing transforms
    batch = [preprocess(img)]
    prediction = pretrained_model(batch)[0]
    labels = [classes[i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    return im

def inference_trainedmodel(modelname, num_classes, classes, checkpointpath, imgpath):
    img = read_image(imgpath)
    model, preprocess = load_trained_model(modelname, num_classes, checkpointpath)
    #Apply inference preprocessing transforms
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    print(prediction["labels"])
    print(prediction["boxes"])
    if classes and len(classes)==num_classes:
        labels = [classes[i] for i in prediction["labels"]]
    else:
        labels = [i for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    return im


def detect_video(args):
    # Define the computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    cap = cv2.VideoCapture(args['input'])
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"
    # Define codec and create VideoWriter object .
    out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # Read until end of video.
    while(cap.isOpened):
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                boxes, classes, labels = predict(
                    frame, model, 
                    device, args['threshold']
                )
            
            # Draw boxes and show current frame on screen.
            image = draw_boxes(boxes, classes, labels, frame)
            # Get the end time.
            end_time = time.time()
            # Get the fps.
            fps = 1 / (end_time - start_time)
            # Add fps to total fps.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            # Write the FPS on the current frame.
            cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # Convert from BGR to RGB color format.
            cv2.imshow('image', image)
            out.write(image)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def predict(image, model, device, transform, class_names, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # Transform the image to tensor.
    image = transform(image).to(device)
    # Add a batch dimension.
    image = image.unsqueeze(0) 
    # Get the predictions on the image.
    with torch.no_grad():
        outputs = model(image) 
    # Get score for all the predicted objects.
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # Get all the predicted bounding boxes.
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score.
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # Get all the predicited class names.
    pred_classes = [class_names[i] for i in labels.cpu().numpy()]
    return boxes, pred_classes, labels

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    for i, box in enumerate(boxes):
        color = 'r' #COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color[::-1], 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[::-1], 2, 
                    lineType=cv2.LINE_AA)
    return image


from DeepDataMiningLearning.detection.models import load_checkpoint
from DeepDataMiningLearning.detection.modeling_rpnfasterrcnn import CustomRCNN
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import requests
def gettensorfromurls(urls):
    # define custom transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    pilimage_list = []
    image_tensorlist = []
    for url in urls:
        image = Image.open(requests.get(url, stream=True).raw)
        #image_tensor1 = pil_to_tensor(image)#did not normalize the image
        # transform the pIL image to tensor image
        image_tensor1 = transform(image)#normalize the image
        pilimage_list.append(image)
        image_tensorlist.append(image_tensor1)
    return image_tensorlist, pilimage_list

def test_Customrcnn():
    model_name = 'resnet50' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #model_name = 'resnet152' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    num_classes = 91 #Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /data/rnd-liu/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
    myrcnn=CustomRCNN(backbone_modulename=model_name,trainable_layers=0,num_classes=num_classes,out_channels=256,min_size=800,max_size=1333)
    summary(model=myrcnn, 
        input_size=(1, 3, 300, 400), #(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 
    device='cuda:3'
    ckpt_file = "/data/cmpe249-fa23/modelzoo/fasterrcnn_resnet50_fpn_v2.pt"
    #ckpt_file = "/data/cmpe249-fa23/trainoutput/waymococo/0923/model_40.pth" #resnet152
    model = load_checkpoint(myrcnn, ckpt_file, fp16=False)
    model=model.to(device)

    model.eval()
    #The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    #image, and should be in 0-1 range. Different images can have different sizes.
    #images = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

    url2 = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_tensorlist, pilimage_list = gettensorfromurls([url1, url2])
    image_tensorlist = [image.to(device) for image in image_tensorlist] #[3, 480, 640] [3, 427, 640]
    predictions = model(image_tensorlist)
    #During testing, it returns list[BoxList] contains additional fields like `scores`, `labels` and `mask` (for Mask R-CNN models).
    print(len(predictions)) #2
    print(predictions[0]) #'boxes' 'labels' 'scores'

    images, targets = model.detcttransform(image_tensorlist, targets=None)
    #images is ImageList, image_sizes:[(800, 1066), (800, 1199)]

    features = model.backbone(images.tensors) #[2, 3, 800, 1216]
    print([(k, v.shape) for k, v in features.items()])
    #[('0', torch.Size([2, 256, 200, 304])), #/4
    # ('1', torch.Size([2, 256, 100, 152])), #/8
    # ('2', torch.Size([2, 256, 50, 76])),   #/16
    # ('3', torch.Size([2, 256, 25, 38])),   #/32
    # ('pool', torch.Size([2, 256, 13, 19]))]

    # anchors = model.rpn_anchor_generator(images, features)#return tensor list
    # for anchor in anchors:
    #     print(anchor.shape)

    # proposals, proposal_losses = model.rpn(images, features, targets)
    # #proposals list of 2, 

    # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    # print(len(detections))

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/video_1.mp4', 
                    help='path to input video')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')
args = vars(parser.parse_args())

def main(args):
    #modelname = 'fasterrcnn_resnet50_fpn_v2'
    imgpath = "../../sampledata/sjsupeople.jpg"
    #im=test_inference(modelname, imgpath)
    #im.save("../../data/testinference.png", "PNG")

    modelname = 'yolov8'
    imgpath = './sampledata/bus.jpg'
    ckpt_file = '/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt'
    device = 'cuda:0'
    multimodel_inference(modelname, imgpath, ckpt_file, device, scale='n')

if __name__ == "__main__":
    test_Customrcnn()
    #main(args)