from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image, ImageDraw
from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
from DeepDataMiningLearning.hfvisionmain import load_visionmodel, load_dataset
from tqdm.auto import tqdm
import requests
import numpy as np
import cv2
import os
import evaluate
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from time import perf_counter

#tasks: "depth-estimation", "image-classification"
class MyVisionInference():
    def __init__(self, model_name, model_type="huggingface", task="image-classification", cache_dir="./output", gpuid='0') -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.transforms = None
        if isinstance(model_name, str) and model_type=="huggingface":
            os.environ['HF_HOME'] = cache_dir #'~/.cache/huggingface/'
            self.model, self.image_processor = load_visionmodel(model_name, task=task, load_only=True, labels=None, mycache_dir=cache_dir, trust_remote_code=True)
        # elif isinstance(model_name, torch.nn.Module):
        #     self.model = model_name
        elif isinstance(model_name, str) and task=="image-classification":#torch model
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True) #'resnet18'
        elif isinstance(model_name, str) and task=="depth-estimation":#torch model
            #https://pytorch.org/hub/intelisl_midas_v2/
            #model_names: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
            self.model = torch.hub.load('intel-isl/MiDaS', model_name, pretrained=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transforms = transforms.dpt_transform #.small_transform resize(384,) Normalized

        #get labels for classification
        if task=="image-classification" and model_type=="huggingface":
            self.id2label = self.model.config.id2label
        elif task=="image-classification":
            labels=load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif task=="depth-estimation":
            pass

        self.model=self.model.to(self.device)
        self.model.eval()
    
    def mypreprocess(self,inp):
        #https://pytorch.org/vision/stable/transforms.html
        #transforms.ToTensor()
        #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        manual_transforms = transforms.Compose([
            #transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
            transforms.ToTensor(), # 2. changes the input arrays from HWC to CHW, Turn image values to between 0 & 1 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                                std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
        ])
        if self.transforms is not None:
            inp = self.transforms(inp)#.unsqueeze(0) already added one dimension
        else:
            #inp = transforms.ToTensor()(inp).unsqueeze(0)
            inp = manual_transforms(inp).unsqueeze(0)
        return inp

    def __call__(self, image):
        self.image = read_image(image, use_pil=True, use_cv2=False, output_format='numpy', plotfig=False)
        #HWC numpy (427, 640, 3)
        if self.model_type=="huggingface":
            inputs = self.image_processor(self.image, return_tensors="pt").pixel_values
            print(inputs.shape) #torch.Size([1, 3, 224, 224]) [1, 3, 350, 518]
        else:
            inputs = self.mypreprocess(self.image)
            print(inputs.shape) #torch.Size([1, 3, 384, 576])
        inputs = inputs.to(self.device)

        start_time = perf_counter()
        with torch.no_grad():
            outputs = self.model(inputs)
        end_time = perf_counter()
        print(f'Elapsed inference time: {end_time-start_time:.3f}s')
        
        results = None
        if self.task=="image-classification":
            results = self.classification_postprocessing(outputs) #return confidence dicts
        elif self.task=="depth-estimation":
            results = self.depth_postprocessing(outputs, recolor=True) #return PIL image
        return results
    
    def depth_postprocessing(self, outputs, recolor=True):
        if self.model_type=="huggingface":
            predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
        else:
            predicted_depth  = outputs
        # interpolate to original size
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        predicted_depth_input = predicted_depth.unsqueeze(1) #[1, 1, 384, 576]

        #read the height/width from image's shape
        if isinstance(self.image, Image.Image):
            self.height = self.image.height
            self.width = self.image.width
            target_size = self.image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
        elif isinstance(self.image, np.ndarray): #HWC format
            self.height = self.image.shape[0]
            self.width = self.image.shape[1]
            target_size = (self.height, self.width) #(427, 640)
        
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        prediction = torch.nn.functional.interpolate(
            predicted_depth_input,
            size=target_size,#(H=427, W=640)
            mode="bicubic", #'bilinear', uses a bicubic interpolation algorithm to compute the values of the new tensor
            align_corners=False,
        ) #[1, 1, 427, 640]
        #output = prediction.squeeze().cpu().numpy() #[427, 640]
        #formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
        formatted = depth.squeeze().cpu().numpy().astype(np.uint8)
        print(formatted.shape) #(427, 640)
        if recolor:
            #Recolor the depth map from grayscale to colored
            #normalized_image = image.astype(np.uint8)
            formatted = cv2.applyColorMap(formatted, cv2.COLORMAP_HOT)[:, :, ::-1]
            print(formatted.shape) #(427, 640, 3)
    
        depth = Image.fromarray(formatted)
        #depth.show()
        depth.save("data/depth_testresult.jpg") 
        return depth

    def classification_postprocessing(self, outputs):
        if self.model_type=="huggingface":
            logits = outputs.logits #torch.Size([1, 1000])
        else:
            logits = outputs
        predictions = torch.nn.functional.softmax(logits[0], dim=0) #[1000]
        #print(predictions.shape) #torch.Size([1000])
        predictions = predictions.cpu().numpy()
        confidences = {self.id2label[i]: float(predictions[i]) for i in range(len(self.id2label))} #len=999

        predmax_idx = np.argmax(predictions, axis=-1)
        predmax_label = self.id2label[predmax_idx]
        predmax_confidence = float(predictions[predmax_idx])
        print(f"predmax_idx: {predmax_idx}, predmax_label: {predmax_label}, predmax_confidence: {predmax_confidence}")
        return confidences

def vision_inferencetest(model_name_or_path, task="image-classification", mycache_dir=None):

    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    #show the PIL image
    #image.show()
    #load the model
    dataset = load_dataset("huggingface/cats-image", cache_dir=mycache_dir)
    image = dataset["test"]["image"][0]
    print(len(dataset["test"]["image"]))
    im1 = image.save("test.jpg") 

    myinference = MyVisionInference(model_name=model_name_or_path, task=task, model_type="huggingface", cache_dir=mycache_dir)
    confidences = myinference(image)
    print(confidences)

    #model_name_or_path = "nielsr/convnext-tiny-finetuned-eurostat"
    #task = "image-classification"
    model, image_processor = load_visionmodel(model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=mycache_dir, trust_remote_code=True)
    id2label = model.config.id2label
    #model = AutoModelForImageClassification.from_pretrained("nielsr/convnext-tiny-finetuned-eurostat")
    #load the image processor
    #image_processor = AutoImageProcessor.from_pretrained("nielsr/convnext-tiny-finetuned-eurostat")
    #preprocess the image (PIL)
    inputs = image_processor(image.convert("RGB"), return_tensors="pt")
    print(inputs.pixel_values.shape) #torch.Size([1, 3, 224, 224])
    #inference
    with torch.no_grad():
        outputs = model(**inputs) #key="logits", "loss"
    #get the prediction
    logits = outputs.logits #torch.Size([1, 1000])

    predictions = torch.nn.functional.softmax(logits[0], dim=0) #[1000]
    confidences = {id2label[i]: float(predictions[i]) for i in range(len(id2label))} #len=999

    pred = logits.argmax(dim=-1) #torch.Size([1])
    #print the prediction
    print(pred[0]) #tensor(646)
    predicted_class_idx = pred[0].item() #1
    print("Predicted class:", id2label[predicted_class_idx])
    return confidences

#https://huggingface.co/docs/transformers/main/en/tasks/zero_shot_image_classification
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
def clip_test(mycache_dir=None):
    #url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
    #candidate_labels = ["fox", "bear", "seagull", "owl"]
    
    url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640" #car
    candidate_labels = ["tree", "car", "bike", "cat"]
    image = Image.open(requests.get(url, stream=True).raw)
    checkpoint = "openai/clip-vit-large-patch14"
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint, cache_dir=mycache_dir)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)

    
    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True) 
    #'input_ids'[4, 3], 'pixel_values'[1, 3, 224, 224], 'attention_mask'[4, 3]

    with torch.no_grad():
        outputs = model(**inputs) #CLIPOutput(loss=None, logits_per_image[1,4], logits_per_text[4,1]

    logits = outputs.logits_per_image[0] #[4]
    probs = logits.softmax(dim=-1).numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": candidate_label}
        for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
    ]

    print(result)

#https://huggingface.co/docs/transformers/main/en/tasks/monocular_depth_estimation
def MyVisionInference_depthtest(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    #image.size()#The size attribute returns a (width, height) tuple
    image.save("data/depth_test.jpg") 

    checkpoint = "Intel/dpt-large" #"LiheYoung/depth-anything-base-hf" #"Intel/dpt-large" #"vinvino02/glpn-nyu"
    depthinference = MyVisionInference(model_name=checkpoint, model_type="huggingface", task="depth-estimation", cache_dir=mycache_dir)
    #checkpoint = "DPT_Large" #pip install timm
    #depthinference = MyVisionInference(model_name=checkpoint, model_type="torch", task="depth-estimation", cache_dir=mycache_dir)
    results = depthinference(image=image)

def depth_test(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    #image.size()#The size attribute returns a (width, height) tuple
    image.save("data/depth_test.jpg") 

    checkpoint = "Intel/dpt-large" #"LiheYoung/depth-anything-base-hf" #"Intel/dpt-large" #"vinvino02/glpn-nyu"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=mycache_dir)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(pixel_values) #only predicted_depth key
        predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
    
    # interpolate to original size
    #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
    predicted_depth_input = predicted_depth.unsqueeze(1) #[1, 1, 384, 384]
    target_size = image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
    prediction = torch.nn.functional.interpolate(
        predicted_depth_input,
        size=target_size,#(H=427, W=640)
        mode="bicubic", #uses a bicubic interpolation algorithm to compute the values of the new tensor
        align_corners=False,
    )
    output = prediction.squeeze().numpy() #[427, 640]
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    #depth.show()
    depth.save("data/depth_testresult.jpg") 

def object_detection(mycache_dir=None):
    #url = "https://i.imgur.com/2lnWoly.jpg"
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)

    checkpoint = "facebook/detr-resnet-50" #"devonho/detr-resnet-50_finetuned_cppe5"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint, cache_dir=mycache_dir)

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]) #[[427, 640]]
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
    image.save("output/ImageDraw.png")

def inference():
    mycache_dir="/home/lkk/Developer/"#r"D:\Cache\huggingface"
    #os.environ['HF_HOME'] = mycache_dir #'~/.cache/huggingface/'
    mycache_dir = os.environ['HF_HOME']
    object_detection(mycache_dir=mycache_dir)
    depth_test(mycache_dir=mycache_dir)
    clip_test(mycache_dir=mycache_dir)
    confidences = vision_inferencetest(model_name_or_path="google/bit-50", task="image-classification", mycache_dir=mycache_dir)

if __name__ == "__main__":
    #"nielsr/convnext-tiny-finetuned-eurostat"
    #"google/bit-50"
    #"microsoft/resnet-50"
    inference()