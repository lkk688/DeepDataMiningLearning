import torch
import numpy as np
import cv2 #pip install opencv-python
import os
import requests
import matplotlib.pyplot as plt
from torchvision import transforms
from time import perf_counter
from PIL import Image, ImageDraw
from torchvision.io import read_image
from transformers import AutoImageProcessor, AutoModelForImageClassification, \
    AutoModelForDepthEstimation, AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection, \
        AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation, AutoModelForInstanceSegmentation
# AutoModelForUniversalSegmentation: Semantic and panoptic segmentation.
# AutoModelForInstanceSegmentation: Instance segmentation.
from DeepDataMiningLearning.vision.visutil import visualize_results
#from DeepDataMiningLearning.detection.models import create_detectionmodel #torch object detection models

#tasks: "depth-estimation", "image-classification", "object-detection", "zeroshot-objectdetection"
class MyVisionInference():
    def __init__(self, model_name, model_path="", model_type="huggingface", task="image-classification", cache_dir="./output", gpuid='0', scale='x', image_maxsize=None) -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = self.get_device(gpuid=gpuid)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.transforms = None
        self.id2label = None
        if isinstance(model_name, str) and model_type=="huggingface":
            #os.environ['HF_HOME'] = cache_dir #'~/.cache/huggingface/'
            if model_path and os.path.exists(model_path):
                model_name_or_path = model_path
            else:
                model_name_or_path = model_name
            self.model, self.image_processor = self.load_hfvisionmodel(model_name_or_path = model_name_or_path, task=task, image_maxsize=image_maxsize)
            self.id2label = self.model.config.id2label
        elif isinstance(model_name, str) and task=="image-classification":#torch model
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True) #'resnet18'
            labels=self.load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task=="object-detection":#torch model
            from DeepDataMiningLearning.detection.models import create_detectionmodel #torch object detection models
            self.model, self.image_processor, labels = create_detectionmodel(modelname=model_name, num_classes=None, ckpt_file=model_path, device=self.device, scale=scale)
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task=="depth-estimation":#torch model
            #https://pytorch.org/hub/intelisl_midas_v2/
            #model_names: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
            self.model = torch.hub.load('intel-isl/MiDaS', model_name, pretrained=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transforms = transforms.dpt_transform #.small_transform resize(384,) Normalized

        self.model=self.model.to(self.device)
        self.model.eval()
    
    def get_device(self, gpuid='0', useaccelerate=False):
        if useaccelerate==False:
            if torch.cuda.is_available():
                device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            from accelerate.test_utils.testing import get_backend
            # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
            device, _, _ = get_backend()
        return device

    #tasks: "depth-estimation", "image-classification", "object-detection", "zeroshot-objectdetection"
    def load_hfvisionmodel(self, model_name_or_path, task="image-classification", image_maxsize=None, trust_remote_code=True, ignore_mismatched_sizes=False):
        if image_maxsize is None:
            image_processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                #cache_dir=mycache_dir,
                trust_remote_code=trust_remote_code,
            )
        else:
            MAX_SIZE = max(image_maxsize, 28)
            image_processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                do_resize=True,
                #size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
                size={"height": MAX_SIZE, "width": MAX_SIZE},
                do_pad=True,
                pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
            )
            
        if task == "image-classification":
            model = AutoModelForImageClassification.from_pretrained(
                model_name_or_path,
                #config=config,
                #cache_dir=mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                trust_remote_code=trust_remote_code,
            )
        elif task == "depth-estimation":
            model = AutoModelForDepthEstimation.from_pretrained(
                model_name_or_path, 
                #config=config,
                #cache_dir=mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                trust_remote_code=trust_remote_code,
            )
        elif task == "object-detection":
            model = AutoModelForObjectDetection.from_pretrained(
                model_name_or_path, 
                #config=config,
                #cache_dir=mycache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                trust_remote_code=trust_remote_code,
            )
        elif task == "zeroshot-objectdetection":
            #"google/owlvit-base-patch32" "google/owlv2-base-patch16-ensemble" "IDEA-Research/grounding-dino-tiny"
            #https://huggingface.co/docs/transformers/v4.22.0/en/main_classes/model, torch_dtype=torch.float16
            model =  AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name_or_path,
                #config = config,
                ignore_mismatched_sizes=ignore_mismatched_sizes)
        elif task == "semantic-segmentation":
            model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes)
        elif "segmentation" in task:
            model = AutoModelForUniversalSegmentation.from_pretrained(
                model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes)
        #model.config.id2label
        return model, image_processor
    
    #output_format="numpy", "pil" in HWC format
    def read_image(self, image, use_pil=True, use_cv2=False, rgb=True, output_format='numpy', plotfig=False):
        if isinstance(image, Image.Image):
            if output_format == 'numpy':
                # Convert PIL image to NumPy array
                image = np.array(image.convert("RGB"))
            elif output_format == 'pil':
                image = image
        elif isinstance(image, str):
            if image.startswith('http'):
                filename = requests.get(image, stream=True).raw
            elif image.endswith('.jpg') or image.endswith('.png'):
                filename = image
            #Read the image to numpy HWC format uint8
            if use_pil:
                #https://pillow.readthedocs.io/en/stable/reference/Image.html
                image = Image.open(filename)
                if rgb:
                    image = image.convert('RGB')
                    size = image.size #size in pixels, as a 2-tuple: (width, height)
                if output_format == 'numpy':
                    image = np.array(image) ##convert PIL image to numpy, HWC format (height, width, color channels)
                elif output_format == 'pil':
                    image = image
                #img.height, img.width
                #image = image.astype(np.float32) / 255.0
            elif use_cv2:
                img = cv2.imread(filename) #row (height) x column (width) x color (3) 'numpy.ndarray'
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #HWC
                #print(img.dtype) # uint8
                if output_format == 'numpy':
                    image = img
                elif output_format == 'pil':
                    image = Image.fromarray(img)
            else:
                #torch read_image get format CHW (color channels, height, width) in tensor
                img = read_image(str(filename)) #3, 900, 900
                img = img.numpy().transpose((1, 2, 0)) #CHW (color channels, height, width) to numpy/matplotlib's HWC
                if output_format == 'numpy':
                    image = img
                elif output_format == 'pil':
                    image = Image.fromarray(img)
        
        #read the height/width from image's shape
        if isinstance(image, Image.Image):
            height = image.height
            width = image.width
            org_sizeHW = image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
        elif isinstance(image, np.ndarray): #HWC format
            height = image.shape[0]
            width = image.shape[1]
            org_sizeHW = (height, width) #(427, 640)
        if plotfig:
            if output_format== 'numpy':
                #image = image.astype(np.float32) / 255.0
                # Plot the image, matplotlib also uses HWC format
                plt.figure(figsize=(10, 7))
                plt.imshow(image) #need HWC
                plt.axis("off")
                #plt.title(image_class, fontsize=14)
            elif output_format== 'pil':
                image.show()
            else:
                # Display the image
                cv2.imshow("Image", image)
        return image, org_sizeHW #HWC in numpy or PIL

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

    def __call__(self, image, text=None):
        self.image, self.org_sizeHW = self.read_image(image, use_pil=True, use_cv2=False, output_format='numpy', plotfig=False)
        print(f"Shape of the NumPy array: {self.image.shape}")
        #numpy/matplotlib's HWC format numpy (427, 640, 3)
        if self.image_processor is not None and self.model_type=="huggingface":
            #require image list in np.ndarray format, match input image HWC (used here) or CHW
            if text is None:
                #Image processors almost always expect a list of images, even if you're processing only one. 
                inputs = self.image_processor([self.image], return_tensors="pt")#.pixel_values
            else:
                #Text processors, when handling a single text input, typically expect a single string. Batching of text is also possible
                inputs = self.image_processor(text=text, images=[self.image], return_tensors="pt")
            print(inputs.keys()) #['pixel_values', 'pixel_mask']
            print(inputs['pixel_values'].shape) #BCHW [1, 3, 480, 480]
        elif self.image_processor is not None:
            inputs = self.image_processor(self.image) #BCHW for tensor
        else:
            #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            inputs = self.mypreprocess(self.image)
            print(inputs.shape) #torch.Size([1, 3, 384, 576])
        
        #Move to GPU
        inputs = inputs.to(self.device) #BCHW

        start_time = perf_counter()
        with torch.no_grad():
            #outputs = self.model(inputs) #if using .pixel_values
            #output: [1, 84, 5880] 84=4(boxes)+80(classes)
            #outputs = self.model(**inputs) #inputs are dicts
            # Forward pass with student model
            if self.model_type=="huggingface":  # Hugging Face model
                outputs = self.model(**inputs) #inputs are dicts
            else:  # Torch model
                outputs = self.model(inputs)
        end_time = perf_counter()
        print(f'Elapsed inference time: {end_time-start_time:.3f}s')
        self.inputs = inputs
        
        results = None
        if self.task=="image-classification":
            results = self.classification_postprocessing(outputs) #return confidence dicts
        elif self.task=="depth-estimation":
            results = self.depth_postprocessing(outputs, recolor=True) #return PIL image
        elif self.task=="object-detection":
            results = self.objectdetection_postprocessing(outputs) #return PIL image
        elif self.task=="semantic-segmentation":
            results = self.segmentation_postprocessing(outputs)
        elif "segmentation" in self.task:
            results = self.universalsegmentation_postprocessing(outputs)
        return results
    
    def zeroshotobjectdetection_postprocessing(self, outputs, text_labels=None, threshold=0.3, text_threshold=0.3, grounding=True):
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        #target_sizes = torch.tensor([[image.height, image.width]])
        target_sizes = torch.tensor([self.org_sizeHW])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)

        if grounding == True:
            results = self.image_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, box_threshold=threshold, text_threshold=text_threshold
            )
        else:
            results = self.image_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
            )
        # Retrieve predictions for the first image for the corresponding text queries
        result = results[0]
        boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
        for box, score, text_label in zip(boxes, scores, text_labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
            
        
    def objectdetection_postprocessing(self, outputs, threshold=0.3):
        target_sizes = torch.tensor([self.org_sizeHW]) #(640, 427)=> [[427, 640]]
        if self.model_type == "huggingface":
            results = self.image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0] #'scores'[30], 'labels', 'boxes'[30,4]
        else: #torch model
            imgsize = self.inputs.shape[2:] #640, 480 HW
            results_list = self.image_processor.postprocess(preds=outputs, newimagesize=imgsize, origimageshapes=target_sizes)
            #result: List[Dict[str, Tensor]]: resdict["boxes"], resdict["scores"], resdict["labels"]
            #bounding boxes in (xmin, ymin, xmax, ymax) format
            results = results_list[0]
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()] #[471.16, 209.09, 536.17, 347.85]#[xmin, ymin, xmax, ymax]
            print(
                f"Detected {self.id2label[int(label.item())]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        #pilimage = self.draw_bbox(self.image, results, outputpath="output/objectdetection_test.png")
        class_names = list(self.id2label.values())
        result_img = visualize_results(image=self.image, boxes=results["boxes"], \
            labels=results["labels"], scores=results["scores"],
            class_names=class_names, output_path="output/objectdetection_testresult1.jpg")
        return result_img

    def draw_bbox(self, image, results, outputpath="output/ImageDraw.png"):
        pilimage=Image.fromarray(image)#numpy HWC
        draw = ImageDraw.Draw(pilimage)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
            draw.text((x, y), self.id2label[int(label.item())], fill="white")
        pilimage.save(outputpath)
        return pilimage
    
    def universalsegmentation_postprocessing(self, outputs):
        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        #These queries are learnable embeddings that the model uses to detect and segment objects.
        #Each value in class_queries_logits represents the raw, unnormalized scores (logits) for the class of a corresponding object query.
        class_queries_logits = outputs.class_queries_logits
        #Applies a softmax function and then argmax along the num_queries dimension to get predicted class indices.
        predicted_classes = torch.nn.functional.softmax(class_queries_logits, dim=1)
        predicted_class_indices = torch.argmax(predicted_classes, dim=1)
        #the resulting tensor will have the shape of (batch_size,), as the num_queries dimension has been reduced to a single index per batch.
    
        #masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        #Each (height, width) slice within masks_queries_logits represents the raw, unnormalized scores (logits) for the segmentation mask of a corresponding object query.
        masks_queries_logits = outputs.masks_queries_logits
        
        result = self.image_processor.post_process_instance_segmentation(outputs, target_sizes=self.org_sizeHW)[0]
        print(result.keys())
        predicted_instance_map = result["segmentation"]
        
        class_names = list(self.id2label.values())
        result_img = visualize_results(image=self.image, instance_seg=predicted_instance_map, \
            class_names=class_names, output_path="output/segmentation_testresult1.jpg",
            label_segments=True)
        
    def segmentation_postprocessing(self, outputs):
        if self.model_type=="huggingface":
            predicted = outputs.logits #shape (batch_size, num_labels, height/4, width/4) [1, 19, 120, 120]
        else:
            predicted  = outputs
        
        upsampled_logits = torch.nn.functional.interpolate(
            predicted,
            size=self.org_sizeHW, #height/width
            mode="bilinear",
            align_corners=False,
        )#[1, 19, 415, 612]

        pred_seg = upsampled_logits.argmax(dim=1)[0] #[415, 612]
        pred_seg_np = pred_seg.cpu().numpy().astype(np.uint8) #ensure correct data type for image creation
        #draw_segmentation(pred_seg)
        class_names = list(self.id2label.values())
        result_img = visualize_results(image=self.image, semantic_seg=pred_seg_np, \
            class_names=class_names, output_path="output/segmentation_testresult1.jpg",
            label_segments=True)
        return result_img
        
    def depth_postprocessing(self, outputs, recolor=True):
        if self.model_type=="huggingface":
            predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
        else:
            predicted_depth  = outputs
        # interpolate to original size
        #The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
        predicted_depth_input = predicted_depth.unsqueeze(1) #[1, 1, 384, 576]

        #read the height/width from image's shape
        target_size = self.org_sizeHW
        
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

    def load_ImageNetlabels(self, filepath='sampledata/imagenet_labels.txt'):
        with open(filepath, 'r') as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]#trim string of labels
        return labels
    #labels=load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')

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

def MyVisionInferencetest_objectdetection(task="object-detection"):

    #url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"

    myinference = MyVisionInference(model_name="facebook/detr-resnet-50", task=task, model_type="huggingface", image_maxsize=480)
    results = myinference(url) #results is PIL image

def MyVisionInferencetest_seg(task="semantic-segmentation"):
    #https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024
    #semantic-segmentation: "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" or instance "facebook/mask2former-swin-large-cityscapes-instance"
    #Panoptic segmentation: facebook/mask2former-swin-large-cityscapes-panoptic
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
    myinference = MyVisionInference(model_name="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", task=task, model_type="huggingface", image_maxsize=480)
    results = myinference(url) #results is PIL image

def MyVisionInferencetest_instanceseg(task="instance-segmentation"):
    #https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024
    #semantic-segmentation: "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" or instance "facebook/mask2former-swin-large-cityscapes-instance"
    #Panoptic segmentation: facebook/mask2former-swin-large-cityscapes-panoptic
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
    myinference = MyVisionInference(model_name="facebook/mask2former-swin-large-cityscapes-instance", task=task, model_type="huggingface", image_maxsize=480)
    results = myinference(url) #results is PIL image

    
def MyVisionInferencetest_zeroshot(task="zeroshot-objectdetection"):
    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    imagepath='./sampledata/bus.jpg'
    #inference
    im0 = cv2.imread(imagepath) #(1080, 810, 3)
    imgs = [im0]

    myinference = MyVisionInference(model_name="huggingface/yolov8-world-m", task=task, model_type="huggingface")
    target_objects = ["person", "car", "bicycle", "motorcycle", "truck", "bus", "traffic light", "stop sign"]
    confidences = myinference(image=imagepath, text=target_objects)
    print(confidences)
    
if __name__ == "__main__":
    MyVisionInferencetest_seg()
    MyVisionInferencetest_objectdetection()
    #MyVisionInferencetest_zeroshot()