import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import HfApi, create_repo, notebook_login
import os
import onnx #pip install onnx onnxruntime-gpu
import onnxruntime
from PIL import Image
import numpy as np
from typing import List, Dict

# ======================
# 1. Custom Config Class
# ======================
class FasterRCNNConfig(PretrainedConfig):
    model_type = "fasterrcnn"
    
    def __init__(
        self,
        num_classes=5,
        min_size=800,
        max_size=1333,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        self.id2label = {i: f"class_{i}" for i in range(num_classes)}
        self.label2id = {f"class_{i}": i for i in range(num_classes)}

# ======================
# 2. Custom Model Class
# ======================
class FasterRCNNModel(PreTrainedModel):
    config_class = FasterRCNNConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Create model architecture
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        self.model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=config.min_size,
            max_size=config.max_size
        )
    
    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        return self.model(images)

# ======================
# 3. Preprocessing Utilities
# ======================
class FasterRCNNProcessor:
    def __init__(self, config):
        self.config = config
        self.size_divisibility = 32
        
    def __call__(self, images: List[Image.Image]):
        # Convert images to tensors and normalize
        processed_images = []
        for image in images:
            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            processed_images.append(image)
        
        # Create image tensors with padding
        images = self.batch_images(processed_images)
        return {"pixel_values": images}
    
    def batch_images(self, images):
        # Pad images to same size
        max_size = max([img.shape[-2:] for img in images], dim=1)
        max_h, max_w = max_size
        
        if self.size_divisibility > 0:
            stride = self.size_divisibility
            max_h = (max_h + stride - 1) // stride * stride
            max_w = (max_w + stride - 1) // stride * stride
        
        batch_shape = (len(images), 3, max_h, max_w)
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        
        return batched_imgs

# ======================
# 4. ONNX Conversion
# ======================
def convert_to_onnx(model, output_path, dummy_input):
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"}
        },
        opset_version=12
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

# ======================
# 5. Inference Pipeline
# ======================
class FasterRCNNPipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def __call__(self, images, threshold=0.5):
        # Preprocess
        inputs = self.processor(images)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = []
        for output in outputs:
            keep = output['scores'] > threshold
            results.append({
                "boxes": output['boxes'][keep].cpu().numpy(),
                "scores": output['scores'][keep].cpu().numpy(),
                "labels": output['labels'][keep].cpu().numpy()
            })
        
        return results

# ======================
# 6. ONNX Inference Class
# ======================
class FasterRCNNOnnxInference:
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, pixel_values):
        outputs = self.session.run(None, {self.input_name: pixel_values.numpy()})
        return {
            "boxes": outputs[0],
            "scores": outputs[1],
            "labels": outputs[2]
        }

# ======================
# 7. Save to Hugging Face
# ======================
def save_to_hub(model, processor, repo_name):
    # Create directory
    os.makedirs(repo_name, exist_ok=True)
    
    # Save model and config
    model.save_pretrained(repo_name)
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 800, 800)
    
    # Convert to ONNX
    convert_to_onnx(
        model.model, 
        os.path.join(repo_name, "model.onnx"), 
        dummy_input
    )
    
    # Save processor (simplified)
    torch.save(processor, os.path.join(repo_name, "processor.bin"))
    
    # Create inference script
    with open(os.path.join(repo_name, "inference.py"), "w") as f:
        f.write("""from PIL import Image
import torch
import numpy as np
from transformers import PreTrainedModel
import onnxruntime

class FasterRCNNInference:
    def __init__(self, model_path):
        # Load ONNX model
        self.ort_session = onnxruntime.InferenceSession(f"{model_path}/model.onnx")
        self.processor = torch.load(f"{model_path}/processor.bin")
        
    def predict(self, images, threshold=0.5):
        # Preprocess
        inputs = self.processor(images)
        pixel_values = inputs["pixel_values"].numpy()
        
        # Run inference
        outputs = self.ort_session.run(None, {"pixel_values": pixel_values})
        
        # Post-process
        results = []
        for boxes, scores, labels in zip(outputs[0], outputs[1], outputs[2]):
            keep = scores > threshold
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep]
            })
        
        return results
""")
    
    # Create README
    with open(os.path.join(repo_name, "README.md"), "w") as f:
        f.write(f"""---
tags:
- object-detection
- vision
- fasterrcnn
- onnx
license: apache-2.0
---

# Custom Faster R-CNN Model

## Usage

### PyTorch
```python
from inference import FasterRCNNInference
from PIL import Image

model = FasterRCNNInference(".")
image = Image.open("your_image.jpg")
results = model.predict([image])""")


def upload2hf():
    # Initialize config and model
    config = FasterRCNNConfig(num_classes=80)
    model = FasterRCNNModel(config)
    
    # Initialize processor
    processor = FasterRCNNProcessor(config)

    # Save to hub
    save_to_hub(model, processor, "myfasterrcnn-detector")

    # Login and upload
    #terminal: huggingface-cli login
    api = HfApi()
    create_repo("lkk688/myfasterrcnn-detector", exist_ok=True)
    api.upload_folder(
        folder_path="myfasterrcnn-detector",
        repo_id="lkk688/myfasterrcnn-detector",
        repo_type="model"
    )

# ======================
# 8. Register Custom Architecture
# ======================
def register_fasterrcnn_architecture():
    """
    Register the FasterRCNN model architecture with the Hugging Face transformers library
    for full integration with the transformers ecosystem.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForObjectDetection
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_FOR_OBJECT_DETECTION_MAPPING
    
    # Register the config
    CONFIG_MAPPING.register("fasterrcnn", FasterRCNNConfig)
    
    # Register the model architecture
    MODEL_MAPPING.register(FasterRCNNConfig, FasterRCNNModel)
    MODEL_FOR_OBJECT_DETECTION_MAPPING.register(FasterRCNNConfig, FasterRCNNModel)
    
    print("FasterRCNN architecture registered successfully with Hugging Face transformers")
    
def hf_pipeline_inference(model_id="lkk688/myfasterrcnn-detector", image_path="sampledata/bus.jpg", 
                          threshold=0.5, device=None, visualize=True, output_path=None):
    """
    Run inference using the Hugging Face pipeline API.
    
    Args:
        model_id (str): Hugging Face model ID or local path
        image_path (str): Path to the input image or directory
        threshold (float): Confidence threshold for detections
        device (int or str): Device to run inference on (-1 for CPU, 0+ for GPU)
        visualize (bool): Whether to visualize and save the results
        output_path (str): Path to save the visualization
        
    Returns:
        list: Detection results
    """
    from transformers import pipeline
    from PIL import Image
    import cv2
    import numpy as np
    import os
    
    # Register architecture if needed
    register_fasterrcnn_architecture()
    
    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    # Create pipeline
    object_detector = pipeline(
        "object-detection", 
        model=model_id,
        device=device,
        threshold=threshold
    )
    
    # Check if input is a directory or file
    if os.path.isdir(image_path):
        # Process all images in directory
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        results = []
        
        for img_file in image_files:
            try:
                # Load image
                image = Image.open(img_file).convert("RGB")
                
                # Run inference
                detections = object_detector(image)
                
                # Visualize if requested
                if visualize:
                    visualized_img = visualize_detections(img_file, detections, output_path)
                
                results.append({
                    "file": img_file,
                    "detections": detections
                })
                
                # Print results
                print(f"\nResults for {os.path.basename(img_file)}:")
                for detection in detections:
                    print(f"Detected {detection['label']} with confidence {round(detection['score'], 3)} "
                          f"at location {detection['box']}")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        return results
    else:
        # Process single image
        image = Image.open(image_path).convert("RGB")
        
        # Run inference
        detections = object_detector(image)
        
        # Visualize if requested
        if visualize:
            visualized_img = visualize_detections(image_path, detections, output_path)
        
        # Print results
        print(f"\nResults for {os.path.basename(image_path)}:")
        for detection in detections:
            print(f"Detected {detection['label']} with confidence {round(detection['score'], 3)} "
                  f"at location {detection['box']}")
        
        return detections

def visualize_detections(image_path, detections, output_path=None):
    """
    Visualize detection results on an image.
    
    Args:
        image_path (str): Path to the input image
        detections (list): Detection results from pipeline
        output_path (str, optional): Path to save the visualization
        
    Returns:
        np.ndarray: Visualization image with bounding boxes
    """
    import cv2
    import numpy as np
    import os
    from PIL import Image
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        # Convert PIL image to OpenCV format
        img = np.array(image_path)
        img = img[:, :, ::-1].copy()  # RGB to BGR
    
    # Draw boxes
    for detection in detections:
        # Get box coordinates
        box = detection['box']
        x1, y1, width, height = int(box['xmin']), int(box['ymin']), int(box['width']), int(box['height'])
        x2, y2 = x1 + width, y1 + height
        
        # Get label and score
        label = detection['label']
        score = detection['score']
        
        # Generate color based on label hash
        label_hash = hash(label) % 255
        color = (label_hash, 255 - label_hash, (label_hash + 125) % 255)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label_text = f"{label}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save if output path is provided
    if output_path:
        # If output_path is a directory, create filename
        if os.path.isdir(output_path):
            if isinstance(image_path, str):
                base_filename = os.path.basename(image_path)
                output_file = os.path.join(output_path, f"{os.path.splitext(base_filename)[0]}_detected{os.path.splitext(base_filename)[1]}")
            else:
                output_file = os.path.join(output_path, "detection_result.jpg")
        else:
            output_file = output_path
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save image
        cv2.imwrite(output_file, img)
        print(f"Visualization saved to {output_file}")
    
    return img

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FasterRCNN Model Operations")
    parser.add_argument("--action", type=str, default="pipeline", 
                        choices=["upload", "inference", "pipeline"],
                        help="Action to perform: upload to HF, run ONNX inference, or run HF pipeline")
    parser.add_argument("--model_id", type=str, default="lkk688/myfasterrcnn-detector",
                        help="Hugging Face model ID or local path")
    parser.add_argument("--image", type=str, default="sampledata/bus.jpg",
                        help="Path to input image or directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output directory or file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=int, default=None,
                        help="Device to run inference on (-1 for CPU, 0+ for GPU)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Visualize detection results")
    
    args = parser.parse_args()
    
    if args.action == "upload":
        upload2hf()
    elif args.action == "inference":
        # Use the ONNX inference
        from inference import FasterRCNNInference
        model = FasterRCNNInference(args.model_id)
        
        if os.path.isdir(args.image):
            results = model.process_directory(args.image, args.threshold, args.visualize, args.output)
        else:
            result = model.process_image_file(args.image, args.threshold, args.visualize, args.output)
    elif args.action == "pipeline":
        # Use the HF pipeline
        hf_pipeline_inference(
            model_id=args.model_id,
            image_path=args.image,
            threshold=args.threshold,
            device=args.device,
            visualize=args.visualize,
            output_path=args.output
        )