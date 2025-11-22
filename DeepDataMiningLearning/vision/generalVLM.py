import os
import time
import torch
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# --- Dependency Checking ---
try:
    import transformers
    from transformers import (
        AutoProcessor, AutoModelForCausalLM, AutoTokenizer,
        Blip2Processor, Blip2ForConditionalGeneration,
        LlavaProcessor, LlavaForConditionalGeneration,
        AutoModelForImageTextToText, # Qwen Standard
        Qwen2_5_VLForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not found.")

try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False
    print("Warning: qwen-vl-utils not found. Qwen advanced features will be limited.")

# --- Constants ---
MODEL_TYPE_BLIP2 = "blip2"
MODEL_TYPE_LLAVA = "llava"
MODEL_TYPE_SMOLVLM = "smolvlm"
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_TEXT = "text"

class GeneralVLMClient:
    """
    A unified wrapper for Text and Vision LLMs.
    Supports advanced tasks: OCR, 2D/3D Detection, Visualization, and Benchmarking.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required.")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._determine_model_type(model_name)
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._initialize_model()

    def _determine_model_type(self, model_name: str) -> str:
        name = model_name.lower()
        if "blip" in name: return MODEL_TYPE_BLIP2
        if "llava" in name: return MODEL_TYPE_LLAVA
        if "smolvlm" in name: return MODEL_TYPE_SMOLVLM
        if "qwen" in name and "vl" in name: return MODEL_TYPE_QWEN
        return MODEL_TYPE_TEXT

    def _initialize_model(self):
        print(f"Loading {self.model_name} ({self.model_type}) on {self.device}...")
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        
        try:
            if self.model_type == MODEL_TYPE_QWEN:
                # Qwen 2.5/3 VL Optimization
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer

                # --- CRITICAL FIX: FORCE LEFT PADDING FOR BATCH GENERATION ---
                # Decoder-only models need left padding to generate valid completions
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            elif self.model_type == MODEL_TYPE_LLAVA:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name, dtype=dtype, device_map="auto"
                )
                self.tokenizer = self.processor.tokenizer
                self.tokenizer.padding_side = "left"

            elif self.model_type == MODEL_TYPE_BLIP2:
                self.processor = Blip2Processor.from_pretrained(self.model_name)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name, dtype=dtype, device_map="auto"
                )

            elif self.model_type == MODEL_TYPE_SMOLVLM:
                from transformers import AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name, dtype=dtype, device_map="auto"
                )
                self.tokenizer = self.processor.tokenizer

            else: # Text Only
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, dtype=dtype, device_map="auto"
                )
                self.tokenizer.padding_side = "left"

            print("Model loaded successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")

    # --- Core Generation Logic ---

    def _measure_performance(self, func, *args, **kwargs):
        """Internal benchmarking wrapper"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        memory_gb = 0.0
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
        return result, latency, memory_gb

    def process_vision(self, 
                       image_input: Union[str, Image.Image, List[Union[str, Image.Image]]], 
                       prompt: str, 
                       max_tokens: int = 1024,
                       system_prompt: Optional[str] = None) -> str:
        """
        Unified vision processing method. Handles single or batch images.
        """
        # Normalization: Ensure input is a list of images
        if not isinstance(image_input, list):
            image_input = [image_input]
        
        # Load images if paths/urls are provided
        loaded_images = []
        for img in image_input:
            if isinstance(img, str):
                if img.startswith("http"):
                    loaded_images.append(img) # Keep URL for Qwen utils, load for others
                else:
                    loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img)

        if self.model_type == MODEL_TYPE_QWEN and QWEN_UTILS_AVAILABLE:
            return self._generate_qwen(loaded_images, prompt, max_tokens, system_prompt)
        elif self.model_type == MODEL_TYPE_LLAVA:
            return self._generate_llava(loaded_images[0], prompt, max_tokens)
        elif self.model_type == MODEL_TYPE_BLIP2:
            return self._generate_blip2(loaded_images[0], prompt, max_tokens)
        else:
            return "Model not supported for vision tasks or generic fallback failed."

    def _generate_qwen(self, images, prompt, max_tokens, system_prompt=None):
        # Qwen-VL-Utils handles image preprocessing nicely
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Apply template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _generate_llava(self, image, prompt, max_tokens):
        # LLaVA logic
        if isinstance(image, str): # LLaVA needs PIL
             image = Image.open(requests.get(image, stream=True).raw if image.startswith("http") else image).convert("RGB")
             
        formatted_prompt = f"<image>\n{prompt}" # Standard LLaVA prompt
        inputs = self.processor(image, formatted_prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True).replace(formatted_prompt, "").strip()

    def _generate_blip2(self, image, prompt, max_tokens):
        if isinstance(image, str):
             image = Image.open(requests.get(image, stream=True).raw if image.startswith("http") else image).convert("RGB")
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True).strip()

    # --- Advanced Task Wrappers ---

    def _clean_json(self, raw_text: str) -> List[Dict]:
        """Helper to robustly extract JSON from model output."""
        try:
            if "```json" in raw_text:
                json_str = re.search(r"```json(.*?)```", raw_text, re.DOTALL).group(1)
            elif "```" in raw_text:
                json_str = re.search(r"```(.*?)```", raw_text, re.DOTALL).group(1)
            else:
                start = raw_text.find('[')
                end = raw_text.rfind(']') + 1
                json_str = raw_text[start:end] if start != -1 else raw_text
            return json.loads(json_str)
        except Exception:
            return [{"error": "Could not parse JSON", "raw_output": raw_text}]

    def benchmark(self, image_path: str, prompt: str = "Describe this image."):
        """Runs inference and returns output + performance stats."""
        print(f"\n--- Benchmarking: {self.model_name} ---")
        output, latency, mem = self._measure_performance(self.process_vision, image_path, prompt)
        print(f"Latency: {latency:.4f}s | VRAM: {mem:.2f} GB")
        return {"output": output, "latency": latency, "memory_gb": mem}

    def parse_document(self, image_path: str) -> str:
        """Extracts text/tables in Markdown."""
        prompt = "Read the text in this image and output it in structured Markdown format."
        return self.process_vision(image_path, prompt, max_tokens=2048)

    def detect_objects_2d(self, image_path: str, target_objects: List[str]) -> List[Dict]:
        """Returns JSON list of 2D bounding boxes."""
        objs = ", ".join(target_objects)
        # FIX: Request native XMIN, YMIN order
        prompt = (
            f"Detect these objects: {objs}. Return a JSON list. "
            "Each item must have 'label' and 'box_2d' [xmin, ymin, xmax, ymax] (0-1000 normalized)."
        )
        raw = self.process_vision(image_path, prompt)
        return self._clean_json(raw)

    def detect_objects_3d(self, image_path: str, target_object: str) -> List[Dict]:
        """
        Returns JSON list of 3D wireframe approximations.
        """
        # FIX: Request native XMIN, YMIN order
        prompt = (
            f"Analyze the 3D structure of '{target_object}'. "
            "Return a JSON list with keys:\n"
            "- 'label': Object name\n"
            "- 'front_face': [xmin, ymin, xmax, ymax] (closest face, 0-1000)\n"
            "- 'back_face': [xmin, ymin, xmax, ymax] (furthest face, 0-1000)\n"
            "- 'description': Orientation details."
        )
        raw = self.process_vision(image_path, prompt)
        return self._clean_json(raw)

    # --- Visualization ---

    def visualize_detections(self, image_input: Union[str, Image.Image], detections: List[Dict], output_path: str, mode: str = '2d'):
        """
        Draws 2D boxes or 3D wireframes on the image.
        """
        # Load Image
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                img = Image.open(BytesIO(requests.get(image_input).content)).convert("RGB")
            else:
                img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.copy()

        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        def scale(box):
            if not box or len(box) != 4: return None
            # FIX: Map index 0->x, index 1->y (Standard Qwen Output)
            # box is [xmin, ymin, xmax, ymax]
            return [
                box[0]/1000*w, # xmin
                box[1]/1000*h, # ymin
                box[2]/1000*w, # xmax
                box[3]/1000*h  # ymax
            ]

        for det in detections:
            if "error" in det: continue
            label = det.get('label', 'Object')
            
            if mode == '2d':
                box = scale(det.get('box_2d'))
                if box:
                    draw.rectangle(box, outline="#00FF00", width=3)
                    draw.text((box[0], box[1]-10), label, fill="#00FF00")
            
            elif mode == '3d':
                f = scale(det.get('front_face'))
                b = scale(det.get('back_face'))
                
                if f and b:
                    # Front Face (Cyan)
                    draw.rectangle(f, outline="cyan", width=3)
                    # Back Face (Blue)
                    draw.rectangle(b, outline="blue", width=2)
                    # Connecting Lines (Wireframe)
                    draw.line([(f[0],f[1]), (b[0],b[1])], fill="cyan", width=1) # Top-Left
                    draw.line([(f[2],f[1]), (b[2],b[1])], fill="cyan", width=1) # Top-Right
                    draw.line([(f[2],f[3]), (b[2],b[3])], fill="cyan", width=1) # Bottom-Right
                    draw.line([(f[0],f[3]), (b[0],b[3])], fill="cyan", width=1) # Bottom-Left
                    draw.text((f[0], f[1]-15), f"{label} (3D)", fill="cyan")

        img.save(output_path)
        print(f"Visualization saved to {output_path}")

# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" #"Qwen/Qwen2.5-VL-7B-Instruct" 
    
    try:
        client = GeneralVLMClient(MODEL_NAME)
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    img_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

    # 2D Object Detection Test
    print("\n=== 3. 2D Object Detection ===")
    if "qwen" in MODEL_NAME.lower():
        d2_res = client.detect_objects_2d(img_url, ["person", "dog"])
        print(f"2D Data: {d2_res}")
        client.visualize_detections(img_url, d2_res, "output_2d.jpg", mode='2d')

    # 3D Object Detection Test
    print("\n=== 4. 3D Object Detection ===")
    if "qwen" in MODEL_NAME.lower():
        # Attempts to understand depth by projecting front/back faces
        d3_res = client.detect_objects_3d(img_url, "person")
        print(f"3D Data: {d3_res}")
        client.visualize_detections(img_url, d3_res, "output_3d.jpg", mode='3d')