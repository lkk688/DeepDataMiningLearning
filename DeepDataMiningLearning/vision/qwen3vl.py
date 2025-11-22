import torch
import json
import re
import time
import os
from typing import List, Dict, Optional, Union, Any
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen3VLWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct", device: str = "cuda"):
        print(f"Loading {model_name} on {device}...")
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("Model loaded successfully.")

    def _measure_performance(self, func, *args, **kwargs):
        """
        Internal helper to measure Latency and GPU Memory of a function call.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        
        # Run the function
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        memory_gb = 0.0
        if torch.cuda.is_available():
            memory_bytes = torch.cuda.max_memory_allocated()
            memory_gb = memory_bytes / (1024 ** 3)
            
        return result, latency, memory_gb

    def _generate_inputs(self, messages_list: List[List[Dict]]) -> Dict:
        """
        Prepares batch inputs for the model.
        """
        texts = []
        all_image_inputs = []
        all_video_inputs = []

        # Process each conversation in the batch individually to get text/images
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            image_input, video_input = process_vision_info(messages)
            if image_input: 
                all_image_inputs.extend(image_input)
            if video_input: 
                all_video_inputs.extend(video_input)

        # Handle case where no images are present
        if not all_image_inputs: all_image_inputs = None
        if not all_video_inputs: all_video_inputs = None

        # Batch process using the processor with padding
        inputs = self.processor(
            text=texts,
            images=all_image_inputs,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _clean_json(self, raw_text: str) -> List[Dict]:
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
        except Exception as e:
            return []

    # --- Single Inference Wrapper ---
    def generate_single(self, image_input: Union[str, Image.Image], prompt: str) -> str:
        """
        Standard single-image inference.
        """
        messages = [[
            {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": prompt}]}
        ]]
        
        inputs = self._generate_inputs(messages)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # --- Batch Inference ---
    def batch_inference(self, image_paths: List[str], prompts: List[str]) -> List[str]:
        """
        Runs inference on a batch of images and prompts simultaneously.
        """
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images must match number of prompts.")

        # Construct batch messages
        messages_batch = []
        for img, prompt in zip(image_paths, prompts):
            messages_batch.append([
                {
                    "role": "user", 
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
                }
            ])

        inputs = self._generate_inputs(messages_batch)
        
        # Generate for the whole batch
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # --- Benchmarked Wrappers ---
    def benchmark_single(self, image_path: str, task_name: str = "Single Inference"):
        print(f"\n--- Benchmarking: {task_name} ---")
        prompt = "Describe this image in detail."
        
        # Run and Measure
        output, latency, mem = self._measure_performance(self.generate_single, image_path, prompt)
        
        print(f"Output Preview: {output[:100]}...")
        print(f"Latency: {latency:.4f} seconds")
        print(f"Peak GPU Memory: {mem:.2f} GB")
        return output

    def benchmark_batch(self, image_paths: List[str], prompts: List[str]):
        print(f"\n--- Benchmarking: Batch Inference (Batch Size: {len(image_paths)}) ---")
        
        # Run and Measure
        outputs, latency, mem = self._measure_performance(self.batch_inference, image_paths, prompts)
        
        for i, out in enumerate(outputs):
            print(f"[Sample {i+1}] Preview: {out[:100]}...")
            
        print(f"Total Latency: {latency:.4f} seconds")
        print(f"Avg Latency per Sample: {latency / len(image_paths):.4f} seconds")
        print(f"Peak GPU Memory: {mem:.2f} GB")
        return outputs

    # --- Task Specific Methods (Re-integrated) ---
    def detect_objects_3d(self, image_path: str, target_object: str) -> List[Dict]:
        prompt = (
            f"Analyze the 3D structure of '{target_object}'. "
            "Return a JSON list with keys: 'label', 'front_face' [ymin, xmin, ymax, xmax], "
            "'back_face' [ymin, xmin, ymax, xmax], and 'description'."
        )
        raw_output = self.generate_single(image_path, prompt)
        return self._clean_json(raw_output)

    def visualize_detections(self, image_path: str, detections: List[Dict], output_path: str, mode: str = '2d'):
        # (Same visualization logic as previous step)
        if image_path.startswith("http"):
            import requests
            from io import BytesIO
            img = Image.open(BytesIO(requests.get(image_path).content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        def scale(box):
            if not box or len(box) != 4: return None
            return [box[1]/1000*width, box[0]/1000*height, box[3]/1000*width, box[2]/1000*height]

        for det in detections:
            label = det.get('label', 'Object')
            if mode == '3d':
                f = scale(det.get('front_face'))
                b = scale(det.get('back_face'))
                if f and b:
                    draw.rectangle(f, outline="cyan", width=3)
                    draw.rectangle(b, outline="blue", width=2)
                    draw.line([(f[0],f[1]), (b[0],b[1])], fill="cyan", width=1) # TL
                    draw.line([(f[2],f[1]), (b[2],b[1])], fill="cyan", width=1) # TR
                    draw.line([(f[2],f[3]), (b[2],b[3])], fill="cyan", width=1) # BR
                    draw.line([(f[0],f[3]), (b[0],b[3])], fill="cyan", width=1) # BL
                    draw.text((f[0], f[1]-15), label, fill="cyan")
        
        img.save(output_path)
        print(f"Saved to {output_path}")

# --- Main Menu ---

def main():
    model_name = 'Qwen/Qwen3-VL-8B-Instruct'
    vlm = Qwen3VLWrapper(model_name=model_name)
    
    # Test Data
    img_url_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    img_url_2 = "https://templates.invoicehome.com/invoice-template-us-classic-white-750px.png"
    
    while True:
        print("\n--- Qwen3-VL Benchmark & Action Menu ---")
        print("1. Benchmark Single Inference (Latency & Memory)")
        print("2. Benchmark Batch Inference (Batch Size = 2)")
        print("3. 3D Object Detection (w/ Viz)")
        print("q. Quit")
        
        choice = input("Select action: ").strip().lower()
        
        if choice == 'q': break
            
        elif choice == '1':
            vlm.benchmark_single(img_url_1)
            
        elif choice == '2':
            # Batch of 2 different images with different prompts
            imgs = [img_url_1, img_url_2]
            prompts = ["Describe the scene.", "Parse this document."]
            vlm.benchmark_batch(imgs, prompts)
            
        elif choice == '3':
            print("Running 3D Detection...")
            # We assume 'person' for the demo image
            res = vlm.detect_objects_3d(img_url_1, "person")
            print(f"Results: {res}")
            vlm.visualize_detections(img_url_1, res, "bench_3d.jpg", mode='3d')

if __name__ == "__main__":
    main()