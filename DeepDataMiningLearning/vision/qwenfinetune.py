import os
import json
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, # Compatible with Qwen3-VL architecture
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct" # Replace with Qwen3-VL path if available
OUTPUT_DIR = "./qwen3_vl_coco_lora"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 64
LORA_ALPHA = 16
BATCH_SIZE = 4 

# Paths to your COCO data
COCO_ANN_FILE = "./data/coco/annotations/instances_train2017.json"
COCO_IMG_DIR = "./data/coco/train2017"

# --- 1. COCO Dataset Class ---
class CocoDetectionDataset(Dataset):
    """
    Parses COCO JSON format:
    bbox: [x_min, y_min, width, height]
    """
    def __init__(self, ann_file: str, img_dir: str, processor: Any):
        self.img_dir = img_dir
        self.processor = processor
        
        print(f"Loading COCO annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        # 1. Map Categories: ID -> Name
        self.cats = {cat['id']: cat['name'] for cat in coco['categories']}
        
        # 2. Map Images: ID -> File Name
        self.imgs = {img['id']: img for img in coco['images']}
        
        # 3. Group Annotations by Image ID
        self.img_to_anns = {img['id']: [] for img in coco['images']}
        for ann in coco['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
            
        # List of valid image IDs (only those with annotations if desired, or all)
        self.ids = list(self.imgs.keys())
        print(f"Loaded {len(self.ids)} images.")

    def __len__(self):
        return len(self.ids)

    def normalize_box(self, bbox, width, height):
        """
        Convert absolute pixels to 0-1000 normalized coordinates.
        """
        # COCO format is [x_min, y_min, width, height]
        x_min, y_min, w_box, h_box = bbox
        
        # Convert to [x_min, y_min, x_max, y_max]
        x_max = x_min + w_box
        y_max = y_min + h_box
        
        return [
            int(x_min / width * 1000),
            int(y_min / height * 1000),
            int(x_max / width * 1000),
            int(y_max / height * 1000)
        ]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        anns = self.img_to_anns[img_id]
        
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load Image
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except (FileNotFoundError, OSError):
            print(f"Warning: Skipping missing/corrupt image {image_path}")
            # Return a fallback or handle generically (simplified here)
            return self.__getitem__((idx + 1) % len(self))
        
        # 1. Create Prompt
        # Collect all unique labels in this image
        target_labels = list(set(self.cats[ann['category_id']] for ann in anns))
        target_str = ", ".join(target_labels)
        
        prompt_text = f"Detect the following objects: {target_str}. Output valid JSON."
        
        # 2. Create Target Response (Ground Truth)
        ground_truth_objs = []
        for ann in anns:
            norm_box = self.normalize_box(ann['bbox'], w, h)
            # Ensure coords are clamped 0-1000
            norm_box = [max(0, min(1000, c)) for c in norm_box]
            
            ground_truth_objs.append({
                "box_2d": norm_box, # [xmin, ymin, xmax, ymax]
                "label": self.cats[ann['category_id']]
            })
        
        response_text = json.dumps(ground_truth_objs, ensure_ascii=False)
        
        # 3. Construct Conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            }
        ]

        # 4. Preprocess using Qwen Utils
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Create Labels (Mask user prompts)
        # Simple approach: Clone input_ids.
        # For production, you should strictly mask the instruction part.
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

# --- 2. Data Collator ---
@dataclass
class DataCollatorForVisualSFT:
    processor: AutoProcessor

    def __call__(self, features):
        first = features[0]
        batch = {}

        for k, v in first.items():
            if k in ["pixel_values", "image_grid_thw"]:
                if k == "pixel_values":
                    batch[k] = [f[k] for f in features] 
                else:
                    batch[k] = torch.stack([f[k] for f in features])
            elif v is not None and not isinstance(v, str):
                batch[k] = torch.stack([f[k] for f in features])

        return batch

# --- 3. Main Training Function ---
def train():
    # A. Load Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # B. Load Model (QLoRA 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # C. Configure LoRA
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # D. Initialize COCO Dataset
    if not os.path.exists(COCO_ANN_FILE):
        print(f"COCO Annotation file not found at {COCO_ANN_FILE}")
        print("Please verify paths or update the 'COCO_ANN_FILE' variable.")
        return

    train_dataset = CocoDetectionDataset(COCO_ANN_FILE, COCO_IMG_DIR, processor)

    # E. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=False,
        bf16=True, 
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit", 
        report_to="none",
        remove_unused_columns=False, 
    )

    # F. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForVisualSFT(processor),
    )

    print("Starting training...")
    trainer.train()
    
    # G. Save
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()