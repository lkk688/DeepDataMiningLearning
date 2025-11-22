import os
import sys
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from DeepDataMiningLearning.llm.rl_dataset import build_rl_dataset, RLDataModule

class RLTrainer:
    """
    Unified Trainer class for TRL (Transformer Reinforcement Learning).
    Supports DPO and PPO (extensible).
    """
    def __init__(self, args, model, tokenizer, train_dataset, valid_dataset):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train(self):
        """
        Execute the training pipeline based on the RL method.
        """
        peft_config = self._get_peft_config()

        if self.args.rl_method == "dpo":
            self._train_dpo(peft_config)
        elif self.args.rl_method == "ppo":
            self._train_ppo(peft_config)
        else:
            raise ValueError(f"Unsupported RL method: {self.args.rl_method}")

    def _get_peft_config(self):
        """
        Setup LoRA/PEFT configuration.
        """
        if not self.args.use_peft:
            return None
            
        print("🛠️  Using LoRA (PEFT)")
        # Default to Qwen/LLaMA modules if not specified
        target_modules = self.args.lora_target_modules if self.args.lora_target_modules else ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules, 
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _train_dpo(self, peft_config):
        """
        Train using Direct Preference Optimization (DPO).
        """
        print(f"\n🏋️  Initializing DPO Trainer...")
        
        training_args = DPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epochs,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            bf16=False, # Use False for CPU/MPS testing, True for Ampere GPU
            remove_unused_columns=False,
            max_length=self.args.max_length,
            max_prompt_length=self.args.max_prompt_length,
            beta=0.1, # DPO beta
        )
        
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None, # DPO trainer creates a copy if None (or uses PEFT adapter)
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        
        print("🚀 Starting Training...")
        trainer.train()
        
        print(f"\n💾 Saving model to {self.args.output_dir}")
        trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print("✅ Done!")

    def _train_ppo(self, peft_config):
        """
        Train using Proximal Policy Optimization (PPO).
        Placeholder for future implementation.
        """
        print("⚠️  PPO training is not yet fully implemented in this wrapper.")
        # Implementation would involve:
        # 1. PPOTrainer setup
        # 2. Rollout generation loop
        # 3. Reward computation
        # 4. Step updates
        raise NotImplementedError("PPO support coming soon.")

def main():
    parser = argparse.ArgumentParser(description="RLHF/RLAIF Training Script using TRL")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face model name")
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset name (e.g., Anthropic/hh-rlhf)")
    parser.add_argument("--dataset_files", type=str, nargs="+", help="Local dataset files (json/jsonl)")
    parser.add_argument("--rl_method", type=str, default="dpo", choices=["dpo"], help="RL method (currently only dpo supported)")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/rl_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    
    # PEFT / LoRA
    parser.add_argument("--use_peft", action="store_true", help="Use LoRA/PEFT")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="Target modules for LoRA (e.g. q_proj v_proj)")
    
    # Quantization (Optional)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")

    args = parser.parse_args()
    
    print(f"🚀 Starting RL Training ({args.rl_method.upper()})")
    print(f"   Model: {args.model_name}")
    print(f"   Output: {args.output_dir}")
    
    # ============================================================
    # 1. Load Data via RLDataModule
    # ============================================================
    print("\n📊 Loading Dataset...")
    # Create a namespace-like object for build_rl_dataset if needed, or just pass args
    # We need to map args to what build_rl_dataset expects
    class DataArgs:
        hf_name = args.dataset_name
        files = args.dataset_files
        tokenizer = "hf:" + args.model_name # Use the model's tokenizer
        vocab_size = None
        max_prompt_length = args.max_prompt_length
        max_response_length = args.max_length - args.max_prompt_length
        batch_size = args.batch_size
        # Default keys
        prompt_key = "prompt"
        chosen_key = "chosen"
        rejected_key = "rejected"
        
    data_module = build_rl_dataset(args.rl_method, DataArgs())
    
    # Convert RLDataModule datasets (list of dicts) to HF Datasets for TRL
    # RLDataModule stores raw data in .data attribute of the dataset
    train_raw = data_module.train_dataset.data
    valid_raw = data_module.valid_dataset.data
    
    # TRL expects columns: prompt, chosen, rejected
    # We need to ensure keys match what TRL expects or map them
    # Our RLDataConfig has the keys used in the raw data.
    
    def to_hf_dataset(raw_data, cfg):
        # Map custom keys to TRL expected keys
        formatted_data = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        for item in raw_data:
            formatted_data["prompt"].append(item.get(cfg.prompt_key, ""))
            formatted_data["chosen"].append(item.get(cfg.chosen_key, ""))
            formatted_data["rejected"].append(item.get(cfg.rejected_key, ""))
        return Dataset.from_dict(formatted_data)

    train_dataset = to_hf_dataset(train_raw, data_module.cfg)
    valid_dataset = to_hf_dataset(valid_raw, data_module.cfg)
    
    print(f"✅ Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")

    # ============================================================
    # 2. Load Model & Tokenizer
    # ============================================================
    print("\n🧠 Loading Model & Tokenizer...")
    
    # Quantization Config
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # ============================================================
    # 3. Initialize & Run Trainer
    # ============================================================
    trainer = RLTrainer(args, model, tokenizer, train_dataset, valid_dataset)
    trainer.train()

if __name__ == "__main__":
    main()
