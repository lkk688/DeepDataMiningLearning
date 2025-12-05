import os
import sys
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# TRL Imports
from trl import (
    SFTTrainer, SFTConfig,
    DPOTrainer, DPOConfig,
    GRPOTrainer, GRPOConfig,
    PPOTrainer, PPOConfig,
    AutoModelForCausalLMWithValueHead
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# 🔗 Import Previous Data Module
# ============================================================
# Ensure this points to where you saved the previous code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# If running in the same folder, just import:
try:
    from DeepDataMiningLearning.llm.rl_dataset import build_rl_dataset, RLDataModule
except ImportError:
    # Fallback if file structure differs
    from rl_dataset import build_rl_dataset, RLDataModule

class RLTrainer:
    """
    Unified Trainer class for TRL (Transformer Reinforcement Learning).
    """
    def __init__(self, args, model, tokenizer, train_dataset, valid_dataset, ref_model=None):
        self.args = args
        self.model = model
        self.ref_model = ref_model # Needed for PPO/DPO sometimes
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train(self):
        """
        Execute the training pipeline based on the RL method.
        """
        peft_config = self._get_peft_config()
        
        # [Image of RLHF training stages diagram]

        if self.args.rl_method == "sft":
            self._train_sft(peft_config)
        elif self.args.rl_method == "dpo":
            self._train_dpo(peft_config)
        elif self.args.rl_method == "grpo":
            self._train_grpo(peft_config)
        elif self.args.rl_method == "ppo":
            self._train_ppo(peft_config)
        else:
            raise ValueError(f"Unsupported RL method: {self.args.rl_method}")

    def _get_peft_config(self):
        if not self.args.use_peft:
            return None
        
        print("🛠️  Using LoRA (PEFT)")
        # [Image of LoRA low rank adaptation architecture]
        
        target_modules = self.args.lora_target_modules if self.args.lora_target_modules else "all-linear"
        
        return LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules, 
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ============================================================
    # 1. SFT (Supervised Fine-Tuning)
    # ============================================================
    def _train_sft(self, peft_config):
        print(f"\n📚 Initializing SFT Trainer...")
        
        # 1. Initialize SFTConfig with strictly generic TrainingArguments
        # We avoid passing 'max_seq_length' or 'dataset_text_field' here to prevent 
        # TypeError in __init__ on certain TRL versions.
        sft_config = SFTConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            learning_rate=self.args.lr,
            logging_steps=10,
            save_steps=100,
            num_train_epochs=self.args.epochs,
            remove_unused_columns=False,
            gradient_accumulation_steps=self.args.grad_accum,
            bf16=torch.cuda.is_available(),
        )

        # 2. Manually set SFT-specific parameters on the config object
        # This ensures they are present for the Trainer to read, 
        # bypassing constructor argument checks.
        sft_config.max_seq_length = self.args.max_length
        sft_config.packing = False
        
        # Handle dataset column mapping
        if "text" in self.train_dataset.column_names:
            sft_config.dataset_text_field = "text"
        
        # 3. Initialize Trainer
        # Do NOT pass max_seq_length here; the trainer will read it from sft_config
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            args=sft_config,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        
        print("🚀 Starting SFT Training...")
        trainer.train()
        self._save(trainer)

    # ============================================================
    # 2. DPO (Direct Preference Optimization)
    # ============================================================
    def _train_dpo(self, peft_config):
        print(f"\n⚖️  Initializing DPO Trainer...")
        
        dpo_config = DPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            logging_steps=10,
            num_train_epochs=self.args.epochs,
            beta=0.1,
            max_length=self.args.max_length,
            max_prompt_length=self.args.max_prompt_length,
            remove_unused_columns=False,
        )
        
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model, # Optional, DPO creates copy if None
            args=dpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        
        print("🚀 Starting DPO Training...")
        trainer.train()
        self._save(trainer)

    # ============================================================
    # 3. GRPO (Group Relative Policy Optimization)
    # ============================================================
    def _train_grpo(self, peft_config):
        print(f"\n🧪 Initializing GRPO Trainer...")
        
        # GRPO expects a reward function. Defining a dummy one for syntax validity.
        # In real usage, pass a function that scores completions.
        def dummy_reward_func(prompts, completions, **kwargs):
            return [len(c) * 0.01 for c in completions] # Reward length

        grpo_config = GRPOConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            learning_rate=self.args.lr,
            logging_steps=10,
            num_train_epochs=self.args.epochs,
            max_prompt_length=self.args.max_prompt_length,
            max_completion_length=self.args.max_length - self.args.max_prompt_length,
            num_generations=4, # Number of samples per prompt for group comparison
        )
        
        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=dummy_reward_func, 
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        print("🚀 Starting GRPO Training...")
        trainer.train()
        self._save(trainer)

    # ============================================================
    # 4. PPO (Proximal Policy Optimization)
    # ============================================================
    def _train_ppo(self, peft_config):
        print(f"\n🎮 Initializing PPO Trainer...")
        
        ppo_config = PPOConfig(
            learning_rate=self.args.lr,
            batch_size=self.args.batch_size,
            mini_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
        )

        # PPO requires a ValueHead wrapper
        if not isinstance(self.model, AutoModelForCausalLMWithValueHead):
            # We assume the model passed in is standard CausalLM, so we wrap it here or earlier
            # Note: Usually PPO requires a specific loading strategy
            pass 

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
            data_collator=lambda x: x, # Custom collation often needed
        )

        print("⚠️ PPO requires a custom generation loop (Rollout -> Calculate Reward -> Step).")
        print("   Starting dummy loop for demonstration...")
        
        # Simplified PPO Loop
        device = ppo_trainer.accelerator.device
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 32,
        }

        for epoch, batch in enumerate(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            
            # 1. Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **generation_kwargs
            )
            
            # 2. Assign Rewards (Dummy)
            batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
            rewards = [torch.tensor(1.0).to(device) for _ in batch["response"]] # Dummy reward 1.0
            
            # 3. Optimization Step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            if epoch >= self.args.epochs: break
            
        self._save(ppo_trainer)

    def _save(self, trainer):
        print(f"\n💾 Saving to {self.args.output_dir}")
        trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print("✅ Done!")

# ============================================================
# 🛠️  Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Unified RLHF/RLAIF Training Script")
    
    # Model & Data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trl-lib/Capybara")
    parser.add_argument("--dataset_files", type=str, nargs="+", help="Local files")
    parser.add_argument("--rl_method", type=str, default="sft", choices=["sft", "dpo", "grpo", "ppo"])
    
    # Config
    parser.add_argument("--output_dir", type=str, default="outputs/rl_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    
    # Columns (To align with RLDataConfig)
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Col name for prompt/input")
    parser.add_argument("--chosen_key", type=str, default="chosen", help="Col name for preferred")
    parser.add_argument("--rejected_key", type=str, default="rejected", help="Col name for rejected")
    
    # PEFT
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="Target modules for LoRA")
    parser.add_argument("--load_in_4bit", action="store_true")

    args = parser.parse_args()
    
    print(f"🚀 Starting {args.rl_method.upper()} Training")
    print(f"   Model: {args.model_name}")

    # 1. Prepare Config for RLDataModule
    class DataArgs:
        hf_name = args.dataset_name
        files = args.dataset_files
        tokenizer_name = "hf:"+args.model_name
        vocab_size = None
        max_prompt_length = args.max_prompt_length
        max_response_length = args.max_length - args.max_prompt_length
        batch_size = args.batch_size
        split_ratio = 0.95
        # Map CLI args to Config keys
        prompt_key = args.prompt_key
        chosen_key = args.chosen_key
        rejected_key = args.rejected_key

    # 2. Load Data via RLDataModule
    print("\n📊 Loading Data Module...")
    # Note: build_rl_dataset is imported from the fixed rl_dataset.py
    data_module = build_rl_dataset(args.rl_method, DataArgs())
    
    def convert_to_hf_dataset(raw_data, method):
        """
        Converts list of dicts from RLDataModule to HF Dataset
        with columns renamed to what TRL expects.
        """
        cols = {}
        
        if not raw_data:
            print("⚠️ Warning: Dataset is empty.")
            return Dataset.from_dict({})

        # SFT: Needs 'text' (completions) or 'messages' (chat)
        if method == "sft":
            # Peek at first item to determine format
            first_content = raw_data[0].get(args.prompt_key, "")
            
            if isinstance(first_content, list): 
                # Chat Format -> TRL expects 'messages'
                cols["messages"] = [item.get(args.prompt_key, []) for item in raw_data]
            else:
                # Plain Text -> TRL expects 'text'
                cols["text"] = [str(item.get(args.prompt_key, "")) for item in raw_data]
            
        # DPO: Needs 'prompt', 'chosen', 'rejected'
        elif method == "dpo":
            cols["prompt"] = [str(i.get(args.prompt_key, "")) for i in raw_data]
            cols["chosen"] = [str(i.get(args.chosen_key, "")) for i in raw_data] # Often list of messages
            cols["rejected"] = [str(i.get(args.rejected_key, "")) for i in raw_data]

        # GRPO/PPO: Needs 'prompt' or 'query'
        elif method in ["grpo", "ppo"]:
            # GRPO expects "prompt" usually
            # PPO expects "query" usually
            prompts = [str(i.get(args.prompt_key, "")) for i in raw_data]
            cols["prompt"] = prompts
            cols["query"] = prompts # Duplicating key to be safe for PPO collation
            
        return Dataset.from_dict(cols)

    train_dataset = convert_to_hf_dataset(data_module.train_dataset.data, args.rl_method)
    valid_dataset = convert_to_hf_dataset(data_module.valid_dataset.data, args.rl_method)
    print(f"✅ Converted to HF Datasets: {len(train_dataset)} train | {len(valid_dataset)} val")

    # 3. Load Model & Tokenizer
    print("\n🧠 Loading Model...")
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    # PPO requires a model with a Value Head (Scalar output)
    if args.rl_method == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, 
            quantization_config=quant_config, 
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # SFT, DPO, GRPO use standard CausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            quantization_config=quant_config, 
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        # Fix for some models (like Llama) where pad token shouldn't be EOS for training
        # But for simple setups, EOS is fine.
    
    # 4. Train
    # Pass None for ref_model (DPO/PPO will handle it internally or creating a copy)
    trainer = RLTrainer(args, model, tokenizer, train_dataset, valid_dataset)
    trainer.train()

if __name__ == "__main__":
    main()