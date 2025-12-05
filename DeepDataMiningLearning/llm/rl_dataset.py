import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import os
import json
from datasets import load_dataset
import transformers

# ============================================================
# 🛠️  Helper / Fallback (To make script runnable standalone)
# ============================================================
class SimpleTokenizer:
    """Fallback tokenizer if the custom factory is missing."""
    def __init__(self, name="gpt2"):
        print(f"⚠️ Using HF AutoTokenizer for {name}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

# ============================================================
# ⚙️  RL Data Configuration
# ============================================================
@dataclass
class RLDataConfig:
    """
    Configuration for RL/SFT datasets.
    """
    # Data Source
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: str = "train"
    files: Optional[List[str]] = None

    # Task Type
    rl_task: str = "dpo"  # "sft", "dpo", "grpo", "ppo"

    # Tokenizer
    tokenizer_name: str = "hf:gpt2"
    vocab_size: int = 50257
    
    # Sequence Lengths
    max_prompt_length: int = 256
    max_response_length: int = 256
    max_seq_length: int = 512 # For SFT
    
    # Column Mapping
    prompt_key: str = "prompt"        # SFT: Text/Msg col | DPO/PPO: Input
    chosen_key: str = "chosen"        # DPO: Preferred
    rejected_key: str = "rejected"    # DPO: Dispreferred

    # Batching
    batch_size: int = 4
    
    # Split
    split_ratio: float = 0.95

# ============================================================
# 📚  SFT Dataset (Supervised Fine-Tuning)
# ============================================================
class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    Handles simple text columns or chat-template style messages.
    Returns: input_ids, labels, attention_mask
    """
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Extract content
        content = item.get(self.cfg.prompt_key, "")
        
        # 2. Handle Chat format (List[Dict]) vs Plain Text
        text_to_tokenize = ""
        if isinstance(content, list):
            # Basic concatenation for chat messages if tokenizer doesn't have apply_chat_template
            # In prod, use tokenizer.apply_chat_template(content, tokenize=False)
            for msg in content:
                role = msg.get('role', '')
                content_txt = msg.get('content', '')
                text_to_tokenize += f"{role}: {content_txt}\n"
        else:
            text_to_tokenize = str(content)

        # 3. Tokenize
        # We use max_seq_length for SFT (prompt + completion combined)
        tokenized = self.tokenizer(
            text_to_tokenize,
            max_length=self.cfg.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # For SFT, labels usually equal input_ids (causal LM)
        labels = input_ids.clone()

        return input_ids, attention_mask, labels

def collate_sft(batch, pad_id=0):
    input_ids, masks, labels = zip(*batch)
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # -100 ignored in loss
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": masks_padded,
        "labels": labels_padded
    }

# ============================================================
# 🧪  DPO Dataset (Direct Preference Optimization)
# ============================================================
class DPODataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str, max_len: int) -> List[int]:
        # Handle if text is a dictionary/list (some datasets have chat structure)
        if isinstance(text, list) or isinstance(text, dict):
            text = str(text) 
            
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item.get(self.cfg.prompt_key, "")
        chosen = item.get(self.cfg.chosen_key, "")
        rejected = item.get(self.cfg.rejected_key, "")
        
        prompt_ids = self._tokenize(prompt, self.cfg.max_prompt_length)
        chosen_ids = self._tokenize(chosen, self.cfg.max_response_length)
        rejected_ids = self._tokenize(rejected, self.cfg.max_response_length)
        
        return (
            torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long)
        )

def collate_dpo(batch, pad_id=0):
    prompts, chosens, rejecteds = zip(*batch)
    
    return {
        "prompt_ids": pad_sequence(prompts, batch_first=True, padding_value=pad_id),
        "chosen_ids": pad_sequence(chosens, batch_first=True, padding_value=pad_id),
        "rejected_ids": pad_sequence(rejecteds, batch_first=True, padding_value=pad_id),
        "prompt_mask": (pad_sequence(prompts, batch_first=True, padding_value=pad_id) != pad_id).long(),
        "chosen_mask": (pad_sequence(chosens, batch_first=True, padding_value=pad_id) != pad_id).long(),
        "rejected_mask": (pad_sequence(rejecteds, batch_first=True, padding_value=pad_id) != pad_id).long()
    }

# ============================================================
# 🧠  GRPO Dataset (Group Relative Policy Optimization)
# ============================================================
class GRPODataset(Dataset):
    """
    GRPO Dataset.
    Often just prompts (like PPO), but structurally distinct as it generates 
    groups of outputs per prompt during training.
    """
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get(self.cfg.prompt_key, "")
        
        if isinstance(prompt, list): prompt = str(prompt)
        
        # Tokenize prompt only
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.cfg.max_prompt_length:
            ids = ids[:self.cfg.max_prompt_length]
            
        return torch.tensor(ids, dtype=torch.long)

# Collator for GRPO is identical to PPO (pad prompts)
def collate_prompt_only(batch, pad_id=0):
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    mask = (padded != pad_id).long()
    return {"input_ids": padded, "attention_mask": mask}

# ============================================================
# 🤖  PPO Dataset (Proximal Policy Optimization)
# ============================================================
class PPODataset(Dataset):
    """
    PPO Dataset. Prompt only.
    """
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get(self.cfg.prompt_key, "")
        
        if isinstance(prompt, list): prompt = str(prompt)

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.cfg.max_prompt_length:
            ids = ids[:self.cfg.max_prompt_length]
            
        return torch.tensor(ids, dtype=torch.long)

# ============================================================
# 📦  RL Data Module
# ============================================================
class RLDataModule:
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg
        
        # 1. Setup Tokenizer
        self._setup_tokenizer()
        
        # 2. Load Data
        data_items = self._load_data()
        
        # 3. Split Data
        split_idx = int(len(data_items) * self.cfg.split_ratio)
        train_data = data_items[:split_idx]
        valid_data = data_items[split_idx:]
        
        print(f"📊 Data Split: {len(train_data):,} train | {len(valid_data):,} valid")
        
        # 4. Build Datasets
        if self.cfg.rl_task == "dpo":
            self.train_dataset = DPODataset(train_data, self.tok, self.cfg)
            self.valid_dataset = DPODataset(valid_data, self.tok, self.cfg)
        elif self.cfg.rl_task == "ppo":
            self.train_dataset = PPODataset(train_data, self.tok, self.cfg)
            self.valid_dataset = PPODataset(valid_data, self.tok, self.cfg)
        elif self.cfg.rl_task == "grpo":
            self.train_dataset = GRPODataset(train_data, self.tok, self.cfg)
            self.valid_dataset = GRPODataset(valid_data, self.tok, self.cfg)
        elif self.cfg.rl_task == "sft":
            self.train_dataset = SFTDataset(train_data, self.tok, self.cfg)
            self.valid_dataset = SFTDataset(valid_data, self.tok, self.cfg)
        else:
            raise ValueError(f"Unknown RL task: {self.cfg.rl_task}")

    def _setup_tokenizer(self):
        try:
            # Try importing the custom factory
            from DeepDataMiningLearning.llm.tokenizer_utils import TokenizerFactory
            print(f"🔤 Setting up tokenizer via Factory: {self.cfg.tokenizer_name}")
            self.tok = TokenizerFactory.build(
                tokenizer=self.cfg.tokenizer_name,
                vocab_size=self.cfg.vocab_size
            )
            # Unwrap logic depending on how TokenizerFactory is implemented
            self.tok = getattr(self.tok, "tokenizer", self.tok)
        except ImportError:
            # Fallback
            self.tok = SimpleTokenizer(self.cfg.tokenizer_name)
            
        self.pad_id = getattr(self.tok, "pad_token_id", 0) or 0

    def _load_data(self) -> List[Dict]:
        items = []
        if self.cfg.hf_name:
            print(f"📚 Loading HF dataset: {self.cfg.hf_name} [{self.cfg.hf_split}]")
            try:
                ds = load_dataset(self.cfg.hf_name, self.cfg.hf_config, split=self.cfg.hf_split)
                
                # Convert to list of dicts immediately to unify format
                # Limit size for testing speed if needed, here taking full
                items = [item for item in ds] 
                
            except Exception as e:
                print(f"⚠️ Failed to load HF dataset {self.cfg.hf_name}: {e}")
        
        return items

    def loaders(self):
        if self.cfg.rl_task == "dpo":
            collate = lambda b: collate_dpo(b, pad_id=self.pad_id)
        elif self.cfg.rl_task == "sft":
            collate = lambda b: collate_sft(b, pad_id=self.pad_id)
        else: # PPO and GRPO
            collate = lambda b: collate_prompt_only(b, pad_id=self.pad_id)
            
        dl_train = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate)
        dl_valid = DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate)
        
        return dl_train, dl_valid

def build_rl_dataset(task: str, args):
    """
    Unified builder for RL datasets.
    Maps generic args (Namespace or Object) to RLDataConfig.
    """
    # Handle case where args might be a Namespace or a Dict
    def get_arg(name, default=None):
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    cfg = RLDataConfig(
        rl_task=task,
        hf_name=get_arg("hf_name", None),
        hf_config=get_arg("hf_config", None),
        hf_split=get_arg("hf_split", "train"),
        files=get_arg("files", None),
        tokenizer_name=get_arg("tokenizer_name", "gpt2"),
        vocab_size=get_arg("vocab_size", 50257),
        max_prompt_length=get_arg("max_prompt_length", 256),
        max_response_length=get_arg("max_response_length", 256),
        batch_size=get_arg("batch_size", 4),
        # Mappings
        prompt_key=get_arg("prompt_key", "prompt"),
        chosen_key=get_arg("chosen_key", "chosen"),
        rejected_key=get_arg("rejected_key", "rejected"),
        split_ratio=get_arg("split_ratio", 0.95),
    )
    
    return RLDataModule(cfg)

# ============================================================
# 🔍  Testing Suite
# ============================================================
def inspect_batch(data_module, name):
    print(f"\n🔍 Inspecting {name} ({data_module.cfg.rl_task})...")
    dl_train, _ = data_module.loaders()
    
    try:
        batch = next(iter(dl_train))
        print(f"  ✅ Batch loaded successfully.")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"     - {k}: {v.shape}")
                
        # Decode Sample
        tok = data_module.tok
        if "input_ids" in batch: # SFT, GRPO, PPO
            text = tok.decode(batch["input_ids"][0], skip_special_tokens=True)
            print(f"     📝 Sample: {text[:100]}...")
        elif "prompt_ids" in batch: # DPO
            text = tok.decode(batch["prompt_ids"][0], skip_special_tokens=True)
            print(f"     📝 Prompt: {text[:100]}...")
            
    except Exception as e:
        print(f"  ❌ Batch loading failed: {e}")

def run_all_tests():
    print("\n🚀 Running Dataset Tests...")
    
    # 1. Test DPO
    # Dataset: trl-lib/ultrafeedback_binarized
    # Columns: prompt, chosen, rejected
    dpo_cfg = RLDataConfig(
        rl_task="dpo",
        hf_name="trl-lib/ultrafeedback_binarized",
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        batch_size=2
    )
    try:
        dpo_module = RLDataModule(dpo_cfg)
        inspect_batch(dpo_module, "DPO Dataset")
    except Exception as e:
        print(f"DPO Test Failed: {e}")

    # 2. Test GRPO
    # Dataset: trl-lib/ultrafeedback-prompt
    # Columns: prompt, prompt_id
    grpo_cfg = RLDataConfig(
        rl_task="grpo",
        hf_name="trl-lib/ultrafeedback-prompt",
        prompt_key="prompt", # Key containing the query
        batch_size=2
    )
    try:
        grpo_module = RLDataModule(grpo_cfg)
        inspect_batch(grpo_module, "GRPO Dataset")
    except Exception as e:
        print(f"GRPO Test Failed: {e}")

    # 3. Test SFT
    # Dataset: trl-lib/Capybara (Likely LDJnr/Capybara)
    # Columns: typically 'conversation' or 'messages'
    # Note: 'trl-lib/Capybara' might not exist publicly, using 'LDJnr/Capybara' structure as backup logic
    # or just treating the user input as valid. 
    # We assume column is "conversation" or "messages" for chat SFT.
    sft_cfg = RLDataConfig(
        rl_task="sft",
        hf_name="LDJnr/Capybara", # Using the actual public dataset name usually referenced
        prompt_key="conversation", # The column containing the chat list
        batch_size=2
    )
    try:
        sft_module = RLDataModule(sft_cfg)
        inspect_batch(sft_module, "SFT Dataset")
    except Exception as e:
        print(f"SFT Test Failed (Checking fallback...): {e}")

    # 4. Test PPO
    # Dataset: lvwerra/stack-exchange-paired (From the link provided)
    # Columns: question, response_j, response_k. PPO only needs the Question.
    ppo_cfg = RLDataConfig(
        rl_task="ppo",
        hf_name="HuggingFaceH4/cherry_picked_prompts",
        hf_split="train[:100]", # Load small subset for speed
        prompt_key="prompt", #"question",
        batch_size=2
    )#https://huggingface.co/docs/trl/v0.7.2/en/ppo_trainer
    try:
        ppo_module = RLDataModule(ppo_cfg)
        inspect_batch(ppo_module, "PPO Dataset")
    except Exception as e:
        print(f"PPO Test Failed: {e}")

if __name__ == "__main__":
    run_all_tests()