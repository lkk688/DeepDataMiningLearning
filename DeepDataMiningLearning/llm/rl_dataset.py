import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import os
import json
from datasets import load_dataset, get_dataset_config_names

# ============================================================
# ⚙️  RL Data Configuration
# ============================================================
@dataclass
class RLDataConfig:
    """
    Configuration for Reinforcement Learning datasets (PPO, DPO, Reward Modeling).
    """
    # Data Source
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: str = "train"
    files: Optional[List[str]] = None

    # Task Type
    rl_task: str = "dpo"  # "dpo", "ppo", "reward"

    # Tokenizer
    tokenizer: str = "hf:gpt2"
    vocab_size: int = 8000
    
    # Sequence Lengths
    max_prompt_length: int = 256
    max_response_length: int = 256
    
    # Column Mapping
    prompt_key: str = "prompt"        # or "instruction"
    chosen_key: str = "chosen"        # or "response_j"
    rejected_key: str = "rejected"    # or "response_k"
    reward_key: str = "score"         # for reward modeling if needed

    # Batching
    batch_size: int = 16
    
    # Split
    split_ratio: float = 0.9


# ============================================================
# 🧪  DPO Dataset (Direct Preference Optimization)
# ============================================================
class DPODataset(Dataset):
    """
    Dataset for DPO training.
    Each item contains:
      - Prompt (Instruction)
      - Chosen Response (Preferred)
      - Rejected Response (Dispreferred)
    
    Returns:
        prompt_ids, chosen_ids, rejected_ids
    """
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        
        # Determine if tokenizer is HF or custom
        self.is_hf = hasattr(tokenizer, "encode_plus") or hasattr(tokenizer, "pad_token_id")
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str, max_len: int) -> List[int]:
        if self.is_hf:
            # HF Tokenizer
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Custom Tokenizer
            out = self.tokenizer.encode(text)
            ids = out.ids if hasattr(out, "ids") else out
            
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract text
        prompt = item.get(self.cfg.prompt_key, "")
        chosen = item.get(self.cfg.chosen_key, "")
        rejected = item.get(self.cfg.rejected_key, "")
        
        # Tokenize
        prompt_ids = self._tokenize(prompt, self.cfg.max_prompt_length)
        chosen_ids = self._tokenize(chosen, self.cfg.max_response_length)
        rejected_ids = self._tokenize(rejected, self.cfg.max_response_length)
        
        # Add EOS if available and not present (simple heuristic)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            if not chosen_ids or chosen_ids[-1] != eos_id:
                chosen_ids.append(eos_id)
            if not rejected_ids or rejected_ids[-1] != eos_id:
                rejected_ids.append(eos_id)

        return (
            torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long)
        )

def collate_dpo(batch, pad_id=0):
    """
    Collate function for DPO.
    Returns:
        prompt_ids, chosen_ids, rejected_ids (all padded)
        prompt_mask, chosen_mask, rejected_mask
    """
    prompts, chosens, rejecteds = zip(*batch)
    
    prompt_batch = pad_sequence(prompts, batch_first=True, padding_value=pad_id)
    chosen_batch = pad_sequence(chosens, batch_first=True, padding_value=pad_id)
    rejected_batch = pad_sequence(rejecteds, batch_first=True, padding_value=pad_id)
    
    prompt_mask = (prompt_batch != pad_id).long()
    chosen_mask = (chosen_batch != pad_id).long()
    rejected_mask = (rejected_batch != pad_id).long()
    
    return {
        "prompt_ids": prompt_batch,
        "chosen_ids": chosen_batch,
        "rejected_ids": rejected_batch,
        "prompt_mask": prompt_mask,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask
    }


# ============================================================
# 🤖  PPO Dataset (Proximal Policy Optimization)
# ============================================================
class PPODataset(Dataset):
    """
    Dataset for PPO training (Phase 3: RL).
    Only requires Prompts (Queries). The model generates responses during training.
    """
    def __init__(self, data: List[Dict], tokenizer, cfg: RLDataConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.is_hf = hasattr(tokenizer, "encode_plus") or hasattr(tokenizer, "pad_token_id")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get(self.cfg.prompt_key, "")
        
        if self.is_hf:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            out = self.tokenizer.encode(prompt)
            ids = out.ids if hasattr(out, "ids") else out
            
        if len(ids) > self.cfg.max_prompt_length:
            ids = ids[:self.cfg.max_prompt_length]
            
        return torch.tensor(ids, dtype=torch.long)

def collate_ppo(batch, pad_id=0):
    """
    Collate for PPO (just prompts).
    """
    # batch is list of 1D tensors
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    mask = (padded != pad_id).long()
    return padded, mask


# ============================================================
# 📦  RL Data Module
# ============================================================
class RLDataModule:
    def __init__(self, cfg: RLDataConfig):
        self.cfg = cfg
        self.train_dataset = None
        self.valid_dataset = None
        
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
        else:
            raise ValueError(f"Unknown RL task: {self.cfg.rl_task}")

    def _setup_tokenizer(self):
        from DeepDataMiningLearning.llm.tokenizer_utils import TokenizerFactory
        print(f"🔤 Setting up tokenizer: {self.cfg.tokenizer}")
        self.tok = TokenizerFactory.build(
            tokenizer=self.cfg.tokenizer,
            vocab_size=self.cfg.vocab_size
        )
        # Unwrap if it's our wrapper, or use as is
        self.tokenizer_obj = getattr(self.tok, "tokenizer", self.tok)
        self.pad_id = getattr(self.tok, "pad_id", 0) or 0

    def _load_data(self) -> List[Dict]:
        items = []
        if self.cfg.hf_name:
            print(f"📚 Loading HF dataset: {self.cfg.hf_name}")
            try:
                ds = load_dataset(self.cfg.hf_name, self.cfg.hf_config, split=self.cfg.hf_split)
                items = [item for item in ds]
            except Exception as e:
                print(f"⚠️ Failed to load HF dataset: {e}")
        
        if self.cfg.files:
            print(f"📄 Loading local files: {self.cfg.files}")
            for fpath in self.cfg.files:
                with open(fpath, 'r', encoding='utf-8') as f:
                    if fpath.endswith('.json'):
                        content = json.load(f)
                        if isinstance(content, list): items.extend(content)
                    elif fpath.endswith('.jsonl'):
                        for line in f:
                            if line.strip(): items.append(json.loads(line))
                            
        print(f"✅ Loaded {len(items):,} total records.")
        return items

    def loaders(self):
        if self.cfg.rl_task == "dpo":
            collate = lambda b: collate_dpo(b, pad_id=self.pad_id)
        else:
            collate = lambda b: collate_ppo(b, pad_id=self.pad_id)
            
        dl_train = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate)
        dl_valid = DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate)
        
        return dl_train, dl_valid


# ============================================================
# 🏗️  Builder Function
# ============================================================
def build_rl_dataset(task: str, args):
    """
    Unified builder for RL datasets.
    """
    cfg = RLDataConfig(
        hf_name=args.hf_name,
        hf_config=getattr(args, "hf_config", None),
        hf_split=getattr(args, "hf_split", "train"),
        files=getattr(args, "files", None),
        rl_task=task, # "dpo" or "ppo"
        tokenizer=getattr(args, "tokenizer", "hf:gpt2"),
        vocab_size=getattr(args, "vocab_size", 8000),
        max_prompt_length=getattr(args, "max_prompt_length", 256),
        max_response_length=getattr(args, "max_response_length", 256),
        prompt_key=getattr(args, "prompt_key", "prompt"),
        chosen_key=getattr(args, "chosen_key", "chosen"),
        rejected_key=getattr(args, "rejected_key", "rejected"),
        batch_size=getattr(args, "batch_size", 16),
        split_ratio=getattr(args, "split_ratio", 0.9),
    )
    
    return RLDataModule(cfg)


# ============================================================
# 🔍  Inspection & Testing
# ============================================================
def inspect_rl_dataset(data: RLDataModule, num_batches=1):
    print(f"\n🔍 Inspecting RL Dataset ({data.cfg.rl_task})...")
    dl_train, _ = data.loaders()
    
    tokenizer = data.tokenizer_obj
    
    for i, batch in enumerate(dl_train):
        if i >= num_batches: break
        print(f"\n📦 Batch {i+1}")
        
        if data.cfg.rl_task == "dpo":
            prompts = batch["prompt_ids"]
            chosens = batch["chosen_ids"]
            rejecteds = batch["rejected_ids"]
            
            print(f"  Shape: Prompt {prompts.shape} | Chosen {chosens.shape} | Rejected {rejecteds.shape}")
            
            # Decode first sample
            p_text = tokenizer.decode(prompts[0], skip_special_tokens=True)
            c_text = tokenizer.decode(chosens[0], skip_special_tokens=True)
            r_text = tokenizer.decode(rejecteds[0], skip_special_tokens=True)
            
            print(f"  📝 Sample 0:")
            print(f"    [Prompt]: {p_text[:100]}...")
            print(f"    [Chosen]: {c_text[:100]}...")
            print(f"    [Rejected]: {r_text[:100]}...")
            
        elif data.cfg.rl_task == "ppo":
            prompts, mask = batch
            print(f"  Shape: Prompt {prompts.shape}")
            p_text = tokenizer.decode(prompts[0], skip_special_tokens=True)
            print(f"  📝 Sample 0:")
            print(f"    [Prompt]: {p_text[:100]}...")


def run_all_rl_dataset_tests():
    print("\n🚀 Running RL Dataset Tests...")
    
    # Mock Args
    class Args:
        pass
    args = Args()
    args.hf_name = None
    args.files = ["dummy_dpo.json"]
    args.tokenizer = "hf:gpt2"
    args.batch_size = 2
    
    # Create dummy data
    dummy_data = [
        {"prompt": "Hello", "chosen": "Hi there!", "rejected": "Bye."},
        {"prompt": "What is 1+1?", "chosen": "It is 2.", "rejected": "It is 3."},
        {"prompt": "Explain AI", "chosen": "AI is Artificial Intelligence.", "rejected": "AI is magic."},
    ]
    with open("dummy_dpo.json", "w") as f:
        json.dump(dummy_data, f)
        
    try:
        # Test DPO
        print("\n🧪 Testing DPO...")
        dpo_data = build_rl_dataset("dpo", args)
        inspect_rl_dataset(dpo_data)
        
        # Test PPO
        print("\n🧪 Testing PPO...")
        ppo_data = build_rl_dataset("ppo", args)
        inspect_rl_dataset(ppo_data)
        
        print("\n✅ RL Dataset Tests Passed!")
        
    except Exception as e:
        print(f"\n❌ Tests Failed: {e}")
    finally:
        if os.path.exists("dummy_dpo.json"):
            os.remove("dummy_dpo.json")

if __name__ == "__main__":
    run_all_rl_dataset_tests()
