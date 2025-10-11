from tqdm.auto import tqdm
import math, os, json, time, random
import math, torch, torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import torch
from dataclasses import dataclass

from DeepDataMiningLearning.llm.lm_dataset import DataModule, DataConfig, CharTokenizer, WordTokenizer, HFTokenizerWrapper
from DeepDataMiningLearning.llm.transformer import TraditionalTransformerLM, TransformerLM, ModelConfig

# ============================================================
# 4) EVALUATOR — accuracy + perplexity for both modes
# ============================================================
class Evaluator:
    """
    Evaluation utility for Transformer language models.

    Supports:
      ✅ Custom TransformerLM models (teacher-forced / final-token)
      ✅ Hugging Face Causal Language Models (AutoModelForCausalLM)
      ✅ Progress bar for evaluation
      ✅ Computes loss, accuracy, and perplexity
    """

    def __init__(self, model, data, mode="teacher-forced", hf_model=False):
        """
        Args:
            model: The model to evaluate (TransformerLM or Hugging Face model)
            data: DataModule (with loaders())
            mode: 'teacher-forced' (predict every token) or 'final-token' (predict next token)
            hf_model: Whether the model follows Hugging Face’s forward signature
        """
        self.model = model
        self.data = data
        self.mode = mode
        self.hf_model = hf_model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, split="valid"):
        """
        Run evaluation on the validation or test split.

        Returns:
            avg_loss (float): Mean cross-entropy loss
            accuracy (float): Overall token-level accuracy
            perplexity (float): exp(avg_loss)
        """
        # Get the appropriate dataloader
        _, dl_valid = self.data.loaders()
        loader = dl_valid
        self.model.eval()

        total_loss, total_correct, total_count = 0.0, 0, 0

        # Wrap in tqdm progress bar
        progress = tqdm(loader, desc=f"🔍 Evaluating ({split})", dynamic_ncols=True)

        for batch in progress:
            if self.hf_model:
                # Hugging Face models handle attention mask and labels automatically
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = logits.argmax(-1)
                labels = inputs["labels"]
                mask = labels != self.model.config.pad_token_id
                total_correct += (preds[mask] == labels[mask]).sum().item()
                total_count += mask.sum().item()

            elif self.mode == "teacher-forced":
                # Evaluate full-sequence next-token prediction
                x, y = [t.to(self.device) for t in batch]
                T = x.size(1)
                mask = TransformerLM.causal_mask(T, x.device)
                logits = self.model(x, attn_mask=mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_count += y.numel()

            else:
                # Evaluate final-token prediction (predict next token after prefix)
                x, y, lengths = [t.to(self.device) for t in batch]
                T = x.size(1)
                mask = TransformerLM.causal_mask(T, x.device)
                logits = self.model(x, attn_mask=mask)
                V = logits.size(-1)

                # Extract last token logits per sequence
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, V)
                last = logits.gather(1, idx).squeeze(1)

                loss = F.cross_entropy(last, y)
                total_loss += loss.item()
                preds = last.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_count += y.numel()

            # Update tqdm bar
            avg_loss = total_loss / max(1, len(progress))
            progress.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        # Compute metrics
        avg_loss = total_loss / max(1, len(loader))
        acc = total_correct / max(1, total_count)
        ppl = math.exp(avg_loss)

        # Restore training mode
        self.model.train()

        print(f"✅ Eval results — loss: {avg_loss:.4f}, acc: {acc*100:.2f}%, ppl: {ppl:.2f}")
        return avg_loss, acc, ppl
    
class Trainer:
    """
    Advanced PyTorch Trainer for language model training (TransformerLM or Hugging Face models).

    Features:
      ✅ Modern progress bar with live updates
      ✅ AMP (mixed precision) using torch.amp
      ✅ Gradient accumulation & clipping
      ✅ Cosine LR scheduler with warmup
      ✅ Supports teacher-forced and final-token modes
      ✅ Optional Hugging Face model compatibility
    """
    def __init__(self, model, data, tcfg, mode="teacher-forced", hf_model=False):
        """
        Args:
            model: TransformerLM or Hugging Face model (e.g. AutoModelForCausalLM)
            data: DataModule providing loaders()
            tcfg: TrainConfig dataclass (lr, epochs, warmup, etc.)
            mode: 'teacher-forced' or 'final-token'
            hf_model: if True, assume Hugging Face forward signature (model(input_ids, labels=...))
        """
        self.model = model
        self.data = data
        self.mode = mode
        self.hf_model = hf_model
        self.tcfg = tcfg

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer and scheduler setup
        no_decay = ["bias", "LayerNorm.weight", "norm", "emb"]
        grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": tcfg.weight_decay,
                "initial_lr": tcfg.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "initial_lr": tcfg.lr,
            },
        ]
        self.opt = torch.optim.AdamW(grouped_params, lr=tcfg.lr, betas=(0.9, 0.95), eps=1e-8)

        # Cosine LR schedule with warmup
        total_steps = (tcfg.epochs * len(self.data.loaders()[0])) // max(1, tcfg.grad_accum)
        self.sched = CosineWithWarmup(self.opt, warmup=tcfg.warmup_steps, total=max(tcfg.warmup_steps + 1, total_steps))

        # Updated AMP scaler (no FutureWarning)
        self.scaler = torch.amp.GradScaler(enabled=tcfg.amp)

        # Tracking
        self.step, self.best_loss = 0, float("inf")

    def _compute_loss(self, batch):
        """
        Compute loss and logits for one batch.

        Handles:
        - Teacher-forced: full sequence next-token prediction
        - Final-token: predict next token after random prefix
        - Hugging Face models (AutoModelForCausalLM)
        """
        if self.hf_model:
            # Hugging Face models handle attention masks and labels internally
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            return outputs.loss, outputs.logits

        if self.mode == "teacher-forced":
            x, y = [t.to(self.device) for t in batch]
            T = x.size(1)
            attn_mask = TransformerLM.causal_mask(T, x.device)
            logits = self.model(x, attn_mask=attn_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss, logits

        else:  # final-token
            x, y, lengths = [t.to(self.device) for t in batch]
            T = x.size(1)
            attn_mask = TransformerLM.causal_mask(T, x.device)
            logits = self.model(x, attn_mask=attn_mask)
            V = logits.size(-1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, V)
            last = logits.gather(1, idx).squeeze(1)
            loss = F.cross_entropy(last, y)
            return loss, last

    def fit(self):
        """
        Main training loop with tqdm progress bar, AMP, gradient accumulation, and evaluation each epoch.
        """
        dl_train, dl_valid = self.data.loaders()
        self.model.train()

        print(f"🚀 Training for {self.tcfg.epochs} epochs on {self.device} (mode={self.mode})")

        for epoch in range(1, self.tcfg.epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(dl_train, desc=f"Epoch {epoch}/{self.tcfg.epochs}", dynamic_ncols=True)
            self.opt.zero_grad(set_to_none=True)

            for i, batch in enumerate(progress_bar, start=1):
                # Forward pass with AMP
                with torch.amp.autocast(device_type=self.device.type, enabled=self.tcfg.amp):
                    loss, _ = self._compute_loss(batch)
                    loss = loss / self.tcfg.grad_accum

                # Backpropagation
                self.scaler.scale(loss).backward()

                if i % self.tcfg.grad_accum == 0:
                    # Gradient clipping
                    if self.tcfg.grad_clip:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)

                    # Step optimizer and scheduler
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.sched.step()
                    self.step += 1

                epoch_loss += loss.item() * self.tcfg.grad_accum
                avg_loss = epoch_loss / i
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.opt.param_groups[0]['lr']:.6e}"})

            # Validation at end of each epoch
            evaluator = Evaluator(self.model, self.data, self.mode)
            val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")
            print(f"✅ Epoch {epoch} done | val_loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                os.makedirs(os.path.dirname(self.tcfg.save_path), exist_ok=True)
                torch.save({"model": self.model.state_dict(), "cfg": getattr(self.model, 'cfg', None)}, self.tcfg.save_path)
                print(f"💾 Saved best model → {self.tcfg.save_path}")


class CosineWithWarmup:
    def __init__(self, optimizer, warmup, total):
        self.opt, self.warmup, self.total = optimizer, warmup, total
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr_scale = self.step_num / max(1, self.warmup)
        else:
            progress = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_scale


# ============================================================
# 3) TRAINER — AMP, grad-accum, cosine warmup, final/teacher modes
# ============================================================
@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accum: int = 1
    amp: bool = True
    save_path: str = "checkpoints/model.pt"
    log_interval: int = 100
    
# ============================================================
# MAIN: Universal entry point for training and evaluation
# ============================================================
if __name__ == "__main__":
    import argparse, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Train Transformer-based LLMs from scratch or Hugging Face models")

    # --- General ---
    parser.add_argument("--files", nargs="*", help="Text file(s) for training (ignored if --hf_name is used)")
    parser.add_argument("--hf_name", type=str, default="wikitext,wikitext-2-raw-v1", help="Hugging Face dataset: 'dataset,config' (e.g., wikitext,wikitext-2-raw-v1)")
    parser.add_argument("--tokenizer", type=str, default="hf:gpt2", help="'char'|'word'|'hf:<name>' (e.g., hf:gpt2)")
    parser.add_argument("--mode", type=str, default="teacher-forced", choices=["teacher-forced", "final-token"])
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save", type=str, default="outputs/LMcheckpoints/model.pt")

    # --- Model Architecture ---
    parser.add_argument("--model_type", type=str, default="TraditionalTransformerLM",
                        choices=["TransformerLM", "TraditionalTransformerLM", "hf"],
                        help="Choose model type: TransformerLM | TraditionalTransformerLM | hf (Hugging Face pretrained)")
    parser.add_argument("--hf_model_name", type=str, default="gpt2", help="Hugging Face model name if --model_type hf")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rope", action="store_true", help="Use Rotary Embeddings (for TransformerLM)")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1️⃣  Dataset and Tokenizer Setup
    # ------------------------------------------------------------
    if args.model_type == "hf":
        # --- Using a Hugging Face pretrained model ---
        print(f"📚 Loading Hugging Face dataset and tokenizer for {args.hf_model_name}...")
        from datasets import load_dataset

        # Parse dataset config (e.g., "wikitext,wikitext-2-raw-v1")
        if args.hf_name:
            parts = args.hf_name.split(",")
            ds_name, ds_config = parts[0], parts[1] if len(parts) > 1 else None
            dataset = load_dataset(ds_name, ds_config)
        elif args.files:
            text_data = "\n".join(open(f, encoding="utf-8").read() for f in args.files)
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": [text_data]})
        else:
            raise ValueError("Provide either --hf_name or --files for Hugging Face model training.")

        # Tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize dataset
        def tok_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.seq_len)
        tokenized = dataset.map(tok_fn, batched=True, remove_columns=["text"])

        # Dataloaders
        from torch.utils.data import DataLoader
        train_ds = tokenized["train"]
        val_ds = tokenized["validation"] if "validation" in tokenized else tokenized["test"]
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        dl_valid = DataLoader(val_ds, batch_size=args.batch_size)

        # Wrap in DataModule-like object for Trainer compatibility
        class HFData:
            def loaders(self): return dl_train, dl_valid
        data = HFData()

        # Load model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)

        hf_mode = True

    else:
        # --- Using custom dataset builder for TransformerLM or TraditionalTransformerLM ---
        dcfg = DataConfig(
            files=args.files,
            hf_name=args.hf_name,
            tokenizer=args.tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            mode=args.mode
        )
        data = DataModule(dcfg)
        hf_mode = False

        # ------------------------------------------------------------
        # 2️⃣  Model Initialization (Custom Models)
        # ------------------------------------------------------------
        if args.model_type == "TransformerLM":
            print("🚀 Initializing modern TransformerLM (RMSNorm + RoPE + SwiGLU)")
            mcfg = ModelConfig(
                vocab_size=data.vocab_size,
                dim=args.dim,
                n_layers=args.layers,
                n_heads=args.heads,
                max_seq_len=args.seq_len,
                rope=args.rope,
                dropout=0.1
            )
            model = TransformerLM(mcfg)

        elif args.model_type == "TraditionalTransformerLM":
            print("🧩 Initializing traditional Transformer (LayerNorm + GELU + AbsPos)")
            model = TraditionalTransformerLM(
                vocab_size=data.vocab_size,
                dim=args.dim,
                n_layers=args.layers,
                n_heads=args.heads,
                ff_dim=args.dim * 4,
                dropout=0.1,
                max_seq_len=args.seq_len
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

    # ------------------------------------------------------------
    # 3️⃣  Trainer and Training Loop
    # ------------------------------------------------------------
    tcfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        amp=args.amp,
        save_path=args.save,
    )

    trainer = Trainer(model, data, tcfg, mode=args.mode, hf_model=hf_mode)
    trainer.fit()

    # ------------------------------------------------------------
    # 4️⃣  Final Evaluation
    # ------------------------------------------------------------
    evaluator = Evaluator(model, data, mode=args.mode, hf_model=hf_mode)
    loss, acc, ppl = evaluator.evaluate(split="valid")
    print(f"🏁 Final Evaluation | loss={loss:.4f} | acc={acc*100:.2f}% | ppl={ppl:.2f}")