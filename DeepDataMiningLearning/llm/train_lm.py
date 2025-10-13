from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math, os, json, time, random
import math, torch, torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import torch
from dataclasses import dataclass
import gc
from DeepDataMiningLearning.llm.lm_dataset import build_dataset, test_tokenizer, inspect_dataset
from DeepDataMiningLearning.llm.transformer import build_model, TransformerLM, PyTorchTransformer, TraditionalTransformerLM, FullTransformer, RNNLanguageModel, LSTMLanguageModel

# ============================================================
# EVALUATOR — Unified Evaluation for All Model Types
# ============================================================
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import math
import inspect


class Evaluator:
    """
    Unified evaluation class supporting:
      ✅ Custom Transformer (decoder-only & encoder–decoder)
      ✅ RNN / LSTM
      ✅ PyTorch nn.Transformer
      ✅ Hugging Face CausalLM and Seq2Seq models
      ✅ Hugging Face `evaluate` integration (optional BLEU/ROUGE)
    """

    def __init__(self, model, data, mode="teacher-forced", hf_model=False):
        self.model = model
        self.data = data
        self.mode = mode
        self.hf_model = hf_model
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------
    # Main Evaluation
    # ------------------------------------------------------------
    def evaluate(self, split="valid", hf_metric=None):
        """
        Evaluate the model on validation or test data.

        Args:
            split (str): "valid" or "test".
            hf_metric (str): Optional Hugging Face metric ("bleu", "rouge", etc.).
        Returns:
            avg_loss, accuracy, perplexity
        """
        print(f"\n🔍 Evaluating split = {split} ...")

        loaders = self.data.loaders()
        # Some data modules have 2 loaders (train, val), some 3 (train, val, test)
        if len(loaders) == 2:
            _, dl_valid = loaders
        elif len(loaders) == 3:
            _, dl_valid, _ = loaders
        else:
            raise ValueError("Invalid number of loaders returned from data.loaders()")

        loader = dl_valid
        self.model.eval()

        total_loss, total_correct, total_count = 0.0, 0, 0

        # Optional Hugging Face metric (e.g., BLEU or ROUGE)
        hf_eval = None
        if hf_metric:
            try:
                from evaluate import load as load_metric
                hf_eval = load_metric(hf_metric)
                print(f"📏 Using Hugging Face metric: {hf_metric}")
            except Exception as e:
                print(f"⚠️ Could not load metric '{hf_metric}': {e}")

        with torch.no_grad():  # 🚫 Disable gradient tracking
            progress = tqdm(loader, desc=f"Eval {split}", dynamic_ncols=True)

            for batch in progress:
                # --------------------------------------------------
                # Case 1 — Hugging Face models
                # --------------------------------------------------
                if self.hf_model:
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits

                    total_loss += loss.item()
                    preds = logits.argmax(-1)
                    labels = inputs.get("labels", None)
                    if labels is not None:
                        mask = labels != self.model.config.pad_token_id
                        total_correct += (preds[mask] == labels[mask]).sum().item()
                        total_count += mask.sum().item()
                    continue

                # --------------------------------------------------
                # Case 2 — Encoder–Decoder (FullTransformer, PyTorchTransformer)
                # --------------------------------------------------
                if isinstance(self.model, (FullTransformer, PyTorchTransformer)):
                    src, tgt_in, tgt_out, _, _ = [t.to(self.device) for t in batch]
                    logits = self.model(src, tgt_in)
                    pad_idx = getattr(self.data.tokenizer_tgt, "pad_idx", -100)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        tgt_out.view(-1),
                        ignore_index=pad_idx,
                    )
                    total_loss += loss.item()

                    preds = logits.argmax(-1)
                    total_correct += (preds == tgt_out).sum().item()
                    total_count += (tgt_out != pad_idx).sum().item()

                    # Optional: BLEU/ROUGE with Hugging Face evaluate
                    if hf_eval:
                        pred_texts = self.data.tokenizer_tgt.decode(preds[0].tolist())
                        ref_texts = self.data.tokenizer_tgt.decode(tgt_out[0].tolist())
                        hf_eval.add(prediction=pred_texts, reference=ref_texts)
                    continue

                # --------------------------------------------------
                # Case 3 — RNN / LSTM (no attn_mask)
                # --------------------------------------------------
                if isinstance(self.model, (RNNLanguageModel, LSTMLanguageModel)):
                    # Handle both (x, y) and (x, y, lengths)
                    if len(batch) == 3:
                        x, y, lengths = batch
                    elif len(batch) == 2:
                        x, y = batch
                        lengths = None
                    else:
                        raise ValueError(f"Unexpected batch format for RNN/LSTM: {type(batch)} ({len(batch)} items)")
                    x, y = x.to(self.device), y.to(self.device)
                    #x, y = [t.to(self.device) for t in batch]
                    logits, _ = self.model(x)
                    pad_idx = getattr(self.data, "pad_idx", -100)
                    # loss = F.cross_entropy(
                    #     logits.view(-1, logits.size(-1)), #[32, 139, 8000]
                    #     y.view(-1), #[32, 5]
                    #     ignore_index=pad_idx,
                    # )
                    # Align lengths (typing target shorter)
                    B, T, V = logits.size()
                    if y.size(1) < T:
                        logits = logits[:, -y.size(1):, :]

                    loss = F.cross_entropy(
                        logits.reshape(-1, V),
                        y.reshape(-1),
                        ignore_index=pad_idx,   # ✅ correct pad ID
                    )
                    total_loss += loss.item()
                    preds = logits.argmax(-1)
                    total_correct += (preds == y).sum().item()
                    total_count += y.numel()
                    continue

                # --------------------------------------------------
                # Case 4 — Decoder-only Transformers
                # --------------------------------------------------
                if self.mode == "teacher-forced":
                    # Each batch is (x, y) or (x, y, lengths)
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch
                    x, y = [t.to(self.device) for t in batch]
                    T = x.size(1)
                    mask = TransformerLM.causal_mask(T, x.device)
                    logits = self.model(x, attn_mask=mask)
                    pad_idx = getattr(self.data, "pad_idx", -100)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=pad_idx,
                    )
                    total_loss += loss.item()
                    preds = logits.argmax(-1)
                    total_correct += (preds == y).sum().item()
                    total_count += y.numel()
                    continue

                elif self.mode == "final-token":
                    x, y, lengths = [t.to(self.device) for t in batch]
                    T = x.size(1)
                    mask = TransformerLM.causal_mask(T, x.device)
                    logits = self.model(x, attn_mask=mask)
                    V = logits.size(-1)
                    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, V)
                    last_logits = logits.gather(1, idx).squeeze(1)
                    loss = F.cross_entropy(last_logits, y)
                    total_loss += loss.item()
                    preds = last_logits.argmax(-1)
                    total_correct += (preds == y).sum().item()
                    total_count += y.numel()

            # end of batch loop
        # end of no_grad

        # --------------------------------------------------
        # Final statistics
        # --------------------------------------------------
        avg_loss = total_loss / max(1, len(loader))
        acc = total_correct / max(1, total_count)
        ppl = math.exp(avg_loss)

        if hf_eval:
            try:
                metric_result = hf_eval.compute()
                print(f"📊 HF metric ({hf_metric}): {metric_result}")
            except Exception as e:
                print(f"⚠️ Metric computation failed: {e}")

        print(f"✅ Eval complete | loss={avg_loss:.4f} | acc={acc*100:.2f}% | ppl={ppl:.2f}\n")
        return avg_loss, acc, ppl
    
class Trainer:
    """
    Advanced PyTorch Trainer for language model training (TransformerLM or Hugging Face models).

    Features:
      ✅ AMP (mixed precision) using torch.amp
      ✅ Gradient accumulation & clipping
      ✅ Cosine LR scheduler with warmup or ReduceLROnPlateau
      ✅ Supports teacher-forced and final-token modes
      ✅ Early stopping
      ✅ Optional Hugging Face model compatibility
    """
    def __init__(self, model, data, tcfg, mode="teacher-forced", hf_model=False, early_stop_patience=3):
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
        print(f"📅 Total training steps: {total_steps}")
        #self.sched = CosineWithWarmup(self.opt, warmup=tcfg.warmup_steps, total=max(tcfg.warmup_steps + 1, total_steps))

        # ------------------------------------------------------------
        # Learning rate scheduler selection (user-configurable)
        # ------------------------------------------------------------
        if hasattr(tcfg, "scheduler_type"):
            scheduler_type = tcfg.scheduler_type.lower()
        else:
            scheduler_type = "reduce"

        if scheduler_type == "reduce":
            print("📉 Using ReduceLROnPlateau scheduler (adaptive LR on validation loss)")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, factor=0.5, patience=1, verbose=True
            )
        elif scheduler_type == "cosine":
            print("🌊 Using CosineAnnealingWarmRestarts scheduler")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt, T_0=5, T_mult=1, eta_min=1e-6
            )
        elif scheduler_type == "none":
            print("🚫 No scheduler selected (fixed learning rate)")
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Updated AMP scaler (no FutureWarning)
        self.amp_enabled = bool(self.tcfg.amp)
        self.amp_dtype = getattr(self.tcfg, "amp_dtype", "bf16")
        self.use_scaler = self.amp_enabled and (self.amp_dtype == "fp16")
        self.scaler = torch.amp.GradScaler(enabled=self.use_scaler)
        #self.scaler = torch.amp.GradScaler(enabled=tcfg.amp)

        # Tracking
        self.step, self.best_loss = 0, float("inf")
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping setup
        self.early_stop_patience = early_stop_patience
        self.no_improve_epochs = 0
        self.stop_training = False

    def _get_pad_id(self):
        """Retrieve pad token ID dynamically from dataset/tokenizer."""
        # Try in several places
        if hasattr(self.data, "pad_id") and self.data.pad_id is not None:
            return self.data.pad_id
        if hasattr(self.data, "tok"):
            tok = self.data.tok
            if hasattr(tok, "pad_id") and tok.pad_id is not None:
                return tok.pad_id
            if hasattr(tok, "tokenizer") and hasattr(tok.tokenizer, "pad_token_id"):
                return tok.tokenizer.pad_token_id
        # Fallback
        return -100 #0

    def _compute_loss(self, batch):
        """
        Compute the training loss and logits for one batch.

        Handles:
        - Hugging Face models (AutoModelForCausalLM)
        - Decoder-only Transformers (teacher-forced / final-token)
        - Encoder–Decoder Transformers (FullTransformer)

        Returns:
            loss (Tensor): Scalar training loss for backprop
            logits (Tensor): Model output logits
        """
        pad_idx = self._get_pad_id()
        # -----------------------------------------------------------
        # Case 1: Hugging Face models (AutoModelForCausalLM)
        # -----------------------------------------------------------
        if self.hf_model:
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
            elif isinstance(batch, (tuple, list)):
                # Support (x, y) or (x, y, lengths)
                x, y = batch[:2]
                if x.size(0) != y.size(0):
                    raise ValueError(
                        f"❌ Input batch_size ({x.size(0)}) "
                        f"!= target batch_size ({y.size(0)}). "
                        "Check dataset: HF models require equal batch and seq length."
                    )
                inputs = {
                    "input_ids": x.to(self.device),
                    "labels": y.to(self.device),
                    "attention_mask": (x != self._get_pad_id()).to(self.device),
                }
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            outputs = self.model(**inputs)
            return outputs.loss, outputs.logits

        # -----------------------------------------------------------
        # Case 2: RNN / LSTM (with optional sequence lengths)
        # -----------------------------------------------------------
        if isinstance(self.model, (RNNLanguageModel, LSTMLanguageModel)):
            # Handle both (x, y) and (x, y, lengths)
            if len(batch) == 3:
                x, y, lengths = batch
            elif len(batch) == 2:
                x, y = batch
                lengths = None
            else:
                raise ValueError(f"Unexpected batch format for RNN/LSTM: {type(batch)} ({len(batch)} items)")

            x, y = x.to(self.device), y.to(self.device)
            if lengths is not None:
                lengths = lengths.to(self.device)

            logits, _ = self.model(x)
            # Align lengths (typing target shorter)
            B, T, V = logits.size()
            if y.size(1) < T:
                logits = logits[:, -y.size(1):, :]

            loss = F.cross_entropy(
                logits.reshape(-1, V),
                y.reshape(-1),
                ignore_index=pad_idx,   # ✅ correct pad ID
            )
            return loss, logits

        
        # -----------------------------------------------------------
        # Case 3: Full Encoder–Decoder Transformer
        # -----------------------------------------------------------
        if isinstance(self.model, (FullTransformer,PyTorchTransformer)):
            # Batch structure: (src, tgt_input, tgt_output, src_lengths, tgt_lengths)
            src, tgt_in, tgt_out, _, _ = [t.to(self.device) for t in batch]

            # Forward pass: feed source & target inputs
            logits = self.model(src, tgt_in)  # [B, T, V]

            # Compute loss over all target positions
            #pad_idx = getattr(self.data.tok_tgt, "pad_idx", -100) if hasattr(self.data, "tok_tgt") else -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1),
                ignore_index=pad_idx,
            )
            return loss, logits

        # -----------------------------------------------------------
        # Case 4: Decoder-only Transformers (TransformerLM)
        # -----------------------------------------------------------
        if isinstance(self.model, (TransformerLM, TraditionalTransformerLM)):
            #pad_idx = getattr(self.data, "pad_id", -100)
            mode = getattr(self, "mode", "teacher-forced")

            # ========================================================
            # Teacher-forced mode (standard autoregressive LM)
            # ========================================================
            if mode == "teacher-forced":
                # Each batch is (x, y) or (x, y, lengths)
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # Build causal (lower-triangular) attention mask
                T = x.size(1)
                attn_mask = TransformerLM.causal_mask(T, x.device)

                # Forward pass through model
                logits = self.model(x, attn_mask=attn_mask)  # [B, T, V]

                # Compute full-sequence cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),  # flatten [B*T, V]
                    y.view(-1),                        # flatten [B*T]
                    ignore_index=pad_idx,
                )
                return loss, logits

            # ========================================================
            # Final-token mode (prefix → next-token prediction)
            # ========================================================
            elif mode == "final-token":
                # Each batch is (x, y, lengths)
                x, y, lengths = [t.to(self.device) for t in batch]

                # Build causal mask
                T = x.size(1)
                attn_mask = TransformerLM.causal_mask(T, x.device)

                # Forward pass through model
                logits = self.model(x, attn_mask=attn_mask)  # [B, T, V]

                # Select logits at each sample’s final valid token position
                V = logits.size(-1)
                idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, V)
                last_logits = logits.gather(1, idx).squeeze(1)  # [B, V]

                # Cross-entropy loss for next-token prediction
                loss = F.cross_entropy(last_logits, y, ignore_index=pad_idx)
                return loss, last_logits

    def fit(self):
        """
        Main training loop with tqdm progress bar, AMP, gradient accumulation, and evaluation each epoch.
        """
        dl_train, dl_valid = self.data.loaders()
        self.model.train()
        
        amp_dtype = torch.float16 if self.amp_dtype == "fp16" else torch.bfloat16
        autocast_ctx = torch.amp.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=self.amp_enabled,
        )
        print(f"\n🔧 Using AMP: {self.amp_enabled} | dtype: {amp_dtype} | scaler: {self.use_scaler}\n")

        print(f"🚀 Training for {self.tcfg.epochs} epochs on {self.device} (mode={self.mode})")

        for epoch in range(1, self.tcfg.epochs + 1):
            if self.stop_training:
                print(f"⏹️ Early stopping triggered at epoch {epoch}.")
                break
            
            # ✅ ensure training mode at epoch start
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(dl_train, desc=f"Epoch {epoch}/{self.tcfg.epochs}", dynamic_ncols=True)
            self.opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            for i, batch in enumerate(progress_bar, start=1):
                
                # Forward pass with AMP
                #with torch.amp.autocast(device_type=self.device.type, enabled=self.tcfg.amp):
                with autocast_ctx:
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
                    # self.sched.step()
                    # self.step += 1

                total_loss += loss.item() * self.tcfg.grad_accum
                avg_loss = total_loss / i
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.opt.param_groups[0]['lr']:.6e}"})

            # Average training loss for epoch
            avg_train_loss = total_loss / max(1, len(dl_train))
            self.train_losses.append(avg_train_loss)
            
            # Validation at end of each epoch
            evaluator = Evaluator(self.model, self.data, self.mode)
            val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")
            self.val_losses.append(val_loss)
            # Scheduler step
            # Scheduler step (only if scheduler exists)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            print(f"✅ Epoch {epoch} done | val_loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}")

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.no_improve_epochs = 0
                os.makedirs(os.path.dirname(self.tcfg.save_path), exist_ok=True)
                torch.save({"model": self.model.state_dict(), "cfg": getattr(self.model, 'cfg', None)}, self.tcfg.save_path)
                print(f"💾 Saved best model → {self.tcfg.save_path}")
            else:
                self.no_improve_epochs += 1
                print(f"⚠️ No improvement for {self.no_improve_epochs} epochs.")
                if self.no_improve_epochs >= self.early_stop_patience:
                    print("⏹️ Early stopping criterion met.")
                    self.stop_training = True

        print("\n✅ Training complete.")
        return self.train_losses, self.val_losses


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
    amp: bool = True                # enable/disable AMP globally
    amp_dtype: str = "bf16"         # 'bf16' (safe) or 'fp16' (scaler)
    scheduler_type: str = "reduce"  # "reduce", "cosine", or "none"
    save_path: str = "checkpoints/model.pt"
    log_interval: int = 100
    


def test_charmodel():
    import torch
    from nltk.corpus import words
    import nltk #pip install nltk
    nltk.download("words")

    # ------------------------------------------------------------
    # Create a text file from the English word corpus
    # ------------------------------------------------------------
    with open("english_words.txt", "w", encoding="utf-8") as f:
        for w in words.words():
            if w.isalpha() and len(w) > 2:
                f.write(w.lower() + "\n")

    # ------------------------------------------------------------
    # Minimal args object for compatibility with build_dataset
    # ------------------------------------------------------------
    class Args:
        def __init__(self, model_type):
            self.model_type = model_type
            self.hf_name = None
            self.files = ["english_words.txt"]
            self.tokenizer = "char"
            self.seq_len = 32
            self.batch_size = 128
            self.mode = "teacher-forced"
            self.dim = 256
            self.layers = 2
            self.heads = 4
            self.rope = False
            self.epochs = 8
            self.lr = 3e-4
            self.weight_decay = 0.01
            self.warmup_steps = 200
            self.grad_clip = 1.0
            self.grad_accum = 1
            self.amp = False
            self.scheduler= "reduce"
            self.save = f"checkpoints/{model_type.lower()}_char_word.pt"

    # ------------------------------------------------------------
    # Helper function for training
    # ------------------------------------------------------------
    def train_char_model(model_type="RNN"):
        print(f"\n🚀 Training {model_type} on English words for next-character prediction...\n")

        # 1️⃣ Build dataset (reusing your general build_dataset)
        args = Args(model_type)
        data = build_dataset(model_type, args)

        # 2️⃣ Build model
        model, hf_mode = build_model(model_type, data, args)

        # 3️⃣ Setup training configuration
        tcfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_clip=args.grad_clip,
            grad_accum=args.grad_accum,
            amp=args.amp,
            scheduler_type=args.scheduler,
            save_path=args.save,
        )

        # 4️⃣ Train
        trainer = Trainer(model, data, tcfg, mode=args.mode, hf_model=hf_mode)
        train_losses, val_losses = trainer.fit()

        # 5️⃣ Evaluate
        evaluator = Evaluator(model, data, mode=args.mode, hf_model=hf_mode)
        loss, acc, ppl = evaluator.evaluate(split="valid")
        print(f"✅ {model_type} — Final | loss={loss:.4f} | acc={acc*100:.2f}% | ppl={ppl:.2f}")

    # ------------------------------------------------------------
    # Train multiple models
    # ------------------------------------------------------------
    for model_type in ["RNN", "LSTM", "TransformerLM"]:
        train_char_model(model_type)

# ============================================================
# Predictive Typing Experiment — RNN, LSTM, Transformer
# ============================================================
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ============================================================
# PREDICTIVE TYPING EXPERIMENT (Optimized per model)
# ============================================================
def run_predictive_typing_experiment(
    hf_name="OpenAssistant/oasst1",
    txt_files=None,
    vocab_size=400, #8000,
    seq_len=96, #128,
    batch_size=32, #64,
    epochs=6,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Runs predictive typing experiments for RNN, LSTM, and TransformerLM.
    Each model uses its own optimized TrainConfig for stability and convergence.
    """
    print("🚀 Starting predictive typing benchmark...\n")

    # ------------------------------------------------------------
    # 1️⃣ Build dataset via unified build_dataset()
    # ------------------------------------------------------------
    class Args:
        pass
    args = Args()
    args.task = "typing"
    args.hf_name = hf_name
    args.files = txt_files
    args.tokenizer = "bpe"
    args.vocab_size = vocab_size
    args.seq_len = seq_len
    args.batch_size = batch_size
    args.num_prefixes_per_sentence = 5 #3
    args.next_token_window = 5
    args.mode = "teacher-forced"

    print(f"📘 Building predictive typing dataset from '{hf_name}'...")
    data = build_dataset(args.task, args)
    print("✅ Dataset ready.\n")
    print(f"Dataset vocab size: {data.tok.vocab_size} | Train samples: {len(data.train_dataset)} | Val samples: {len(data.valid_dataset)}\n")
    # Check tokenization
    sample = "I really love coding in Python!"
    encoded = data.tok.encode(sample)
    print(encoded)
    print(data.tok.decode(encoded))

    # ------------------------------------------------------------
    # 2️⃣ Define model configurations
    # ------------------------------------------------------------
    # model_configs = {
    #     "RNN": dict(model_type="RNN", dim=256, layers=2, lr=1e-3, weight_decay=0.01, grad_clip=1.0, epochs=6),
    #     "LSTM": dict(model_type="LSTM", dim=256, layers=2, lr=5e-4, weight_decay=0.02, grad_clip=0.8, epochs=8),
    #     "TransformerLM": dict(model_type="TransformerLM", dim=384, layers=6, heads=6, lr=3e-4, weight_decay=0.05, grad_clip=1.0, epochs=10),
    # }
    model_configs = {
        "RNN": dict(model_type="RNN", dim=192, layers=2, lr=1e-3, weight_decay=0.01, grad_clip=1.0, epochs=6),
        "LSTM": dict(model_type="LSTM", dim=192, layers=2, lr=5e-4, weight_decay=0.02, grad_clip=0.8, epochs=8),
        "TransformerLM": dict(model_type="TransformerLM", dim=192, layers=2, heads=4, lr=3e-4, weight_decay=0.05, grad_clip=1.0, epochs=10),
    }
    #RNN: Fast to converge, benefits from slightly higher LR and smaller warmup.
    #LSTM: More stable but slower — needs lower LR and stronger gradient clipping.
    #TransformerLM: Large model, benefits from cosine scheduling and warmup for stability.

    results = {}

    # ------------------------------------------------------------
    # 3️⃣ Train and evaluate each model with custom TrainConfig
    # ------------------------------------------------------------
    for name, cfg in model_configs.items():
        print(f"\n🧠 Training {name} model...")

        # Build args for the model
        args.model_type = cfg["model_type"]
        args.dim = cfg["dim"]
        args.layers = cfg["layers"]
        args.heads = cfg.get("heads", 4)
        args.lr = cfg["lr"]
        args.save = f"checkpoints/{name.lower()}_typing.pt"

        # Build model
        model, hf_mode = build_model(args.model_type, data, args)

        # --------------------------------------------------------
        # Custom TrainConfig per model type
        # --------------------------------------------------------
        if name == "RNN":
            tcfg = TrainConfig(
                epochs=epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                warmup_steps=100,
                grad_clip=cfg["grad_clip"],
                grad_accum=2,
                amp=False,
                scheduler_type="reduce",
                save_path=args.save,
            )

        elif name == "LSTM":
            tcfg = TrainConfig(
                epochs=epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                warmup_steps=150,
                grad_clip=cfg["grad_clip"],
                grad_accum=1,
                amp=False,
                scheduler_type="reduce",
                save_path=args.save,
            )

        elif name == "TransformerLM":
            tcfg = TrainConfig(
                epochs=epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                warmup_steps=400,
                grad_clip=cfg["grad_clip"],
                grad_accum=2,
                amp=True,                     # ✅ AMP helps Transformers
                scheduler_type="cosine",      # ✅ Cosine Annealing + Warmup
                save_path=args.save,
            )

        # --------------------------------------------------------
        # Train model
        # --------------------------------------------------------
        trainer = Trainer(model, data, tcfg, mode="teacher-forced", hf_model=hf_mode)
        train_losses, val_losses = trainer.fit()

        # --------------------------------------------------------
        # Evaluate model
        # --------------------------------------------------------
        evaluator = Evaluator(model, data, mode="teacher-forced", hf_model=hf_mode)
        val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")

        results[name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_loss": val_loss,
            "final_acc": val_acc,
            "final_ppl": val_ppl,
        }

        print(f"✅ {name} — loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}\n")
        
        # ------------------------------------------------------------
        # 🧹 Memory cleanup between models
        # ------------------------------------------------------------
        del model, trainer, evaluator
        torch.cuda.empty_cache()
        gc.collect()

        print("🧹 Cleared memory.\n")

    # ------------------------------------------------------------
    # 4️⃣ Plot results
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for name, metrics in results.items():
        plt.plot(metrics["train_losses"], label=f"{name} Train", linestyle="-")
        plt.plot(metrics["val_losses"], label=f"{name} Val", linestyle="--")
    plt.title("Predictive Typing — Training & Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 5️⃣ Print summary
    # ------------------------------------------------------------
    print("\n📊 Final Comparison Summary:")
    print(f"{'Model':<15} {'Loss':<10} {'Acc (%)':<10} {'PPL':<10}")
    print("-" * 45)
    for name, m in results.items():
        print(f"{name:<15} {m['final_loss']:<10.4f} {m['final_acc']*100:<10.2f} {m['final_ppl']:<10.2f}")

    print("\n✅ Benchmark complete!\n")
    return results


import torch
import matplotlib.pyplot as plt


def run_char_typing_experiment(
    hf_name="OpenAssistant/oasst1",
    seq_len=128,
    batch_size=64,
    epochs=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Simple predictive typing experiment with char-level tokenization.

    Task:
        - Build a small predictive typing dataset (prefix → next few chars)
        - Train RNN, LSTM, and Transformer models on the same data
        - Compare their training/validation loss and accuracy

    Dataset:
        - English subsamples from OpenAssistant/oasst1
        - Tokenized at character level (vocab ~100)
    """
    print("\n🚀 ===== Character-Level Typing Experiment =====")

    # ------------------------------------------------------------
    # 1️⃣ Build dataset (char-level typing)
    # ------------------------------------------------------------
    class Args:
        pass

    args = Args()
    args.task = "typing"
    args.hf_name = hf_name
    args.hf_split = "train"
    args.tokenizer = "char"
    args.vocab_size = 256
    args.seq_len = seq_len
    args.batch_size = batch_size
    args.lowercase = True
    args.mode = "teacher-forced"
    args.num_prefixes_per_sentence = 3
    args.next_token_window = 5

    print(f"📘 Building char-level typing dataset from '{hf_name}'...")
    data = build_dataset(args.task, args)
    train_loader, val_loader = data.loaders()
    print("✅ Dataset ready for training.\n")

    # ------------------------------------------------------------
    # 2️⃣ Define model configurations
    # ------------------------------------------------------------
    model_configs = {
        "RNN": dict(model_type="RNN", dim=128, layers=1, lr=1e-3, epochs=epochs),
        "LSTM": dict(model_type="LSTM", dim=128, layers=1, lr=1e-3, epochs=epochs),
        "Transformer": dict(model_type="TransformerLM", dim=192, layers=2, heads=4, lr=5e-4, epochs=epochs),
    }

    results = {}

    # ------------------------------------------------------------
    # 3️⃣ Train and compare models
    # ------------------------------------------------------------
    for name, cfg in model_configs.items():
        print(f"\n🧠 Training {name} model...")

        class ModelArgs:
            pass

        margs = ModelArgs()
        margs.model_type = cfg["model_type"]
        margs.dim = cfg["dim"]
        margs.layers = cfg["layers"]
        margs.heads = cfg.get("heads", 2)
        margs.lr = cfg["lr"]
        margs.seq_len = seq_len
        margs.save = f"checkpoints/{name.lower()}_char_typing.pt"

        model, hf_mode = build_model(margs.model_type, data, margs)

        # Training configuration
        tcfg = TrainConfig(
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=0.01,
            warmup_steps=100,
            grad_clip=1.0,
            grad_accum=1,
            amp=False,
            scheduler_type="reduce",
            save_path=margs.save,
        )

        trainer = Trainer(model, data, tcfg, mode="teacher-forced", hf_model=hf_mode)
        train_losses, val_losses = trainer.fit()

        evaluator = Evaluator(model, data, mode="teacher-forced", hf_model=hf_mode)
        val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")

        results[name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_loss": val_loss,
            "final_acc": val_acc,
            "final_ppl": val_ppl,
        }

        print(f"✅ {name} — loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}\n")

        # cleanup between models
        del model, trainer, evaluator
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 4️⃣ Plot comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name, metrics in results.items():
        plt.plot(metrics["train_losses"], label=f"{name} Train")
        plt.plot(metrics["val_losses"], label=f"{name} Val", linestyle="--")
    plt.title("Char-level Typing — Model Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 5️⃣ Print summary
    # ------------------------------------------------------------
    print("\n📊 Final Comparison Summary:")
    print(f"{'Model':<15} {'Loss':<10} {'Acc (%)':<10} {'PPL':<10}")
    print("-" * 45)
    for name, m in results.items():
        print(f"{name:<15} {m['final_loss']:<10.4f} {m['final_acc']*100:<10.2f} {m['final_ppl']:<10.2f}")
    print("\n✅ ===== Experiment Complete =====\n")

    return results

# ============================================================
# QWEN 2.5–3B FINE-TUNING EXPERIMENT
# ============================================================
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_qwen_finetune_experiment(
    hf_name="OpenAssistant/oasst1",
    txt_files=None,
    vocab_size=400, #8000,
    seq_len=128, #512,
    batch_size=2,
    epochs=3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Fine-tune the Hugging Face pretrained model Qwen/Qwen2.5-3B
    on the predictive typing dataset built from HF or local text.
    """
    print("🚀 Starting Qwen2.5-3B fine-tuning experiment...\n")

    # ------------------------------------------------------------
    # 1️⃣  Build the predictive typing dataset (Byte-level BPE)
    # ------------------------------------------------------------
    class Args:
        pass
    args = Args()
    args.task = "lm" #"typing"
    args.hf_name = hf_name
    args.files = txt_files
    args.tokenizer = "hf:Qwen/Qwen2.5-3B" #"bpe"
    args.vocab_size = vocab_size
    args.seq_len = seq_len
    args.stride = 256
    args.max_train_samples = 100 #use for debug
    args.batch_size = batch_size
    args.mode = "teacher-forced"

    print(f"📘 Building predictive typing dataset from '{hf_name}'...")
    data = build_dataset(args.task, args)
    train_loader, val_loader = data.loaders()
    inspect_dataset(data)
    test_tokenizer(data)
    print("✅ Dataset ready for fine-tuning.\n")
    

    # ------------------------------------------------------------
    # 2️⃣  Load the pretrained Qwen2.5-3B model and tokenizer
    # ------------------------------------------------------------
    model_name = "Qwen/Qwen2.5-3B"
    print(f"🧠 Loading pretrained model: {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    #model.resize_token_embeddings(len(tokenizer))
    print("✅ Model and tokenizer ready.\n")

    # ------------------------------------------------------------
    # 3️⃣  Configure fine-tuning hyperparameters
    # ------------------------------------------------------------
    tcfg = TrainConfig(
        epochs=epochs,
        lr=1e-5,                   # ✅ lower LR for full fine-tune
        weight_decay=0.05,
        warmup_steps=100,
        grad_clip=1.0,
        grad_accum=4,              # ✅ accumulate gradients to keep global batch large
        amp=True,                  # ✅ mixed precision
        amp_dtype="bf16",          # ✅ 'bf16' (safe) or 'fp16' (scaler)
        scheduler_type="cosine",
        save_path="checkpoints/qwen2.5_finetune.pt",
    )

    # ------------------------------------------------------------
    # 4️⃣  Initialize Trainer
    # ------------------------------------------------------------
    trainer = Trainer(
        model,
        data,
        tcfg,
        mode="teacher-forced",
        hf_model=True,   # ✅ Hugging Face model path
    )

    # ------------------------------------------------------------
    # 5️⃣  Fine-tune
    # ------------------------------------------------------------
    print("🚀 Starting fine-tuning...")
    train_losses, val_losses = trainer.fit()

    # ------------------------------------------------------------
    # 6️⃣  Evaluate model
    # ------------------------------------------------------------
    evaluator = Evaluator(model, data, mode="teacher-forced", hf_model=True)
    val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")

    print(f"\n✅ Qwen2.5-3B — loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}\n")

    # ------------------------------------------------------------
    # 7️⃣  Clean up memory
    # ------------------------------------------------------------
    del model, trainer, evaluator
    torch.cuda.empty_cache()
    gc.collect()

    return dict(
        train_losses=train_losses,
        val_losses=val_losses,
        final_loss=val_loss,
        final_acc=val_acc,
        final_ppl=val_ppl,
    )
                
# ============================================================
# MAIN: Universal entry point for training and evaluation
# ============================================================
def main():
    import argparse, torch

    parser = argparse.ArgumentParser(description="Universal Transformer & RNN Trainer")

    # --- General arguments ---
    parser.add_argument("--model_type", type=str, default="TransformerLM",
                        choices=["TransformerLM", "TraditionalTransformerLM", "FullTransformer", "RNN", "LSTM", "hf"])
    parser.add_argument("--hf_model_name", type=str, default="gpt2")
    parser.add_argument("--hf_name", type=str, default=None)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="reduce",
        choices=["reduce", "cosine", "none"],
        help="Select learning rate scheduler: 'reduce' | 'cosine' | 'none'"
    )
    parser.add_argument("--files", nargs="*", help="Text file(s) for training")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--mode", type=str, default="teacher-forced")
    parser.add_argument("--save", type=str, default="checkpoints/model.pt")
    # --- Training hyperparams ---
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1️⃣ Dataset Creation
    # ------------------------------------------------------------
    data = build_dataset(args.model_type, args)

    # ------------------------------------------------------------
    # 2️⃣ Model Initialization
    # ------------------------------------------------------------
    model, hf_mode = build_model(args.model_type, data, args)

    # ------------------------------------------------------------
    # 3️⃣ Training setup
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
    # 4️⃣ Evaluation
    # ------------------------------------------------------------
    evaluator = Evaluator(model, data, mode=args.mode, hf_model=hf_mode)
    loss, acc, ppl = evaluator.evaluate(split="valid")
    print(f"🏁 Final Evaluation | loss={loss:.4f} | acc={acc*100:.2f}% | ppl={ppl:.2f}")
    
if __name__ == "__main__":
    #test_charmodel()
    #main()
    
    #results = run_predictive_typing_experiment()
    run_qwen_finetune_experiment()