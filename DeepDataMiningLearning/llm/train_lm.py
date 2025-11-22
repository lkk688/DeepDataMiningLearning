from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math, os, json, time, random
import math, torch, torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import torch
from dataclasses import dataclass
import gc
from DeepDataMiningLearning.llm.lm_dataset import build_dataset #, test_tokenizer, inspect_dataset
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

    def __init__(self, model, data, mode="teacher-forced", hf_model=False, task="lm"):
        self.model = model
        self.data = data
        self.mode = mode
        self.hf_model = hf_model
        self.task = task
        self.device = next(model.parameters()).device
    
    def _get_pad_id(self):
        """Retrieve pad token ID dynamically from dataset/tokenizer."""
        # For instruction tuning, we use -100 for masking, but we still need the pad_id for the model input
        # However, the loss calculation should ignore -100.
        # If task is instruction, the labels already contain -100 where appropriate.
        
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
        return 0 # Default to 0 if not found


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
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                    elif isinstance(batch, (tuple, list)):
                        #With Hugging Face AutoModelForCausalLM, the model already shifts internally (it uses logits[:, :-1] vs labels[:, 1:]). 
                        # So if your dataset returns (x, y=x→next), don’t feed y as labels to the HF model. Feed labels = x and let the model do the shift.
                        # Support (x, y) or (x, y, lengths)
                        x, y = batch[:2]
                        # if x.size(0) != y.size(0):
                        #     raise ValueError(
                        #         f"❌ Input batch_size ({x.size(0)}) "
                        #         f"!= target batch_size ({y.size(0)}). "
                        #         "Check dataset: HF models require equal batch and seq length."
                        #     )
                        pad_id = self._get_pad_id()
                        x = x.to(self.device)
                        labels = x.clone()
                        # HF causal LM expects ignore_index == -100 at padded positions
                        # If task is instruction, labels might already have -100.
                        # If task is lm, we need to mask padding.
                        if self.task != "instruction":
                             labels[labels == pad_id] = -100
                        
                        inputs = {
                            "input_ids": x,
                            "labels": labels, # same as the input y.to(self.device),
                            "attention_mask": (x != pad_id).to(self.device),
                        }
                    else:
                        raise TypeError(f"Unsupported batch type: {type(batch)}")
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits # (B, T, V)

                    total_loss += float(loss.item())
                    #preds = logits.argmax(-1)
                    labels = inputs.get("labels", None)
                    # logits: (B, T, V) from the model
                    # labels: should be (B, T); may come as list/tensor
                    # IMPORTANT: for HF causal LM, ignored labels are usually -100
                    #ignore_index = -100  # getattr(self.model.config, "ignore_index", -100) if you prefer

                    if labels is not None:
                        # Ensure tensor on the right device/dtype
                        if not torch.is_tensor(labels):
                            labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
                        else:
                            labels = labels.to(logits.device).long()
                        ignore_index = getattr(self.model.config, "ignore_index", -100)

                        # Predictions
                        #preds = logits.argmax(dim=-1)  # (B, T)

                        # If shapes somehow differ, align conservatively
                        # if preds.shape != labels.shape:
                        #     T = min(preds.size(-1), labels.size(-1))
                        #     preds = preds[..., :T]
                        #     labels = labels[..., :T]
                            
                        # ---- shift to match the loss target ----
                        shift_logits = logits[:, :-1, :]      # predicts token at t+1
                        shift_labels = labels[:, 1:]          # gold next tokens

                        # Build a boolean tensor mask (labels we care about)
                        #mask = labels.ne(ignore_index)   # (B, T) boolean tensor
                        # build mask from shifted labels (valid supervision positions)
                        mask = shift_labels.ne(ignore_index)  # (B, T-1) boolean
                        # top-1 predictions
                        preds = shift_logits.argmax(dim=-1)   # (B, T-1)

                        # Accumulate counts
                        #total_correct += (preds[mask] == labels[mask]).sum().item()
                        total_correct += (preds.eq(shift_labels) & mask).sum().item()
                        total_count   += mask.sum().item()
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
                if self.hf_model==False and self.mode == "teacher-forced":
                    # Each batch is (x, y) or (x, y, lengths)
                    if len(batch) == 3:
                        x, y, _ = batch
                    else:
                        x, y = batch
                    #x, y = [t.to(self.device) for t in batch]
                    x, y = x.to(self.device), y.to(self.device)
                    T = x.size(1)
                    mask = TransformerLM.causal_mask(T, x.device)
                    logits = self.model(x, attn_mask=mask)
                    
                    # Determine ignore_index
                    pad_idx = self._get_pad_id()
                    ignore_index = -100 if self.task == "instruction" else pad_idx

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=ignore_index,
                    )
                    total_loss += loss.item()
                    preds = logits.argmax(-1)
                    
                    # Correctness check
                    # For instruction, y has -100. For LM, y has pad_idx.
                    # We should mask out ignore_index for accuracy calc.
                    valid_mask = (y != ignore_index)
                    total_correct += (preds == y).masked_select(valid_mask).sum().item()
                    total_count += valid_mask.sum().item()
                    continue


                elif self.hf_model==False and self.mode == "final-token":
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
        #ppl = math.exp(avg_loss)
        ppl = math.exp(min(20.0, avg_loss))  # clamp to avoid overflow

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
      ✅ Torch Compile support
    """
    def __init__(self, model, data, tcfg, mode="teacher-forced", hf_model=False, early_stop_patience=2, task="lm", peft_config=None):
        """
        Args:
            model: TransformerLM or Hugging Face model (e.g. AutoModelForCausalLM)
            data: DataModule providing loaders()
            tcfg: TrainConfig dataclass (lr, epochs, warmup, etc.)
            mode: 'teacher-forced' or 'final-token'
            hf_model: if True, assume Hugging Face forward signature (model(input_ids, labels=...))
            task: 'lm', 'typing', 'instruction', 'seq2seq'
            peft_config: Optional LoraConfig or PEFT config
        """
        self.model = model
        self.data = data
        self.mode = mode
        self.hf_model = hf_model
        self.tcfg = tcfg
        self.task = task

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Keep reference to raw model for isinstance checks (in case of compilation)
        self.raw_model = self.model

        # ------------------------------------------------------------
        # PEFT / LoRA Integration
        # ------------------------------------------------------------
        if peft_config is not None:
            try:
                from peft import get_peft_model
                print("�️  Wrapping model with PEFT/LoRA...")
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
                # Update raw_model reference if needed, though usually we check base model type
                # self.raw_model = self.model.base_model.model # depends on structure
            except ImportError:
                print("⚠️  PEFT library not found. Skipping PEFT wrapping.")

        # Optional: Compile model (PyTorch 2.0+)
        if hasattr(torch, "compile") and os.name != "nt": # Skip on Windows if problematic
             print("🚀 Compiling model with torch.compile()...")
             try:
                 self.model = torch.compile(self.model)
             except Exception as e:
                 print(f"⚠️ torch.compile failed: {e}. Proceeding without compilation.")


        # Optimizer and scheduler setup
        no_decay = ["bias", "LayerNorm.weight", "norm", "emb"]

        # Prepare two lists for the two parameter groups
        params_decay = []
        params_no_decay = []

        # --- Single pass over all named parameters ---
        for n, p in self.model.named_parameters():
            # 1. Skip parameters that are frozen
            if not p.requires_grad:
                continue

            # 2. Check if the parameter name matches any in the no_decay list
            if any(nd in n for nd in no_decay):
                # Add to the no_decay list
                params_no_decay.append(p)
            else:
                # Add to the decay list
                params_decay.append(p)

        # Create the final grouped_params list
        grouped_params = [
            {
                "params": params_decay,
                "weight_decay": tcfg.weight_decay,
                "initial_lr": tcfg.lr,  # Your original code used 'initial_lr'
            },
            {
                "params": params_no_decay,
                "weight_decay": 0.0,
                "initial_lr": tcfg.lr,  # Your original code used 'initial_lr'
            },
        ]

        # Initialize the optimizer
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
                self.opt, factor=0.5, patience=1
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
        return 0 # Default to 0

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
        ignore_index = -100 if self.task == "instruction" else pad_idx
        
        # -----------------------------------------------------------
        # Case 1: Hugging Face models (AutoModelForCausalLM)
        # -----------------------------------------------------------
        if self.hf_model:
            #outputs = self.model(**inputs)
            # Normalize batch -> x (input_ids)
            if isinstance(batch, dict):
                x = batch["input_ids"].to(self.device)
                # Always recompute labels from x to avoid "double shift"
                # even if the dict contains 'labels'.
            elif isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)  # ignore y here; model will shift internally
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            # Attention mask
            attention_mask = (x != pad_idx).to(self.device)

            # Labels = input_ids, with pads -> -100 (ignore_index)
            # For instruction tuning, if batch has labels, use them!
            if self.task == "instruction" and isinstance(batch, (tuple, list)) and len(batch) >= 2:
                 labels = batch[1].to(self.device)
            else:
                 labels = x.clone()
                 labels[labels == pad_idx] = -100

            outputs = self.model(
                input_ids=x,
                attention_mask=attention_mask,
                labels=labels,  # HF loss will compare logits[:, :-1] to labels[:, 1:]
            )
            return outputs.loss, outputs.logits


        # -----------------------------------------------------------
        # Case 2: RNN / LSTM (with optional sequence lengths)
        # -----------------------------------------------------------
        if isinstance(self.raw_model, (RNNLanguageModel, LSTMLanguageModel)):
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
        # -----------------------------------------------------------
        # Case 3: Full Encoder–Decoder Transformer
        # -----------------------------------------------------------
        if isinstance(self.raw_model, (FullTransformer,PyTorchTransformer)):
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
        # -----------------------------------------------------------
        # Case 4: Decoder-only Transformers (TransformerLM)
        # -----------------------------------------------------------
        if isinstance(self.raw_model, (TransformerLM, TraditionalTransformerLM)):
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
                # logits: [B, T_pred, V]
                # y: [B, T_label]
                # Align lengths (typing target shorter)
                B, T_pred, V = logits.shape
                T_label = y.size(1)
                # Align lengths if mismatch
                if T_pred != T_label:
                    min_len = min(T_pred, T_label)
                    logits = logits[:, :min_len, :]
                    y = y[:, :min_len]

                loss = F.cross_entropy(
                    logits.reshape(-1, V),
                    y.reshape(-1),
                    ignore_index=ignore_index,   # ✅ correct pad ID or -100
                )

                # Compute full-sequence cross-entropy loss
                # loss = F.cross_entropy(
                #     logits.view(-1, logits.size(-1)),  # flatten [B*T, V]
                #     y.view(-1),                        # flatten [B*T]
                #     ignore_index=pad_idx,
                # )
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
            #evaluator = Evaluator(self.model, self.data, self.mode, hf_model=self.hf_model)
            evaluator = Evaluator(model=self.model, data=self.data, mode=self.mode, hf_model=self.hf_model, task=self.task)
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
     

                
# ============================================================
# MAIN: Universal entry point for training and evaluation
# ============================================================
def main():
    import argparse, torch

    parser = argparse.ArgumentParser(description="Universal Transformer & RNN Trainer")

    # --- General arguments ---
    parser.add_argument("--model_type", type=str, default="TransformerLM",
                        choices=["TransformerLM", "TraditionalTransformerLM", "FullTransformer", "RNN", "LSTM", "hf"])
    parser.add_argument("--task", type=str, default="lm",
                        choices=["lm", "typing", "instruction", "seq2seq"],
                        help="Task type: 'lm', 'typing', 'instruction', 'seq2seq'")
    parser.add_argument("--hf_model_name", type=str, default="gpt2")
    parser.add_argument("--hf_name", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="char", help="Tokenizer type: 'char', 'hf:gpt2', etc.")


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
    # Pass 'task' explicitly to build_dataset
    data = build_dataset(args.task, args)

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

    trainer = Trainer(model, data, tcfg, mode=args.mode, hf_model=hf_mode, task=args.task)
    trainer.fit()

    # ------------------------------------------------------------
    # 4️⃣ Evaluation
    # ------------------------------------------------------------
    evaluator = Evaluator(model, data, mode=args.mode, hf_model=hf_mode, task=args.task)
    loss, acc, ppl = evaluator.evaluate(split="valid")
    print(f"🏁 Final Evaluation | loss={loss:.4f} | acc={acc*100:.2f}% | ppl={ppl:.2f}")

    
if __name__ == "__main__":
    #test_charmodel()
    main()
    
    #results = run_predictive_typing_experiment()
    #run_qwen_finetune_experiment()
    #run_char_typing_experiment()