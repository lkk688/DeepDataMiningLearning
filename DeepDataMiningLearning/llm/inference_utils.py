

import torch
import matplotlib.pyplot as plt
import os, json, torch, random
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

import os
import json
import torch

# ------------------------------------------------------------
# Import your framework utilities
# ------------------------------------------------------------
from transformer import build_model           # model builder
from data_utils import CharTokenizer          # or whatever tokenizer you used
from inference_utils import run_inference     # the independent inference function

# ------------------------------------------------------------
# 1️⃣ Load experiment configuration
# ------------------------------------------------------------
def load_experiment_config(config_path):
    """
    Load experiment configuration JSON that includes dataset args and model settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✅ Loaded experiment config from {config_path}")
    return cfg


# ------------------------------------------------------------
# 2️⃣ Simple inference pipeline
# ------------------------------------------------------------
def simple_inference(
    config_path,
    model_name="TransformerLM",
    prompt="The cat",
    output_dir="outputs",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_samples=3,
):
    """
    Load trained model and tokenizer using the experiment config,
    then run inference to generate text completions.
    """
    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    exp_cfg = load_experiment_config(config_path)
    args = exp_cfg["args"]
    model_cfg = exp_cfg["model_configs"][model_name]
    seq_len = args.get("seq_len", 128)
    model_path = os.path.join(output_dir, "checkpoints", f"{model_name.lower()}_char_typing.pt")

    # --------------------------------------------------------
    # Rebuild tokenizer
    # --------------------------------------------------------
    if args["tokenizer"] == "char":
        # You may have a saved tokenizer.json if you used BPE; for char, rebuild from dataset
        tokenizer = CharTokenizer.load(os.path.join(output_dir, "tokenizer.json")) \
                    if os.path.exists(os.path.join(output_dir, "tokenizer.json")) \
                    else CharTokenizer([prompt])   # fallback minimal tokenizer
        print("🔤 Loaded character-level tokenizer.")
    else:
        raise ValueError(f"Unsupported tokenizer type: {args['tokenizer']}")

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------
    run_inference(
        model_path=model_path,
        model_type=model_cfg["model_type"],
        prompt=prompt,
        tokenizer=tokenizer,
        seq_len=seq_len,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_samples=num_samples,
    )

    
def run_inference(
    model_path,
    model_type,
    prompt,
    tokenizer,
    seq_len=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    num_samples=3,
):
    """
    General-purpose inference function for trained language models.

    This function:
        - Loads a trained model checkpoint
        - Tokenizes the given prompt
        - Generates new tokens autoregressively
        - Supports multiple sampling methods:
            * Greedy decoding
            * Temperature sampling
            * Top-K sampling
            * Top-P (nucleus) sampling
        - Prints multiple output samples for qualitative comparison

    Args:
        model_path (str): Path to the trained model checkpoint (.pt file).
        model_type (str): Model type string, e.g., "TransformerLM", "RNN", "LSTM".
        prompt (str): Text prompt or prefix to condition generation.
        tokenizer (Tokenizer): Tokenizer object with encode() / decode() methods.
        seq_len (int): Max context length (window size during generation).
        device (str): "cuda" or "cpu".
        max_new_tokens (int): Number of tokens to generate beyond the prompt.
        temperature (float): Temperature for sampling. Lower = more deterministic.
        top_k (int): If > 0, keep only top-K highest probability tokens.
        top_p (float): Nucleus sampling threshold (keep top tokens with cumulative prob ≤ top_p).
        num_samples (int): How many completions to generate for comparison.

    Example:
        run_inference(
            model_path="outputs/checkpoints/transformerlm_char_typing.pt",
            model_type="TransformerLM",
            prompt="The cat",
            tokenizer=data.tok,
            seq_len=64,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            num_samples=3,
        )
    """
    print(f"\n🔮 Loading model checkpoint from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found: {model_path}")

    # ----------------------------------------------------------------------
    # 1️⃣ Build and load model
    # ----------------------------------------------------------------------
    # We assume build_model(model_type, data, args) is available globally
    # and returns a (model, hf_mode) pair.
    model, _ = build_model(model_type, None, None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 2️⃣ Helper: sample next token from logits with multiple strategies
    # ----------------------------------------------------------------------
    def sample_next_token(logits):
        """
        Sample a token index from logits using temperature, top-k, and top-p.
        """
        # Temperature scaling
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        # --- Top-K sampling (keep only K most probable tokens)
        if top_k > 0:
            top_vals, top_idx = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_idx, top_vals)
            probs = probs / probs.sum()

        # --- Top-P (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            if torch.any(cutoff):
                cutoff_idx = torch.nonzero(cutoff)[0].item()
                keep_idx = sorted_idx[: cutoff_idx + 1]
                probs = torch.zeros_like(probs).scatter_(0, keep_idx, probs[keep_idx])
                probs = probs / probs.sum()

        # Draw one token from the resulting distribution
        next_id = torch.multinomial(probs, 1).item()
        return next_id

    # ----------------------------------------------------------------------
    # 3️⃣ Prepare prompt tokens
    # ----------------------------------------------------------------------
    prompt_ids = tokenizer.encode(prompt)
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔢 Encoded prompt length: {len(prompt_ids)} tokens")

    # ----------------------------------------------------------------------
    # 4️⃣ Generate samples
    # ----------------------------------------------------------------------
    for i in range(num_samples):
        ids = prompt_ids.copy()
        print(f"\n🎯 Generating Sample {i + 1}/{num_samples}...")
        for step in range(max_new_tokens):
            # Use only the last seq_len tokens as input context
            x = torch.tensor(ids[-seq_len:], dtype=torch.long, device=device)[None, :]

            with torch.no_grad():
                logits = model(x)
                if isinstance(logits, tuple):  # handle RNN/LSTM returning (output, hidden)
                    logits = logits[0]
                logits = logits[0, -1, :]  # last token logits

            next_id = sample_next_token(logits.cpu())
            ids.append(next_id)

        # ------------------------------------------------------------------
        # Decode and print generated text
        # ------------------------------------------------------------------
        text = tokenizer.decode(ids)
        print(f"🧩 Sample {i+1} Output:\n{text}")
        print("-" * 60)

    print("✨ Inference complete.\n")

import torch
from transformer import TransformerLM, TraditionalTransformerLM, ModelConfig
from data_utils import CharTokenizer  # or your tokenizer class

def simple_inference_from_checkpoint(
    model_path,
    prompt="The cat",
    tokenizer=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=50,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    num_samples=3,
):
    """
    Inference directly from a checkpoint that contains both:
        { "model": state_dict, "cfg": model.cfg }

    Automatically:
      - Loads model config and weights
      - Rebuilds the right Transformer type
      - Generates samples from a text prompt
    """
    print(f"🔮 Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    model_cfg = ckpt.get("cfg", None)

    if model_cfg is None:
        raise ValueError("❌ This checkpoint does not include model.cfg. Re-train or provide a config manually.")

    print("✅ Config loaded from checkpoint:")
    print(model_cfg.__dict__ if hasattr(model_cfg, "__dict__") else model_cfg)

    # ----------------------------------------------------------------------
    # 1️⃣  Prepare tokenizer
    # ----------------------------------------------------------------------
    if tokenizer is None:
        print("🔤 Loading/rebuilding tokenizer ...")
        tokenizer = CharTokenizer.load("outputs/tokenizer.json") \
            if hasattr(CharTokenizer, "load") and \
               os.path.exists("outputs/tokenizer.json") else CharTokenizer([prompt])

    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size

    # ----------------------------------------------------------------------
    # 2️⃣  Build model directly from cfg
    # ----------------------------------------------------------------------
    model_type = getattr(model_cfg, "model_type", "TransformerLM")
    seq_len = getattr(model_cfg, "seq_len", 128)

    # If cfg is a dict, unpack it
    if isinstance(model_cfg, dict):
        cfg_obj = ModelConfig(**model_cfg)
    else:
        cfg_obj = model_cfg

    cfg_obj.vocab_size = vocab_size  # ensure vocab matches tokenizer

    if model_type == "TransformerLM":
        print("🧠 Building modern TransformerLM ...")
        model = TransformerLM(cfg_obj)
    elif model_type == "TraditionalTransformerLM":
        print("🧩 Building TraditionalTransformerLM ...")
        model = TraditionalTransformerLM(cfg_obj)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("✅ Weights loaded successfully.")

    # ----------------------------------------------------------------------
    # 3️⃣  Sampling helper
    # ----------------------------------------------------------------------
    def sample_next_token(logits):
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        if top_k > 0:
            vals, idx = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, idx, vals)
            probs /= probs.sum()
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            if torch.any(cutoff):
                cutoff_idx = torch.nonzero(cutoff)[0].item()
                keep_idx = sorted_idx[: cutoff_idx + 1]
                probs = torch.zeros_like(probs).scatter_(0, keep_idx, probs[keep_idx])
                probs /= probs.sum()
        return torch.multinomial(probs, 1).item()

    # ----------------------------------------------------------------------
    # 4️⃣  Generate completions
    # ----------------------------------------------------------------------
    prompt_ids = tokenizer.encode(prompt)
    print(f"\n📝 Prompt: '{prompt}' ({len(prompt_ids)} tokens)\n")

    for i in range(num_samples):
        ids = prompt_ids.copy()
        for _ in range(max_new_tokens):
            x = torch.tensor(ids[-seq_len:], dtype=torch.long, device=device)[None, :]
            with torch.no_grad():
                logits = model(x)
                if isinstance(logits, tuple):  # for RNNs
                    logits = logits[0]
                logits = logits[0, -1, :]
            next_id = sample_next_token(logits.cpu())
            ids.append(next_id)
        text = tokenizer.decode(ids)
        print(f"🧩 Sample {i+1}:\n{text}\n" + "-" * 60)

    print("✨ Inference complete.\n")
# ------------------------------------------------------------
# 3️⃣ Run directly (example)
# ------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from transformer import TransformerLM, ModelConfig
    from data_utils import CharTokenizer  # adjust if your tokenizer class differs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # Load checkpoint (contains both weights and cfg)
    # ------------------------------------------------------------
    ckpt_path = "outputs/checkpoints/transformerlm_char_typing.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    cfg_dict = ckpt["cfg"]

    print("✅ Config loaded from checkpoint:", cfg_dict)

    # ------------------------------------------------------------
    # Build ModelConfig directly from the checkpoint cfg
    # ------------------------------------------------------------
    model_cfg = ModelConfig(
        vocab_size=cfg_dict["vocab_size"],   # ✅ 69 from checkpoint
        dim=cfg_dict["dim"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        dropout=cfg_dict.get("dropout", 0.1),
        seq_len=cfg_dict.get("max_seq_len", 64),
        rope=cfg_dict.get("rope", False),
    )

    # ------------------------------------------------------------
    # Initialize the model using the saved configuration
    # ------------------------------------------------------------
    model = TransformerLM(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print("✅ Model rebuilt successfully with vocab_size =", model_cfg.vocab_size)

    # ------------------------------------------------------------
    # Load tokenizer (char-level tokenizer from training)
    # ------------------------------------------------------------
    tokenizer = CharTokenizer.load("outputs/tokenizer.json")
    print(f"✅ Tokenizer loaded (vocab size = {len(tokenizer.vocab)})")

    # ------------------------------------------------------------
    # Perform inference
    # ------------------------------------------------------------
    prompt = "The cat"
    input_ids = tokenizer.encode(prompt)
    print(f"\n📝 Prompt: '{prompt}' ({len(input_ids)} tokens)")

    seq_len = model_cfg.seq_len
    max_new_tokens = 50

    for _ in range(max_new_tokens):
        x = torch.tensor(input_ids[-seq_len:], dtype=torch.long, device=device)[None, :]
        with torch.no_grad():
            logits = model(x)[0, -1, :]
        next_token = torch.argmax(logits, dim=-1).item()
        input_ids.append(next_token)

    output_text = tokenizer.decode(input_ids)
    print("\n🧩 Generated text:\n", output_text)

    # path to saved experiment config
    config_path = "outputs/experiment_config_20250214_112045.json"

    simple_inference(
        config_path=config_path,
        model_name="TransformerLM",   # or "TraditionalTransformerLM", "LSTM", etc.
        prompt="The cat",
        output_dir="outputs",
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        num_samples=3,
    )

    best_model_path = "outputs/checkpoints/transformerlm_char_typing.pt"
    run_inference(
        model_path=best_model_path,
        model_type="TransformerLM",
        prompt="The cat",
        tokenizer=data.tok,
        seq_len=64,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        num_samples=3,
    )
    
    simple_inference_from_checkpoint(
        model_path="outputs/checkpoints/transformerlm_char_typing.pt",
        prompt="The cat",
        max_new_tokens=60,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        num_samples=3,
    )