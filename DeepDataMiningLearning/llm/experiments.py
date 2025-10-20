
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os   


def test_charmodel():
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


def run_char_typing_experiment(
    hf_name="npvinHnivqn/EnglishDictionary",
    seq_len=64,
    batch_size=64,
    epochs=6,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs"
):
    """
    Character-level predictive typing experiment with configuration saving
    and general-purpose inference testing.
    """
    print("\n🚀 ===== Character-Level Typing Experiment =====")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # ------------------------------------------------------------
    # 1️⃣ Build dataset (char-level typing)
    # ------------------------------------------------------------
    class Args:
        pass

    args = Args()
    args.files = None
    args.batch_size = batch_size
    args.task = "lm"
    args.hf_name = hf_name
    args.hf_split = "train"
    args.hf_features = ["definition"] #["word"]
    args.tokenizer = "char"
    args.seq_len = seq_len
    args.stride = 1
    args.next_token_window = 1
    args.num_prefixes_per_sentence = 5
    args.max_prefix_len = 20

    print(f"📘 Building char-level typing dataset from '{hf_name}'...")
    data = build_dataset(args.task, args)
    train_loader, val_loader = data.loaders()
    inspect_dataset(data, task=args.task, num_batches=4, num_samples=4)
    print("✅ Dataset ready for training.\n")

    # ------------------------------------------------------------
    # 2️⃣ Define model configurations
    # ------------------------------------------------------------
    # model_configs = {
    #     "RNN": dict(model_type="RNN", dim=64, layers=1, lr=1e-3, epochs=epochs),
    #     "LSTM": dict(model_type="LSTM", dim=64, layers=1, lr=1e-3, epochs=epochs),
    #     "TraditionalTransformerLM": dict(
    #         model_type="TraditionalTransformerLM",
    #         dim=128, layers=2, heads=2, ff_dim=256, lr=8e-4, epochs=epochs),
    #     "TransformerLM": dict(
    #         model_type="TransformerLM",
    #         dim=128, layers=2, heads=2, lr=8e-4, epochs=epochs),
    # }
    model_configs = {
        "TraditionalTransformerLM": dict(
            model_type="TraditionalTransformerLM",
            dim=128, layers=2, heads=2, ff_dim=256, lr=8e-4, epochs=epochs),
        "TransformerLM": dict(
            model_type="TransformerLM",
            dim=128, layers=2, heads=2, lr=8e-4, epochs=epochs),
        "RNN": dict(model_type="RNN", dim=64, layers=1, lr=1e-3, epochs=epochs),
        "LSTM": dict(model_type="LSTM", dim=64, layers=1, lr=1e-3, epochs=epochs),
    }

    results = {}
    all_configs = {"dataset_args": vars(args), "models": {}}

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
        margs.rope = False
        margs.save = os.path.join(output_dir, "checkpoints", f"{name.lower()}_char_typing.pt")

        all_configs["models"][name] = {**cfg, "save_path": margs.save}

        model, hf_mode = build_model(margs.model_type, data, margs)

        tcfg = TrainConfig(
            epochs=epochs,
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

        del model, trainer, evaluator
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 4️⃣ Save configurations and results to JSON
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = os.path.join(output_dir, f"experiment_config_{timestamp}.json")
    summary = {
        "args": vars(args),
        "model_configs": model_configs,
        "results": results,
        "timestamp": timestamp,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"💾 Saved experiment configuration and results → {cfg_path}")

    # ------------------------------------------------------------
    # 5️⃣ Plot comparison
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
    # 6️⃣ Summary print
    # ------------------------------------------------------------
    print("\n📊 Final Comparison Summary:")
    print(f"{'Model':<15} {'Loss':<10} {'Acc (%)':<10} {'PPL':<10}")
    print("-" * 45)
    for name, m in results.items():
        print(f"{name:<15} {m['final_loss']:<10.4f} {m['final_acc']*100:<10.2f} {m['final_ppl']:<10.2f}")
    print("\n✅ ===== Experiment Complete =====\n")


    # Example inference call
    for name, cfg in model_configs.items():
        print(f"\n🔮 Running inference with {name} model...")
        best_model_path = os.path.join(output_dir, "checkpoints", f"{name.lower()}_char_typing.pt")
        run_inference(
            model_path=best_model_path,
            model_type=cfg["model_type"],
            prompt="The cat",
            tokenizer=data.tok,
            seq_len=64,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            num_samples=2,
        )

    return results



# ============================================================
# QWEN 2.5–3B FINE-TUNING EXPERIMENT
# ============================================================
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
    args.files = None
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

if __name__ == "__main__":
    #test_charmodel()
    #main()
    
    #results = run_predictive_typing_experiment()
    #run_qwen_finetune_experiment()
    run_char_typing_experiment()