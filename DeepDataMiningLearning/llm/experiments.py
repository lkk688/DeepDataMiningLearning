
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os   
import json
from datetime import datetime
from DeepDataMiningLearning.llm.tokenizer_utils import EN_WORDS, test_tokenizers, TokenizerFactory
from DeepDataMiningLearning.llm.lm_dataset import build_dataset, inspect_dataset
from DeepDataMiningLearning.llm.train_lm import TrainConfig, build_model, Trainer, Evaluator
from DeepDataMiningLearning.llm.inference_utils import run_inference

def test_charmodel():
    # ------------------------------------------------------------
    # Create a text file from the English word corpus
    # ------------------------------------------------------------
    with open("english_words.txt", "w", encoding="utf-8") as f:
        for w in EN_WORDS: #.words():
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


def run_tokenizer_comparison_experiment(
    hf_name="npvinHnivqn/EnglishDictionary",
    seq_len=128,
    batch_size=32,
    epochs=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="outputs/tokenizer_comparison"
):
    """
    Compare TraditionalTransformerLM performance across different tokenizers:
    char, word, hftokenizer, custom:sp-unigram, and custom:tiktoken-bpe
    """
    print("\n🚀 ===== Tokenizer Comparison Experiment =====")
    print(f"Testing TraditionalTransformerLM with 5 different tokenizers")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Define tokenizers to test
    tokenizer_configs = {
        "sp-unigram": {"tokenizer": "custom:sp-unigram", "color": "orange"},
        "tiktoken-bpe": {"tokenizer": "custom:tiktoken-bpe", "color": "purple"},
        "hftokenizer": {"tokenizer": "hf:Qwen/Qwen2-1.5B", "color": "red"},
        "char": {"tokenizer": "char", "color": "blue"},
        "word": {"tokenizer": "word", "color": "green"}, 
    }
    
    results = {}
    all_configs = {"experiment_type": "tokenizer_comparison", "tokenizers": {}}
    
    # Prepare sample texts for custom tokenizers
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of NLP."
    ]
    
    # ------------------------------------------------------------
    # Train model with each tokenizer
    # ------------------------------------------------------------
    for tok_name, tok_config in tokenizer_configs.items():
        print(f"\n🧠 Training TraditionalTransformerLM with {tok_name} tokenizer...")
        
        # Build dataset with specific tokenizer
        class Args:
            pass
        
        args = Args()
        args.files = None
        args.batch_size = batch_size
        args.task = "lm"
        args.hf_name = hf_name
        args.hf_split = "train"
        args.hf_features = ["definition"]
        args.tokenizer = tok_config["tokenizer"]
        args.seq_len = seq_len
        args.stride = 1
        # args.next_token_window = 1
        # args.num_prefixes_per_sentence = 5
        # args.max_prefix_len = 20
        
        # For custom tokenizers, we need to prepare them first
        if tok_config["tokenizer"].startswith("custom:"):
            print(f"📘 Preparing custom tokenizer: {tok_config['tokenizer']}")
            try:
                # Build custom tokenizer with sample texts
                custom_tok = TokenizerFactory.build(
                    tokenizer=tok_config["tokenizer"],
                    texts=sample_texts,
                    vocab_size=5000
                )
                print(f"✅ Custom tokenizer {tok_name} prepared successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not prepare custom tokenizer {tok_name}: {e}")
                print(f"Skipping {tok_name} tokenizer...")
                continue
        
        print(f"📘 Building dataset with {tok_name} tokenizer...")
        try:
            data = build_dataset(args.task, args)
            train_loader, val_loader = data.loaders()
            print(f"✅ Dataset ready with {tok_name} tokenizer (vocab_size: {data.tok.vocab_size})")
        except Exception as e:
            print(f"❌ Failed to build dataset with {tok_name} tokenizer: {e}")
            continue
        
        # Model configuration
        class ModelArgs:
            pass
        
        margs = ModelArgs()
        margs.model_type = "TraditionalTransformerLM"
        margs.dim = 128
        margs.layers = 4
        margs.heads = 4
        margs.lr = 8e-4
        margs.seq_len = seq_len
        margs.rope = False
        margs.save = os.path.join(output_dir, "checkpoints", f"transformer_{tok_name}.pt")
        
        all_configs["tokenizers"][tok_name] = {
            "tokenizer": tok_config["tokenizer"],
            "vocab_size": data.tok.vocab_size,
            "save_path": margs.save,
            "model_config": {
                "dim": margs.dim,
                "layers": margs.layers,
                "heads": margs.heads,
                "lr": margs.lr
            }
        }
        
        # Build and train model
        try:
            model, hf_mode = build_model(margs.model_type, data, margs)
            
            tcfg = TrainConfig(
                epochs=epochs,
                lr=margs.lr,
                weight_decay=0.01,
                warmup_steps=100,
                grad_clip=1.0,
                grad_accum=4,
                amp=True,
                scheduler_type="reduce",
                save_path=margs.save,
            )
            
            trainer = Trainer(model, data, tcfg, mode="teacher-forced", hf_model=hf_mode)
            train_losses, val_losses = trainer.fit()
            
            evaluator = Evaluator(model, data, mode="teacher-forced", hf_model=hf_mode)
            val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")
            
            results[tok_name] = {
                "tokenizer": tok_config["tokenizer"],
                "vocab_size": data.tok.vocab_size,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "final_loss": val_loss,
                "final_acc": val_acc,
                "final_ppl": val_ppl,
                "color": tok_config["color"]
            }
            
            print(f"✅ {tok_name} — loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f} | vocab={data.tok.vocab_size}")
            
            # Save individual training curves
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label="Train Loss", color=tok_config["color"])
            plt.plot(val_losses, label="Val Loss", color=tok_config["color"], linestyle="--")
            plt.title(f"{tok_name} Tokenizer - Training Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(range(len(train_losses)), [val_acc] * len(train_losses), 
                    label=f"Final Acc: {val_acc*100:.2f}%", color=tok_config["color"])
            plt.title(f"{tok_name} Tokenizer - Final Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            curve_path = os.path.join(output_dir, "plots", f"{tok_name}_training_curves.png")
            plt.savefig(curve_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved training curves → {curve_path}")
            
        except Exception as e:
            print(f"❌ Failed to train model with {tok_name} tokenizer: {e}")
            continue
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            if 'evaluator' in locals():
                del evaluator
            torch.cuda.empty_cache()
            gc.collect()
    
    # ------------------------------------------------------------
    # Save experiment configuration and results
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_path = os.path.join(output_dir, f"tokenizer_comparison_results_{timestamp}.json")
    summary = {
        "experiment_type": "tokenizer_comparison",
        "model_type": "TraditionalTransformerLM",
        "dataset": hf_name,
        "hyperparameters": {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "epochs": epochs,
            "dim": 128,
            "layers": 2,
            "heads": 2,
            "lr": 8e-4
        },
        "tokenizer_configs": all_configs["tokenizers"],
        "results": results,
        "timestamp": timestamp,
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"💾 Saved experiment results → {results_path}")
    
    # ------------------------------------------------------------
    # Create comparison plots
    # ------------------------------------------------------------
    if len(results) > 0:
        # Training loss comparison
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training losses
        plt.subplot(2, 3, 1)
        for tok_name, metrics in results.items():
            plt.plot(metrics["train_losses"], label=f"{tok_name}", 
                    color=metrics["color"], linewidth=2)
        plt.title("Training Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Validation losses
        plt.subplot(2, 3, 2)
        for tok_name, metrics in results.items():
            plt.plot(metrics["val_losses"], label=f"{tok_name}", 
                    color=metrics["color"], linewidth=2, linestyle="--")
        plt.title("Validation Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Final metrics comparison
        plt.subplot(2, 3, 3)
        tokenizers = list(results.keys())
        final_losses = [results[tok]["final_loss"] for tok in tokenizers]
        colors = [results[tok]["color"] for tok in tokenizers]
        bars = plt.bar(tokenizers, final_losses, color=colors, alpha=0.7)
        plt.title("Final Validation Loss")
        plt.ylabel("Loss")
        plt.xticks(rotation=45)
        for bar, loss in zip(bars, final_losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{loss:.3f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy comparison
        plt.subplot(2, 3, 4)
        final_accs = [results[tok]["final_acc"] * 100 for tok in tokenizers]
        bars = plt.bar(tokenizers, final_accs, color=colors, alpha=0.7)
        plt.title("Final Accuracy")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        for bar, acc in zip(bars, final_accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Perplexity comparison
        plt.subplot(2, 3, 5)
        final_ppls = [results[tok]["final_ppl"] for tok in tokenizers]
        bars = plt.bar(tokenizers, final_ppls, color=colors, alpha=0.7)
        plt.title("Final Perplexity")
        plt.ylabel("Perplexity")
        plt.xticks(rotation=45)
        for bar, ppl in zip(bars, final_ppls):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{ppl:.2f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Vocabulary size comparison
        plt.subplot(2, 3, 6)
        vocab_sizes = [results[tok]["vocab_size"] for tok in tokenizers]
        bars = plt.bar(tokenizers, vocab_sizes, color=colors, alpha=0.7)
        plt.title("Vocabulary Size")
        plt.ylabel("Vocab Size")
        plt.xticks(rotation=45)
        for bar, size in zip(bars, vocab_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{size}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "plots", f"tokenizer_comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved comparison plots → {comparison_path}")
        
        # Create a summary table plot
        plt.figure(figsize=(12, 8))
        plt.axis('tight')
        plt.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ["Tokenizer", "Vocab Size", "Final Loss", "Accuracy (%)", "Perplexity"]
        
        for tok_name in tokenizers:
            metrics = results[tok_name]
            row = [
                tok_name,
                f"{metrics['vocab_size']:,}",
                f"{metrics['final_loss']:.4f}",
                f"{metrics['final_acc']*100:.2f}%",
                f"{metrics['final_ppl']:.2f}"
            ]
            table_data.append(row)
        
        table = plt.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Color code the rows
        for i, tok_name in enumerate(tokenizers):
            color = results[tok_name]["color"]
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
        
        plt.title("Tokenizer Comparison Summary", fontsize=16, fontweight='bold', pad=20)
        
        table_path = os.path.join(output_dir, "plots", f"tokenizer_summary_table_{timestamp}.png")
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved summary table → {table_path}")
    
    # ------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------
    print("\n📊 ===== Tokenizer Comparison Summary =====")
    if len(results) > 0:
        print(f"{'Tokenizer':<15} {'Vocab':<8} {'Loss':<8} {'Acc (%)':<8} {'PPL':<8}")
        print("-" * 55)
        for tok_name, metrics in results.items():
            print(f"{tok_name:<15} {metrics['vocab_size']:<8} {metrics['final_loss']:<8.4f} "
                  f"{metrics['final_acc']*100:<8.2f} {metrics['final_ppl']:<8.2f}")
        
        # Find best performing tokenizer
        best_loss_tok = min(results.keys(), key=lambda x: results[x]["final_loss"])
        best_acc_tok = max(results.keys(), key=lambda x: results[x]["final_acc"])
        best_ppl_tok = min(results.keys(), key=lambda x: results[x]["final_ppl"])
        
        print(f"\n🏆 Best Results:")
        print(f"   Lowest Loss: {best_loss_tok} ({results[best_loss_tok]['final_loss']:.4f})")
        print(f"   Highest Acc: {best_acc_tok} ({results[best_acc_tok]['final_acc']*100:.2f}%)")
        print(f"   Lowest PPL:  {best_ppl_tok} ({results[best_ppl_tok]['final_ppl']:.2f})")
    else:
        print("❌ No successful experiments completed.")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("✅ ===== Tokenizer Comparison Experiment Complete =====\n")
    
    return results

# 📊 ===== Tokenizer Comparison Summary =====
# Tokenizer       Vocab    Loss     Acc (%)  PPL     
# -------------------------------------------------------
# hftokenizer     151643   4.0864   31.15    59.53   
# sp-unigram      50000    3.6963   34.25    40.30   
# tiktoken-bpe    100277   4.0721   31.35    58.68   
# char            70       1.3486   57.98    3.85    
# word            75389    4.1900   36.28    66.02   

# 🏆 Best Results:
#    Lowest Loss: char (1.3486)
#    Highest Acc: char (57.98%)
#    Lowest PPL:  char (3.85)



# ============================================================
# QWEN 2.5–3B FINE-TUNING EXPERIMENT
# ============================================================
def run_qwen_finetune_experiment(
    hf_name="OpenAssistant/oasst1",
    txt_files=None,
    vocab_size=None, #400, #8000,
    seq_len=256, #512,
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
    args.stride = 64 #256 #reduce stride means more data pairs
    args.max_train_samples = None #not actually used
    args.batch_size = batch_size
    args.mode = "teacher-forced"

    print(f"📘 Building predictive typing dataset from '{hf_name}'...")
    data = build_dataset(args.task, args)
    train_loader, val_loader = data.loaders()
    inspect_dataset(data)
    #test_tokenizers(data)
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
    # 1. Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    print("✅ All parameters frozen.")

    # 2. Unfreeze the last two transformer blocks and the language modeling head
    # The Qwen2.5 model structure typically has a 'model.model.layers' attribute for transformer blocks
    # and 'model.lm_head' for the final output layer.
    num_layers_to_unfreeze = 2
    total_layers = len(model.model.layers)

    # Unfreeze the last 'num_layers_to_unfreeze' layers
    for i in range(total_layers - num_layers_to_unfreeze, total_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = True
        print(f"✅ Unfroze layer {i}.")

    # Unfreeze the language modeling head (the final prediction layer)
    for param in model.lm_head.parameters():
        param.requires_grad = True
    print("✅ Unfroze language modeling head.")

    # Verify which layers are trainable (optional)
    print("\nTrainable parameters check:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"🔹 {name} is trainable.")
        # else:
        #     print(f"🔸 {name} is frozen.")

    # After this, you can proceed with your fine-tuning setup (e.g., configure optimizer, data, and trainer).
    # Make sure your optimizer is initialized *after* setting the requires_grad flags.
    # The optimizer should only include parameters where requires_grad is True.
    #trainable_params = filter(lambda p: p.requires_grad, model.parameters())


    #model.resize_token_embeddings(len(tokenizer))
    print("✅ Model and tokenizer ready.\n")
    tokenizer = data.tok.tokenizer
    # Sanity: make sure nobody shrank the vocab to 400
    print("tokenizer:", len(tokenizer))
    print("embed:", model.get_input_embeddings().num_embeddings)
    print("lm_head:", model.lm_head.weight.shape)
    #Qwen2.5-3B’s model vocab is typically 151,936.
	#Your tokenizer length is 151,665, which is a bit smaller because Qwen ships with extra reserved tokens (e.g., <|extra_0|>… placeholders, tool tokens, multimodal markers) that live in the model’s embedding table but aren’t all exposed by the tokenizer by default.
    
    evaluator = Evaluator(model, data, mode="teacher-forced", hf_model=True)
    val_loss, val_acc, val_ppl = evaluator.evaluate(split="valid")
    print(f"\n✅ Qwen2.5-3B Original — loss={val_loss:.4f} | acc={val_acc*100:.2f}% | ppl={val_ppl:.2f}\n")

    
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
    run_qwen_finetune_experiment()
    # run_char_typing_experiment()
    
    # Run the new tokenizer comparison experiment
    # run_tokenizer_comparison_experiment(
    #     hf_name="npvinHnivqn/EnglishDictionary",
    #     seq_len=64,
    #     batch_size=128, #32,
    #     epochs=4,  # Reduced for faster testing
    #     output_dir="outputs/tokenizer_comparison"
    # )