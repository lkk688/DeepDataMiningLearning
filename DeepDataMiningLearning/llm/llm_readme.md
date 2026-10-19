# LLM teaching code

| Start here | |
|---|---|
| **Students with a laptop / Mac** | [`LAPTOP_QUICKSTART.md`](LAPTOP_QUICKSTART.md): download, run, expected time / memory / results |
| **The course** | [`TUTORIAL_nanolm.md`](TUTORIAL_nanolm.md). Part 1: LM basics, RNN → LSTM → Transformer, tokenizers, sampling. Part 2: pre-training on web text, using / SFT / DPO-tuning pretrained LLMs. Part 3: Parameter Golf. Part 4: methodology |
| **Serving big models on a Mac** | [`TUTORIAL_mac_inference.md`](TUTORIAL_mac_inference.md): Qwen3.8-27B on Apple Silicon with MLX-LM, mlx-vlm, oMLX and lithos-metal; memory, prefill vs decode, speculative decoding, prefix caching, vision |
| **The code** | [`nanolm/`](nanolm/README.md): one config-driven package for every part (CUDA / Apple MPS / CPU) |
| the previous course code | [`legacy/`](legacy/README.md), with a map from each old function to its replacement |

---

## Legacy run log (char-level typing experiment with the old pipeline, `legacy/experiments.py`)



(py312) lkk@newalienware:~/Developer/DeepDataMiningLearning$  cd /home/lkk/Developer/DeepDataMiningLearning ; /usr/bin/env /home/lkk/miniconda3/envs/py312/bin/python /home/lkk/.vscode-server/extensions/ms-python.debugpy-2025.14.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 52291 -- /home/lkk/Developer/DeepDataMiningLearning/DeepDataMiningLearning/llm/train_lm.py 

🚀 ===== Character-Level Typing Experiment =====
📘 Building char-level typing dataset from 'npvinHnivqn/EnglishDictionary'...
📗 Building DataModule for task='lm' (tokenizer=char, seq_len=64)
📚 Loading Hugging Face dataset: 'npvinHnivqn/EnglishDictionary'
🧩 Detected configs: ['default']
✅ Using first available config: 'default'
📗 Using split: 'train'
🔤 Using text field(s): ['word', 'definition']
✅ Loaded 223,200 text entries from 'npvinHnivqn/EnglishDictionary' (config=default)
🧹 Cleaning raw text ...
✅ clean_texts: kept 235,572/235,689 lines (99.95% retained)
✅ Cleaned text length: 13,748,802 chars (235,572 lines kept)
🔤 Setting up tokenizer: char
✅ Tokenizer initialized | vocab_size=69 | pad_id=0
🧩 Building LM dataset ...
🧩 Encoding text in chunks (chunk_size=50000) ...
Encoding: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 12.21it/s]
✅ Encoded 13,748,803 tokens total
⚙️  Using sliding window stride=1
📏 Built 13,748,739 sequences of len=64
🧾 Sample types: type(sequences[0])=<class 'list'>
  Inner element type: <class 'int'>
✅ Dataset split: 12,373,865 train | 1,374,874 val sequences

🔍 ===== DATASET INSPECTION =====
🧩 Task type: lm
📦 Train batches: 193341 | Validation batches: 21483
🔡 Vocabulary size: 69

🧩 --- Inspecting train batch 1/2 ---
🔹 Batch type: tuple (len=3)
  • tensor[0]: shape=(64, 63), dtype=torch.int64
  • tensor[1]: shape=(64, 63), dtype=torch.int64
  • tensor[2]: shape=(64,), dtype=torch.int64
  • Avg length: 62.0 | Min: 57 | Max: 63
  • Pad ratio: 1.61%  (pad_id=0)
  • Label unique tokens: 40
  • Label vocab coverage: 57.97% of vocab
  • Label pad ratio: 1.59%
  • Top 5 label IDs: [(np.int64(1), 580), (np.int64(41), 333), (np.int64(51), 291), (np.int64(37), 290), (np.int64(56), 256)]

🗣️ Decoded examples:
   1. tellation; -- called also informed, or unformed, stars.sporadi
      → target: ellation; -- called also informed, or unformed, stars.sporadia
   2. ; the roll of waves.roll (v.) that which rolls; a roller.roll
      → target:  the roll of waves.roll (v.) that which rolls; a roller.roll 

🧩 --- Inspecting train batch 2/2 ---
🔹 Batch type: tuple (len=3)
  • tensor[0]: shape=(64, 63), dtype=torch.int64
  • tensor[1]: shape=(64, 63), dtype=torch.int64
  • tensor[2]: shape=(64,), dtype=torch.int64
  • Avg length: 61.8 | Min: 58 | Max: 63
  • Pad ratio: 1.84%  (pad_id=0)
  • Label unique tokens: 36
  • Label vocab coverage: 52.17% of vocab
  • Label pad ratio: 1.84%
  • Top 5 label IDs: [(np.int64(1), 580), (np.int64(41), 373), (np.int64(51), 318), (np.int64(37), 270), (np.int64(56), 264)]

🗣️ Decoded examples:
   1. rity; especially, a renunciation of allegiance and subjection t
      → target: ity; especially, a renunciation of allegiance and subjection to
   2. arts of enticing or corrupting.seducerone who, or that which,
      → target: rts of enticing or corrupting.seducerone who, or that which, 

📊 ===== DATASET SUMMARY =====
  • Training samples: Unknown
  • Validation samples: Unknown
✅ Dataset inspection complete.

✅ Dataset ready for training.


🧠 Training RNN model...
🧠 Initializing traditional RNN Language Model...
📅 Total training steps: 1160046
📉 Using ReduceLROnPlateau scheduler (adaptive LR on validation loss)

🔧 Using AMP: False | dtype: torch.bfloat16 | scaler: False

🚀 Training for 6 epochs on cuda (mode=teacher-forced)
Epoch 1/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [12:06<00:00, 266.22it/s, loss=2.0000, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:46<00:00, 462.40it/s]
✅ Eval complete | loss=2.4405 | acc=42.31% | ppl=11.48

✅ Epoch 1 done | val_loss=2.4405 | acc=42.31% | ppl=11.48
💾 Saved best model → checkpoints/rnn_char_typing.pt
Epoch 2/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [12:18<00:00, 261.77it/s, loss=1.9727, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:46<00:00, 465.00it/s]
✅ Eval complete | loss=2.4864 | acc=42.38% | ppl=12.02

✅ Epoch 2 done | val_loss=2.4864 | acc=42.38% | ppl=12.02
⚠️ No improvement for 1 epochs.
Epoch 3/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [12:22<00:00, 260.38it/s, loss=1.9687, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:45<00:00, 468.46it/s]
✅ Eval complete | loss=2.4799 | acc=42.45% | ppl=11.94

✅ Epoch 3 done | val_loss=2.4799 | acc=42.45% | ppl=11.94
⚠️ No improvement for 2 epochs.
Epoch 4/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [12:46<00:00, 252.15it/s, loss=1.9648, lr=5.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:46<00:00, 464.12it/s]
✅ Eval complete | loss=2.4773 | acc=42.54% | ppl=11.91

✅ Epoch 4 done | val_loss=2.4773 | acc=42.54% | ppl=11.91
⚠️ No improvement for 3 epochs.
⏹️ Early stopping criterion met.
⏹️ Early stopping triggered at epoch 5.

✅ Training complete.

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:46<00:00, 461.83it/s]
✅ Eval complete | loss=2.4773 | acc=42.54% | ppl=11.91

✅ RNN — loss=2.4773 | acc=42.54% | ppl=11.91


🧠 Training LSTM model...
🧠 Initializing LSTM Language Model...
📅 Total training steps: 1160046
📉 Using ReduceLROnPlateau scheduler (adaptive LR on validation loss)

🔧 Using AMP: False | dtype: torch.bfloat16 | scaler: False

🚀 Training for 6 epochs on cuda (mode=teacher-forced)
Epoch 1/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [13:10<00:00, 244.70it/s, loss=1.7851, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:47<00:00, 456.33it/s]
✅ Eval complete | loss=2.5260 | acc=48.17% | ppl=12.50

✅ Epoch 1 done | val_loss=2.5260 | acc=48.17% | ppl=12.50
💾 Saved best model → checkpoints/lstm_char_typing.pt
Epoch 2/6: 100%|███████████████████████████████████████████████████████████████████| 193341/193341 [52:19<00:00, 61.57it/s, loss=1.7516, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:45<00:00, 467.69it/s]
✅ Eval complete | loss=2.8844 | acc=48.49% | ppl=17.89

✅ Epoch 2 done | val_loss=2.8844 | acc=48.49% | ppl=17.89
⚠️ No improvement for 1 epochs.
Epoch 3/6: 100%|█████████████████████████████████████████████████████████████████| 193341/193341 [1:15:35<00:00, 42.63it/s, loss=1.7460, lr=1.000000e-03]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:45<00:00, 476.76it/s]
✅ Eval complete | loss=3.0996 | acc=48.56% | ppl=22.19

✅ Epoch 3 done | val_loss=3.0996 | acc=48.56% | ppl=22.19
⚠️ No improvement for 2 epochs.
Epoch 4/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [24:13<00:00, 133.00it/s, loss=1.7411, lr=5.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:45<00:00, 477.10it/s]
✅ Eval complete | loss=3.2053 | acc=48.63% | ppl=24.66

✅ Epoch 4 done | val_loss=3.2053 | acc=48.63% | ppl=24.66
⚠️ No improvement for 3 epochs.
⏹️ Early stopping criterion met.
⏹️ Early stopping triggered at epoch 5.

✅ Training complete.

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:46<00:00, 462.34it/s]
✅ Eval complete | loss=3.2053 | acc=48.63% | ppl=24.66

✅ LSTM — loss=3.2053 | acc=48.63% | ppl=24.66


🧠 Training TraditionalTransformerLM model...
🧩 Initializing Traditional Transformer (LayerNorm + GELU + AbsPos)
📅 Total training steps: 1160046
📉 Using ReduceLROnPlateau scheduler (adaptive LR on validation loss)

🔧 Using AMP: False | dtype: torch.bfloat16 | scaler: False

🚀 Training for 6 epochs on cuda (mode=teacher-forced)
Epoch 1/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [22:02<00:00, 146.20it/s, loss=1.4642, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:49<00:00, 434.42it/s]
✅ Eval complete | loss=1.5357 | acc=57.01% | ppl=4.64

✅ Epoch 1 done | val_loss=1.5357 | acc=57.01% | ppl=4.64
💾 Saved best model → checkpoints/traditionaltransformerlm_char_typing.pt
Epoch 2/6: 100%|█████████████████████████████████████████████████████████████████| 193341/193341 [2:01:19<00:00, 26.56it/s, loss=1.4095, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:50<00:00, 425.43it/s]
✅ Eval complete | loss=1.5299 | acc=57.26% | ppl=4.62

✅ Epoch 2 done | val_loss=1.5299 | acc=57.26% | ppl=4.62
💾 Saved best model → checkpoints/traditionaltransformerlm_char_typing.pt
Epoch 3/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [22:02<00:00, 146.21it/s, loss=1.3980, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 409.83it/s]
✅ Eval complete | loss=1.5331 | acc=57.44% | ppl=4.63

✅ Epoch 3 done | val_loss=1.5331 | acc=57.44% | ppl=4.63
⚠️ No improvement for 1 epochs.
Epoch 4/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [22:19<00:00, 144.36it/s, loss=1.3917, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:49<00:00, 430.82it/s]
✅ Eval complete | loss=1.5358 | acc=57.53% | ppl=4.65

✅ Epoch 4 done | val_loss=1.5358 | acc=57.53% | ppl=4.65
⚠️ No improvement for 2 epochs.
Epoch 5/6: 100%|███████████████████████████████████████████████████████████████████| 193341/193341 [59:13<00:00, 54.41it/s, loss=1.3760, lr=4.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:49<00:00, 430.20it/s]
✅ Eval complete | loss=1.5285 | acc=57.83% | ppl=4.61

✅ Epoch 5 done | val_loss=1.5285 | acc=57.83% | ppl=4.61
💾 Saved best model → checkpoints/traditionaltransformerlm_char_typing.pt
Epoch 6/6: 100%|█████████████████████████████████████████████████████████████████| 193341/193341 [1:20:01<00:00, 40.27it/s, loss=1.3752, lr=4.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:51<00:00, 416.28it/s]
✅ Eval complete | loss=1.5292 | acc=57.75% | ppl=4.61

✅ Epoch 6 done | val_loss=1.5292 | acc=57.75% | ppl=4.61
⚠️ No improvement for 1 epochs.

✅ Training complete.

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:51<00:00, 416.47it/s]
✅ Eval complete | loss=1.5292 | acc=57.75% | ppl=4.61

✅ TraditionalTransformerLM — loss=1.5292 | acc=57.75% | ppl=4.61


🧠 Training TransformerLM model...
🚀 Initializing modern TransformerLM (RMSNorm + RoPE + SwiGLU)
📅 Total training steps: 1160046
📉 Using ReduceLROnPlateau scheduler (adaptive LR on validation loss)

🔧 Using AMP: False | dtype: torch.bfloat16 | scaler: False

🚀 Training for 6 epochs on cuda (mode=teacher-forced)
Epoch 1/6: 100%|███████████████████████████████████████████████████████████████████| 193341/193341 [34:31<00:00, 93.34it/s, loss=1.4176, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 405.93it/s]
✅ Eval complete | loss=1.5758 | acc=56.83% | ppl=4.83

✅ Epoch 1 done | val_loss=1.5758 | acc=56.83% | ppl=4.83
💾 Saved best model → checkpoints/transformerlm_char_typing.pt
Epoch 2/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [22:57<00:00, 140.38it/s, loss=1.3577, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 407.41it/s]
✅ Eval complete | loss=1.5594 | acc=57.21% | ppl=4.76

✅ Epoch 2 done | val_loss=1.5594 | acc=57.21% | ppl=4.76
💾 Saved best model → checkpoints/transformerlm_char_typing.pt
Epoch 3/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [22:55<00:00, 140.56it/s, loss=1.3478, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 409.63it/s]
✅ Eval complete | loss=1.5641 | acc=57.30% | ppl=4.78

✅ Epoch 3 done | val_loss=1.5641 | acc=57.30% | ppl=4.78
⚠️ No improvement for 1 epochs.
Epoch 4/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [23:11<00:00, 138.95it/s, loss=1.3428, lr=8.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 407.50it/s]
✅ Eval complete | loss=1.5631 | acc=57.37% | ppl=4.77

✅ Epoch 4 done | val_loss=1.5631 | acc=57.37% | ppl=4.77
⚠️ No improvement for 2 epochs.
Epoch 5/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [23:07<00:00, 139.33it/s, loss=1.3269, lr=4.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:51<00:00, 414.28it/s]
✅ Eval complete | loss=1.5574 | acc=57.79% | ppl=4.75

✅ Epoch 5 done | val_loss=1.5574 | acc=57.79% | ppl=4.75
💾 Saved best model → checkpoints/transformerlm_char_typing.pt
Epoch 6/6: 100%|██████████████████████████████████████████████████████████████████| 193341/193341 [23:25<00:00, 137.54it/s, loss=1.3272, lr=4.000000e-04]

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 406.99it/s]
✅ Eval complete | loss=1.5616 | acc=57.71% | ppl=4.77

✅ Epoch 6 done | val_loss=1.5616 | acc=57.71% | ppl=4.77
⚠️ No improvement for 1 epochs.

✅ Training complete.

🔍 Evaluating split = valid ...
Eval valid: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 21483/21483 [00:52<00:00, 411.41it/s]
✅ Eval complete | loss=1.5617 | acc=57.71% | ppl=4.77

✅ TransformerLM — loss=1.5617 | acc=57.71% | ppl=4.77