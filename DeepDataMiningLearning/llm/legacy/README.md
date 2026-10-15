# Legacy LLM teaching code (pre-2026-10)

These files are the previous version of the course code. They still run (imports now go through
`DeepDataMiningLearning.llm.legacy.*`), but **new teaching material uses [`../nanolm`](../nanolm/README.md)
and [`../TUTORIAL_nanolm.md`](../TUTORIAL_nanolm.md).** This table says where each piece went.

| old file / function | new home | notes |
|---|---|---|
| `transformer.py: TransformerLM` (RMSNorm+RoPE+SwiGLU) | `nanolm/models/gpt.py: GPT` with `norm: rmsnorm, pos: rope, mlp: swiglu` | one class; each design choice is a config key |
| `transformer.py: TraditionalTransformerLM` (LayerNorm+abs pos+GELU) | `GPT` with `norm: layernorm, pos: learned, mlp: gelu, bias: true` | |
| `transformer.py: RNNLanguageModel / LSTMLanguageModel` | `nanolm/models/rnn.py` (`arch: rnn / gru / lstm / lstm_scratch`) | + a hand-written LSTM cell |
| `transformer.py: FullTransformer, PyTorchTransformer` (encoder-decoder) | not ported | seq2seq is outside the LM course; `FullTransformer`'s shared-layer bug was fixed here |
| `train_lm.py: Trainer, Evaluator` | `nanolm/train.py`, `nanolm/evaluate.py` | step/token budgets instead of epochs; bits-per-byte; cuda/mps/cpu |
| `train_lm.py` HF-model branch | `nanolm/finetune.py` | |
| `lm_dataset.py: DataModule (task=lm)` | `nanolm/data.py` | tokenize once into memory-mapped uint16 shards; no stride-1 window lists |
| `lm_dataset.py` (task=typing) | next-token accuracy (`val_acc`) reported for every run | |
| `lm_dataset.py: InstructionDataset` | `nanolm/finetune.py: chat_example / load_examples` | response-only loss via the model's chat template |
| `lm_dataset.py: Seq2SeqDataModuleHF` | not ported | |
| `tokenizer_utils.py: CharTokenizer` | `data.py: CharTokenizer` (`tokenizer: char`) | |
| `tokenizer_utils.py: WordTokenizer` | `data.py: WordTokenizer` (`tokenizer: word:<V>`) | now lossless: OOV words fall back to bytes, so bpb is exact |
| `tokenizer_utils.py: CustomTokenizer` (SentencePiece / tiktoken) | `tokenizer: sp_train:<bpe|unigram>:<V>`, `sp:<model>`, `gpt2` | trained on the TRAIN split only, lossless settings |
| `tokenizer_utils.py: HFTokenizerWrapper` | `tokenizer: hf:<name>` | exact byte accounting for byte-level BPE |
| `inference_utils.py` | `nanolm/generate.py` | temperature / top-k / top-p, interactive mode |
| `experiments.py: run_char_typing_experiment` | `configs/01_char_rnn.yaml`, `configs/02_*.yaml`, `sweeps/01_char_models.yaml` | |
| `experiments.py: run_tokenizer_comparison_experiment` | `sweeps/02_tokenizers.yaml`, `sweeps/laptop_tokenizers.yaml` | compared in bpb (per-token loss is not comparable across tokenizers) |
| `experiments.py: run_qwen_finetune_experiment` | `configs/part2/P2_sft_alpaca.yaml` | LoRA instead of unfreezing the last 2 layers of a 3B model |
| `rl_train_main.py`, `rl_dataset.py` (TRL: SFT/DPO/GRPO/PPO) | `nanolm/finetune.py` (SFT, DPO written out) | GRPO/PPO not ported: too slow for a laptop course; still runnable here |
| `experiments.ipynb` | `TUTORIAL_nanolm.md` | |
