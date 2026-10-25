"""Put on PYTHONPATH by llmtrain's automodel backend so that every child Python process (the automodel CLI and
any torchrun workers) applies llmtrain's attention-kernel choice before training starts. See TrainSpec.cudnn_sdpa."""
import os

if os.environ.get("LLMTRAIN_CUDNN_SDPA") == "0":
    try:
        import torch
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:          # never break interpreter start-up
        pass
