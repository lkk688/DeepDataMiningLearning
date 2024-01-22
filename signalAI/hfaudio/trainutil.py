import torch
from torch.optim import AdamW

def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    if weight_decay>0:
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    else:
        adam_beta1=0.9
        adam_beta2=0.999
        adam_epsilon=1e-8
        optimizer = AdamW(
            list(model.parameters()),
            lr=learning_rate,
            betas=[adam_beta1, adam_beta2],
            eps=adam_epsilon,
        )
    return optimizer
