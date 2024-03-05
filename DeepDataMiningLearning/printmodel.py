try:
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision not available.")
try:
    #https://github.com/TylerYep/torchinfo
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo
    #import os
    #os.system("pip install -q torchinfo")

from torchvision.models import get_model, get_model_weights, get_weight, list_models
model_names=list_models(module=torchvision.models)
print("Torchvision buildin models:", model_names)

model_resnet50 = get_model('resnet50', weights="DEFAULT")
summary(model=model_resnet50, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 