#modified based on https://github.com/lkk688/MultiModalClassifier/blob/main/TorchClassifier/myTorchModels/TorchCNNmodels.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#new approach: https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/
from torchvision.models import get_model, get_model_weights, get_weight, list_models
#print("Torch buildin models:", list_models())
model_names=list_models(module=torchvision.models)
#print("Torchvision buildin models:", model_names)

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    # !pip install -q torchinfo
    # from torchinfo import summary

def create_torchclassifiermodel(model_name, numclasses=None, model_type='torchvision', torchhublink=None, freezeparameters=True, pretrained=True, dropoutp=0.2):
    pretrained_model = None
    preprocess = None
    imagenet_classes = None
    #different model_type: 'torchvision', 'torchhub'
    if model_type == 'torchvision':
        model_names=list_models(module=torchvision.models)
        if model_name in model_names:
            # Step 1: Initialize model with the best available weights
            weights_enum = get_model_weights(model_name)
            weights = weights_enum.IMAGENET1K_V1
            #print([weight for weight in weights_enum])
            #weights = get_weight("ResNet50_Weights.IMAGENET1K_V2")#ResNet50_Weights.DEFAULT
            if pretrained == True:
                pretrained_model=get_model(model_name, weights=weights)#weights="DEFAULT"
                #pretrained_model=get_model(model_name, weights="DEFAULT")
                # Freeze the base parameters
                if freezeparameters == True :
                    print('Freeze parameters')
                    for parameter in pretrained_model.parameters():
                        parameter.requires_grad = False
            else:
                pretrained_model=get_model(model_name, weights=None)
            #print(pretrained_model)
            # Step 2: Initialize the inference transforms
            preprocess = weights.transforms()#preprocess.crop_size
            imagenet_classes = weights.meta["categories"]
            # Step 3: Apply inference preprocessing transforms
            #batch = preprocess(img).unsqueeze(0)
            if numclasses is not None and len(imagenet_classes) !=numclasses:
                pretrained_model = modify_classifier(pretrained_model=pretrained_model, numclasses=numclasses, dropoutp=dropoutp)
            else:
                numclasses = len(imagenet_classes)
            return pretrained_model
        else:
            print('Model name not exist.')
    elif model_type == 'torchhub' and torchhublink is not None:
        #'deit_base_patch16_224'
        #pretrained_model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        pretrained_model = torch.hub.load(torchhublink, model_name, pretrained=pretrained)
    elif model_type == 'timm':
        import timm #pip install timm
        model_names = timm.list_models(pretrained=True)
        if model_name in model_names:
            #'mobilenetv3_large_100' 'resnet50' 'resnest26d'
            pretrained_model = timm.create_model(model_name, pretrained=pretrained, num_classes=numclasses)
            data_cfg = timm.data.resolve_data_config(pretrained_model.pretrained_cfg)
            preprocess = timm.data.create_transform(**data_cfg)
        else:
            print('Model name not exist.')
        
    return pretrained_model, preprocess, numclasses, imagenet_classes

def modify_classifier(pretrained_model, numclasses, dropoutp=0.3, classifiername=None):
    #display model architecture
    lastmoduleinlist=list(pretrained_model.named_children())[-1]
    #print("lastmoduleinlist len:",len(lastmoduleinlist))
    lastmodulename=lastmoduleinlist[0]
    print("lastmodulename:",lastmodulename)
    lastlayer=lastmoduleinlist[-1]
    if isinstance(lastlayer, nn.Linear):
        print('Linear layer')
        newclassifier = nn.Linear(in_features=lastlayer.in_features, out_features=numclasses)
    elif isinstance(lastlayer, nn.Sequential):
        print('Sequential layer')
        lastlayerlist=list(lastlayer) #[-1] #last layer
        #print("lastlayerlist type:",type(lastlayerlist))
        if isinstance(lastlayerlist, list):
            #print("your object is a list !")
            lastlayer=lastlayerlist[-1]
            newclassifier = torch.nn.Sequential(
                torch.nn.Dropout(p=dropoutp, inplace=True), 
                torch.nn.Linear(in_features=lastlayer.in_features, 
                            out_features=numclasses, # same number of output units as our number of classes
                            bias=True))
        else:
            print("Error: Sequential layer is not list:",lastlayer)
            #newclassifier = nn.Linear(in_features=lastlayer.in_features, out_features=classnum)
    if lastmodulename=='heads':
        pretrained_model.heads = newclassifier #.to(device)
    elif lastmodulename=='classifier':
        pretrained_model.classifier = newclassifier #.to(device)
    elif lastmodulename=='fc':
        pretrained_model.fc = newclassifier #.to(device)
    elif classifiername is not None:
        lastlayer = newclassifier #not tested!!
    else:
        print('Please check the last module name of the model.')
    
    return pretrained_model

if __name__ == "__main__":
    model_name='efficientnet_b1'
    model = create_torchclassifiermodel(model_name, numclasses=None, model_type='torchvision', freezeparameters=False, pretrained=True)
    summary(model=model,
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )