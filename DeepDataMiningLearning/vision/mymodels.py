import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from transformers import ViTForImageClassification, AutoModelForImageClassification

# Torchvision buildin models: ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', \
#                              'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', \
#                              'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', \
#                              'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', \
#                              'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', \
#                             'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', \
#                             'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', \
#                             'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', \
#                             'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', \
#                             'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', \
#                             'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', \
#                             'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', \
#                             'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t',\
#                             'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 
#                             'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']
def create_torchvisionmodel(modulename, numclasses, freezeparameters=True, pretrained=True, dropoutp=0.2):
    model_names=list_models(module=models)
    if modulename in model_names:
        if pretrained == True:
            pretrained_model=get_model(modulename, weights="DEFAULT")
            # Freeze the base parameters
            if freezeparameters == True :
                print('Freeze parameters')
                for parameter in pretrained_model.parameters():
                    parameter.requires_grad = False
        else:
            pretrained_model=get_model(modulename, weights=None)
        #print(pretrained_model)
        
        #display model architecture
        lastmoduleinlist=list(pretrained_model.named_children())[-1]
        #print("lastmoduleinlist len:",len(lastmoduleinlist))
        lastmodulename=lastmoduleinlist[0]
        print("lastmodulename:",lastmodulename)
        lastlayer=lastmoduleinlist[-1]
        if isinstance(lastlayer, nn.Linear):
            print('Linear layer')
            newclassifier = nn.Linear(in_features=lastlayer.in_features, out_features=classnum)
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
        else:
            print('Please check the last module name of the model.')
        
        return pretrained_model
    else:
        print('Model name not exist.')
        raise ValueError(f"Unsupported torchvision model: {modulename}")
        
def load_mymodel(model_name, num_classes, source="torchvision"):
    """
    Load a model from either torchvision or Hugging Face transformers library.
    
    Args:
        model_name (str): Name of the model (e.g., "resnet18", "mobilenet_v2", "vit").
        num_classes (int): Number of output classes for the model.
        source (str): Source of the model ("torchvision" or "huggingface").
    
    Returns:
        model: Loaded model with adjusted output layer.
    """
    if source == "torchvision":
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Adjust output layer
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer
        else:
            model = create_torchvisionmodel(model_name, num_classes, freezeparameters=False, pretrained=True)
    
    elif source == "huggingface":
        if model_name == "vit":
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            model.classifier = nn.Linear(model.config.hidden_size, num_classes)  # Adjust output layer
        elif model_name == "deit":
            model = ViTForImageClassification.from_pretrained("facebook/deit-base-patch16-224")
            model.classifier = nn.Linear(model.config.hidden_size, num_classes)  # Adjust output layer
        else:
            try:
                model = AutoModelForImageClassification.from_pretrained(model_name)
            except Exception as e:
                print(f"Error loading model from Hugging Face: {e}")
                raise ValueError(f"Unsupported Hugging Face model: {model_name}")
    
    else:
        raise ValueError(f"Unsupported model source: {source}")
    
    return model

# Testing code for the model
def test_model_loading():
    # Test torchvision model
    print("Testing torchvision ResNet-18 model...")
    model = load_mymodel("resnet18", num_classes=10, source="torchvision")
    print(f"Model architecture: {model}")
    print(f"Output layer: {model.fc}")
    print()
    
    # Test Hugging Face model
    print("Testing Hugging Face ViT model...")
    model = load_mymodel("vit", num_classes=10, source="huggingface")
    print(f"Model architecture: {model}")
    print(f"Output layer: {model.classifier}")

if __name__ == "__main__":
    test_model_loading()