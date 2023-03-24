from PIL import Image #can solve the error of Glibc
import torch
from pprint import pprint
import os
os.environ['TORCH_HOME'] = '/data/cmpe249-fa22/torchhome/'

import timm #pip install timm
m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()
x = torch.randn(1, 3, 224, 224)
features = m.forward_features(x)
print(features.shape)

m = timm.create_model('resnet50', pretrained=True)

m = timm.create_model('resnest26d', pretrained=True)



model_names = timm.list_models(pretrained=True)
pprint(model_names)