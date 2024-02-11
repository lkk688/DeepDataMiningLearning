#This code should be run in the head node with internet access to download the data
from PIL import Image
import torch
import torchvision
import torchvision.datasets as datasets
import os
mytorchvisiondata='data/'#'/data/cmpe249-fa23/torchvisiondata/'

print(torch.__version__)

from torchvision.transforms import v2

H, W = 32, 32
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img)
print(img.shape)
print(type(img)) #<class 'torch.Tensor'>

# Detection (re-using imports and transforms from above)
from torchvision import tv_tensors
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
boxes = torch.randint(0, H // 2, size=(3, 4))
boxes[:, 2:] += boxes[:, :2]
boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))

# The same transforms can be used!
img, boxes = transforms(img, boxes)
# And you can pass arbitrary input structures
output_dict = transforms({"image": img, "boxes": boxes})


cifar_trainset = datasets.CIFAR10(root=mytorchvisiondata, train=True, download=True, transform=None)
print(cifar_trainset)
#Extracting /data/cmpe249-fa22/torchvisiondata/cifar-10-python.tar.gz to /data/cmpe249-fa22/torchvisiondata/
cifar_testset = datasets.CIFAR10(root=mytorchvisiondata, train=False, download=True, transform=None)
print(cifar_testset)
#Extracting ./data/cifar-10-python.tar.gz to ./data



mnist_testset = datasets.MNIST(root=mytorchvisiondata, train=False, download=True, transform=None)

training_data = datasets.FashionMNIST(
    root=mytorchvisiondata,
    train=True,
    download=True,
    transform=None,
)
test_data = datasets.FashionMNIST(
    root=mytorchvisiondata,
    train=False,
    download=True,
    transform=None,
)

# imagenet_data = datasets.ImageNet(mytorchvisiondata, split = 'train', download = True)
# print(imagenet_data)

#http://places2.csail.mit.edu/index.html
# places365=torchvision.datasets.Places365(mytorchvisiondata, split = 'train-standard', small = False, download = True)
# print(places365)

#Download model
os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/' #setting the environment variable
resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)
resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)#torchvision.models.resnet50(pretrained=True)
print(resnet50)
#Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /data/cmpe249-fa22/torchhome/hub/checkpoints/resnet50-19c8e357.pth