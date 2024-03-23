import json
import os
import requests
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt
from torchvision.io import read_image
#%matplotlib inline
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import torch

def get_device(gpuid='0', useamp=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpuid))  # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        useamp = False
    else:
        device = torch.device("cpu")
        useamp = False
    return device, useamp

def savedict2file(data_dict, filename):
    # save vocab dict to be loaded into tokenizer
    with open(filename, "w") as file:
        json.dump(data_dict, file)

def saveargs2file(args, trainoutput):
    args_dict={}
    args_str=' '
    for k, v in vars(args).items():
        args_dict[k]=v
        args_str.join(f'{k}={v}, ')
    print(args_str)
    savedict2file(data_dict=args_dict, filename=os.path.join(trainoutput,'args.json'))

def save_ImageNetlabels(filepath='sampledata/imagenet_labels.txt'):#https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    # Download human-readable labels for ImageNet.
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")
    #save python lists to txt file
    with open(filepath, 'w') as f:
        for label in labels:
            f.write(label + '\n')
#save_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
#write a function to load labels.txt file into a python list
def load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt'):
    with open(filepath, 'r') as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]#trim string of labels
    return labels
#labels=load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')

#output_format="numpy", "pil" in HWC format
def read_image(image, use_pil=True, use_cv2=False, rgb=True, output_format='numpy', plotfig=False):
    if isinstance(image, Image.Image):
        if output_format == 'numpy':
            # Convert PIL image to NumPy array
            image = np.array(image.convert("RGB"))
        elif output_format == 'pil':
            image = image
    if isinstance(image, str):
        if image.startswith('http'):
            filename = requests.get(image, stream=True).raw
        elif image.endswith('.jpg') or image.endswith('.png'):
            filename = image
        #Read the image to numpy HWC format uint8
        if use_pil:
            #https://pillow.readthedocs.io/en/stable/reference/Image.html
            image = Image.open(filename)
            if rgb:
                image = image.convert('RGB')
                size = image.size #size in pixels, as a 2-tuple: (width, height)
            if output_format == 'numpy':
                image = np.array(image) ##convert PIL image to numpy, HWC format (height, width, color channels)
            elif output_format == 'pil':
                image = image
            #img.height, img.width
            #image = image.astype(np.float32) / 255.0
        elif use_cv2:
            img = cv2.imread(filename) #row (height) x column (width) x color (3) 'numpy.ndarray'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #HWC
            #print(img.dtype) # uint8
            if output_format == 'numpy':
                image = img
            elif output_format == 'pil':
                image = Image.fromarray(img)
        else:
            #torch read_image get format CHW (color channels, height, width) in tensor
            img = read_image(str(filename)) #3, 900, 900
            img = img.numpy().transpose((1, 2, 0)) #CHW (color channels, height, width) to numpy/matplotlib's HWC
            if output_format == 'numpy':
                image = img
            elif output_format == 'pil':
                image = Image.fromarray(img)
    #read the height/width from image's shape
    if isinstance(image, Image.Image):
        height = image.height
        width = image.width
        org_sizeHW = image.size[::-1]#(width, height) (W=640, H=427)->(H=427, W=640)
    elif isinstance(image, np.ndarray): #HWC format
        height = image.shape[0]
        width = image.shape[1]
        org_sizeHW = (height, width) #(427, 640)
    if plotfig:
        if output_format== 'numpy':
            #image = image.astype(np.float32) / 255.0
            # Plot the image, matplotlib also uses HWC format
            plt.figure(figsize=(10, 7))
            plt.imshow(image) #need HWC
            plt.axis("off")
            #plt.title(image_class, fontsize=14)
        elif output_format== 'pil':
            image.show()
        else:
            # Display the image
            cv2.imshow("Image", image)
    return image, org_sizeHW #HWC in numpy or PIL